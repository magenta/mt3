# Copyright 2023 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for MT3 inference."""

import functools
import json

from typing import Any, Optional, Sequence

import gin

from mt3 import metrics_utils
from mt3 import note_sequences
from mt3 import tasks
from mt3 import vocabularies

import note_seq
import seqio
import tensorflow as tf


def write_inferences_to_file(
    path: str,
    inferences: Sequence[Any],
    task_ds: tf.data.Dataset,
    mode: str,
    vocabulary: Optional[seqio.Vocabulary] = None,
    vocab_config=gin.REQUIRED,
    onsets_only=gin.REQUIRED,
    use_ties=gin.REQUIRED) -> None:
  """Writes model predictions, ground truth transcriptions, and input audio.

  For now this only works for transcription tasks with ties.

  Args:
    path: File path to write to.
    inferences: Model inferences, output of predict_batch.
    task_ds: Original task dataset.
    mode: Prediction mode; must be 'predict' as 'score' is not supported.
    vocabulary: Task output vocabulary.
    vocab_config: Vocabulary config object.
    onsets_only: If True, only predict onsets.
    use_ties: If True, use "tie" representation.
  """
  if mode == 'score':
    raise ValueError('`score` mode currently not supported in MT3')
  if not vocabulary:
    raise ValueError('`vocabulary` parameter required in `predict` mode')

  if onsets_only and use_ties:
    raise ValueError('ties not compatible with onset-only transcription')
  if onsets_only:
    encoding_spec = note_sequences.NoteOnsetEncodingSpec
  elif not use_ties:
    encoding_spec = note_sequences.NoteEncodingSpec
  else:
    encoding_spec = note_sequences.NoteEncodingWithTiesSpec

  codec = vocabularies.build_codec(vocab_config)

  targets = []
  predictions = []

  for inp, output in zip(task_ds.as_numpy_iterator(), inferences):
    tokens = tasks.trim_eos(vocabulary.decode_tf(output).numpy())

    start_time = inp['input_times'][0]
    # Round down to nearest symbolic token step.
    start_time -= start_time % (1 / codec.steps_per_second)

    targets.append({
        'unique_id': inp['unique_id'][0],
        'ref_ns': inp['sequence'][0] if inp['sequence'][0] else None,
    })

    predictions.append({
        'unique_id': inp['unique_id'][0],
        'est_tokens': tokens,
        'start_time': start_time,
        # Input audio is not part of the "prediction" but the below call to
        # metrics_utils.event_predictions_to_ns handles the concatenation.
        'raw_inputs': inp['raw_inputs']
    })

  # The first target for each full example contains the NoteSequence; just
  # organize by ID.
  full_targets = {}
  for target in targets:
    if target['ref_ns']:
      full_targets[target['unique_id']] = {
          'ref_ns': note_seq.NoteSequence.FromString(target['ref_ns'])
      }

  full_predictions = metrics_utils.combine_predictions_by_id(
      predictions=predictions,
      combine_predictions_fn=functools.partial(
          metrics_utils.event_predictions_to_ns,
          codec=codec,
          encoding_spec=encoding_spec))

  assert sorted(full_targets.keys()) == sorted(full_predictions.keys())

  full_target_prediction_pairs = [
      (full_targets[id], full_predictions[id])
      for id in sorted(full_targets.keys())
  ]

  def note_to_dict(note):
    return {
        'start_time': note.start_time,
        'end_time': note.end_time,
        'pitch': note.pitch,
        'velocity': note.velocity,
        'program': note.program,
        'is_drum': note.is_drum
    }

  with tf.io.gfile.GFile(path, 'w') as f:
    for target, prediction in full_target_prediction_pairs:
      json_dict = {
          'id': target['ref_ns'].id,
          'est_notes':
              [note_to_dict(note) for note in prediction['est_ns'].notes]
      }
      json_str = json.dumps(json_dict, cls=seqio.TensorAndNumpyEncoder)
      f.write(json_str + '\n')
