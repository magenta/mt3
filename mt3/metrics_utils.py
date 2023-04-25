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

"""Utilities for transcription metrics."""

import collections
import functools

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar

from mt3 import event_codec
from mt3 import note_sequences
from mt3 import run_length_encoding

import note_seq
import numpy as np
import pretty_midi
import sklearn

S = TypeVar('S')
T = TypeVar('T')

CombineExamplesFunctionType = Callable[[Sequence[Mapping[str, Any]]],
                                       Mapping[str, Any]]


def _group_predictions_by_id(
    predictions: Sequence[Mapping[str, T]]
) -> Mapping[str, Sequence[T]]:
  predictions_by_id = collections.defaultdict(list)
  for pred in predictions:
    predictions_by_id[pred['unique_id']].append(pred)
  return predictions_by_id


def combine_predictions_by_id(
    predictions: Sequence[Mapping[str, Any]],
    combine_predictions_fn: CombineExamplesFunctionType
) -> Mapping[str, Mapping[str, Any]]:
  """Concatenate predicted examples, grouping by ID and sorting by time."""
  predictions_by_id = _group_predictions_by_id(predictions)
  return {
      id: combine_predictions_fn(preds)
      for id, preds in predictions_by_id.items()
  }


def decode_and_combine_predictions(
    predictions: Sequence[Mapping[str, Any]],
    init_state_fn: Callable[[], S],
    begin_segment_fn: Callable[[S], None],
    decode_tokens_fn: Callable[[S, Sequence[int], int, Optional[int]],
                               Tuple[int, int]],
    flush_state_fn: Callable[[S], T]
) -> Tuple[T, int, int]:
  """Decode and combine a sequence of predictions to a full result.

  For time-based events, this usually means concatenation.

  Args:
    predictions: List of predictions, each of which is a dictionary containing
        estimated tokens ('est_tokens') and start time ('start_time') fields.
    init_state_fn: Function that takes no arguments and returns an initial
        decoding state.
    begin_segment_fn: Function that updates the decoding state at the beginning
        of a segment.
    decode_tokens_fn: Function that takes a decoding state, estimated tokens
        (for a single segment), start time, and max time, and processes the
        tokens, updating the decoding state in place. Also returns the number of
        invalid and dropped events for the segment.
    flush_state_fn: Function that flushes the final decoding state into the
        result.

  Returns:
    result: The full combined decoding.
    total_invalid_events: Total number of invalid event tokens across all
        predictions.
    total_dropped_events: Total number of dropped event tokens across all
        predictions.
  """
  sorted_predictions = sorted(predictions, key=lambda pred: pred['start_time'])

  state = init_state_fn()
  total_invalid_events = 0
  total_dropped_events = 0

  for pred_idx, pred in enumerate(sorted_predictions):
    begin_segment_fn(state)

    # Depending on the audio token hop length, each symbolic token could be
    # associated with multiple audio frames. Since we split up the audio frames
    # into segments for prediction, this could lead to overlap. To prevent
    # overlap issues, ensure that the current segment does not make any
    # predictions for the time period covered by the subsequent segment.
    max_decode_time = None
    if pred_idx < len(sorted_predictions) - 1:
      max_decode_time = sorted_predictions[pred_idx + 1]['start_time']

    invalid_events, dropped_events = decode_tokens_fn(
        state, pred['est_tokens'], pred['start_time'], max_decode_time)

    total_invalid_events += invalid_events
    total_dropped_events += dropped_events

  return flush_state_fn(state), total_invalid_events, total_dropped_events


def event_predictions_to_ns(
    predictions: Sequence[Mapping[str, Any]], codec: event_codec.Codec,
    encoding_spec: note_sequences.NoteEncodingSpecType
) -> Mapping[str, Any]:
  """Convert a sequence of predictions to a combined NoteSequence."""
  ns, total_invalid_events, total_dropped_events = decode_and_combine_predictions(
      predictions=predictions,
      init_state_fn=encoding_spec.init_decoding_state_fn,
      begin_segment_fn=encoding_spec.begin_decoding_segment_fn,
      decode_tokens_fn=functools.partial(
          run_length_encoding.decode_events,
          codec=codec,
          decode_event_fn=encoding_spec.decode_event_fn),
      flush_state_fn=encoding_spec.flush_decoding_state_fn)

  # Also concatenate raw inputs from all predictions.
  sorted_predictions = sorted(predictions, key=lambda pred: pred['start_time'])
  raw_inputs = np.concatenate(
      [pred['raw_inputs'] for pred in sorted_predictions], axis=0)
  start_times = [pred['start_time'] for pred in sorted_predictions]

  return {
      'raw_inputs': raw_inputs,
      'start_times': start_times,
      'est_ns': ns,
      'est_invalid_events': total_invalid_events,
      'est_dropped_events': total_dropped_events,
  }


def get_prettymidi_pianoroll(ns: note_seq.NoteSequence, fps: float,
                             is_drum: bool):
  """Convert NoteSequence to pianoroll through pretty_midi."""
  for note in ns.notes:
    if is_drum or note.end_time - note.start_time < 0.05:
      # Give all drum notes a fixed length, and all others a min length
      note.end_time = note.start_time + 0.05

  pm = note_seq.note_sequence_to_pretty_midi(ns)
  end_time = pm.get_end_time()
  cc = [
      # all sound off
      pretty_midi.ControlChange(number=120, value=0, time=end_time),
      # all notes off
      pretty_midi.ControlChange(number=123, value=0, time=end_time)
  ]
  pm.instruments[0].control_changes = cc
  if is_drum:
    # If inst.is_drum is set, pretty_midi will return an all zero pianoroll.
    for inst in pm.instruments:
      inst.is_drum = False
  pianoroll = pm.get_piano_roll(fs=fps)
  return pianoroll


def frame_metrics(ref_pianoroll: np.ndarray,
                  est_pianoroll: np.ndarray,
                  velocity_threshold: int) -> Tuple[float, float, float]:
  """Frame Precision, Recall, and F1."""
  # Pad to same length
  if ref_pianoroll.shape[1] > est_pianoroll.shape[1]:
    diff = ref_pianoroll.shape[1] - est_pianoroll.shape[1]
    est_pianoroll = np.pad(est_pianoroll, [(0, 0), (0, diff)], mode='constant')
  elif est_pianoroll.shape[1] > ref_pianoroll.shape[1]:
    diff = est_pianoroll.shape[1] - ref_pianoroll.shape[1]
    ref_pianoroll = np.pad(ref_pianoroll, [(0, 0), (0, diff)], mode='constant')

  # For ref, remove any notes that are too quiet (consistent with Cerberus.)
  ref_frames_bool = ref_pianoroll > velocity_threshold
  # For est, keep all predicted notes.
  est_frames_bool = est_pianoroll > 0

  precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
      ref_frames_bool.flatten(),
      est_frames_bool.flatten(),
      labels=[True, False])

  return precision[0], recall[0], f1[0]
