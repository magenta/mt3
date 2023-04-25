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

"""Functions for mixing (in the audio sense) multiple transcription examples."""

from typing import Callable, Optional, Sequence

import gin

from mt3 import event_codec
from mt3 import run_length_encoding

import numpy as np
import seqio
import tensorflow as tf


@gin.configurable
def mix_transcription_examples(
    ds: tf.data.Dataset,
    sequence_length: seqio.preprocessors.SequenceLengthType,
    output_features: seqio.preprocessors.OutputFeaturesType,
    codec: event_codec.Codec,
    inputs_feature_key: str = 'inputs',
    targets_feature_keys: Sequence[str] = ('targets',),
    max_examples_per_mix: Optional[int] = None,
    shuffle_buffer_size: int = seqio.SHUFFLE_BUFFER_SIZE
) -> Callable[..., tf.data.Dataset]:
  """Preprocessor that mixes together "batches" of transcription examples.

  Args:
    ds: Dataset of individual transcription examples, each of which should
        have an 'inputs' field containing 1D audio samples (currently only
        audio encoders that use raw samples as an intermediate representation
        are supported), and a 'targets' field containing run-length encoded
        note events.
    sequence_length: Dictionary mapping feature key to length.
    output_features: Dictionary mapping feature key to spec.
    codec: An event_codec.Codec used to interpret the target events.
    inputs_feature_key: Feature key for inputs which will be mixed as audio.
    targets_feature_keys: List of feature keys for targets, each of which will
        be merged (separately) as run-length encoded note events.
    max_examples_per_mix: Maximum number of individual examples to mix together.
    shuffle_buffer_size: Size of shuffle buffer to use for shuffle prior to
        mixing.

  Returns:
    Dataset containing mixed examples.
  """
  if max_examples_per_mix is None:
    return ds

  # TODO(iansimon): is there a way to use seqio's seed?
  ds = tf.data.Dataset.sample_from_datasets([
      ds.shuffle(
          buffer_size=shuffle_buffer_size // max_examples_per_mix
      ).padded_batch(batch_size=i) for i in range(1, max_examples_per_mix + 1)
  ])

  def mix_inputs(ex):
    samples = tf.reduce_sum(ex[inputs_feature_key], axis=0)
    norm = tf.linalg.norm(samples, ord=np.inf)
    ex[inputs_feature_key] = tf.math.divide_no_nan(samples, norm)
    return ex
  ds = ds.map(mix_inputs, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  max_tokens = sequence_length['targets']
  if output_features['targets'].add_eos:
    # Leave room to insert an EOS token.
    max_tokens -= 1

  def mix_targets(ex):
    for k in targets_feature_keys:
      ex[k] = run_length_encoding.merge_run_length_encoded_targets(
          targets=ex[k],
          codec=codec)
    return ex
  ds = ds.map(mix_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return ds
