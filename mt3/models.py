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

"""Feature converter and model for continuous inputs."""

from typing import Mapping
import seqio
from t5x import decoding
from t5x import models
import tensorflow as tf


class ContinuousInputsEncDecFeatureConverter(seqio.FeatureConverter):
  """Feature converter for an encoder-decoder with continuous inputs."""

  TASK_FEATURES = {
      "inputs": seqio.FeatureConverter.FeatureSpec(dtype=tf.float32, rank=2),
      "targets": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "encoder_input_tokens":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.float32, rank=2),
      "decoder_target_tokens":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "encoder_segment_ids": tf.int32,
      "decoder_segment_ids": tf.int32,
      "encoder_positions": tf.int32,
      "decoder_positions": tf.int32
  }

  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Convert the dataset to be fed to the encoder-decoder model.

    The conversion process involves three steps

    1. Each feature in the `task_feature_lengths` is trimmed/padded and
       optionally packed depending on the value of self.pack.
    2. "inputs" fields are mapped to the encoder input and "targets" are mapped
       to decoder input (after being shifted) and target.

    All the keys in the `task_feature_lengths` should be present in the input
    dataset, which may contain some extra features that are not in the
    `task_feature_lengths`. They will not be included in the output dataset.
    One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
    fields.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from feature to its length.

    Returns:
      ds: the converted dataset.
    """

    def convert_example(
        features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
      # targets_segment_id is present only for a packed dataset.
      decoder_input_tokens = seqio.autoregressive_inputs(
          features["targets"],
          sequence_id=features.get("targets_segment_ids", None))

      d = {"encoder_input_tokens": features["inputs"],
           "decoder_target_tokens": features["targets"],
           "decoder_input_tokens": decoder_input_tokens,
           # Loss is computed for all but the padding positions.
           "decoder_loss_weights":
               seqio.non_padding_position(features["targets"])}

      if self.pack:
        d["encoder_segment_ids"] = features["inputs_segment_ids"]
        d["decoder_segment_ids"] = features["targets_segment_ids"]
        d["encoder_positions"] = features["inputs_positions"]
        d["decoder_positions"] = features["targets_positions"]

      return d

    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
        convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    encoder_length = task_feature_lengths["inputs"]
    decoder_length = task_feature_lengths["targets"]

    model_feature_lengths = {
        "encoder_input_tokens": encoder_length,
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens": decoder_length,
        "decoder_loss_weights": decoder_length
    }
    if self.pack:
      model_feature_lengths["encoder_segment_ids"] = encoder_length
      model_feature_lengths["decoder_segment_ids"] = decoder_length
      model_feature_lengths["encoder_positions"] = encoder_length
      model_feature_lengths["decoder_positions"] = decoder_length

    return model_feature_lengths


class ContinuousInputsEncoderDecoderModel(models.EncoderDecoderModel):
  """Encoder-decoder model with continuous inputs."""

  FEATURE_CONVERTER_CLS = ContinuousInputsEncDecFeatureConverter

  def __init__(self, module, input_vocabulary, output_vocabulary, optimizer_def,
               input_depth, decode_fn=decoding.beam_search, label_smoothing=0.0,
               z_loss=0.0, loss_normalizing_factor=None):
    super().__init__(
        module=module,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor)
    self._input_depth = input_depth

  def get_initial_variables(self, rng, input_shapes, input_types=None):
    """Hacky override to bypass eval/infer inability to handle rank-3 inputs."""
    encoder_shape = input_shapes["encoder_input_tokens"]
    if len(encoder_shape) == 2:
      input_shapes = {
          "encoder_input_tokens": (*encoder_shape, self._input_depth),
          **{k: v for k, v in input_shapes.items()
             if k != "encoder_input_tokens"}
      }
    else:
      assert encoder_shape[-1] == self._input_depth
    return super().get_initial_variables(
        rng=rng, input_shapes=input_shapes, input_types=input_types)
