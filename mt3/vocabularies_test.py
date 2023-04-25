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

"""Tests for vocabularies."""

from absl.testing import absltest
from mt3 import vocabularies

import numpy as np
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()


class VocabulariesTest(absltest.TestCase):

  def test_velocity_quantization(self):
    self.assertEqual(0, vocabularies.velocity_to_bin(0, num_velocity_bins=1))
    self.assertEqual(0, vocabularies.velocity_to_bin(0, num_velocity_bins=127))
    self.assertEqual(0, vocabularies.bin_to_velocity(0, num_velocity_bins=1))
    self.assertEqual(0, vocabularies.bin_to_velocity(0, num_velocity_bins=127))

    self.assertEqual(
        1,
        vocabularies.velocity_to_bin(
            vocabularies.bin_to_velocity(1, num_velocity_bins=1),
            num_velocity_bins=1))

    for velocity_bin in range(1, 128):
      self.assertEqual(
          velocity_bin,
          vocabularies.velocity_to_bin(
              vocabularies.bin_to_velocity(velocity_bin, num_velocity_bins=127),
              num_velocity_bins=127))

  def test_encode_decode(self):
    vocab = vocabularies.GenericTokenVocabulary(32)
    input_tokens = [1, 2, 3]
    expected_encoded = [4, 5, 6]

    # Encode
    self.assertSequenceEqual(vocab.encode(input_tokens), expected_encoded)
    np.testing.assert_array_equal(
        vocab.encode_tf(tf.convert_to_tensor(input_tokens)).numpy(),
        expected_encoded)

    # Decode
    self.assertSequenceEqual(vocab.decode(expected_encoded), input_tokens)
    np.testing.assert_array_equal(
        vocab.decode_tf(tf.convert_to_tensor(expected_encoded)).numpy(),
        input_tokens)

  def test_decode_invalid_ids(self):
    vocab = vocabularies.GenericTokenVocabulary(32, extra_ids=4)
    encoded = [0, 2, 3, 4, 34, 35]
    expected_decoded = [-2, -2, 0, 1, 31, -2]
    self.assertSequenceEqual(vocab.decode(encoded), expected_decoded)
    np.testing.assert_array_equal(
        vocab.decode_tf(tf.convert_to_tensor(encoded)).numpy(),
        expected_decoded)

  def test_decode_eos(self):
    vocab = vocabularies.GenericTokenVocabulary(32)
    encoded = [0, 2, 3, 4, 1, 0, 1, 0]
    # Python decode function truncates everything after first EOS.
    expected_decoded = [-2, -2, 0, 1, -1]
    self.assertSequenceEqual(vocab.decode(encoded), expected_decoded)
    # TF decode function preserves array length.
    expected_decoded_tf = [-2, -2, 0, 1, -1, -1, -1, -1]
    np.testing.assert_array_equal(
        vocab.decode_tf(tf.convert_to_tensor(encoded)).numpy(),
        expected_decoded_tf)

  def test_encode_invalid_id(self):
    vocab = vocabularies.GenericTokenVocabulary(32)
    inputs = [0, 15, 31]
    # No exception expected.
    vocab.encode(inputs)
    vocab.encode_tf(tf.convert_to_tensor(inputs))

    inputs_too_low = [-1, 15, 31]
    with self.assertRaises(ValueError):
      vocab.encode(inputs_too_low)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      vocab.encode_tf(tf.convert_to_tensor(inputs_too_low))

    inputs_too_high = [0, 15, 32]
    with self.assertRaises(ValueError):
      vocab.encode(inputs_too_high)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      vocab.encode_tf(tf.convert_to_tensor(inputs_too_high))

  def test_encode_dtypes(self):
    vocab = vocabularies.GenericTokenVocabulary(32)
    inputs = [0, 15, 31]
    encoded32 = vocab.encode_tf(tf.convert_to_tensor(inputs, tf.int32))
    self.assertEqual(tf.int32, encoded32.dtype)
    encoded64 = vocab.encode_tf(tf.convert_to_tensor(inputs, tf.int64))
    self.assertEqual(tf.int64, encoded64.dtype)


if __name__ == '__main__':
  absltest.main()
