# Copyright 2022 The MT3 Authors.
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

"""Tests for run_length_encoding."""

from mt3 import event_codec
from mt3 import run_length_encoding

import note_seq
import seqio
import tensorflow as tf

assert_dataset = seqio.test_utils.assert_dataset
codec = event_codec.Codec(
    max_shift_steps=100,
    steps_per_second=100,
    event_ranges=[
        event_codec.EventRange('pitch', note_seq.MIN_MIDI_PITCH,
                               note_seq.MAX_MIDI_PITCH),
        event_codec.EventRange('velocity', 0, 127),
        event_codec.EventRange('drum', note_seq.MIN_MIDI_PITCH,
                               note_seq.MAX_MIDI_PITCH),
        event_codec.EventRange('program', note_seq.MIN_MIDI_PROGRAM,
                               note_seq.MAX_MIDI_PROGRAM),
        event_codec.EventRange('tie', 0, 0)
    ])
run_length_encode_shifts = run_length_encoding.run_length_encode_shifts_fn(
    codec=codec)
run_length_encode_shifts_drop_redundant = run_length_encoding.run_length_encode_shifts_fn(
    codec=codec, state_change_event_types=['velocity', 'program'])


class RunLengthEncodingTest(tf.test.TestCase):

  def test_run_length_encode_shifts(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'targets': [1, 1, 1, 161, 1, 1, 1, 162, 1, 1, 1]
    })

    assert_dataset(
        run_length_encode_shifts(og_dataset),
        {
            'targets': [3, 161, 6, 162],
        })

  def test_run_length_encode_shifts_beyond_max_length(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'targets': [1] * 202 + [161, 1, 1, 1]
    })

    assert_dataset(
        run_length_encode_shifts(og_dataset),
        {
            'targets': [100, 100, 2, 161],
        })

  def test_run_length_encode_shifts_simultaneous(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'targets': [1, 1, 1, 161, 162, 1, 1, 1]
    })

    assert_dataset(
        run_length_encode_shifts(og_dataset),
        {
            'targets': [3, 161, 162],
        })

  def test_run_length_encode_shifts_drop_redundant(self):
    og_dataset = tf.data.Dataset.from_tensors({
        'targets': [1, 1, 1, 525, 356, 161, 1, 1, 525, 356, 161, 355, 394]
    })

    assert_dataset(
        run_length_encode_shifts_drop_redundant(og_dataset),
        {
            'targets': [3, 525, 356, 161, 5, 161, 355, 394],
        })


if __name__ == '__main__':
  tf.test.main()
