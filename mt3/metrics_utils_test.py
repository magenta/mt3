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

"""Tests for metrics_utils."""

from mt3 import event_codec
from mt3 import metrics_utils
from mt3 import note_sequences

import note_seq
import numpy as np
import tensorflow as tf


class MetricsUtilsTest(tf.test.TestCase):

  def test_event_predictions_to_ns(self):
    predictions = [
        {
            'raw_inputs': [0, 0],
            'start_time': 0.0,
            'est_tokens': [20, 160],
        },
        {
            'raw_inputs': [1, 1],
            'start_time': 0.4,
            # These last 2 events should be dropped.
            'est_tokens': [20, 161, 50, 162],
        },
        {
            'raw_inputs': [2, 2],
            'start_time': 0.8,
            'est_tokens': [163, 20, 164]
        },
    ]
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=59,
        velocity=100,
        start_time=0.20,
        end_time=0.21)
    expected_ns.notes.add(
        pitch=60,
        velocity=100,
        start_time=0.60,
        end_time=0.61)
    expected_ns.notes.add(
        pitch=62,
        velocity=100,
        start_time=0.80,
        end_time=0.81)
    expected_ns.notes.add(
        pitch=63,
        velocity=100,
        start_time=1.00,
        end_time=1.01)
    expected_ns.total_time = 1.01

    codec = event_codec.Codec(
        max_shift_steps=100,
        steps_per_second=100,
        event_ranges=[
            event_codec.EventRange('pitch', note_seq.MIN_MIDI_PITCH,
                                   note_seq.MAX_MIDI_PITCH)])
    res = metrics_utils.event_predictions_to_ns(
        predictions, codec=codec,
        encoding_spec=note_sequences.NoteOnsetEncodingSpec)
    self.assertProtoEquals(expected_ns, res['est_ns'])
    self.assertEqual(0, res['est_invalid_events'])
    self.assertEqual(2, res['est_dropped_events'])
    np.testing.assert_array_equal([0, 0, 1, 1, 2, 2], res['raw_inputs'])

  def test_event_predictions_to_ns_with_offsets(self):
    predictions = [
        {
            'raw_inputs': [0, 0],
            'start_time': 0.0,
            'est_tokens': [20, 356, 160],
        },
        {
            'raw_inputs': [1, 1],
            'start_time': 0.4,
            'est_tokens': [20, 292, 161],
        },
        {
            'raw_inputs': [2, 2],
            'start_time': 0.8,
            'est_tokens': [20, 229, 160, 161]
        },
    ]
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=59,
        velocity=127,
        start_time=0.20,
        end_time=1.00)
    expected_ns.notes.add(
        pitch=60,
        velocity=63,
        start_time=0.60,
        end_time=1.00)
    expected_ns.total_time = 1.00

    codec = event_codec.Codec(
        max_shift_steps=100,
        steps_per_second=100,
        event_ranges=[
            event_codec.EventRange('pitch', note_seq.MIN_MIDI_PITCH,
                                   note_seq.MAX_MIDI_PITCH),
            event_codec.EventRange('velocity', 0, 127)
        ])
    res = metrics_utils.event_predictions_to_ns(
        predictions, codec=codec, encoding_spec=note_sequences.NoteEncodingSpec)
    self.assertProtoEquals(expected_ns, res['est_ns'])
    self.assertEqual(0, res['est_invalid_events'])
    self.assertEqual(0, res['est_dropped_events'])
    np.testing.assert_array_equal([0, 0, 1, 1, 2, 2], res['raw_inputs'])

  def test_event_predictions_to_ns_multitrack(self):
    predictions = [
        {
            'raw_inputs': [0, 0],
            'start_time': 0.0,
            'est_tokens': [20, 517, 356, 160],
        },
        {
            'raw_inputs': [1, 1],
            'start_time': 0.4,
            'est_tokens': [20, 356, 399],
        },
        {
            'raw_inputs': [2, 2],
            'start_time': 0.8,
            'est_tokens': [20, 517, 229, 160]
        },
    ]
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=42,
        velocity=127,
        start_time=0.60,
        end_time=0.61,
        is_drum=True,
        instrument=9)
    expected_ns.notes.add(
        pitch=59,
        velocity=127,
        start_time=0.20,
        end_time=1.00,
        program=32)
    expected_ns.total_time = 1.00

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
                                   note_seq.MAX_MIDI_PROGRAM)
        ])
    res = metrics_utils.event_predictions_to_ns(
        predictions, codec=codec, encoding_spec=note_sequences.NoteEncodingSpec)
    self.assertProtoEquals(expected_ns, res['est_ns'])
    self.assertEqual(0, res['est_invalid_events'])
    self.assertEqual(0, res['est_dropped_events'])
    np.testing.assert_array_equal([0, 0, 1, 1, 2, 2], res['raw_inputs'])

  def test_event_predictions_to_ns_multitrack_ties(self):
    predictions = [
        {
            'raw_inputs': [0, 0],
            'start_time': 0.0,
            'est_tokens': [613,  # no tied notes
                           20, 517, 356, 160],
        },
        {
            'raw_inputs': [1, 1],
            'start_time': 0.4,
            'est_tokens': [517, 160, 613,  # tied note
                           20, 356, 399],
        },
        {
            'raw_inputs': [2, 2],
            'start_time': 0.8,
            'est_tokens': [613]  # no tied notes, causing active note to end
        },
    ]
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=42,
        velocity=127,
        start_time=0.60,
        end_time=0.61,
        is_drum=True,
        instrument=9)
    expected_ns.notes.add(
        pitch=59,
        velocity=127,
        start_time=0.20,
        end_time=0.80,
        program=32)
    expected_ns.total_time = 0.80

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
    res = metrics_utils.event_predictions_to_ns(
        predictions, codec=codec,
        encoding_spec=note_sequences.NoteEncodingWithTiesSpec)
    self.assertProtoEquals(expected_ns, res['est_ns'])
    self.assertEqual(0, res['est_invalid_events'])
    self.assertEqual(0, res['est_dropped_events'])
    np.testing.assert_array_equal([0, 0, 1, 1, 2, 2], res['raw_inputs'])

  def test_frame_metrics(self):
    ref = np.zeros(shape=(128, 5))
    est = np.zeros(shape=(128, 5))

    # one overlapping note, two false positives, two false negatives
    ref[10, 0] = 127
    ref[10, 1] = 127
    ref[10, 2] = 127

    est[10, 2] = 127
    est[10, 3] = 127
    est[10, 4] = 127

    prec, rec, _ = metrics_utils.frame_metrics(ref, est, velocity_threshold=1)
    np.testing.assert_approx_equal(prec, 1/3)
    np.testing.assert_approx_equal(rec, 1/3)


if __name__ == '__main__':
  tf.test.main()
