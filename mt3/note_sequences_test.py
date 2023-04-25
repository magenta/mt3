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

"""Tests for note_sequences."""

from mt3 import event_codec
from mt3 import note_sequences
from mt3 import run_length_encoding

import note_seq
import numpy as np
import tensorflow as tf

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


class RunLengthEncodingTest(tf.test.TestCase):

  def test_encode_and_index_note_sequence(self):
    ns = note_seq.NoteSequence()
    ns.notes.add(start_time=1.0,
                 end_time=1.1,
                 pitch=61,
                 velocity=100)
    ns.notes.add(start_time=2.0,
                 end_time=2.1,
                 pitch=62,
                 velocity=100)
    ns.notes.add(start_time=3.0,
                 end_time=3.1,
                 pitch=63,
                 velocity=100)
    ns.total_time = ns.notes[-1].end_time

    frame_times = np.arange(0, 4, step=.001)

    event_times, event_values = note_sequences.note_sequence_to_onsets(ns)
    events, event_start_indices, event_end_indices, _, _ = run_length_encoding.encode_and_index_events(
        state=None, event_times=event_times, event_values=event_values,
        encode_event_fn=note_sequences.note_event_data_to_events,
        codec=codec, frame_times=frame_times)

    self.assertEqual(len(frame_times), len(event_start_indices))
    self.assertEqual(len(frame_times), len(event_end_indices))
    self.assertLen(events, 403)
    expected_events = ([1] * 100 +
                       [162] +
                       [1] * 100 +
                       [163] +
                       [1] * 100 +
                       [164] +
                       [1] * 100)
    np.testing.assert_array_equal(expected_events, events)

    self.assertEqual(event_start_indices[0], 0)
    self.assertEqual(event_end_indices[0], 0)

    self.assertEqual(162, events[100])
    self.assertEqual(1.0, frame_times[1000])
    self.assertEqual(event_start_indices[1000], 100)
    self.assertEqual(event_end_indices[1000], 100)

    self.assertEqual(163, events[201])
    self.assertEqual(2.0, frame_times[2000])
    self.assertEqual(event_start_indices[2000], 201)
    self.assertEqual(event_end_indices[2000], 201)

    self.assertEqual(164, events[302])
    self.assertEqual(3.0, frame_times[3000])
    self.assertEqual(event_start_indices[3000], 302)
    self.assertEqual(event_end_indices[3000], 302)

    self.assertEqual(1, events[-1])
    self.assertEqual(3.999, frame_times[-1])
    self.assertEqual(event_start_indices[-1], 402)
    self.assertEqual(event_end_indices[-1], len(expected_events))

  def test_encode_and_index_note_sequence_velocity(self):
    ns = note_seq.NoteSequence()
    ns.notes.add(start_time=1.0,
                 end_time=3.0,
                 pitch=61,
                 velocity=1)
    ns.notes.add(start_time=2.0,
                 end_time=4.0,
                 pitch=62,
                 velocity=127)
    ns.total_time = ns.notes[-1].end_time

    frame_times = np.arange(0, 4, step=.001)

    event_times, event_values = (
        note_sequences.note_sequence_to_onsets_and_offsets(ns))
    events, event_start_indices, event_end_indices, _, _ = run_length_encoding.encode_and_index_events(
        state=None, event_times=event_times, event_values=event_values,
        encode_event_fn=note_sequences.note_event_data_to_events,
        codec=codec, frame_times=frame_times)

    self.assertEqual(len(frame_times), len(event_start_indices))
    self.assertEqual(len(frame_times), len(event_end_indices))
    self.assertLen(events, 408)
    expected_events = ([1] * 100 +
                       [230, 162] +
                       [1] * 100 +
                       [356, 163] +
                       [1] * 100 +
                       [229, 162] +
                       [1] * 100 +
                       [229, 163])
    np.testing.assert_array_equal(expected_events, events)

    self.assertEqual(event_start_indices[0], 0)
    self.assertEqual(event_end_indices[0], 0)

    self.assertEqual(230, events[100])
    self.assertEqual(162, events[101])
    self.assertEqual(1.0, frame_times[1000])
    self.assertEqual(event_start_indices[1000], 100)
    self.assertEqual(event_end_indices[1000], 100)

    self.assertEqual(356, events[202])
    self.assertEqual(163, events[203])
    self.assertEqual(2.0, frame_times[2000])
    self.assertEqual(event_start_indices[2000], 202)
    self.assertEqual(event_end_indices[2000], 202)

    self.assertEqual(229, events[304])
    self.assertEqual(162, events[305])
    self.assertEqual(3.0, frame_times[3000])
    self.assertEqual(event_start_indices[3000], 304)
    self.assertEqual(event_end_indices[3000], 304)

    self.assertEqual(229, events[406])
    self.assertEqual(163, events[407])
    self.assertEqual(3.999, frame_times[-1])
    self.assertEqual(event_start_indices[-1], 405)
    self.assertEqual(event_end_indices[-1], len(expected_events))

  def test_encode_and_index_note_sequence_multitrack(self):
    ns = note_seq.NoteSequence()
    ns.notes.add(start_time=0.0,
                 end_time=1.0,
                 pitch=37,
                 velocity=127,
                 is_drum=True)
    ns.notes.add(start_time=1.0,
                 end_time=3.0,
                 pitch=61,
                 velocity=127,
                 program=0)
    ns.notes.add(start_time=2.0,
                 end_time=4.0,
                 pitch=62,
                 velocity=127,
                 program=40)
    ns.total_time = ns.notes[-1].end_time

    frame_times = np.arange(0, 4, step=.001)

    event_times, event_values = (
        note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns))
    (tokens, event_start_indices, event_end_indices, state_tokens,
     state_event_indices) = run_length_encoding.encode_and_index_events(
         state=note_sequences.NoteEncodingState(),
         event_times=event_times, event_values=event_values,
         encode_event_fn=note_sequences.note_event_data_to_events,
         codec=codec, frame_times=frame_times,
         encoding_state_to_events_fn=(
             note_sequences.note_encoding_state_to_events))

    self.assertEqual(len(frame_times), len(event_start_indices))
    self.assertEqual(len(frame_times), len(event_end_indices))
    self.assertEqual(len(frame_times), len(state_event_indices))
    self.assertLen(tokens, 414)

    expected_events = (
        [event_codec.Event('velocity', 127), event_codec.Event('drum', 37)] +
        [event_codec.Event('shift', 1)] * 100 +
        [event_codec.Event('program', 0),
         event_codec.Event('velocity', 127), event_codec.Event('pitch', 61)] +
        [event_codec.Event('shift', 1)] * 100 +
        [event_codec.Event('program', 40),
         event_codec.Event('velocity', 127), event_codec.Event('pitch', 62)] +
        [event_codec.Event('shift', 1)] * 100 +
        [event_codec.Event('program', 0),
         event_codec.Event('velocity', 0), event_codec.Event('pitch', 61)] +
        [event_codec.Event('shift', 1)] * 100 +
        [event_codec.Event('program', 40),
         event_codec.Event('velocity', 0), event_codec.Event('pitch', 62)])
    expected_tokens = [codec.encode_event(e) for e in expected_events]
    np.testing.assert_array_equal(expected_tokens, tokens)

    expected_state_events = [
        event_codec.Event('tie', 0),       # state prior to first drum
        event_codec.Event('tie', 0),       # state prior to first onset
        event_codec.Event('program', 0),   # state prior to second onset
        event_codec.Event('pitch', 61),    # |
        event_codec.Event('tie', 0),       # |
        event_codec.Event('program', 0),   # state prior to first offset
        event_codec.Event('pitch', 61),    # |
        event_codec.Event('program', 40),  # |
        event_codec.Event('pitch', 62),    # |
        event_codec.Event('tie', 0),       # |
        event_codec.Event('program', 40),  # state prior to second offset
        event_codec.Event('pitch', 62),    # |
        event_codec.Event('tie', 0)        # |
    ]
    expected_state_tokens = [codec.encode_event(e)
                             for e in expected_state_events]
    np.testing.assert_array_equal(expected_state_tokens, state_tokens)

    self.assertEqual(event_start_indices[0], 0)
    self.assertEqual(event_end_indices[0], 0)
    self.assertEqual(state_event_indices[0], 0)

    self.assertEqual(1.0, frame_times[1000])
    self.assertEqual(event_start_indices[1000], 102)
    self.assertEqual(event_end_indices[1000], 102)
    self.assertEqual(state_event_indices[1000], 1)

    self.assertEqual(2.0, frame_times[2000])
    self.assertEqual(event_start_indices[2000], 205)
    self.assertEqual(event_end_indices[2000], 205)
    self.assertEqual(state_event_indices[2000], 2)

    self.assertEqual(3.0, frame_times[3000])
    self.assertEqual(event_start_indices[3000], 308)
    self.assertEqual(event_end_indices[3000], 308)
    self.assertEqual(state_event_indices[3000], 5)

    self.assertEqual(3.999, frame_times[-1])
    self.assertEqual(event_start_indices[-1], 410)
    self.assertEqual(event_end_indices[-1], len(expected_events))
    self.assertEqual(state_event_indices[-1], 10)

  def test_encode_and_index_note_sequence_last_token_alignment(self):
    ns = note_seq.NoteSequence()
    ns.notes.add(start_time=0.0,
                 end_time=0.1,
                 pitch=60,
                 velocity=100)
    ns.total_time = ns.notes[-1].end_time

    frame_times = np.arange(0, 1.008, step=.008)

    event_times, event_values = note_sequences.note_sequence_to_onsets(ns)
    events, event_start_indices, event_end_indices, _, _ = run_length_encoding.encode_and_index_events(
        state=None,
        event_times=event_times,
        event_values=event_values,
        encode_event_fn=note_sequences.note_event_data_to_events,
        codec=codec,
        frame_times=frame_times)

    self.assertEqual(len(frame_times), len(event_start_indices))
    self.assertEqual(len(frame_times), len(event_end_indices))
    self.assertLen(events, 102)
    expected_events = [161] + [1] * 101

    np.testing.assert_array_equal(expected_events, events)

    self.assertEqual(event_start_indices[0], 0)
    self.assertEqual(event_end_indices[0], 0)
    self.assertEqual(event_start_indices[125], 101)
    self.assertEqual(event_end_indices[125], 102)

  def test_decode_note_sequence_events(self):
    events = [25, 161, 50, 162]

    decoding_state = note_sequences.NoteDecodingState()
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=events, start_time=0, max_time=None,
        codec=codec, decode_event_fn=note_sequences.decode_note_onset_event)
    ns = note_sequences.flush_note_decoding_state(decoding_state)

    self.assertEqual(0, invalid_ids)
    self.assertEqual(0, dropped_events)
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=60,
        velocity=100,
        start_time=0.25,
        end_time=0.26)
    expected_ns.notes.add(
        pitch=61,
        velocity=100,
        start_time=0.50,
        end_time=0.51)
    expected_ns.total_time = 0.51
    self.assertProtoEquals(expected_ns, ns)

  def test_decode_note_sequence_events_onsets_only(self):
    events = [5, 161, 25, 162]

    decoding_state = note_sequences.NoteDecodingState()
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=events, start_time=0, max_time=None,
        codec=codec, decode_event_fn=note_sequences.decode_note_onset_event)
    ns = note_sequences.flush_note_decoding_state(decoding_state)

    self.assertEqual(0, invalid_ids)
    self.assertEqual(0, dropped_events)
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=60,
        velocity=100,
        start_time=0.05,
        end_time=0.06)
    expected_ns.notes.add(
        pitch=61,
        velocity=100,
        start_time=0.25,
        end_time=0.26)
    expected_ns.total_time = 0.26
    self.assertProtoEquals(expected_ns, ns)

  def test_decode_note_sequence_events_velocity(self):
    events = [5, 356, 161, 25, 229, 161]

    decoding_state = note_sequences.NoteDecodingState()
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=events, start_time=0, max_time=None,
        codec=codec, decode_event_fn=note_sequences.decode_note_event)
    ns = note_sequences.flush_note_decoding_state(decoding_state)

    self.assertEqual(0, invalid_ids)
    self.assertEqual(0, dropped_events)
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=60,
        velocity=127,
        start_time=0.05,
        end_time=0.25)
    expected_ns.total_time = 0.25
    self.assertProtoEquals(expected_ns, ns)

  def test_decode_note_sequence_events_missing_offset(self):
    events = [5, 356, 161, 10, 161, 25, 229, 161]

    decoding_state = note_sequences.NoteDecodingState()
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=events, start_time=0, max_time=None,
        codec=codec, decode_event_fn=note_sequences.decode_note_event)
    ns = note_sequences.flush_note_decoding_state(decoding_state)

    self.assertEqual(0, invalid_ids)
    self.assertEqual(0, dropped_events)
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=60,
        velocity=127,
        start_time=0.05,
        end_time=0.10)
    expected_ns.notes.add(
        pitch=60,
        velocity=127,
        start_time=0.10,
        end_time=0.25)
    expected_ns.total_time = 0.25
    self.assertProtoEquals(expected_ns, ns)

  def test_decode_note_sequence_events_multitrack(self):
    events = [5, 525, 356, 161, 15, 356, 394, 25, 525, 229, 161]

    decoding_state = note_sequences.NoteDecodingState()
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=events, start_time=0, max_time=None,
        codec=codec, decode_event_fn=note_sequences.decode_note_event)
    ns = note_sequences.flush_note_decoding_state(decoding_state)

    self.assertEqual(0, invalid_ids)
    self.assertEqual(0, dropped_events)
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=37,
        velocity=127,
        start_time=0.15,
        end_time=0.16,
        instrument=9,
        is_drum=True)
    expected_ns.notes.add(
        pitch=60,
        velocity=127,
        start_time=0.05,
        end_time=0.25,
        program=40)
    expected_ns.total_time = 0.25
    self.assertProtoEquals(expected_ns, ns)

  def test_decode_note_sequence_events_invalid_tokens(self):
    events = [5, -1, 161, -2, 25, 162, 9999]

    decoding_state = note_sequences.NoteDecodingState()
    invalid_events, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=events, start_time=0, max_time=None,
        codec=codec, decode_event_fn=note_sequences.decode_note_onset_event)
    ns = note_sequences.flush_note_decoding_state(decoding_state)

    self.assertEqual(3, invalid_events)
    self.assertEqual(0, dropped_events)
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=60,
        velocity=100,
        start_time=0.05,
        end_time=0.06)
    expected_ns.notes.add(
        pitch=61,
        velocity=100,
        start_time=0.25,
        end_time=0.26)
    expected_ns.total_time = 0.26
    self.assertProtoEquals(expected_ns, ns)

  def test_decode_note_sequence_events_allow_event_at_exactly_max_time(self):
    events = [161, 25, 162]

    decoding_state = note_sequences.NoteDecodingState()
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=events, start_time=1.0, max_time=1.25,
        codec=codec, decode_event_fn=note_sequences.decode_note_onset_event)
    ns = note_sequences.flush_note_decoding_state(decoding_state)

    self.assertEqual(0, invalid_ids)
    self.assertEqual(0, dropped_events)
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=60,
        velocity=100,
        start_time=1.00,
        end_time=1.01)
    expected_ns.notes.add(
        pitch=61,
        velocity=100,
        start_time=1.25,
        end_time=1.26)
    expected_ns.total_time = 1.26
    self.assertProtoEquals(expected_ns, ns)

  def test_decode_note_sequence_events_dropped_events(self):
    events = [5, 161, 30, 162]

    decoding_state = note_sequences.NoteDecodingState()
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=events, start_time=1.0, max_time=1.25,
        codec=codec, decode_event_fn=note_sequences.decode_note_onset_event)
    ns = note_sequences.flush_note_decoding_state(decoding_state)

    self.assertEqual(0, invalid_ids)
    self.assertEqual(2, dropped_events)
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=60,
        velocity=100,
        start_time=1.05,
        end_time=1.06)
    expected_ns.total_time = 1.06
    self.assertProtoEquals(expected_ns, ns)

  def test_decode_note_sequence_events_invalid_events(self):
    events = [25, 230, 50, 161]

    decoding_state = note_sequences.NoteDecodingState()
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=events, start_time=0, max_time=None,
        codec=codec, decode_event_fn=note_sequences.decode_note_onset_event)
    ns = note_sequences.flush_note_decoding_state(decoding_state)

    self.assertEqual(1, invalid_ids)
    self.assertEqual(0, dropped_events)
    expected_ns = note_seq.NoteSequence(ticks_per_quarter=220)
    expected_ns.notes.add(
        pitch=60,
        velocity=100,
        start_time=0.50,
        end_time=0.51)
    expected_ns.total_time = 0.51
    self.assertProtoEquals(expected_ns, ns)


if __name__ == '__main__':
  tf.test.main()
