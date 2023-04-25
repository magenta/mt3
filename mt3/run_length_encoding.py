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

"""Tools for run length encoding."""

import dataclasses
from typing import Any, Callable, Mapping, MutableMapping, Tuple, Optional, Sequence, TypeVar

from absl import logging
from mt3 import event_codec

import numpy as np
import seqio
import tensorflow as tf

Event = event_codec.Event

# These should be type variables, but unfortunately those are incompatible with
# dataclasses.
EventData = Any
EncodingState = Any
DecodingState = Any
DecodeResult = Any

T = TypeVar('T', bound=EventData)
ES = TypeVar('ES', bound=EncodingState)
DS = TypeVar('DS', bound=DecodingState)


@dataclasses.dataclass
class EventEncodingSpec:
  """Spec for encoding events."""
  # initialize encoding state
  init_encoding_state_fn: Callable[[], EncodingState]
  # convert EventData into zero or more events, updating encoding state
  encode_event_fn: Callable[[EncodingState, EventData, event_codec.Codec],
                            Sequence[event_codec.Event]]
  # convert encoding state (at beginning of segment) into events
  encoding_state_to_events_fn: Optional[Callable[[EncodingState],
                                                 Sequence[event_codec.Event]]]
  # create empty decoding state
  init_decoding_state_fn: Callable[[], DecodingState]
  # update decoding state when entering new segment
  begin_decoding_segment_fn: Callable[[DecodingState], None]
  # consume time and Event and update decoding state
  decode_event_fn: Callable[
      [DecodingState, float, event_codec.Event, event_codec.Codec], None]
  # flush decoding state into result
  flush_decoding_state_fn: Callable[[DecodingState], DecodeResult]


def encode_and_index_events(
    state: ES,
    event_times: Sequence[float],
    event_values: Sequence[T],
    encode_event_fn: Callable[[ES, T, event_codec.Codec],
                              Sequence[event_codec.Event]],
    codec: event_codec.Codec,
    frame_times: Sequence[float],
    encoding_state_to_events_fn: Optional[
        Callable[[ES], Sequence[event_codec.Event]]] = None,
) -> Tuple[Sequence[int], Sequence[int], Sequence[int],
           Sequence[int], Sequence[int]]:
  """Encode a sequence of timed events and index to audio frame times.

  Encodes time shifts as repeated single step shifts for later run length
  encoding.

  Optionally, also encodes a sequence of "state events", keeping track of the
  current encoding state at each audio frame. This can be used e.g. to prepend
  events representing the current state to a targets segment.

  Args:
    state: Initial event encoding state.
    event_times: Sequence of event times.
    event_values: Sequence of event values.
    encode_event_fn: Function that transforms event value into a sequence of one
        or more event_codec.Event objects.
    codec: An event_codec.Codec object that maps Event objects to indices.
    frame_times: Time for every audio frame.
    encoding_state_to_events_fn: Function that transforms encoding state into a
        sequence of one or more event_codec.Event objects.

  Returns:
    events: Encoded events and shifts.
    event_start_indices: Corresponding start event index for every audio frame.
        Note: one event can correspond to multiple audio indices due to sampling
        rate differences. This makes splitting sequences tricky because the same
        event can appear at the end of one sequence and the beginning of
        another.
    event_end_indices: Corresponding end event index for every audio frame. Used
        to ensure when slicing that one chunk ends where the next begins. Should
        always be true that event_end_indices[i] = event_start_indices[i + 1].
    state_events: Encoded "state" events representing the encoding state before
        each event.
    state_event_indices: Corresponding state event index for every audio frame.
  """
  indices = np.argsort(event_times, kind='stable')
  event_steps = [round(event_times[i] * codec.steps_per_second)
                 for i in indices]
  event_values = [event_values[i] for i in indices]

  events = []
  state_events = []
  event_start_indices = []
  state_event_indices = []

  cur_step = 0
  cur_event_idx = 0
  cur_state_event_idx = 0

  def fill_event_start_indices_to_cur_step():
    while(len(event_start_indices) < len(frame_times) and
          frame_times[len(event_start_indices)] <
          cur_step / codec.steps_per_second):
      event_start_indices.append(cur_event_idx)
      state_event_indices.append(cur_state_event_idx)

  for event_step, event_value in zip(event_steps, event_values):
    while event_step > cur_step:
      events.append(codec.encode_event(Event(type='shift', value=1)))
      cur_step += 1
      fill_event_start_indices_to_cur_step()
      cur_event_idx = len(events)
      cur_state_event_idx = len(state_events)
    if encoding_state_to_events_fn:
      # Dump state to state events *before* processing the next event, because
      # we want to capture the state prior to the occurrence of the event.
      for e in encoding_state_to_events_fn(state):
        state_events.append(codec.encode_event(e))
    for e in encode_event_fn(state, event_value, codec):
      events.append(codec.encode_event(e))

  # After the last event, continue filling out the event_start_indices array.
  # The inequality is not strict because if our current step lines up exactly
  # with (the start of) an audio frame, we need to add an additional shift event
  # to "cover" that frame.
  while cur_step / codec.steps_per_second <= frame_times[-1]:
    events.append(codec.encode_event(Event(type='shift', value=1)))
    cur_step += 1
    fill_event_start_indices_to_cur_step()
    cur_event_idx = len(events)

  # Now fill in event_end_indices. We need this extra array to make sure that
  # when we slice events, each slice ends exactly where the subsequent slice
  # begins.
  event_end_indices = event_start_indices[1:] + [len(events)]

  events = np.array(events)
  state_events = np.array(state_events)
  event_start_indices = np.array(event_start_indices)
  event_end_indices = np.array(event_end_indices)
  state_event_indices = np.array(state_event_indices)

  return (events, event_start_indices, event_end_indices,
          state_events, state_event_indices)


@seqio.map_over_dataset
def extract_target_sequence_with_indices(features, state_events_end_token=None):
  """Extract target sequence corresponding to audio token segment."""
  target_start_idx = features['input_event_start_indices'][0]
  target_end_idx = features['input_event_end_indices'][-1]

  features['targets'] = features['targets'][target_start_idx:target_end_idx]

  if state_events_end_token is not None:
    # Extract the state events corresponding to the audio start token, and
    # prepend them to the targets array.
    state_event_start_idx = features['input_state_event_indices'][0]
    state_event_end_idx = state_event_start_idx + 1
    while features['state_events'][
        state_event_end_idx - 1] != state_events_end_token:
      state_event_end_idx += 1
    features['targets'] = tf.concat([
        features['state_events'][state_event_start_idx:state_event_end_idx],
        features['targets']
    ], axis=0)

  return features


def remove_redundant_state_changes_fn(
    codec: event_codec.Codec,
    feature_key: str = 'targets',
    state_change_event_types: Sequence[str] = ()
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
  """Return preprocessing function that removes redundant state change events.

  Args:
    codec: The event_codec.Codec used to interpret the events.
    feature_key: The feature key for which to remove redundant state changes.
    state_change_event_types: A list of event types that represent state
        changes; tokens corresponding to these event types will be interpreted
        as state changes and redundant ones will be removed.

  Returns:
    A preprocessing function that removes redundant state change events.
  """
  state_change_event_ranges = [codec.event_type_range(event_type)
                               for event_type in state_change_event_types]

  def remove_redundant_state_changes(
      features: MutableMapping[str, Any],
  ) -> Mapping[str, Any]:
    """Remove redundant tokens e.g. duplicate velocity changes from sequence."""
    current_state = tf.zeros(len(state_change_event_ranges), dtype=tf.int32)
    output = tf.constant([], dtype=tf.int32)

    for event in features[feature_key]:
      # Let autograph know that the shape of 'output' will change during the
      # loop.
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(output, tf.TensorShape([None]))])
      is_redundant = False
      for i, (min_index, max_index) in enumerate(state_change_event_ranges):
        if (min_index <= event) and (event <= max_index):
          if current_state[i] == event:
            is_redundant = True
          current_state = tf.tensor_scatter_nd_update(
              current_state, indices=[[i]], updates=[event])
      if not is_redundant:
        output = tf.concat([output, [event]], axis=0)

    features[feature_key] = output
    return features

  return seqio.map_over_dataset(remove_redundant_state_changes)


def run_length_encode_shifts_fn(
    codec: event_codec.Codec,
    feature_key: str = 'targets'
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
  """Return a function that run-length encodes shifts for a given codec.

  Args:
    codec: The Codec to use for shift events.
    feature_key: The feature key for which to run-length encode shifts.

  Returns:
    A preprocessing function that run-length encodes single-step shifts.
  """
  def run_length_encode_shifts(
      features: MutableMapping[str, Any]
  ) -> Mapping[str, Any]:
    """Combine leading/interior shifts, trim trailing shifts.

    Args:
      features: Dict of features to process.

    Returns:
      A dict of features.
    """
    events = features[feature_key]

    shift_steps = 0
    total_shift_steps = 0
    output = tf.constant([], dtype=tf.int32)

    for event in events:
      # Let autograph know that the shape of 'output' will change during the
      # loop.
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[(output, tf.TensorShape([None]))])
      if codec.is_shift_event_index(event):
        shift_steps += 1
        total_shift_steps += 1

      else:
        # Once we've reached a non-shift event, RLE all previous shift events
        # before outputting the non-shift event.
        if shift_steps > 0:
          shift_steps = total_shift_steps
          while shift_steps > 0:
            output_steps = tf.minimum(codec.max_shift_steps, shift_steps)
            output = tf.concat([output, [output_steps]], axis=0)
            shift_steps -= output_steps
        output = tf.concat([output, [event]], axis=0)

    features[feature_key] = output
    return features

  return seqio.map_over_dataset(run_length_encode_shifts)


def merge_run_length_encoded_targets(
    targets: np.ndarray,
    codec: event_codec.Codec
) -> Sequence[int]:
  """Merge multiple tracks of target events into a single stream.

  Args:
    targets: A 2D array (# tracks by # events) of integer event values.
    codec: The event_codec.Codec used to interpret the events.

  Returns:
    A 1D array of merged events.
  """
  num_tracks = tf.shape(targets)[0]
  targets_length = tf.shape(targets)[1]

  current_step = 0
  current_offsets = tf.zeros(num_tracks, dtype=tf.int32)

  output = tf.constant([], dtype=tf.int32)
  done = tf.constant(False)

  while not done:
    # Let autograph know that the shape of 'output' will change during the loop.
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(output, tf.TensorShape([None]))])

    # Determine which targets track has the earliest next step.
    next_step = codec.max_shift_steps + 1
    next_track = -1
    for i in range(num_tracks):
      if (current_offsets[i] == targets_length or
          targets[i][current_offsets[i]] == 0):
        # Already reached the end of this targets track.
        # (Zero is technically a valid shift event but we never actually use it;
        #  it is always padding.)
        continue
      if not codec.is_shift_event_index(targets[i][current_offsets[i]]):
        # The only way we would be at a non-shift event is if we have not yet
        # reached the first shift event, which means we're at step zero.
        next_step = 0
        next_track = i
      elif targets[i][current_offsets[i]] < next_step:
        next_step = targets[i][current_offsets[i]]
        next_track = i

    if next_track == -1:
      # We've already merged all of the target tracks in their entirety.
      done = tf.constant(True)
      break

    if next_step == current_step and next_step > 0:
      # We don't need to include the shift event itself as it's the same step as
      # the previous shift.
      start_offset = current_offsets[next_track] + 1
    else:
      start_offset = current_offsets[next_track]

    # Merge in events up to but not including the next shift.
    end_offset = start_offset + 1
    while end_offset < targets_length and not codec.is_shift_event_index(
        targets[next_track][end_offset]):
      end_offset += 1
    output = tf.concat(
        [output, targets[next_track][start_offset:end_offset]], axis=0)

    current_step = next_step
    current_offsets = tf.tensor_scatter_nd_update(
        current_offsets, indices=[[next_track]], updates=[end_offset])

  return output


def decode_events(
    state: DS,
    tokens: np.ndarray,
    start_time: int,
    max_time: Optional[int],
    codec: event_codec.Codec,
    decode_event_fn: Callable[[DS, float, event_codec.Event, event_codec.Codec],
                              None],
) -> Tuple[int, int]:
  """Decode a series of tokens, maintaining a decoding state object.

  Args:
    state: Decoding state object; will be modified in-place.
    tokens: event tokens to convert.
    start_time: offset start time if decoding in the middle of a sequence.
    max_time: Events at or beyond this time will be dropped.
    codec: An event_codec.Codec object that maps indices to Event objects.
    decode_event_fn: Function that consumes an Event (and the current time) and
        updates the decoding state.

  Returns:
    invalid_events: number of events that could not be decoded.
    dropped_events: number of events dropped due to max_time restriction.
  """
  invalid_events = 0
  dropped_events = 0
  cur_steps = 0
  cur_time = start_time
  token_idx = 0
  for token_idx, token in enumerate(tokens):
    try:
      event = codec.decode_event_index(token)
    except ValueError:
      invalid_events += 1
      continue
    if event.type == 'shift':
      cur_steps += event.value
      cur_time = start_time + cur_steps / codec.steps_per_second
      if max_time and cur_time > max_time:
        dropped_events = len(tokens) - token_idx
        break
    else:
      cur_steps = 0
      try:
        decode_event_fn(state, cur_time, event, codec)
      except ValueError:
        invalid_events += 1
        logging.info(
            'Got invalid event when decoding event %s at time %f. '
            'Invalid event counter now at %d.',
            event, cur_time, invalid_events, exc_info=True)
        continue
  return invalid_events, dropped_events
