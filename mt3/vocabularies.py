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

"""Model vocabulary."""

import dataclasses
import math

from typing import Callable, Optional, Sequence
from mt3 import event_codec

import note_seq
import seqio
import t5.data
import tensorflow as tf


DECODED_EOS_ID = -1
DECODED_INVALID_ID = -2

# defaults for vocabulary config
DEFAULT_STEPS_PER_SECOND = 100
DEFAULT_MAX_SHIFT_SECONDS = 10
DEFAULT_NUM_VELOCITY_BINS = 127


@dataclasses.dataclass
class VocabularyConfig:
  """Vocabulary configuration parameters."""
  steps_per_second: int = DEFAULT_STEPS_PER_SECOND
  max_shift_seconds: int = DEFAULT_MAX_SHIFT_SECONDS
  num_velocity_bins: int = DEFAULT_NUM_VELOCITY_BINS

  @property
  def abbrev_str(self):
    s = ''
    if self.steps_per_second != DEFAULT_STEPS_PER_SECOND:
      s += 'ss%d' % self.steps_per_second
    if self.max_shift_seconds != DEFAULT_MAX_SHIFT_SECONDS:
      s += 'ms%d' % self.max_shift_seconds
    if self.num_velocity_bins != DEFAULT_NUM_VELOCITY_BINS:
      s += 'vb%d' % self.num_velocity_bins
    return s


def num_velocity_bins_from_codec(codec: event_codec.Codec):
  """Get number of velocity bins from event codec."""
  lo, hi = codec.event_type_range('velocity')
  return hi - lo


def velocity_to_bin(velocity, num_velocity_bins):
  if velocity == 0:
    return 0
  else:
    return math.ceil(num_velocity_bins * velocity / note_seq.MAX_MIDI_VELOCITY)


def bin_to_velocity(velocity_bin, num_velocity_bins):
  if velocity_bin == 0:
    return 0
  else:
    return int(note_seq.MAX_MIDI_VELOCITY * velocity_bin / num_velocity_bins)


def drop_programs(tokens, codec: event_codec.Codec):
  """Drops program change events from a token sequence."""
  min_program_id, max_program_id = codec.event_type_range('program')
  return tokens[(tokens < min_program_id) | (tokens > max_program_id)]


def programs_to_midi_classes(tokens, codec):
  """Modifies program events to be the first program in the MIDI class."""
  min_program_id, max_program_id = codec.event_type_range('program')
  is_program = (tokens >= min_program_id) & (tokens <= max_program_id)
  return tf.where(
      is_program,
      min_program_id + 8 * ((tokens - min_program_id) // 8),
      tokens)


@dataclasses.dataclass
class ProgramGranularity:
  # both tokens_map_fn and program_map_fn should be idempotent
  tokens_map_fn: Callable[[Sequence[int], event_codec.Codec], Sequence[int]]
  program_map_fn: Callable[[int], int]


PROGRAM_GRANULARITIES = {
    # "flat" granularity; drop program change tokens and set NoteSequence
    # programs to zero
    'flat': ProgramGranularity(
        tokens_map_fn=drop_programs,
        program_map_fn=lambda program: 0),

    # map each program to the first program in its MIDI class
    'midi_class': ProgramGranularity(
        tokens_map_fn=programs_to_midi_classes,
        program_map_fn=lambda program: 8 * (program // 8)),

    # leave programs as is
    'full': ProgramGranularity(
        tokens_map_fn=lambda tokens, codec: tokens,
        program_map_fn=lambda program: program)
}


def build_codec(vocab_config: VocabularyConfig):
  """Build event codec."""
  event_ranges = [
      event_codec.EventRange('pitch', note_seq.MIN_MIDI_PITCH,
                             note_seq.MAX_MIDI_PITCH),
      # velocity bin 0 is used for note-off
      event_codec.EventRange('velocity', 0, vocab_config.num_velocity_bins),
      # used to indicate that a pitch is present at the beginning of a segment
      # (only has an "off" event as when using ties all pitch events until the
      # "tie" event belong to the tie section)
      event_codec.EventRange('tie', 0, 0),
      event_codec.EventRange('program', note_seq.MIN_MIDI_PROGRAM,
                             note_seq.MAX_MIDI_PROGRAM),
      event_codec.EventRange('drum', note_seq.MIN_MIDI_PITCH,
                             note_seq.MAX_MIDI_PITCH),
  ]

  return event_codec.Codec(
      max_shift_steps=(vocab_config.steps_per_second *
                       vocab_config.max_shift_seconds),
      steps_per_second=vocab_config.steps_per_second,
      event_ranges=event_ranges)


def vocabulary_from_codec(codec: event_codec.Codec) -> seqio.Vocabulary:
  return GenericTokenVocabulary(
      codec.num_classes, extra_ids=t5.data.DEFAULT_EXTRA_IDS)


class GenericTokenVocabulary(seqio.Vocabulary):
  """Vocabulary with pass-through encoding of tokens."""

  def __init__(self, regular_ids: int, extra_ids: int = 0):
    # The special tokens: 0=PAD, 1=EOS, and 2=UNK
    self._num_special_tokens = 3
    self._num_regular_tokens = regular_ids
    super().__init__(extra_ids=extra_ids)

  @property
  def eos_id(self) -> Optional[int]:
    return 1

  @property
  def unk_id(self) -> Optional[int]:
    return 2

  @property
  def _base_vocab_size(self) -> int:
    """Number of ids.

    Returns:
      an integer, the vocabulary size
    """
    return self._num_special_tokens + self._num_regular_tokens

  def _encode(self, token_ids: Sequence[int]) -> Sequence[int]:
    """Encode a list of tokens ids as a list of integers.

    To keep the first few ids for special tokens, increase ids by the number
    of special tokens.

    Args:
      token_ids: array of token ids.

    Returns:
      a list of integers (not terminated by EOS)
    """
    encoded = []
    for token_id in token_ids:
      if not 0 <= token_id < self._num_regular_tokens:
        raise ValueError(
            f'token_id {token_id} does not fall within valid range of '
            f'[0, {self._num_regular_tokens})')
      encoded.append(token_id + self._num_special_tokens)

    return encoded

  def _decode(self, ids: Sequence[int]) -> Sequence[int]:
    """Decode a list of integers to a list of token ids.

    The special tokens of PAD and UNK as well as extra_ids will be
    replaced with DECODED_INVALID_ID in the output. If EOS is present, it will
    be the final token in the decoded output and will be represented by
    DECODED_EOS_ID.

    Args:
      ids: a list of integers

    Returns:
      a list of token ids.
    """
    # convert all the extra ids  to INVALID_ID
    def _decode_id(encoded_id):
      if encoded_id == self.eos_id:
        return DECODED_EOS_ID
      elif encoded_id < self._num_special_tokens:
        return DECODED_INVALID_ID
      elif encoded_id >= self._base_vocab_size:
        return DECODED_INVALID_ID
      else:
        return encoded_id - self._num_special_tokens
    ids = [_decode_id(int(i)) for i in ids]
    return ids

  def _encode_tf(self, token_ids: tf.Tensor) -> tf.Tensor:
    """Encode a list of tokens to a tf.Tensor.

    Args:
      token_ids: array of audio token ids.

    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    with tf.control_dependencies(
        [tf.debugging.assert_less(
            token_ids, tf.cast(self._num_regular_tokens, token_ids.dtype)),
         tf.debugging.assert_greater_equal(
             token_ids, tf.cast(0, token_ids.dtype))
         ]):
      tf_ids = token_ids + self._num_special_tokens
    return tf_ids

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    """Decode in TensorFlow.

    The special tokens of PAD and UNK as well as extra_ids will be
    replaced with DECODED_INVALID_ID in the output. If EOS is present, it and
    all following tokens in the decoded output and will be represented by
    DECODED_EOS_ID.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32

    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    # Create a mask that is true from the first EOS position onward.
    # First, create an array that is True whenever there is an EOS, then cumsum
    # that array so that every position after and including the first True is
    # >1, then cast back to bool for the final mask.
    eos_and_after = tf.cumsum(
        tf.cast(tf.equal(ids, self.eos_id), tf.int32), exclusive=False, axis=-1)
    eos_and_after = tf.cast(eos_and_after, tf.bool)

    return tf.where(
        eos_and_after,
        DECODED_EOS_ID,
        tf.where(
            tf.logical_and(
                tf.greater_equal(ids, self._num_special_tokens),
                tf.less(ids, self._base_vocab_size)),
            ids - self._num_special_tokens,
            DECODED_INVALID_ID))

  def __eq__(self, other):
    their_extra_ids = other.extra_ids
    their_num_regular_tokens = other._num_regular_tokens
    return (self.extra_ids == their_extra_ids and
            self._num_regular_tokens == their_num_regular_tokens)


def num_embeddings(vocabulary: GenericTokenVocabulary) -> int:
  """Vocabulary size as a multiple of 128 for TPU efficiency."""
  return 128 * math.ceil(vocabulary.vocab_size / 128)
