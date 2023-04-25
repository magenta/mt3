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

"""Transcription preprocessors."""

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from absl import logging
import gin
from immutabledict import immutabledict
import librosa

from mt3 import event_codec
from mt3 import note_sequences
from mt3 import run_length_encoding
from mt3 import spectrograms
from mt3 import vocabularies

import note_seq
import numpy as np
import seqio
import tensorflow as tf


def add_unique_id(ds: tf.data.Dataset) -> tf.data.Dataset:
  """Add unique integer ID to each example in a dataset."""
  def add_id_field(i, ex):
    ex['unique_id'] = [i]
    return ex
  return ds.enumerate().map(
      add_id_field, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@seqio.map_over_dataset
def pad_notesequence_array(ex):
  """Pad the NoteSequence array so that it can later be "split"."""
  ex['sequence'] = tf.pad(tf.expand_dims(ex['sequence'], 0),
                          [[0, len(ex['input_times']) - 1]])
  return ex


@seqio.map_over_dataset
def add_dummy_targets(ex):
  """Add dummy targets; used in eval when targets are not actually used."""
  ex['targets'] = np.array([], dtype=np.int32)
  return ex


def _audio_to_frames(
    samples: Sequence[float],
    spectrogram_config: spectrograms.SpectrogramConfig,
) -> Tuple[Sequence[Sequence[int]], np.ndarray]:
  """Convert audio samples to non-overlapping frames and frame times."""
  frame_size = spectrogram_config.hop_width
  logging.info('Padding %d samples to multiple of %d', len(samples), frame_size)
  samples = np.pad(samples,
                   [0, frame_size - len(samples) % frame_size],
                   mode='constant')

  frames = spectrograms.split_audio(samples, spectrogram_config)

  num_frames = len(samples) // frame_size
  logging.info('Encoded %d samples to %d frames (%d samples each)',
               len(samples), num_frames, frame_size)

  times = np.arange(num_frames) / spectrogram_config.frames_per_second
  return frames, times


def _include_inputs(ds, input_record, fields_to_omit=('audio',)):
  """Include fields from input record (other than audio) in dataset records."""
  def include_inputs_fn(output_record):
    for key in set(input_record.keys()) - set(output_record.keys()):
      output_record[key] = input_record[key]
    for key in fields_to_omit:
      del output_record[key]
    return output_record
  return ds.map(include_inputs_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


def tokenize_transcription_example(
    ds: tf.data.Dataset, spectrogram_config: spectrograms.SpectrogramConfig,
    codec: event_codec.Codec, is_training_data: bool,
    onsets_only: bool, include_ties: bool, audio_is_samples: bool,
    id_feature_key: Optional[str] = None
) -> tf.data.Dataset:
  """Tokenize a note transcription example for run-length encoding.

  Outputs include:
    inputs: audio sample frames, num_frames-by-frame_size
    input_time: timestamp for each frame
    targets: symbolic sequence of note-related events
    input_event_start_indices: start target index for every input index
    input_event_end_indices: end target index for every input index

  Args:
    ds: Input dataset.
    spectrogram_config: Spectrogram configuration.
    codec: Event vocabulary codec.
    is_training_data: Unused.
    onsets_only: If True, include only onset events (not offset, velocity, or
        program).
    include_ties: If True, also write state events containing active notes to
        support a "tie" section after run-length encoding.
    audio_is_samples: If True, audio is floating-point samples instead of
        serialized WAV.
    id_feature_key: If not None, replace sequence ID with specified key field
        from the dataset.

  Returns:
    Dataset with the outputs described above.
  """
  del is_training_data

  if onsets_only and include_ties:
    raise ValueError('Ties not supported when only modeling onsets.')

  def tokenize(sequence, audio, sample_rate, example_id=None):
    ns = note_seq.NoteSequence.FromString(sequence)
    note_sequences.validate_note_sequence(ns)

    if example_id is not None:
      ns.id = example_id

    if audio_is_samples:
      samples = audio
      if sample_rate != spectrogram_config.sample_rate:
        samples = librosa.resample(
            samples, sample_rate, spectrogram_config.sample_rate)
    else:
      samples = note_seq.audio_io.wav_data_to_samples_librosa(
          audio, sample_rate=spectrogram_config.sample_rate)

    logging.info('Got samples for %s::%s with length %d',
                 ns.id, ns.filename, len(samples))

    frames, frame_times = _audio_to_frames(samples, spectrogram_config)

    if onsets_only:
      times, values = note_sequences.note_sequence_to_onsets(ns)
    else:
      ns = note_seq.apply_sustain_control_changes(ns)
      times, values = (
          note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns))

    # The original NoteSequence can have a lot of control changes we don't need;
    # delete them.
    del ns.control_changes[:]

    (events, event_start_indices, event_end_indices,
     state_events, state_event_indices) = (
         run_length_encoding.encode_and_index_events(
             state=note_sequences.NoteEncodingState() if include_ties else None,
             event_times=times,
             event_values=values,
             encode_event_fn=note_sequences.note_event_data_to_events,
             codec=codec,
             frame_times=frame_times,
             encoding_state_to_events_fn=(
                 note_sequences.note_encoding_state_to_events
                 if include_ties else None)))

    yield {
        'inputs': frames,
        'input_times': frame_times,
        'targets': events,
        'input_event_start_indices': event_start_indices,
        'input_event_end_indices': event_end_indices,
        'state_events': state_events,
        'input_state_event_indices': state_event_indices,
        'sequence': ns.SerializeToString()
    }

  def process_record(input_record):
    if audio_is_samples and 'sample_rate' not in input_record:
      raise ValueError('Must provide sample rate when audio is samples.')

    args = [
        input_record['sequence'],
        input_record['audio'],
        input_record['sample_rate'] if 'sample_rate' in input_record else 0
    ]
    if id_feature_key is not None:
      args.append(input_record[id_feature_key])

    ds = tf.data.Dataset.from_generator(
        tokenize,
        output_signature={
            'inputs':
                tf.TensorSpec(
                    shape=(None, spectrogram_config.hop_width),
                    dtype=tf.float32),
            'input_times':
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'targets':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'input_event_start_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'input_event_end_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'state_events':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'input_state_event_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'sequence':
                tf.TensorSpec(shape=(), dtype=tf.string)
        },
        args=args)

    ds = _include_inputs(ds, input_record)
    return ds

  tokenized_records = ds.flat_map(process_record)
  return tokenized_records


def tokenize_guitarset_example(
    ds: tf.data.Dataset, spectrogram_config: spectrograms.SpectrogramConfig,
    codec: event_codec.Codec, is_training_data: bool,
    onsets_only: bool, include_ties: bool
) -> tf.data.Dataset:
  """Tokenize a GuitarSet transcription example."""
  def _preprocess_example(ex, name):
    assert 'inst_names' not in ex, 'Key `inst_names` is already populated.'
    ex['inst_names'] = [name]
    ex['instrument_sequences'] = [ex.pop('sequence')]
    return ex

  ds = ds.map(
      lambda x: _preprocess_example(x, 'Clean Guitar'),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = tokenize_example_with_program_lookup(
      ds,
      spectrogram_config=spectrogram_config,
      codec=codec,
      is_training_data=is_training_data,
      inst_name_to_program_fn=guitarset_instrument_to_program,
      onsets_only=onsets_only,
      include_ties=include_ties,
      id_feature_key='id')
  return ds


def guitarset_instrument_to_program(instrument: str) -> int:
  """GuitarSet is all guitar, return the first MIDI guitar program."""
  if instrument == 'Clean Guitar':
    return 24
  else:
    raise ValueError('Unknown GuitarSet instrument: %s' % instrument)


def tokenize_example_with_program_lookup(
    ds: tf.data.Dataset,
    spectrogram_config: spectrograms.SpectrogramConfig,
    codec: event_codec.Codec,
    is_training_data: bool,
    onsets_only: bool,
    include_ties: bool,
    inst_name_to_program_fn: Callable[[str], int],
    id_feature_key: Optional[str] = None
) -> tf.data.Dataset:
  """Tokenize an example, optionally looking up and assigning program numbers.

  This can be used by any dataset where a mapping function can be used to
  map from the inst_names feature to a set of program numbers.

  Args:
    ds: Input dataset.
    spectrogram_config: Spectrogram configuration.
    codec: Event vocabulary codec.
    is_training_data: Unused.
    onsets_only: If True, include only onset events (not offset & velocity).
    include_ties: If True, include tie events.
    inst_name_to_program_fn: A function used to map the instrument names
      in the `inst_names` feature of each example to a MIDI program number.
    id_feature_key: If not None, replace sequence ID with specified key field
        from the dataset.

  Returns:
    Dataset with the outputs described above.
  """
  del is_training_data

  def tokenize(sequences, inst_names, audio, example_id=None):
    # Add all the notes from the tracks to a single NoteSequence.
    ns = note_seq.NoteSequence(ticks_per_quarter=220)
    tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
    assert len(tracks) == len(inst_names)
    for track, inst_name in zip(tracks, inst_names):
      program = inst_name_to_program_fn(
          inst_name.decode())

      # Note that there are no pitch bends in URMP data; the below block will
      # raise PitchBendError if one is encountered.
      add_track_to_notesequence(ns, track, program=program, is_drum=False,
                                ignore_pitch_bends=False)

    note_sequences.assign_instruments(ns)
    note_sequences.validate_note_sequence(ns)

    if example_id is not None:
      ns.id = example_id

    samples = note_seq.audio_io.wav_data_to_samples_librosa(
        audio, sample_rate=spectrogram_config.sample_rate)

    logging.info('Got samples for %s::%s with length %d',
                 ns.id, ns.filename, len(samples))

    frames, frame_times = _audio_to_frames(samples, spectrogram_config)

    if onsets_only:
      times, values = note_sequences.note_sequence_to_onsets(ns)
    else:
      times, values = (
          note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns))

    # The original NoteSequence can have a lot of control changes we don't need;
    # delete them.
    del ns.control_changes[:]

    (events, event_start_indices, event_end_indices,
     state_events, state_event_indices) = (
         run_length_encoding.encode_and_index_events(
             state=note_sequences.NoteEncodingState() if include_ties else None,
             event_times=times,
             event_values=values,
             encode_event_fn=note_sequences.note_event_data_to_events,
             codec=codec,
             frame_times=frame_times,
             encoding_state_to_events_fn=(
                 note_sequences.note_encoding_state_to_events
                 if include_ties else None)))

    yield {
        'inputs': frames,
        'input_times': frame_times,
        'targets': events,
        'input_event_start_indices': event_start_indices,
        'input_event_end_indices': event_end_indices,
        'state_events': state_events,
        'input_state_event_indices': state_event_indices,
        'sequence': ns.SerializeToString()
    }

  def process_record(input_record):
    args = [
        input_record['instrument_sequences'],
        input_record['inst_names'],
        input_record['audio'],
    ]
    if id_feature_key is not None:
      args.append(input_record[id_feature_key])

    ds = tf.data.Dataset.from_generator(
        tokenize,
        output_signature={
            'inputs':
                tf.TensorSpec(
                    shape=(None, spectrogram_config.hop_width),
                    dtype=tf.float32),
            'input_times':
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'targets':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'input_event_start_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'input_event_end_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'state_events':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'input_state_event_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'sequence':
                tf.TensorSpec(shape=(), dtype=tf.string)
        },
        args=args)

    ds = _include_inputs(ds, input_record)
    return ds

  tokenized_records = ds.flat_map(process_record)
  return tokenized_records


_URMP_INSTRUMENT_PROGRAMS = immutabledict({
    'vn': 40,   # violin
    'va': 41,   # viola
    'vc': 42,   # cello
    'db': 43,   # double bass
    'tpt': 56,  # trumpet
    'tbn': 57,  # trombone
    'tba': 58,  # tuba
    'hn': 60,   # French horn
    'sax': 64,  # saxophone
    'ob': 68,   # oboe
    'bn': 70,   # bassoon
    'cl': 71,   # clarinet
    'fl': 73    # flute
})


def urmp_instrument_to_program(urmp_instrument: str) -> int:
  """Fetch the program number associated with a given URMP instrument code."""
  if urmp_instrument not in _URMP_INSTRUMENT_PROGRAMS:
    raise ValueError('unknown URMP instrument: %s' % urmp_instrument)
  return _URMP_INSTRUMENT_PROGRAMS[urmp_instrument]


_SLAKH_CLASS_PROGRAMS = immutabledict({
    'Acoustic Piano': 0,
    'Electric Piano': 4,
    'Chromatic Percussion': 8,
    'Organ': 16,
    'Acoustic Guitar': 24,
    'Clean Electric Guitar': 26,
    'Distorted Electric Guitar': 29,
    'Acoustic Bass': 32,
    'Electric Bass': 33,
    'Violin': 40,
    'Viola': 41,
    'Cello': 42,
    'Contrabass': 43,
    'Orchestral Harp': 46,
    'Timpani': 47,
    'String Ensemble': 48,
    'Synth Strings': 50,
    'Choir and Voice': 52,
    'Orchestral Hit': 55,
    'Trumpet': 56,
    'Trombone': 57,
    'Tuba': 58,
    'French Horn': 60,
    'Brass Section': 61,
    'Soprano/Alto Sax': 64,
    'Tenor Sax': 66,
    'Baritone Sax': 67,
    'Oboe': 68,
    'English Horn': 69,
    'Bassoon': 70,
    'Clarinet': 71,
    'Pipe': 73,
    'Synth Lead': 80,
    'Synth Pad': 88
})


def slakh_class_to_program_and_is_drum(slakh_class: str) -> Tuple[int, bool]:
  """Map Slakh class string to program number and boolean indicating drums."""
  if slakh_class == 'Drums':
    return 0, True
  elif slakh_class not in _SLAKH_CLASS_PROGRAMS:
    raise ValueError('unknown Slakh class: %s' % slakh_class)
  else:
    return _SLAKH_CLASS_PROGRAMS[slakh_class], False


class PitchBendError(Exception):
  pass


def add_track_to_notesequence(ns: note_seq.NoteSequence,
                              track: note_seq.NoteSequence,
                              program: int, is_drum: bool,
                              ignore_pitch_bends: bool):
  """Add a track to a NoteSequence."""
  if track.pitch_bends and not ignore_pitch_bends:
    raise PitchBendError
  track_sus = note_seq.apply_sustain_control_changes(track)
  for note in track_sus.notes:
    note.program = program
    note.is_drum = is_drum
    ns.notes.extend([note])
    ns.total_time = max(ns.total_time, note.end_time)


def tokenize_slakh_example(
    ds: tf.data.Dataset,
    spectrogram_config: spectrograms.SpectrogramConfig,
    codec: event_codec.Codec,
    is_training_data: bool,
    onsets_only: bool,
    include_ties: bool,
    track_specs: Optional[Sequence[note_sequences.TrackSpec]],
    ignore_pitch_bends: bool
) -> tf.data.Dataset:
  """Tokenize a Slakh multitrack note transcription example."""
  def tokenize(sequences, samples, sample_rate, inst_names, example_id):
    if sample_rate != spectrogram_config.sample_rate:
      samples = librosa.resample(
          samples, sample_rate, spectrogram_config.sample_rate)

    frames, frame_times = _audio_to_frames(samples, spectrogram_config)

    # Add all the notes from the tracks to a single NoteSequence.
    ns = note_seq.NoteSequence(ticks_per_quarter=220)
    tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
    assert len(tracks) == len(inst_names)
    if track_specs:
      # Specific tracks expected.
      assert len(tracks) == len(track_specs)
      for track, spec, inst_name in zip(tracks, track_specs, inst_names):
        # Make sure the instrument name matches what we expect.
        assert inst_name.decode() == spec.name
        try:
          add_track_to_notesequence(ns, track,
                                    program=spec.program, is_drum=spec.is_drum,
                                    ignore_pitch_bends=ignore_pitch_bends)
        except PitchBendError:
          # TODO(iansimon): is there a way to count these?
          return
    else:
      for track, inst_name in zip(tracks, inst_names):
        # Instrument name should be Slakh class.
        program, is_drum = slakh_class_to_program_and_is_drum(
            inst_name.decode())
        try:
          add_track_to_notesequence(ns, track, program=program, is_drum=is_drum,
                                    ignore_pitch_bends=ignore_pitch_bends)
        except PitchBendError:
          # TODO(iansimon): is there a way to count these?
          return

    note_sequences.assign_instruments(ns)
    note_sequences.validate_note_sequence(ns)
    if is_training_data:
      # Trim overlapping notes in training (as our event vocabulary cannot
      # represent them), but preserve original NoteSequence for eval.
      ns = note_sequences.trim_overlapping_notes(ns)

    ns.id = example_id

    if onsets_only:
      times, values = note_sequences.note_sequence_to_onsets(ns)
    else:
      times, values = (
          note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns))

    (events, event_start_indices, event_end_indices,
     state_events, state_event_indices) = (
         run_length_encoding.encode_and_index_events(
             state=note_sequences.NoteEncodingState() if include_ties else None,
             event_times=times,
             event_values=values,
             encode_event_fn=note_sequences.note_event_data_to_events,
             codec=codec,
             frame_times=frame_times,
             encoding_state_to_events_fn=(
                 note_sequences.note_encoding_state_to_events
                 if include_ties else None)))

    yield {
        'inputs': frames,
        'input_times': frame_times,
        'targets': events,
        'input_event_start_indices': event_start_indices,
        'input_event_end_indices': event_end_indices,
        'state_events': state_events,
        'input_state_event_indices': state_event_indices,
        'sequence': ns.SerializeToString()
    }

  def process_record(input_record):
    ds = tf.data.Dataset.from_generator(
        tokenize,
        output_signature={
            'inputs':
                tf.TensorSpec(
                    shape=(None, spectrogram_config.hop_width),
                    dtype=tf.float32),
            'input_times':
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'targets':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'input_event_start_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'input_event_end_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'state_events':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'input_state_event_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'sequence':
                tf.TensorSpec(shape=(), dtype=tf.string)
        },
        args=[
            input_record['note_sequences'], input_record['mix'],
            input_record['audio_sample_rate'], input_record['inst_names'],
            input_record['track_id']
        ])

    ds = _include_inputs(ds, input_record, fields_to_omit=['mix', 'stems'])
    return ds

  tokenized_records = ds.flat_map(process_record)
  return tokenized_records




@seqio.map_over_dataset
def compute_spectrograms(ex, spectrogram_config):
  samples = spectrograms.flatten_frames(ex['inputs'])
  ex['inputs'] = spectrograms.compute_spectrogram(samples, spectrogram_config)
  ex['raw_inputs'] = samples
  return ex


def handle_too_long(dataset: tf.data.Dataset,
                    output_features: seqio.preprocessors.OutputFeaturesType,
                    sequence_length: seqio.preprocessors.SequenceLengthType,
                    skip: bool = False) -> tf.data.Dataset:
  """Handle sequences that are too long, by either failing or skipping them."""
  def max_length_for_key(key):
    max_length = sequence_length[key]
    if output_features[key].add_eos:
      max_length -= 1
    return max_length

  if skip:
    # Drop examples where one of the features is longer than its maximum
    # sequence length.
    def is_not_too_long(ex):
      return not tf.reduce_any(
          [k in output_features and len(v) > max_length_for_key(k)
           for k, v in ex.items()])
    dataset = dataset.filter(is_not_too_long)

  def assert_not_too_long(key: str, value: tf.Tensor) -> tf.Tensor:
    if key in output_features:
      max_length = max_length_for_key(key)
      tf.debugging.assert_less_equal(
          tf.shape(value)[0], max_length,
          f'Value for "{key}" field exceeds maximum length')
    return value

  # Assert that no examples have features longer than their maximum sequence
  # length.
  return dataset.map(
      lambda ex: {k: assert_not_too_long(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable
def map_midi_programs(
    ds: tf.data.Dataset,
    codec: event_codec.Codec,
    granularity_type: str = 'full',
    feature_key: str = 'targets'
) -> Mapping[str, Any]:
  """Apply MIDI program map to token sequences."""
  granularity = vocabularies.PROGRAM_GRANULARITIES[granularity_type]
  def _map_program_tokens(ex):
    ex[feature_key] = granularity.tokens_map_fn(ex[feature_key], codec)
    return ex
  return ds.map(_map_program_tokens,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
