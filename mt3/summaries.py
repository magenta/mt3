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

"""TensorBoard summaries and utilities."""

from typing import Any, Mapping, Optional, Sequence, Tuple

import librosa

from mt3 import note_sequences
from mt3 import spectrograms

import note_seq
from note_seq import midi_synth
from note_seq import sequences_lib
from note_seq.protobuf import music_pb2

import numpy as np
import seqio


_DEFAULT_AUDIO_SECONDS = 30.0
_DEFAULT_PIANOROLL_FRAMES_PER_SECOND = 15

# TODO(iansimon): pick a SoundFont; for some reason the default is all organ


def _extract_example_audio(
    examples: Sequence[Mapping[str, Any]],
    sample_rate: float,
    num_seconds: float,
    audio_key: str = 'raw_inputs'
) -> np.ndarray:
  """Extract audio from examples.

  Args:
    examples: List of examples containing raw audio.
    sample_rate: Number of samples per second.
    num_seconds: Number of seconds of audio to include.
    audio_key: Dictionary key for the raw audio.

  Returns:
    An n-by-num_samples numpy array of samples.
  """
  n = len(examples)
  num_samples = round(num_seconds * sample_rate)
  all_samples = np.zeros([n, num_samples])
  for i, ex in enumerate(examples):
    samples = ex[audio_key][:num_samples]
    all_samples[i, :len(samples)] = samples
  return all_samples


def _example_to_note_sequence(
    example: Mapping[str, Sequence[float]],
    ns_feature_name: str,
    note_onset_feature_name: str,
    note_offset_feature_name: str,
    note_frequency_feature_name: str,
    note_confidence_feature_name: str,
    num_seconds: float
) -> music_pb2.NoteSequence:
  """Extract NoteSequence from example."""
  if ns_feature_name:
    ns = example[ns_feature_name]

  else:
    onset_times = np.array(example[note_onset_feature_name])
    pitches = librosa.hz_to_midi(
        example[note_frequency_feature_name]).round().astype(int)
    assert len(onset_times) == len(pitches)

    if note_offset_feature_name or note_confidence_feature_name:
      offset_times = (
          example[note_offset_feature_name]
          if note_offset_feature_name
          else onset_times + note_sequences.DEFAULT_NOTE_DURATION
      )
      assert len(onset_times) == len(offset_times)

      confidences = (np.array(example[note_confidence_feature_name])
                     if note_confidence_feature_name else None)
      velocities = np.ceil(
          note_seq.MAX_MIDI_VELOCITY * confidences if confidences is not None
          else note_sequences.DEFAULT_VELOCITY * np.ones_like(onset_times)
      ).astype(int)
      assert len(onset_times) == len(velocities)

      ns = note_sequences.note_arrays_to_note_sequence(
          onset_times=onset_times, offset_times=offset_times,
          pitches=pitches, velocities=velocities)

    else:
      ns = note_sequences.note_arrays_to_note_sequence(
          onset_times=onset_times, pitches=pitches)

  return sequences_lib.trim_note_sequence(ns, 0, num_seconds)


def _synthesize_example_notes(
    examples: Sequence[Mapping[str, Sequence[float]]],
    ns_feature_name: str,
    note_onset_feature_name: str,
    note_offset_feature_name: str,
    note_frequency_feature_name: str,
    note_confidence_feature_name: str,
    sample_rate: float,
    num_seconds: float,
) -> np.ndarray:
  """Synthesize example notes to audio.

  Args:
    examples: List of example dictionaries, containing either serialized
        NoteSequence protos or note onset times and pitches.
    ns_feature_name: Name of serialized NoteSequence feature.
    note_onset_feature_name: Name of note onset times feature.
    note_offset_feature_name: Name of note offset times feature.
    note_frequency_feature_name: Name of note frequencies feature.
    note_confidence_feature_name: Name of note confidences (velocities) feature.
    sample_rate: Sample rate at which to synthesize.
    num_seconds: Number of seconds to synthesize for each example.

  Returns:
    An n-by-num_samples numpy array of samples.
  """
  if (ns_feature_name is not None) == (note_onset_feature_name is not None):
    raise ValueError(
        'must specify exactly one of NoteSequence feature and onset feature')

  n = len(examples)
  num_samples = round(num_seconds * sample_rate)

  all_samples = np.zeros([n, num_samples])

  for i, ex in enumerate(examples):
    ns = _example_to_note_sequence(
        ex,
        ns_feature_name=ns_feature_name,
        note_onset_feature_name=note_onset_feature_name,
        note_offset_feature_name=note_offset_feature_name,
        note_frequency_feature_name=note_frequency_feature_name,
        note_confidence_feature_name=note_confidence_feature_name,
        num_seconds=num_seconds)
    fluidsynth = midi_synth.fluidsynth
    samples = fluidsynth(ns, sample_rate=sample_rate)
    if len(samples) > num_samples:
      samples = samples[:num_samples]
    all_samples[i, :len(samples)] = samples

  return all_samples


def _examples_to_pianorolls(
    targets: Sequence[Mapping[str, Sequence[float]]],
    predictions: Sequence[Mapping[str, Sequence[float]]],
    ns_feature_suffix: str,
    note_onset_feature_suffix: str,
    note_offset_feature_suffix: str,
    note_frequency_feature_suffix: str,
    note_confidence_feature_suffix: str,
    track_specs: Optional[Sequence[note_sequences.TrackSpec]],
    num_seconds: float,
    frames_per_second: float
) -> Tuple[np.ndarray, np.ndarray]:
  """Generate pianoroll images from example notes.

  Args:
    targets: List of target dictionaries, containing either serialized
        NoteSequence protos or note onset times and pitches.
    predictions: List of prediction dictionaries, containing either serialized
        NoteSequence protos or note onset times and pitches.
    ns_feature_suffix: Suffix of serialized NoteSequence feature.
    note_onset_feature_suffix: Suffix of note onset times feature.
    note_offset_feature_suffix: Suffix of note offset times feature.
    note_frequency_feature_suffix: Suffix of note frequencies feature.
    note_confidence_feature_suffix: Suffix of note confidences (velocities)
        feature.
    track_specs: Optional list of TrackSpec objects to indicate a set of tracks
        into which each NoteSequence should be split. Tracks will be stacked
        vertically in the pianorolls
    num_seconds: Number of seconds to show for each example.
    frames_per_second: Number of pianoroll frames per second.

  Returns:
    onset_pianorolls: An n-by-num_pitches-by-num_frames-by-4 numpy array of
        pianoroll images showing only onsets.
    full_pianorolls: An n-by-num_pitches-by-num_frames-by-4 numpy array of
        pianoroll images.
  """
  if (ns_feature_suffix is not None) == (note_onset_feature_suffix is not None):
    raise ValueError(
        'must specify exactly one of NoteSequence feature and onset feature')

  def ex_to_ns(example, prefix):
    return _example_to_note_sequence(
        example=example,
        ns_feature_name=(prefix + ns_feature_suffix
                         if ns_feature_suffix else None),
        note_onset_feature_name=(prefix + note_onset_feature_suffix
                                 if note_onset_feature_suffix else None),
        note_offset_feature_name=(prefix + note_offset_feature_suffix
                                  if note_offset_feature_suffix else None),
        note_frequency_feature_name=(
            prefix + note_frequency_feature_suffix
            if note_frequency_feature_suffix else None),
        note_confidence_feature_name=(
            prefix + note_confidence_feature_suffix
            if note_confidence_feature_suffix else None),
        num_seconds=num_seconds)

  n = len(targets)
  num_pitches = note_seq.MAX_MIDI_PITCH - note_seq.MIN_MIDI_PITCH + 1
  num_frames = round(num_seconds * frames_per_second)
  num_tracks = len(track_specs) if track_specs else 1
  pianoroll_height = num_tracks * num_pitches + (num_tracks - 1)

  onset_images = np.zeros([n, pianoroll_height, num_frames, 3])
  full_images = np.zeros([n, pianoroll_height, num_frames, 3])

  for i, (target, pred) in enumerate(zip(targets, predictions)):
    target_ns, pred_ns = [
        ex_to_ns(ex, prefix)
        for (ex, prefix) in [(target, 'ref_'), (pred, 'est_')]
    ]

    # Show lines at frame boundaries. To ensure that these lines are drawn with
    # the same downsampling and frame selection logic as the real NoteSequences,
    # use this hack to draw the lines with a NoteSequence that contains notes
    # across all pitches at all frame start times.
    start_times_ns = note_seq.NoteSequence()
    start_times_ns.CopyFrom(target_ns)
    del start_times_ns.notes[:]
    for start_time in pred['start_times']:
      if start_time < target_ns.total_time:
        for pitch in range(
            note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH + 1):
          start_times_ns.notes.add(
              pitch=pitch,
              velocity=100,
              start_time=start_time,
              end_time=start_time + (1 / frames_per_second))

    start_time_roll = sequences_lib.sequence_to_pianoroll(
        start_times_ns,
        frames_per_second=frames_per_second,
        min_pitch=note_seq.MIN_MIDI_PITCH,
        max_pitch=note_seq.MAX_MIDI_PITCH,
        onset_mode='length_ms')
    num_start_time_frames = min(len(start_time_roll.onsets), num_frames)

    if track_specs is not None:
      target_tracks = [note_sequences.extract_track(target_ns,
                                                    spec.program, spec.is_drum)
                       for spec in track_specs]
      pred_tracks = [note_sequences.extract_track(pred_ns,
                                                  spec.program, spec.is_drum)
                     for spec in track_specs]
    else:
      target_tracks = [target_ns]
      pred_tracks = [pred_ns]

    for j, (target_track, pred_track) in enumerate(zip(target_tracks[::-1],
                                                       pred_tracks[::-1])):
      target_roll = sequences_lib.sequence_to_pianoroll(
          target_track,
          frames_per_second=frames_per_second,
          min_pitch=note_seq.MIN_MIDI_PITCH,
          max_pitch=note_seq.MAX_MIDI_PITCH,
          onset_mode='length_ms')
      pred_roll = sequences_lib.sequence_to_pianoroll(
          pred_track,
          frames_per_second=frames_per_second,
          min_pitch=note_seq.MIN_MIDI_PITCH,
          max_pitch=note_seq.MAX_MIDI_PITCH,
          onset_mode='length_ms')

      num_target_frames = min(len(target_roll.onsets), num_frames)
      num_pred_frames = min(len(pred_roll.onsets), num_frames)

      start_offset = j * (num_pitches + 1)
      end_offset = (j + 1) * (num_pitches + 1) - 1

      # Onsets
      onset_images[
          i, start_offset:end_offset, :num_start_time_frames, 0
      ] = start_time_roll.onsets[:num_start_time_frames, :].T
      onset_images[
          i, start_offset:end_offset, :num_target_frames, 1
      ] = target_roll.onsets[:num_target_frames, :].T
      onset_images[
          i, start_offset:end_offset, :num_pred_frames, 2
      ] = pred_roll.onsets[:num_pred_frames, :].T

      # Full notes
      full_images[
          i, start_offset:end_offset, :num_start_time_frames, 0
      ] = start_time_roll.onsets[:num_start_time_frames, :].T
      full_images[
          i, start_offset:end_offset, :num_target_frames, 1
      ] = target_roll.active[:num_target_frames, :].T
      full_images[
          i, start_offset:end_offset, :num_pred_frames, 2
      ] = pred_roll.active[:num_pred_frames, :].T

      # Add separator between tracks.
      if j < num_tracks - 1:
        onset_images[i, end_offset, :, 0] = 1
        full_images[i, end_offset, :, 0] = 1

  return onset_images[:, ::-1, :, :], full_images[:, ::-1, :, :]


def prettymidi_pianoroll(
    track_pianorolls: Mapping[str, Sequence[Tuple[np.ndarray, np.ndarray]]],
    fps: float,
    num_seconds=_DEFAULT_AUDIO_SECONDS
) -> Mapping[str, seqio.metrics.MetricValue]:
  """Create summary from given pianorolls."""
  max_len = int(num_seconds * fps)
  summaries = {}
  for inst_name, all_prs in track_pianorolls.items():

    est_prs, ref_prs = zip(*all_prs)

    bs = len(ref_prs)
    pianoroll_image_batch = np.zeros(shape=(bs, 128, max_len, 3))
    for i in range(bs):
      ref_pr = ref_prs[i][:, :max_len]
      est_pr = est_prs[i][:, :max_len]

      pianoroll_image_batch[i, :, :est_pr.shape[1], 2] = est_pr
      pianoroll_image_batch[i, :, :ref_pr.shape[1], 1] = ref_pr
    if not inst_name:
      inst_name = 'all instruments'

    summaries[f'{inst_name} pretty_midi pianoroll'] = seqio.metrics.Image(
        image=pianoroll_image_batch, max_outputs=bs)

  return summaries


def audio_summaries(
    targets: Sequence[Mapping[str, Sequence[float]]],
    predictions: Sequence[Mapping[str, Sequence[float]]],
    spectrogram_config: spectrograms.SpectrogramConfig,
    num_seconds: float = _DEFAULT_AUDIO_SECONDS
) -> Mapping[str, seqio.metrics.MetricValue]:
  """Compute audio summaries for a list of examples.

  Args:
    targets: List of targets, unused as we pass the input audio tokens via
        predictions.
    predictions: List of predictions, including input audio tokens.
    spectrogram_config: Spectrogram configuration.
    num_seconds: Number of seconds of audio to include in the summaries.
        Longer audio will be cropped (from the beginning), shorter audio will be
        padded with silence (at the end).

  Returns:
    A dictionary mapping "audio" to the audio summaries.
  """
  del targets
  samples = _extract_example_audio(
      examples=predictions,
      sample_rate=spectrogram_config.sample_rate,
      num_seconds=num_seconds)
  return {
      'audio': seqio.metrics.Audio(
          audiodata=samples[:, :, np.newaxis],
          sample_rate=spectrogram_config.sample_rate,
          max_outputs=samples.shape[0])
  }


def transcription_summaries(
    targets: Sequence[Mapping[str, Sequence[float]]],
    predictions: Sequence[Mapping[str, Sequence[float]]],
    spectrogram_config: spectrograms.SpectrogramConfig,
    ns_feature_suffix: Optional[str] = None,
    note_onset_feature_suffix: Optional[str] = None,
    note_offset_feature_suffix: Optional[str] = None,
    note_frequency_feature_suffix: Optional[str] = None,
    note_confidence_feature_suffix: Optional[str] = None,
    track_specs: Optional[Sequence[note_sequences.TrackSpec]] = None,
    num_seconds: float = _DEFAULT_AUDIO_SECONDS,
    pianoroll_frames_per_second: float = _DEFAULT_PIANOROLL_FRAMES_PER_SECOND,
) -> Mapping[str, seqio.metrics.MetricValue]:
  """Compute note transcription summaries for multiple examples.

  Args:
    targets: List of targets containing ground truth.
    predictions: List of predictions, including raw input audio.
    spectrogram_config: The spectrogram configuration.
    ns_feature_suffix: Suffix of serialized NoteSequence feature.
    note_onset_feature_suffix: Suffix of note onset times feature.
    note_offset_feature_suffix: Suffix of note offset times feature.
    note_frequency_feature_suffix: Suffix of note frequencies feature.
    note_confidence_feature_suffix: Suffix of note confidences (velocities)
        feature.
    track_specs: Optional list of TrackSpec objects to indicate a set of tracks
        into which each NoteSequence should be split.
    num_seconds: Number of seconds of audio to include in the summaries.
        Longer audio will be cropped (from the beginning), shorter audio will be
        padded with silence (at the end).
    pianoroll_frames_per_second: Temporal resolution of pianoroll images.

  Returns:
    A dictionary of input, ground truth, and transcription summaries.
  """
  audio_samples = _extract_example_audio(
      examples=predictions,
      sample_rate=spectrogram_config.sample_rate,
      num_seconds=num_seconds)

  def synthesize(examples, prefix):
    return _synthesize_example_notes(
        examples=examples,
        ns_feature_name=(prefix + ns_feature_suffix
                         if ns_feature_suffix else None),
        note_onset_feature_name=(prefix + note_onset_feature_suffix
                                 if note_onset_feature_suffix else None),
        note_offset_feature_name=(prefix + note_offset_feature_suffix
                                  if note_offset_feature_suffix else None),
        note_frequency_feature_name=(
            prefix + note_frequency_feature_suffix
            if note_frequency_feature_suffix else None),
        note_confidence_feature_name=(
            prefix + note_confidence_feature_suffix
            if note_confidence_feature_suffix else None),
        sample_rate=spectrogram_config.sample_rate,
        num_seconds=num_seconds)

  synthesized_predictions = synthesize(predictions, 'est_')

  onset_pianoroll_images, full_pianoroll_images = _examples_to_pianorolls(
      targets=targets,
      predictions=predictions,
      ns_feature_suffix=ns_feature_suffix,
      note_onset_feature_suffix=note_onset_feature_suffix,
      note_offset_feature_suffix=note_offset_feature_suffix,
      note_frequency_feature_suffix=note_frequency_feature_suffix,
      note_confidence_feature_suffix=note_confidence_feature_suffix,
      track_specs=track_specs,
      num_seconds=num_seconds,
      frames_per_second=pianoroll_frames_per_second)

  return {
      'input_with_transcription': seqio.metrics.Audio(
          audiodata=np.stack([audio_samples, synthesized_predictions], axis=2),
          sample_rate=spectrogram_config.sample_rate,
          max_outputs=audio_samples.shape[0]),

      'pianoroll': seqio.metrics.Image(
          image=full_pianoroll_images,
          max_outputs=full_pianoroll_images.shape[0]),

      'onset_pianoroll': seqio.metrics.Image(
          image=onset_pianoroll_images,
          max_outputs=onset_pianoroll_images.shape[0]),
  }
