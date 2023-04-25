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

"""Detect monophonic tracks and extract notes."""

import collections
import os

from absl import app
from absl import flags
from absl import logging

import ddsp
import librosa
import note_seq
import numpy as np
import scipy
import tensorflow as tf


_INPUT_DIR = flags.DEFINE_string(
    'input_dir', None,
    'Input directory containing WAV files.')
_OUTPUT_TFRECORD_PATH = flags.DEFINE_string(
    'output_tfrecord_path', None,
    'Path to the output TFRecord containing tf.train.Example protos with '
    'monophonic tracks and inferred NoteSequence protos.')


CREPE_SAMPLE_RATE = 16000
CREPE_FRAME_RATE = 100

MONOPHONIC_CONFIDENCE_THRESHOLD = 0.95  # confidence must be greater than this
MONOPHONIC_CONFIDENCE_FRAC = 0.2        # for this fraction of frames

# split input audio into clips
CLIP_LENGTH_SECONDS = 5


def is_monophonic_heuristic(f0_confidence):
  """Heuristic to check for monophonicity using f0 confidence."""
  return (np.sum(f0_confidence >= MONOPHONIC_CONFIDENCE_THRESHOLD) /
          len(f0_confidence) >= MONOPHONIC_CONFIDENCE_FRAC)


# HMM parameters for modeling notes and F0 tracks.
F0_MIDI_SIGMA = 0.2
OCTAVE_ERROR_PROB = 0.05
NOTES_PER_SECOND = 2
NOTE_CHANGE_PROB = NOTES_PER_SECOND / CREPE_FRAME_RATE
F0_CONFIDENCE_EXP = 7.5


def f0_hmm_matrices(f0_hz, f0_confidence):
  """Observation and transition matrices for hidden Markov model of F0."""
  f0_midi = librosa.hz_to_midi(f0_hz)
  f0_midi_diff = f0_midi[:, np.newaxis] - np.arange(128)[np.newaxis, :]

  # Compute the probability of each pitch at each frame, taking octave errors
  # into account.
  f0_midi_prob_octave_correct = scipy.stats.norm.pdf(
      f0_midi_diff, scale=F0_MIDI_SIGMA)
  f0_midi_prob_octave_low = scipy.stats.norm.pdf(
      f0_midi_diff + 12, scale=F0_MIDI_SIGMA)
  f0_midi_prob_octave_high = scipy.stats.norm.pdf(
      f0_midi_diff - 12, scale=F0_MIDI_SIGMA)

  # distribution of pitch values given note
  f0_midi_loglik = ((1 - OCTAVE_ERROR_PROB) * f0_midi_prob_octave_correct +
                    0.5 * OCTAVE_ERROR_PROB * f0_midi_prob_octave_low +
                    0.5 * OCTAVE_ERROR_PROB * f0_midi_prob_octave_high)
  # (uniform) distribution of pitch values given rest
  f0_midi_rest_loglik = -np.log(128)

  # Here we interpret confidence, after adjusting by exponent, as P(not rest).
  f0_confidence_prob = np.power(f0_confidence, F0_CONFIDENCE_EXP)[:, np.newaxis]

  obs_loglik = np.concatenate([
      # probability of note (normalized by number of possible notes)
      f0_midi_loglik + np.log(f0_confidence_prob) - np.log(128),
      # probability of rest
      f0_midi_rest_loglik + np.log(1.0 - f0_confidence_prob)
  ], axis=1)

  # Normalize to adjust P(confidence | note) by uniform P(note).
  # TODO(iansimon): Not sure how correct this is but it doesn't affect the path.
  obs_loglik += np.log(129)

  trans_prob = ((NOTE_CHANGE_PROB / 128) * np.ones(129) +
                (1 - NOTE_CHANGE_PROB - NOTE_CHANGE_PROB / 128) * np.eye(129))
  trans_loglik = np.log(trans_prob)

  return obs_loglik, trans_loglik


def hmm_forward(obs_loglik, trans_loglik):
  """Forward algorithm for a hidden Markov model."""
  n, k = obs_loglik.shape
  trans = np.exp(trans_loglik)

  loglik = 0.0

  l = obs_loglik[0] - np.log(k)
  c = scipy.special.logsumexp(l)
  loglik += c

  for i in range(1, n):
    p = np.exp(l - c)
    l = np.log(np.dot(p, trans)) + obs_loglik[i]
    c = scipy.special.logsumexp(l)
    loglik += c

  return loglik


def hmm_viterbi(obs_loglik, trans_loglik):
  """Viterbi algorithm for a hidden Markov model."""
  n, k = obs_loglik.shape

  loglik_matrix = np.zeros_like(obs_loglik)
  path_matrix = np.zeros_like(obs_loglik, dtype=np.int32)

  loglik_matrix[0, :] = obs_loglik[0, :] - np.log(k)

  for i in range(1, n):
    mat = np.tile(loglik_matrix[i - 1][:, np.newaxis], [1, 129]) + trans_loglik
    path_matrix[i, :] = mat.argmax(axis=0)
    loglik_matrix[i, :] = mat[path_matrix[i, :], range(129)] + obs_loglik[i]

  path = [np.argmax(loglik_matrix[-1])]
  for i in range(n, 1, -1):
    path.append(path_matrix[i - 1, path[-1]])

  return [(pitch if pitch < 128 else None) for pitch in path[::-1]]


def pitches_to_notesequence(pitches):
  """Convert sequence of pitches output by Viterbi to NoteSequence proto."""
  ns = note_seq.NoteSequence(ticks_per_quarter=220)
  current_pitch = None
  start_time = None
  for frame, pitch in enumerate(pitches):
    time = frame / CREPE_FRAME_RATE
    if pitch != current_pitch:
      if current_pitch is not None:
        ns.notes.add(
            pitch=current_pitch, velocity=100,
            start_time=start_time, end_time=time)
      current_pitch = pitch
      start_time = time
  if current_pitch is not None:
    ns.notes.add(
        pitch=current_pitch, velocity=100,
        start_time=start_time, end_time=len(pitches) / CREPE_FRAME_RATE)
  if ns.notes:
    ns.total_time = ns.notes[-1].end_time
  return ns


# Per-frame log likelihood threshold below which an F0 track will be discarded.
# Note that this is dependent on the HMM parameters specified above, so if those
# change then this threshold should also change.
PER_FRAME_LOGLIK_THRESHOLD = 0.3


def extract_note_sequence(crepe, samples, counters):
  """Use CREPE to attempt to extract a monophonic NoteSequence from audio."""
  f0_hz, f0_confidence = crepe.predict_f0_and_confidence(
      samples[np.newaxis, :], viterbi=False)

  f0_hz = f0_hz[0].numpy()
  f0_confidence = f0_confidence[0].numpy()

  if not is_monophonic_heuristic(f0_confidence):
    counters['not_monophonic'] += 1
    return None

  obs_loglik, trans_loglik = f0_hmm_matrices(f0_hz, f0_confidence)

  loglik = hmm_forward(obs_loglik, trans_loglik)
  if loglik / len(obs_loglik) < PER_FRAME_LOGLIK_THRESHOLD:
    counters['low_likelihood'] += 1
    return None

  pitches = hmm_viterbi(obs_loglik, trans_loglik)
  ns = pitches_to_notesequence(pitches)

  counters['extracted_monophonic_sequence'] += 1
  return ns


def process_wav_file(wav_filename, crepe, counters):
  """Extract monophonic transcription examples from a WAV file."""
  wav_data = tf.io.gfile.GFile(wav_filename, 'rb').read()
  samples = note_seq.audio_io.wav_data_to_samples_librosa(
      wav_data, sample_rate=CREPE_SAMPLE_RATE)
  clip_length_samples = int(CREPE_SAMPLE_RATE * CLIP_LENGTH_SECONDS)
  for start_sample in range(0, len(samples), clip_length_samples):
    clip_samples = samples[start_sample:start_sample + clip_length_samples]
    if len(clip_samples) < clip_length_samples:
      clip_samples = np.pad(
          clip_samples, [(0, clip_length_samples - len(clip_samples))])
    ns = extract_note_sequence(crepe, clip_samples, counters)
    if ns:
      feature = {
          'audio': tf.train.Feature(
              float_list=tf.train.FloatList(value=clip_samples.tolist())),
          'filename': tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[wav_filename.encode()])),
          'offset': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[start_sample])),
          'sampling_rate': tf.train.Feature(
              float_list=tf.train.FloatList(value=[CREPE_SAMPLE_RATE])),
          'sequence': tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[ns.SerializeToString()]))
      }
      yield tf.train.Example(features=tf.train.Features(feature=feature))


def main(unused_argv):
  flags.mark_flags_as_required(['input_dir', 'output_tfrecord_path'])
  crepe = ddsp.spectral_ops.PretrainedCREPE('full')
  counters = collections.defaultdict(int)
  with tf.io.TFRecordWriter(_OUTPUT_TFRECORD_PATH.value) as writer:
    for filename in tf.io.gfile.listdir(_INPUT_DIR.value):
      if not filename.endswith('.wav'):
        logging.info('skipping %s...', filename)
        counters['non_wav_files_skipped'] += 1
        continue
      logging.info('processing %s...', filename)
      for ex in process_wav_file(
          os.path.join(_INPUT_DIR.value, filename), crepe, counters):
        writer.write(ex.SerializeToString())
      counters['wav_files_processed'] += 1
  for k, v in counters.items():
    logging.info('COUNTER: %s = %d', k, v)


if __name__ == '__main__':
  app.run(main)
