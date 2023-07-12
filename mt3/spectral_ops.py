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

"""Forked from DDSP spectral_ops.py (just for compute_logmel)."""

import gin
import tensorflow.compat.v2 as tf


def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32)


def safe_log(x, eps=1e-5):
  """Avoid taking the log of a non-positive number."""
  safe_x = tf.where(x <= 0.0, eps, x)
  return tf.math.log(safe_x)


def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
  """Differentiable stft in tensorflow, computed in batch."""
  # Remove channel dim if present.
  audio = tf_float32(audio)
  if len(audio.shape) == 3:
    audio = tf.squeeze(audio, axis=-1)

  s = tf.signal.stft(
      signals=audio,
      frame_length=int(frame_size),
      frame_step=int(frame_size * (1.0 - overlap)),
      fft_length=None,  # Use enclosing power of 2.
      pad_end=pad_end)
  return s


@gin.register
def compute_mag(audio, size=2048, overlap=0.75, pad_end=True):
  mag = tf.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end))
  return tf_float32(mag)


@gin.register
def compute_mel(audio,
                lo_hz=0.0,
                hi_hz=8000.0,
                bins=64,
                fft_size=2048,
                overlap=0.75,
                pad_end=True,
                sample_rate=16000):
  """Calculate Mel Spectrogram."""
  mag = compute_mag(audio, fft_size, overlap, pad_end)
  num_spectrogram_bins = int(mag.shape[-1])
  linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
      bins, num_spectrogram_bins, sample_rate, lo_hz, hi_hz)
  mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
  mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
  return mel


@gin.register
def compute_logmel(audio,
                   lo_hz=80.0,
                   hi_hz=7600.0,
                   bins=64,
                   fft_size=2048,
                   overlap=0.75,
                   pad_end=True,
                   sample_rate=16000):
  """Logarithmic amplitude of mel-scaled spectrogram."""
  mel = compute_mel(audio, lo_hz, hi_hz, bins,
                    fft_size, overlap, pad_end, sample_rate)
  return safe_log(mel)
