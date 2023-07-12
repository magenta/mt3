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

"""Audio spectrogram functions."""

import dataclasses

from mt3 import spectral_ops
import tensorflow as tf

# defaults for spectrogram config
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512

# fixed constants; add these to SpectrogramConfig before changing
FFT_SIZE = 2048
MEL_LO_HZ = 20.0


@dataclasses.dataclass
class SpectrogramConfig:
  """Spectrogram configuration parameters."""
  sample_rate: int = DEFAULT_SAMPLE_RATE
  hop_width: int = DEFAULT_HOP_WIDTH
  num_mel_bins: int = DEFAULT_NUM_MEL_BINS

  @property
  def abbrev_str(self):
    s = ''
    if self.sample_rate != DEFAULT_SAMPLE_RATE:
      s += 'sr%d' % self.sample_rate
    if self.hop_width != DEFAULT_HOP_WIDTH:
      s += 'hw%d' % self.hop_width
    if self.num_mel_bins != DEFAULT_NUM_MEL_BINS:
      s += 'mb%d' % self.num_mel_bins
    return s

  @property
  def frames_per_second(self):
    return self.sample_rate / self.hop_width


def split_audio(samples, spectrogram_config):
  """Split audio into frames."""
  return tf.signal.frame(
      samples,
      frame_length=spectrogram_config.hop_width,
      frame_step=spectrogram_config.hop_width,
      pad_end=True)


def compute_spectrogram(samples, spectrogram_config):
  """Compute a mel spectrogram."""
  overlap = 1 - (spectrogram_config.hop_width / FFT_SIZE)
  return spectral_ops.compute_logmel(
      samples,
      bins=spectrogram_config.num_mel_bins,
      lo_hz=MEL_LO_HZ,
      overlap=overlap,
      fft_size=FFT_SIZE,
      sample_rate=spectrogram_config.sample_rate)


def flatten_frames(frames):
  """Convert frames back into a flat array of samples."""
  return tf.reshape(frames, [-1])


def input_depth(spectrogram_config):
  return spectrogram_config.num_mel_bins
