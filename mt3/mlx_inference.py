# Copyright 2026 The MT3 Authors.
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

"""Apple Silicon MLX inference model for MT3 music transcription."""

import os
from typing import Any, Mapping, Sequence, Tuple, Union, cast

import flax.traverse_util
import mlx.core as mx
import mlx.nn as nn
from mt3 import metrics_utils
from mt3 import network
from mt3 import note_sequences
from mt3 import spectrograms
from mt3 import vocabularies
import note_seq
import numpy as np
import t5x.checkpoints

_MEL_HI_HZ = 7600.0
_LAYER_NORM_EPS = 1e-6
_KV_CACHE_BUCKETS = (64, 128, 192, 256, 384, 512, 1024)
_DTYPE_MAP = {
    'bfloat16': mx.bfloat16,
    'float16': mx.float16,
    'float32': mx.float32,
}


def _load_t5x_checkpoint(checkpoint_path: str) -> dict[str, np.ndarray]:
  """Load a T5X checkpoint directory into a flat dict of float32 arrays."""
  ckpt = t5x.checkpoints.load_t5x_checkpoint(checkpoint_path)
  flat = flax.traverse_util.flatten_dict(ckpt['target'], sep='.')
  return {k: np.asarray(v, dtype=np.float32) for k, v in flat.items()}


def _hertz_to_mel(frequencies_hertz: np.ndarray) -> np.ndarray:
  """Convert frequencies in Hz to the HTK mel scale matching tf.signal."""
  return 1127.0 * np.log(1.0 + (frequencies_hertz / 700.0))


def _linear_to_mel_weight_matrix(
    num_mel_bins: int = spectrograms.DEFAULT_NUM_MEL_BINS,
    num_spectrogram_bins: int = spectrograms.FFT_SIZE // 2 + 1,
    sample_rate: int = spectrograms.DEFAULT_SAMPLE_RATE,
    lower_edge_hertz: float = spectrograms.MEL_LO_HZ,
    upper_edge_hertz: float = _MEL_HI_HZ,
) -> np.ndarray:
  """Construct an HTK mel filterbank matrix matching tf.signal."""
  bands_to_zero = 1
  nyquist_hertz = sample_rate / 2.0
  linear_frequencies = np.linspace(
      0.0, nyquist_hertz, num_spectrogram_bins, dtype=np.float32
  )[bands_to_zero:]
  spectrogram_bins_mel = _hertz_to_mel(linear_frequencies)[:, None]

  band_edges_mel = np.linspace(
      _hertz_to_mel(np.array(lower_edge_hertz, dtype=np.float32)),
      _hertz_to_mel(np.array(upper_edge_hertz, dtype=np.float32)),
      num_mel_bins + 2,
      dtype=np.float32,
  )
  lower_edge_mel = band_edges_mel[0:-2][None, :]
  center_mel = band_edges_mel[1:-1][None, :]
  upper_edge_mel = band_edges_mel[2:][None, :]

  lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
      center_mel - lower_edge_mel
  )
  upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
      upper_edge_mel - center_mel
  )
  mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
  return np.pad(
      mel_weights_matrix, [[bands_to_zero, 0], [0, 0]], mode='constant'
  ).astype(np.float32)


def _compute_logmel(
    audio_batch: mx.array,
    mel_matrix: mx.array,
    window: mx.array,
    fft_size: int = spectrograms.FFT_SIZE,
    hop_width: int = spectrograms.DEFAULT_HOP_WIDTH,
) -> mx.array:
  """Compute log-mel spectrograms in MLX for [B, L] or [L] audio samples."""
  is_1d = audio_batch.ndim == 1
  if is_1d:
    audio_batch = audio_batch[None, :]
  b_sz, length = audio_batch.shape
  num_frames = int(np.ceil(length / hop_width))
  pad_len = (num_frames - 1) * hop_width + fft_size - length
  if pad_len > 0:
    audio_batch = mx.pad(audio_batch, [(0, 0), (0, pad_len)])

  padded_len = audio_batch.shape[1]
  frames = mx.as_strided(
      audio_batch,
      shape=(b_sz, num_frames, fft_size),
      strides=(padded_len, hop_width, 1),
  )
  windowed = frames * window
  stft = mx.fft.rfft(windowed, n=fft_size, axis=-1)
  mag = mx.abs(stft)
  mel = mag @ mel_matrix
  logmel = mx.log(mx.where(mel <= 0.0, 1e-5, mel))
  return logmel[0] if is_1d else logmel


def _sinusoidal_embeddings(
    max_len: int = 2048,
    features: int = network.T5Config.emb_dim,
) -> mx.array:
  """Create fixed sinusoidal positional embeddings matching mt3.layers."""
  pe = np.zeros((max_len, features), dtype=np.float32)
  position = np.arange(0, max_len)[:, np.newaxis]
  scale_factor = -np.log(10000.0 / 1.0) / (features // 2 - 1)
  div_term = np.exp(np.arange(0, features // 2) * scale_factor)
  pe[:, : features // 2] = np.sin(position * div_term)
  pe[:, features // 2 : 2 * (features // 2)] = np.cos(position * div_term)
  return mx.array(pe)


class _EncoderLayer(nn.Module):
  """T5.1.1 Encoder layer with fused QKV and fused GEGLU weights."""

  def __init__(self, config: network.T5Config):
    super().__init__()
    self.num_heads = config.num_heads
    self.head_dim = config.head_dim
    self.qkv_dim = config.num_heads * config.head_dim
    emb_dim = config.emb_dim
    mlp_dim = config.mlp_dim

    self.pre_attn_norm_weight = mx.ones((emb_dim,))
    self.qkv_weight = mx.zeros((emb_dim, 3 * self.qkv_dim))
    self.out_weight = mx.zeros((self.qkv_dim, emb_dim))
    self.pre_mlp_norm_weight = mx.ones((emb_dim,))
    self.wi_fused_weight = mx.zeros((emb_dim, 2 * mlp_dim))
    self.wo_weight = mx.zeros((mlp_dim, emb_dim))

  def __call__(self, x: mx.array) -> mx.array:
    b_sz, seq_len, _ = x.shape
    h = mx.fast.rms_norm(x, self.pre_attn_norm_weight, eps=_LAYER_NORM_EPS)
    qkv = h @ self.qkv_weight
    q, k, v = mx.split(qkv, 3, axis=-1)
    q = q.reshape(b_sz, seq_len, self.num_heads, self.head_dim).transpose(
        0, 2, 1, 3
    )
    k = k.reshape(b_sz, seq_len, self.num_heads, self.head_dim).transpose(
        0, 2, 1, 3
    )
    v = v.reshape(b_sz, seq_len, self.num_heads, self.head_dim).transpose(
        0, 2, 1, 3
    )
    attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
    attn = attn.transpose(0, 2, 1, 3).reshape(b_sz, seq_len, self.qkv_dim)
    x = x + (attn @ self.out_weight)

    h = mx.fast.rms_norm(x, self.pre_mlp_norm_weight, eps=_LAYER_NORM_EPS)
    wi = h @ self.wi_fused_weight
    w0, w1 = mx.split(wi, 2, axis=-1)
    return x + ((nn.gelu_approx(w0) * w1) @ self.wo_weight)


class _DecoderLayer(nn.Module):
  """T5.1.1 Decoder layer with fused self-QKV, cross-KV, and GEGLU weights."""

  def __init__(self, config: network.T5Config):
    super().__init__()
    self.num_heads = config.num_heads
    self.head_dim = config.head_dim
    self.qkv_dim = config.num_heads * config.head_dim
    emb_dim = config.emb_dim
    mlp_dim = config.mlp_dim

    self.pre_self_attn_norm_weight = mx.ones((emb_dim,))
    self.self_qkv_weight = mx.zeros((emb_dim, 3 * self.qkv_dim))
    self.self_out_weight = mx.zeros((self.qkv_dim, emb_dim))

    self.pre_cross_attn_norm_weight = mx.ones((emb_dim,))
    self.cross_q_weight = mx.zeros((emb_dim, self.qkv_dim))
    self.cross_kv_weight = mx.zeros((emb_dim, 2 * self.qkv_dim))
    self.cross_out_weight = mx.zeros((self.qkv_dim, emb_dim))

    self.pre_mlp_norm_weight = mx.ones((emb_dim,))
    self.wi_fused_weight = mx.zeros((emb_dim, 2 * mlp_dim))
    self.wo_weight = mx.zeros((mlp_dim, emb_dim))

  def project_cross_kv(self, encoded: mx.array) -> Tuple[mx.array, mx.array]:
    """Precompute cross-attention K and V tensors from encoder outputs."""
    b_sz, seq_len, _ = encoded.shape
    kv = encoded @ self.cross_kv_weight
    ck, cv = mx.split(kv, 2, axis=-1)
    ck = ck.reshape(b_sz, seq_len, self.num_heads, self.head_dim).transpose(
        0, 2, 1, 3
    )
    cv = cv.reshape(b_sz, seq_len, self.num_heads, self.head_dim).transpose(
        0, 2, 1, 3
    )
    return ck, cv

  def __call__(
      self,
      y: mx.array,
      step_idx: mx.array,
      mask: mx.array,
      k_cache: mx.array,
      v_cache: mx.array,
      cross_k: mx.array,
      cross_v: mx.array,
  ) -> Tuple[mx.array, mx.array, mx.array]:
    b_cur = y.shape[0]
    h = mx.fast.rms_norm(
        y, self.pre_self_attn_norm_weight, eps=_LAYER_NORM_EPS
    )
    qkv = h @ self.self_qkv_weight
    q, k, v = mx.split(qkv, 3, axis=-1)
    q = q.reshape(b_cur, self.num_heads, 1, self.head_dim)
    k = k.reshape(b_cur, self.num_heads, 1, self.head_dim)
    v = v.reshape(b_cur, self.num_heads, 1, self.head_dim)

    k_cache[:, :, step_idx, :] = k
    v_cache[:, :, step_idx, :] = v

    attn = mx.fast.scaled_dot_product_attention(
        q, k_cache, v_cache, scale=1.0, mask=mask
    )
    y = y + (attn.reshape(b_cur, self.qkv_dim) @ self.self_out_weight)

    h = mx.fast.rms_norm(
        y, self.pre_cross_attn_norm_weight, eps=_LAYER_NORM_EPS
    )
    q_cross = (h @ self.cross_q_weight).reshape(
        b_cur, self.num_heads, 1, self.head_dim
    )
    cross_attn = mx.fast.scaled_dot_product_attention(
        q_cross, cross_k, cross_v, scale=1.0
    )
    y = y + (cross_attn.reshape(b_cur, self.qkv_dim) @ self.cross_out_weight)

    h = mx.fast.rms_norm(y, self.pre_mlp_norm_weight, eps=_LAYER_NORM_EPS)
    wi = h @ self.wi_fused_weight
    w0, w1 = mx.split(wi, 2, axis=-1)
    y = y + ((nn.gelu_approx(w0) * w1) @ self.wo_weight)
    return y, k_cache, v_cache


class _Transformer(nn.Module):
  """MT3 Encoder-Decoder Transformer in MLX configured via network.T5Config."""

  def __init__(
      self,
      config: network.T5Config,
      input_depth: int = spectrograms.DEFAULT_NUM_MEL_BINS,
  ):
    super().__init__()
    self.config = config
    self.vocab_size = config.vocab_size
    self.emb_dim = config.emb_dim
    self.num_heads = config.num_heads
    self.head_dim = config.head_dim
    self.mlp_dim = config.mlp_dim
    self.num_encoder_layers = config.num_encoder_layers
    self.num_decoder_layers = config.num_decoder_layers

    self.pe = _sinusoidal_embeddings(features=config.emb_dim)
    self.enc_proj_weight = mx.zeros((input_depth, config.emb_dim))
    self.enc_layers = [
        _EncoderLayer(config) for _ in range(config.num_encoder_layers)
    ]
    self.enc_norm_weight = mx.ones((config.emb_dim,))

    self.token_embedding = mx.zeros((config.vocab_size, config.emb_dim))
    self.dec_layers = [
        _DecoderLayer(config) for _ in range(config.num_decoder_layers)
    ]
    self.dec_norm_weight = mx.ones((config.emb_dim,))
    self.logits_weight = mx.zeros((config.emb_dim, config.vocab_size))

  def _load_from_t5x_dict(self, p: Mapping[str, np.ndarray]) -> None:
    """Populate fused MLX weights from a flat T5X checkpoint dictionary."""
    self.enc_proj_weight = mx.array(
        p['encoder.continuous_inputs_projection.kernel']
    )
    self.enc_norm_weight = mx.array(p['encoder.encoder_norm.scale'])
    for i, lyr in enumerate(self.enc_layers):
      pfx = f'encoder.layers_{i}'
      lyr.pre_attn_norm_weight = mx.array(
          p[f'{pfx}.pre_attention_layer_norm.scale']
      )
      lyr.qkv_weight = mx.array(
          np.concatenate(
              [
                  p[f'{pfx}.attention.query.kernel'],
                  p[f'{pfx}.attention.key.kernel'],
                  p[f'{pfx}.attention.value.kernel'],
              ],
              axis=-1,
          )
      )
      lyr.out_weight = mx.array(p[f'{pfx}.attention.out.kernel'])
      lyr.pre_mlp_norm_weight = mx.array(p[f'{pfx}.pre_mlp_layer_norm.scale'])
      lyr.wi_fused_weight = mx.array(
          np.concatenate(
              [p[f'{pfx}.mlp.wi_0.kernel'], p[f'{pfx}.mlp.wi_1.kernel']],
              axis=-1,
          )
      )
      lyr.wo_weight = mx.array(p[f'{pfx}.mlp.wo.kernel'])

    self.token_embedding = mx.array(p['decoder.token_embedder.embedding'])
    self.dec_norm_weight = mx.array(p['decoder.decoder_norm.scale'])
    self.logits_weight = mx.array(p['decoder.logits_dense.kernel'])
    for i, lyr in enumerate(self.dec_layers):
      pfx = f'decoder.layers_{i}'
      lyr.pre_self_attn_norm_weight = mx.array(
          p[f'{pfx}.pre_self_attention_layer_norm.scale']
      )
      lyr.self_qkv_weight = mx.array(
          np.concatenate(
              [
                  p[f'{pfx}.self_attention.query.kernel'],
                  p[f'{pfx}.self_attention.key.kernel'],
                  p[f'{pfx}.self_attention.value.kernel'],
              ],
              axis=-1,
          )
      )
      lyr.self_out_weight = mx.array(p[f'{pfx}.self_attention.out.kernel'])

      lyr.pre_cross_attn_norm_weight = mx.array(
          p[f'{pfx}.pre_cross_attention_layer_norm.scale']
      )
      lyr.cross_q_weight = mx.array(
          p[f'{pfx}.encoder_decoder_attention.query.kernel']
      )
      lyr.cross_kv_weight = mx.array(
          np.concatenate(
              [
                  p[f'{pfx}.encoder_decoder_attention.key.kernel'],
                  p[f'{pfx}.encoder_decoder_attention.value.kernel'],
              ],
              axis=-1,
          )
      )
      lyr.cross_out_weight = mx.array(
          p[f'{pfx}.encoder_decoder_attention.out.kernel']
      )

      lyr.pre_mlp_norm_weight = mx.array(p[f'{pfx}.pre_mlp_layer_norm.scale'])
      lyr.wi_fused_weight = mx.array(
          np.concatenate(
              [p[f'{pfx}.mlp.wi_0.kernel'], p[f'{pfx}.mlp.wi_1.kernel']],
              axis=-1,
          )
      )
      lyr.wo_weight = mx.array(p[f'{pfx}.mlp.wo.kernel'])

  def _cast_weights(self, dtype: mx.Dtype) -> None:
    """Cast all model weights and positional embeddings to the target dtype."""
    self.apply(lambda x: x.astype(dtype))

  def encode(self, encoder_input_tokens: mx.array) -> mx.array:
    """Encode continuous spectrogram inputs of shape [batch, length, depth]."""
    seq_len = encoder_input_tokens.shape[1]
    x = (encoder_input_tokens @ self.enc_proj_weight) + self.pe[:seq_len]
    for lyr in self.enc_layers:
      x = lyr(x)
    return mx.fast.rms_norm(x, self.enc_norm_weight, eps=_LAYER_NORM_EPS)

  def decode_step(
      self,
      cur_tokens_1d: mx.array,
      step_idx: mx.array,
      mask: mx.array,
      k_caches: list[mx.array],
      v_caches: list[mx.array],
      c_ks: list[mx.array],
      c_vs: list[mx.array],
  ) -> Tuple[mx.array, list[mx.array], list[mx.array]]:
    """Run a single autoregressive decoder step across all decoder layers."""
    y = self.token_embedding[cur_tokens_1d] + self.pe[step_idx]
    new_k_caches = []
    new_v_caches = []
    for i, lyr in enumerate(self.dec_layers):
      y, kc, vc = lyr(
          y, step_idx, mask, k_caches[i], v_caches[i], c_ks[i], c_vs[i]
      )
      new_k_caches.append(kc)
      new_v_caches.append(vc)
    y = mx.fast.rms_norm(y, self.dec_norm_weight, eps=_LAYER_NORM_EPS)
    logits = y @ self.logits_weight
    next_tokens_1d = mx.argmax(logits, axis=-1).astype(mx.int32)
    return next_tokens_1d, new_k_caches, new_v_caches


def _infer_config_from_params(
    p: Mapping[str, Any],
    head_dim: int = network.T5Config.head_dim,
) -> Tuple[network.T5Config, int]:
  """Infer a network.T5Config and input_depth from T5X or fused MLX weights."""
  if 'enc_proj_weight' in p:
    input_depth, emb_dim = p['enc_proj_weight'].shape
    vocab_size = p['token_embedding'].shape[0]
    qkv_dim = p['enc_layers.0.qkv_weight'].shape[1] // 3
    num_heads = qkv_dim // head_dim
    mlp_dim = p['enc_layers.0.wi_fused_weight'].shape[1] // 2
    num_enc_layers = sum(
        1 for k in p if k.startswith('enc_layers.') and k.endswith('.qkv_weight')
    )
    num_dec_layers = sum(
        1
        for k in p
        if k.startswith('dec_layers.') and k.endswith('.self_qkv_weight')
    )
  else:
    input_depth = int(
        p['encoder.continuous_inputs_projection.kernel'].shape[0]
    )
    vocab_size, emb_dim = p['decoder.token_embedder.embedding'].shape
    q_dim = p['encoder.layers_0.attention.query.kernel'].shape[1]
    num_heads = q_dim // head_dim
    mlp_dim = p['encoder.layers_0.mlp.wi_0.kernel'].shape[1]
    num_enc_layers = sum(
        1
        for k in p
        if k.startswith('encoder.layers_')
        and k.endswith('.pre_attention_layer_norm.scale')
    )
    num_dec_layers = sum(
        1
        for k in p
        if k.startswith('decoder.layers_')
        and k.endswith('.pre_self_attention_layer_norm.scale')
    )
  cfg = network.T5Config(
      vocab_size=int(vocab_size),
      emb_dim=int(emb_dim),
      num_heads=int(num_heads),
      num_encoder_layers=int(num_enc_layers),
      num_decoder_layers=int(num_dec_layers),
      head_dim=int(head_dim),
      mlp_dim=int(mlp_dim),
      mlp_activations=('gelu', 'linear'),
  )
  return cfg, int(input_depth)


def _make_compiled_fns(model: _Transformer):
  """Build @mx.compile functions for encoder+cross-KV and 4-step decoding."""

  @mx.compile
  def encode_and_cross_kv(specs: mx.array):
    encoded = model.encode(specs)
    cross_ks = []
    cross_vs = []
    for lyr in model.dec_layers:
      ck, cv = lyr.project_cross_kv(encoded)
      cross_ks.append(ck)
      cross_vs.append(cv)
    return cross_ks, cross_vs

  @mx.compile
  def decode_4_steps(
      cur_tokens_1d,
      s0,
      s1,
      s2,
      s3,
      m0,
      m1,
      m2,
      m3,
      k_caches,
      v_caches,
      c_ks,
      c_vs,
  ):
    t1, k_caches, v_caches = model.decode_step(
        cur_tokens_1d, s0, m0, k_caches, v_caches, c_ks, c_vs
    )
    t2, k_caches, v_caches = model.decode_step(
        t1, s1, m1, k_caches, v_caches, c_ks, c_vs
    )
    t3, k_caches, v_caches = model.decode_step(
        t2, s2, m2, k_caches, v_caches, c_ks, c_vs
    )
    t4, k_caches, v_caches = model.decode_step(
        t3, s3, m3, k_caches, v_caches, c_ks, c_vs
    )
    return mx.stack([t1, t2, t3, t4], axis=1), t4, k_caches, v_caches

  return encode_and_cross_kv, decode_4_steps


class InferenceModel:
  """Drop-in MLX wrapper of the MT3 model for music transcription.

  Provides the same interface as the JAX/T5X `InferenceModel` while using
  Apple Silicon MLX with precomputed cross-attention KV caches, fused
  projections, compiled 4-step decoding, and dynamic EOS batch compaction.
  """

  def __init__(
      self,
      checkpoint_path: Union[str, Mapping[str, Any]],
      model_type: str = 'mt3',
      dtype: str = 'bfloat16',
  ):
    """Initialize the MLX inference model and restore weights from checkpoint.

    Args:
      checkpoint_path: Path to the T5X checkpoint directory, path to a saved MLX
        weights file (`.npz` / `.safetensors`), or a flat parameter mapping.
      model_type: One of 'mt3', 'ismir2021', 'ismir2022_small', or
        'ismir2022_base'.
      dtype: Computation and weight dtype ('bfloat16', 'float16', or 'float32').
    """
    if model_type == 'ismir2021':
      num_velocity_bins = vocabularies.DEFAULT_NUM_VELOCITY_BINS
      self.encoding_spec = note_sequences.NoteEncodingSpec
      self.inputs_length = 512
    elif model_type in ('mt3', 'ismir2022_small', 'ismir2022_base'):
      num_velocity_bins = 1
      self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
      self.inputs_length = 256
    else:
      raise ValueError('unknown model_type: %s' % model_type)

    if dtype not in _DTYPE_MAP:
      raise ValueError(
          f'Unsupported dtype {dtype!r}; choose from {list(_DTYPE_MAP)}'
      )
    self.dtype_name = dtype
    self.dtype = _DTYPE_MAP[dtype]
    self.model_type = model_type
    self.batch_size = 8
    self.outputs_length = 1024
    self.sequence_length = {
        'inputs': self.inputs_length,
        'targets': self.outputs_length,
    }

    self.spectrogram_config = spectrograms.SpectrogramConfig()
    self.codec = vocabularies.build_codec(
        vocab_config=vocabularies.VocabularyConfig(
            num_velocity_bins=num_velocity_bins
        )
    )
    self.vocabulary = cast(
        vocabularies.GenericTokenVocabulary,
        vocabularies.vocabulary_from_codec(self.codec),
    )

    self._mel_mat = mx.array(
        _linear_to_mel_weight_matrix(
            num_mel_bins=self.spectrogram_config.num_mel_bins,
            sample_rate=self.spectrogram_config.sample_rate,
        )
    )
    fft_size = spectrograms.FFT_SIZE
    self._hann_win = mx.array(
        0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(fft_size) / fft_size),
        dtype=mx.float32,
    )

    self.restore_from_checkpoint(checkpoint_path)

  @property
  def input_shapes(self) -> dict[str, Tuple[int, int]]:
    """Return nominal encoder and decoder input shapes."""
    return {
        'encoder_input_tokens': (self.batch_size, self.inputs_length),
        'decoder_input_tokens': (self.batch_size, self.outputs_length),
    }

  def restore_from_checkpoint(
      self, checkpoint_path: Union[str, Mapping[str, Any]]
  ) -> None:
    """Restore model weights from a T5X directory, MLX weights file, or dict."""
    if isinstance(checkpoint_path, Mapping):
      flat_params = checkpoint_path
      config, input_depth = _infer_config_from_params(flat_params)
      self.model = _Transformer(config=config, input_depth=input_depth)
      self.model._load_from_t5x_dict(flat_params)
    elif os.path.isfile(checkpoint_path):
      weights = mx.load(checkpoint_path)
      config, input_depth = _infer_config_from_params(weights)
      self.model = _Transformer(config=config, input_depth=input_depth)
      self.model.load_weights(list(weights.items()))
    else:
      flat_params = _load_t5x_checkpoint(checkpoint_path)
      config, input_depth = _infer_config_from_params(flat_params)
      self.model = _Transformer(config=config, input_depth=input_depth)
      self.model._load_from_t5x_dict(flat_params)

    self.model._cast_weights(self.dtype)
    mx.eval(self.model.parameters())
    self._encode_fn, self._decode_4_steps_fn = _make_compiled_fns(self.model)

  def save_weights(self, path: str) -> None:
    """Save fused MLX model weights to a single .safetensors or .npz file."""
    self.model.save_weights(path)

  def predict_tokens(
      self, batch: Union[Mapping[str, Any], np.ndarray, mx.array], seed: int = 0
  ) -> np.ndarray:
    """Predict decoded vocabulary tokens from a batch of spectrogram frames.

    Args:
      batch: Either a mapping containing `'encoder_input_tokens'` of shape
        `[batch, inputs_length, depth]` or a raw array of the same shape.
      seed: Unused random seed kept for API compatibility with the JAX model.

    Returns:
      An int32 numpy array of decoded token IDs (`DECODED_EOS_ID` for EOS).
    """
    del seed
    if isinstance(batch, Mapping):
      enc_in = batch['encoder_input_tokens']
    else:
      enc_in = batch
    specs = mx.array(enc_in).astype(self.dtype)
    cross_ks, cross_vs = self._encode_fn(specs)
    mx.eval(*cross_ks, *cross_vs)
    raw_tokens = self._decode_segments(
        cross_ks, cross_vs, batch_size=specs.shape[0]
    )
    return self._decode_vocab_ids(raw_tokens)

  def __call__(self, audio: np.ndarray) -> note_seq.NoteSequence:
    """Infer a transcribed NoteSequence from 16 kHz mono audio samples.

    Args:
      audio: 1-D numpy array of 16 kHz mono audio samples.

    Returns:
      A `note_seq.NoteSequence` of the transcribed audio.
    """
    specs, start_times = self._preprocess_audio(audio)
    cross_ks, cross_vs = self._encode_fn(specs)
    mx.eval(*cross_ks, *cross_vs)
    raw_tokens = self._decode_segments(
        cross_ks, cross_vs, batch_size=specs.shape[0]
    )
    decoded_tokens = self._decode_vocab_ids(raw_tokens)
    predictions = [
        self._postprocess(tokens, start_time)
        for tokens, start_time in zip(decoded_tokens, start_times)
    ]
    result = metrics_utils.event_predictions_to_ns(
        predictions, codec=self.codec, encoding_spec=self.encoding_spec
    )
    return result['est_ns']

  def _audio_to_frames(
      self, audio: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Pad 1-D audio to a multiple of hop_width and split into frames."""
    frame_size = self.spectrogram_config.hop_width
    padding = [0, frame_size - len(audio) % frame_size]
    audio_padded = np.pad(audio, padding, mode='constant').astype(np.float32)
    num_frames = len(audio_padded) // frame_size
    frames = audio_padded.reshape(num_frames, frame_size)
    times = (
        np.arange(num_frames, dtype=np.float64)
        / self.spectrogram_config.frames_per_second
    )
    return frames, times

  def _preprocess_audio(
      self, audio: np.ndarray
  ) -> Tuple[mx.array, list[float]]:
    """Split audio into segments and compute log-mel spectrograms in MLX."""
    frames, times = self._audio_to_frames(audio)
    num_frames = len(frames)
    frame_size = self.spectrogram_config.hop_width
    inputs_length = self.inputs_length
    num_segments = int(np.ceil(num_frames / inputs_length))
    start_times = [float(times[i * inputs_length]) for i in range(num_segments)]
    seg_lengths = [
        min(inputs_length, num_frames - i * inputs_length)
        for i in range(num_segments)
    ]

    total_padded_frames = num_segments * inputs_length
    if total_padded_frames > num_frames:
      frames_padded = np.pad(
          frames,
          ((0, total_padded_frames - num_frames), (0, 0)),
          mode='constant',
      )
    else:
      frames_padded = frames

    audio_segs = mx.array(
        frames_padded.reshape(num_segments, inputs_length * frame_size)
    )
    specs = _compute_logmel(
        audio_segs,
        self._mel_mat,
        self._hann_win,
        fft_size=spectrograms.FFT_SIZE,
        hop_width=frame_size,
    )

    last_len = seg_lengths[-1]
    if last_len < inputs_length:
      mask = mx.arange(inputs_length) < last_len
      last_spec = mx.where(
          mask[:, None], specs[-1], mx.array(0.0, dtype=specs.dtype)
      )
      specs = mx.concatenate([specs[:-1], last_spec[None, :, :]], axis=0)

    specs = specs.astype(self.dtype)
    mx.eval(specs)
    return specs, start_times

  def _decode_segments(
      self, cross_ks: list[mx.array], cross_vs: list[mx.array], batch_size: int
  ) -> np.ndarray:
    """Autoregressively decode token sequences with bucketed KV & EOS compaction."""
    bucket_idx = 0
    cur_bucket = _KV_CACHE_BUCKETS[0]
    n_heads = self.model.num_heads
    h_dim = self.model.head_dim
    n_layers = self.model.num_decoder_layers
    eos_id = self.vocabulary.eos_id
    k_caches = [
        mx.zeros((batch_size, n_heads, cur_bucket, h_dim), dtype=self.dtype)
        for _ in range(n_layers)
    ]
    v_caches = [
        mx.zeros((batch_size, n_heads, cur_bucket, h_dim), dtype=self.dtype)
        for _ in range(n_layers)
    ]
    arange_cache = mx.arange(cur_bucket)[None, None, None, :]

    cur_tokens_1d = mx.zeros((batch_size,), dtype=mx.int32)
    active_indices = np.arange(batch_size, dtype=np.int32)
    finished_tokens = np.full(
        (batch_size, self.outputs_length), eos_id, dtype=np.int32
    )
    chunk_tokens = []
    chunk_start_step = 0

    for step in range(0, self.outputs_length, 4):
      if step == cur_bucket:
        bucket_idx += 1
        next_bucket = _KV_CACHE_BUCKETS[bucket_idx]
        b_cur = cur_tokens_1d.shape[0]
        pad_zeros = mx.zeros(
            (b_cur, n_heads, next_bucket - cur_bucket, h_dim), dtype=self.dtype
        )
        k_caches = [mx.concatenate([kc, pad_zeros], axis=2) for kc in k_caches]
        v_caches = [mx.concatenate([vc, pad_zeros], axis=2) for vc in v_caches]
        cur_bucket = next_bucket
        arange_cache = mx.arange(cur_bucket)[None, None, None, :]

      s0 = mx.array([step], dtype=mx.int32)
      s1 = mx.array([step + 1], dtype=mx.int32)
      s2 = mx.array([step + 2], dtype=mx.int32)
      s3 = mx.array([step + 3], dtype=mx.int32)
      m0 = arange_cache <= s0
      m1 = arange_cache <= s1
      m2 = arange_cache <= s2
      m3 = arange_cache <= s3

      toks_4, cur_tokens_1d, k_caches, v_caches = self._decode_4_steps_fn(
          cur_tokens_1d,
          s0,
          s1,
          s2,
          s3,
          m0,
          m1,
          m2,
          m3,
          k_caches,
          v_caches,
          cross_ks,
          cross_vs,
      )
      chunk_tokens.append(toks_4)

      next_step = step + 4
      check_interval = 16 if next_step >= 128 else 64
      if next_step % check_interval == 0 or next_step == self.outputs_length:
        chunk_mat = mx.concatenate(chunk_tokens, axis=1)
        mx.eval(chunk_mat, cur_tokens_1d, *k_caches, *v_caches)
        chunk_np = np.array(chunk_mat)
        finished_tokens[active_indices, chunk_start_step:next_step] = chunk_np
        chunk_tokens = []
        chunk_start_step = next_step

        has_eos = np.any(chunk_np == eos_id, axis=1)
        if np.all(has_eos):
          break
        if np.any(has_eos):
          keep_mask = ~has_eos
          keep_idx_np = np.where(keep_mask)[0]
          keep_idx_mx = mx.array(keep_idx_np, dtype=mx.int32)
          active_indices = active_indices[keep_idx_np]
          cur_tokens_1d = mx.take(cur_tokens_1d, keep_idx_mx, axis=0)
          k_caches = [mx.take(kc, keep_idx_mx, axis=0) for kc in k_caches]
          v_caches = [mx.take(vc, keep_idx_mx, axis=0) for vc in v_caches]
          cross_ks = [mx.take(ck, keep_idx_mx, axis=0) for ck in cross_ks]
          cross_vs = [mx.take(cv, keep_idx_mx, axis=0) for cv in cross_vs]

    return finished_tokens[:, :chunk_start_step]

  def _decode_vocab_ids(self, raw_tokens: np.ndarray) -> np.ndarray:
    """Map raw vocabulary token IDs to decoded event IDs."""
    raw_tokens = np.asarray(raw_tokens, dtype=np.int32)
    num_special = self.vocabulary._num_special_tokens  # pylint: disable=protected-access
    eos_id = self.vocabulary.eos_id
    num_classes = self.codec.num_classes
    valid_mask = (raw_tokens >= num_special) & (
        raw_tokens < num_special + num_classes
    )
    decoded = np.where(
        valid_mask,
        raw_tokens - num_special,
        vocabularies.DECODED_INVALID_ID,
    )
    return np.where(
        raw_tokens == eos_id, vocabularies.DECODED_EOS_ID, decoded
    ).astype(np.int32)

  def _postprocess(
      self, tokens: Sequence[int], start_time: float
  ) -> dict[str, Any]:
    """Trim EOS and align segment start_time to the codec step grid."""
    tokens = self._trim_eos(tokens)
    start_time = float(start_time)
    start_time -= start_time % (1 / self.codec.steps_per_second)
    return {
        'est_tokens': tokens,
        'start_time': start_time,
        'raw_inputs': [],
    }

  @staticmethod
  def _trim_eos(tokens: Sequence[int]) -> np.ndarray:
    """Truncate a decoded token sequence at the first DECODED_EOS_ID."""
    tokens = np.array(tokens, np.int32)
    if vocabularies.DECODED_EOS_ID in tokens:
      tokens = tokens[: np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
    return tokens


MLXInferenceModel = InferenceModel
