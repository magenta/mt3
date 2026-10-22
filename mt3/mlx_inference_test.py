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

"""Tests for mt3.mlx_inference."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import flax.traverse_util
import jax
import jax.numpy as jnp
import mlx.core as mx
from mt3 import layers
from mt3 import mlx_inference
from mt3 import network
from mt3 import spectrograms
from mt3 import vocabularies
import note_seq
import numpy as np
import tensorflow as tf


def _flatten_flax_params(params_tree) -> dict[str, np.ndarray]:
  flat = flax.traverse_util.flatten_dict(params_tree, sep='.')
  return {k: np.array(v, dtype=np.float32, copy=True) for k, v in flat.items()}


class MlxInferenceTest(parameterized.TestCase):

  def test_mel_weight_matrix_matches_tf_signal(self):
    num_spec_bins = spectrograms.FFT_SIZE // 2 + 1
    tf_mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=spectrograms.DEFAULT_NUM_MEL_BINS,
        num_spectrogram_bins=num_spec_bins,
        sample_rate=spectrograms.DEFAULT_SAMPLE_RATE,
        lower_edge_hertz=spectrograms.MEL_LO_HZ,
        upper_edge_hertz=mlx_inference._MEL_HI_HZ,
    ).numpy()
    mlx_mel = mlx_inference._linear_to_mel_weight_matrix()
    self.assertEqual(
        mlx_mel.shape, (num_spec_bins, spectrograms.DEFAULT_NUM_MEL_BINS)
    )
    np.testing.assert_allclose(mlx_mel, tf_mel, rtol=1e-4, atol=1e-4)

  def test_compute_logmel_matches_mt3_spectrograms(self):
    rng = np.random.RandomState(0)
    audio = (rng.randn(32 * 128) * 0.1).astype(np.float32)
    spec_cfg = spectrograms.SpectrogramConfig()
    tf_spec = spectrograms.compute_spectrogram(audio, spec_cfg).numpy()

    mel_mat = mx.array(mlx_inference._linear_to_mel_weight_matrix())
    fft_size = spectrograms.FFT_SIZE
    hann_win = mx.array(
        0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(fft_size) / fft_size),
        dtype=mx.float32,
    )
    with mx.stream(mx.cpu):
      mlx_spec = mlx_inference._compute_logmel(
          mx.array(audio), mel_mat, hann_win
      )
      mx.eval(mlx_spec)
    mlx_spec_np = np.array(mlx_spec)

    self.assertEqual(mlx_spec_np.shape, tf_spec.shape)
    np.testing.assert_allclose(mlx_spec_np, tf_spec, rtol=1e-4, atol=1e-4)

  def test_sinusoidal_embeddings_match_flax_layers(self):
    flax_pe = np.asarray(layers.sinusoidal()(None, (2048, 512), jnp.float32))
    mlx_pe = np.array(mlx_inference._sinusoidal_embeddings(2048, 512))
    np.testing.assert_allclose(mlx_pe, flax_pe, rtol=1e-5, atol=1e-5)

  def test_encoder_and_decoder_parity_with_flax(self):
    cfg = network.T5Config(
        vocab_size=64,
        dtype=jnp.float32,
        emb_dim=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        head_dim=8,
        mlp_dim=64,
        mlp_activations=('gelu', 'linear'),
        dropout_rate=0.0,
        logits_via_embedding=False,
    )
    flax_module = network.Transformer(config=cfg)
    rng = jax.random.PRNGKey(42)
    dummy_enc_in = jnp.ones((2, 16, 512), dtype=jnp.float32)
    dummy_dec_in = jnp.zeros((2, 8), dtype=jnp.int32)
    init_vars = flax_module.init(
        rng,
        dummy_enc_in,
        dummy_dec_in,
        dummy_dec_in,
        enable_dropout=False,
    )
    flat_params = _flatten_flax_params(init_vars['params'])

    inferred_cfg, input_depth = mlx_inference._infer_config_from_params(
        flat_params, head_dim=8
    )
    self.assertEqual(inferred_cfg.vocab_size, 64)
    self.assertEqual(inferred_cfg.emb_dim, 32)
    self.assertEqual(inferred_cfg.num_heads, 4)
    self.assertEqual(inferred_cfg.head_dim, 8)
    self.assertEqual(inferred_cfg.mlp_dim, 64)
    self.assertEqual(inferred_cfg.num_encoder_layers, 2)
    self.assertEqual(inferred_cfg.num_decoder_layers, 2)
    self.assertEqual(input_depth, 512)

    mlx_model = mlx_inference._Transformer(
        config=inferred_cfg, input_depth=input_depth
    )
    mlx_model._load_from_t5x_dict(flat_params)

    np_rng = np.random.RandomState(7)
    test_specs = np_rng.randn(2, 16, 512).astype(np.float32) * 0.1
    flax_encoded = np.asarray(
        flax_module.apply(
            {'params': init_vars['params']},
            jnp.array(test_specs),
            enable_dropout=False,
            method=flax_module.encode,
        )
    )

    with mx.stream(mx.cpu):
      mlx_encoded = mlx_model.encode(mx.array(test_specs))
      mx.eval(mlx_encoded)
      enc_fn, dec_4_fn = mlx_inference._make_compiled_fns(mlx_model)
      cross_ks, cross_vs = enc_fn(mx.array(test_specs))
      mx.eval(*cross_ks, *cross_vs)

      k_caches = [mx.zeros((2, 4, 8, 8)) for _ in range(2)]
      v_caches = [mx.zeros((2, 4, 8, 8)) for _ in range(2)]
      arange_8 = mx.arange(8)[None, None, None, :]
      s0 = mx.array([0], dtype=mx.int32)
      s1 = mx.array([1], dtype=mx.int32)
      s2 = mx.array([2], dtype=mx.int32)
      s3 = mx.array([3], dtype=mx.int32)
      toks_4, _, _, _ = dec_4_fn(
          mx.zeros((2,), dtype=mx.int32),
          s0,
          s1,
          s2,
          s3,
          arange_8 <= s0,
          arange_8 <= s1,
          arange_8 <= s2,
          arange_8 <= s3,
          k_caches,
          v_caches,
          cross_ks,
          cross_vs,
      )
      mx.eval(toks_4)

    np.testing.assert_allclose(
        np.array(mlx_encoded), flax_encoded, rtol=1e-5, atol=1e-5
    )

    # Verify step-0 greedy token parity against Flax decode.
    flax_logits_0 = flax_module.apply(
        {'params': init_vars['params']},
        jnp.array(flax_encoded),
        jnp.array(test_specs),
        jnp.zeros((2, 1), dtype=jnp.int32),
        jnp.zeros((2, 1), dtype=jnp.int32),
        enable_dropout=False,
        decode=False,
        method=flax_module.decode,
    )
    flax_tok0 = np.argmax(np.asarray(flax_logits_0[:, 0, :]), axis=-1)
    np.testing.assert_array_equal(np.array(toks_4)[:, 0], flax_tok0)

  @parameterized.parameters('bfloat16', 'float16', 'float32')
  def test_inference_model_api_and_call(self, dtype):
    cfg = network.T5Config(
        vocab_size=1536,
        dtype=jnp.float32,
        emb_dim=64,
        num_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        head_dim=64,
        mlp_dim=128,
        mlp_activations=('gelu', 'linear'),
        dropout_rate=0.0,
        logits_via_embedding=False,
    )
    flax_module = network.Transformer(config=cfg)
    init_vars = flax_module.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, 16, 512), dtype=jnp.float32),
        jnp.zeros((1, 8), dtype=jnp.int32),
        jnp.zeros((1, 8), dtype=jnp.int32),
        enable_dropout=False,
    )
    flat_params = _flatten_flax_params(init_vars['params'])
    # Bias EOS token (index 1) strongly so synthetic model terminates in 1 chunk.
    flat_params['decoder.logits_dense.kernel'][:, 1] = 100.0

    model = mlx_inference.InferenceModel(
        flat_params, model_type='mt3', dtype=dtype
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      weights_file = os.path.join(tmp_dir, 'model.npz')
      model.save_weights(weights_file)
      model = mlx_inference.InferenceModel(
          weights_file, model_type='mt3', dtype=dtype
      )

    self.assertEqual(
        model.input_shapes,
        {
            'encoder_input_tokens': (8, 256),
            'decoder_input_tokens': (8, 1024),
        },
    )

    dummy_batch = {
        'encoder_input_tokens': np.zeros((2, 256, 512), dtype=np.float32)
    }
    tokens = model.predict_tokens(dummy_batch)
    self.assertEqual(tokens.shape[0], 2)
    self.assertEqual(tokens[0, 0], vocabularies.DECODED_EOS_ID)

    dummy_audio = np.zeros(16000, dtype=np.float32)
    est_ns = model(dummy_audio)
    self.assertIsInstance(est_ns, note_seq.NoteSequence)


if __name__ == '__main__':
  absltest.main()
