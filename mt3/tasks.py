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

"""Transcription task definitions."""

import functools
from typing import Optional, Sequence

from mt3 import datasets
from mt3 import event_codec
from mt3 import metrics
from mt3 import mixing
from mt3 import preprocessors
from mt3 import run_length_encoding
from mt3 import spectrograms
from mt3 import vocabularies

import note_seq
import numpy as np
import seqio
import t5
import tensorflow as tf

# Split audio frame sequences into this length before the cache placeholder.
MAX_NUM_CACHED_FRAMES = 2000

seqio.add_global_cache_dirs(['gs://mt3/data/cache_tasks/'])


def construct_task_name(
    task_prefix: str,
    spectrogram_config=spectrograms.SpectrogramConfig(),
    vocab_config=vocabularies.VocabularyConfig(),
    task_suffix: Optional[str] = None
) -> str:
  """Construct task name from prefix, config, and optional suffix."""
  fields = [task_prefix]
  if spectrogram_config.abbrev_str:
    fields.append(spectrogram_config.abbrev_str)
  if vocab_config.abbrev_str:
    fields.append(vocab_config.abbrev_str)
  if task_suffix:
    fields.append(task_suffix)
  return '_'.join(fields)


def trim_eos(tokens: Sequence[int]) -> np.ndarray:
  """If EOS is present, remove it and everything after."""
  tokens = np.array(tokens, np.int32)
  if vocabularies.DECODED_EOS_ID in tokens:
    tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
  return tokens


def postprocess(tokens, example, is_target, codec):
  """Transcription postprocessing function."""
  tokens = trim_eos(tokens)

  if is_target:
    return {
        'unique_id': example['unique_id'][0],
        'ref_ns': (note_seq.NoteSequence.FromString(example['sequence'][0])
                   if example['sequence'][0] else None),
        'ref_tokens': tokens,
    }

  start_time = example['input_times'][0]
  # Round down to nearest symbolic token step.
  start_time -= start_time % (1 / codec.steps_per_second)

  return {
      'unique_id': example['unique_id'][0],
      'raw_inputs': example['raw_inputs'],
      'est_tokens': tokens,
      'start_time': start_time
  }


def add_transcription_task_to_registry(
    dataset_config: datasets.DatasetConfig,
    spectrogram_config: spectrograms.SpectrogramConfig,
    vocab_config: vocabularies.VocabularyConfig,
    tokenize_fn,  # TODO(iansimon): add type signature
    onsets_only: bool,
    include_ties: bool,
    skip_too_long: bool = False
) -> None:
  """Add note transcription task to seqio.TaskRegistry."""
  codec = vocabularies.build_codec(vocab_config)
  vocabulary = vocabularies.vocabulary_from_codec(codec)

  output_features = {
      'targets': seqio.Feature(vocabulary=vocabulary),
      'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2)
  }

  task_name = 'onsets' if onsets_only else 'notes'
  if include_ties:
    task_name += '_ties'
  task_prefix = f'{dataset_config.name}_{task_name}'

  train_task_name = construct_task_name(
      task_prefix=task_prefix,
      spectrogram_config=spectrogram_config,
      vocab_config=vocab_config,
      task_suffix='train')

  mixture_task_names = []

  tie_token = codec.encode_event(event_codec.Event('tie', 0))
  track_specs = (dataset_config.track_specs
                 if dataset_config.track_specs else None)

  # Add transcription training task.
  seqio.TaskRegistry.add(
      train_task_name,
      source=seqio.TFExampleDataSource(
          split_to_filepattern={
              'train': dataset_config.paths[dataset_config.train_split],
              'eval': dataset_config.paths[dataset_config.train_eval_split]
          },
          feature_description=dataset_config.features),
      output_features=output_features,
      preprocessors=[
          functools.partial(
              tokenize_fn,
              spectrogram_config=spectrogram_config, codec=codec,
              is_training_data=True, onsets_only=onsets_only,
              include_ties=include_ties),
          functools.partial(
              t5.data.preprocessors.split_tokens,
              max_tokens_per_segment=MAX_NUM_CACHED_FRAMES,
              feature_key='inputs',
              additional_feature_keys=[
                  'input_event_start_indices', 'input_event_end_indices',
                  'input_state_event_indices'
              ],
              passthrough_feature_keys=['targets', 'state_events']),
          seqio.CacheDatasetPlaceholder(),
          functools.partial(
              t5.data.preprocessors.select_random_chunk,
              feature_key='inputs',
              additional_feature_keys=[
                  'input_event_start_indices', 'input_event_end_indices',
                  'input_state_event_indices'
              ],
              passthrough_feature_keys=['targets', 'state_events'],
              uniform_random_start=True),
          functools.partial(
              run_length_encoding.extract_target_sequence_with_indices,
              state_events_end_token=tie_token if include_ties else None),
          functools.partial(preprocessors.map_midi_programs, codec=codec),
          run_length_encoding.run_length_encode_shifts_fn(
              codec,
              feature_key='targets'),
          functools.partial(
              mixing.mix_transcription_examples,
              codec=codec,
              targets_feature_keys=['targets']),
          run_length_encoding.remove_redundant_state_changes_fn(
              feature_key='targets', codec=codec,
              state_change_event_types=['velocity', 'program']),
          functools.partial(
              preprocessors.compute_spectrograms,
              spectrogram_config=spectrogram_config),
          functools.partial(preprocessors.handle_too_long, skip=skip_too_long),
          functools.partial(
              seqio.preprocessors.tokenize_and_append_eos,
              copy_pretokenized=False)
      ],
      postprocess_fn=None,
      metric_fns=[],
  )

  # Add transcription eval tasks.
  for split in dataset_config.infer_eval_splits:
    eval_task_name = construct_task_name(
        task_prefix=task_prefix,
        spectrogram_config=spectrogram_config,
        vocab_config=vocab_config,
        task_suffix=split.suffix)

    if split.include_in_mixture:
      mixture_task_names.append(eval_task_name)

    seqio.TaskRegistry.add(
        eval_task_name,
        source=seqio.TFExampleDataSource(
            split_to_filepattern={'eval': dataset_config.paths[split.name]},
            feature_description=dataset_config.features),
        output_features=output_features,
        preprocessors=[
            functools.partial(
                tokenize_fn,
                spectrogram_config=spectrogram_config, codec=codec,
                is_training_data='train' in split.name, onsets_only=onsets_only,
                include_ties=include_ties),
            seqio.CacheDatasetPlaceholder(),
            preprocessors.add_unique_id,
            preprocessors.pad_notesequence_array,
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                feature_key='inputs',
                additional_feature_keys=['input_times', 'sequence'],
                passthrough_feature_keys=['unique_id']),
            # Add dummy targets as they are dropped during the above split to
            # avoid memory blowups, but expected to be present by seqio; the
            # evaluation metrics currently only use the target NoteSequence.
            preprocessors.add_dummy_targets,
            functools.partial(
                preprocessors.compute_spectrograms,
                spectrogram_config=spectrogram_config),
            functools.partial(preprocessors.handle_too_long, skip=False),
            functools.partial(
                seqio.preprocessors.tokenize_and_append_eos,
                copy_pretokenized=False)
        ],
        postprocess_fn=functools.partial(postprocess, codec=codec),
        metric_fns=[
            functools.partial(
                metrics.transcription_metrics,
                codec=codec,
                spectrogram_config=spectrogram_config,
                onsets_only=onsets_only,
                use_ties=include_ties,
                track_specs=track_specs)
        ],
    )

  seqio.MixtureRegistry.add(
      construct_task_name(
          task_prefix=task_prefix, spectrogram_config=spectrogram_config,
          vocab_config=vocab_config, task_suffix='eval'),
      mixture_task_names,
      default_rate=1)


# Just use default spectrogram config.
SPECTROGRAM_CONFIG = spectrograms.SpectrogramConfig()

# Create two vocabulary configs, one default and one with only on-off velocity.
VOCAB_CONFIG_FULL = vocabularies.VocabularyConfig()
VOCAB_CONFIG_NOVELOCITY = vocabularies.VocabularyConfig(num_velocity_bins=1)

# Transcribe MAESTRO v1.
add_transcription_task_to_registry(
    dataset_config=datasets.MAESTROV1_CONFIG,
    spectrogram_config=SPECTROGRAM_CONFIG,
    vocab_config=VOCAB_CONFIG_FULL,
    tokenize_fn=functools.partial(
        preprocessors.tokenize_transcription_example,
        audio_is_samples=False,
        id_feature_key='id'),
    onsets_only=False,
    include_ties=False)

# Transcribe MAESTRO v3.
add_transcription_task_to_registry(
    dataset_config=datasets.MAESTROV3_CONFIG,
    spectrogram_config=SPECTROGRAM_CONFIG,
    vocab_config=VOCAB_CONFIG_FULL,
    tokenize_fn=functools.partial(
        preprocessors.tokenize_transcription_example,
        audio_is_samples=False,
        id_feature_key='id'),
    onsets_only=False,
    include_ties=False)

# Transcribe MAESTRO v3 without velocities, with ties.
add_transcription_task_to_registry(
    dataset_config=datasets.MAESTROV3_CONFIG,
    spectrogram_config=SPECTROGRAM_CONFIG,
    vocab_config=VOCAB_CONFIG_NOVELOCITY,
    tokenize_fn=functools.partial(
        preprocessors.tokenize_transcription_example,
        audio_is_samples=False,
        id_feature_key='id'),
    onsets_only=False,
    include_ties=True)

# Transcribe GuitarSet, with ties.
add_transcription_task_to_registry(
    dataset_config=datasets.GUITARSET_CONFIG,
    spectrogram_config=SPECTROGRAM_CONFIG,
    vocab_config=VOCAB_CONFIG_NOVELOCITY,
    tokenize_fn=preprocessors.tokenize_guitarset_example,
    onsets_only=False,
    include_ties=True)

# Transcribe URMP mixes, with ties.
add_transcription_task_to_registry(
    dataset_config=datasets.URMP_CONFIG,
    spectrogram_config=SPECTROGRAM_CONFIG,
    vocab_config=VOCAB_CONFIG_NOVELOCITY,
    tokenize_fn=functools.partial(
        preprocessors.tokenize_example_with_program_lookup,
        inst_name_to_program_fn=preprocessors.urmp_instrument_to_program,
        id_feature_key='id'),
    onsets_only=False,
    include_ties=True)

# Transcribe MusicNet, with ties.
add_transcription_task_to_registry(
    dataset_config=datasets.MUSICNET_CONFIG,
    spectrogram_config=SPECTROGRAM_CONFIG,
    vocab_config=VOCAB_CONFIG_NOVELOCITY,
    tokenize_fn=functools.partial(
        preprocessors.tokenize_transcription_example,
        audio_is_samples=True,
        id_feature_key='id'),
    onsets_only=False,
    include_ties=True)

# Transcribe MusicNetEM, with ties.
add_transcription_task_to_registry(
    dataset_config=datasets.MUSICNET_EM_CONFIG,
    spectrogram_config=SPECTROGRAM_CONFIG,
    vocab_config=VOCAB_CONFIG_NOVELOCITY,
    tokenize_fn=functools.partial(
        preprocessors.tokenize_transcription_example,
        audio_is_samples=True,
        id_feature_key='id'),
    onsets_only=False,
    include_ties=True)

# Transcribe Cerberus4 (piano-guitar-bass-drums quartets), with ties.
add_transcription_task_to_registry(
    dataset_config=datasets.CERBERUS4_CONFIG,
    spectrogram_config=SPECTROGRAM_CONFIG,
    vocab_config=VOCAB_CONFIG_NOVELOCITY,
    tokenize_fn=functools.partial(
        preprocessors.tokenize_slakh_example,
        track_specs=datasets.CERBERUS4_CONFIG.track_specs,
        ignore_pitch_bends=True),
    onsets_only=False,
    include_ties=True)

# Transcribe 10 random sub-mixes of each song from Slakh, with ties.
add_transcription_task_to_registry(
    dataset_config=datasets.SLAKH_CONFIG,
    spectrogram_config=SPECTROGRAM_CONFIG,
    vocab_config=VOCAB_CONFIG_NOVELOCITY,
    tokenize_fn=functools.partial(
        preprocessors.tokenize_slakh_example,
        track_specs=None,
        ignore_pitch_bends=True),
    onsets_only=False,
    include_ties=True)


# Construct task names to include in transcription mixture.
MIXTURE_DATASET_NAMES = [
    'maestrov3', 'guitarset', 'urmp', 'musicnet_em', 'cerberus4', 'slakh'
]
MIXTURE_TRAIN_TASK_NAMES = []
MIXTURE_EVAL_TASK_NAMES = []
MIXTURE_TEST_TASK_NAMES = []
for dataset_name in MIXTURE_DATASET_NAMES:
  MIXTURE_TRAIN_TASK_NAMES.append(
      construct_task_name(task_prefix=f'{dataset_name}_notes_ties',
                          spectrogram_config=SPECTROGRAM_CONFIG,
                          vocab_config=VOCAB_CONFIG_NOVELOCITY,
                          task_suffix='train'))
  MIXTURE_EVAL_TASK_NAMES.append(
      construct_task_name(task_prefix=f'{dataset_name}_notes_ties',
                          spectrogram_config=SPECTROGRAM_CONFIG,
                          vocab_config=VOCAB_CONFIG_NOVELOCITY,
                          task_suffix='validation'))
MIXING_TEMPERATURE = 10 / 3

# Add the mixture of all transcription tasks, with ties.
seqio.MixtureRegistry.add(
    construct_task_name(
        task_prefix='mega_notes_ties',
        spectrogram_config=SPECTROGRAM_CONFIG,
        vocab_config=VOCAB_CONFIG_NOVELOCITY,
        task_suffix='train'),
    MIXTURE_TRAIN_TASK_NAMES,
    default_rate=functools.partial(
        seqio.mixing_rate_num_examples,
        temperature=MIXING_TEMPERATURE))
seqio.MixtureRegistry.add(
    construct_task_name(
        task_prefix='mega_notes_ties',
        spectrogram_config=SPECTROGRAM_CONFIG,
        vocab_config=VOCAB_CONFIG_NOVELOCITY,
        task_suffix='eval'),
    MIXTURE_EVAL_TASK_NAMES,
    default_rate=functools.partial(
        seqio.mixing_rate_num_examples,
        temperature=MIXING_TEMPERATURE))
