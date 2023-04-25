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

"""Dataset configurations."""

import dataclasses
from typing import Mapping, Sequence, Union

from mt3 import note_sequences
import tensorflow as tf



@dataclasses.dataclass
class InferEvalSplit:
  # key in dictionary containing all dataset splits
  name: str
  # task name suffix (each eval split is a separate task)
  suffix: str
  # whether or not to include in the mixture of all eval tasks
  include_in_mixture: bool = True


@dataclasses.dataclass
class DatasetConfig:
  """Configuration for a transcription dataset."""
  # dataset name
  name: str
  # mapping from split name to path
  paths: Mapping[str, str]
  # mapping from feature name to feature
  features: Mapping[str, Union[tf.io.FixedLenFeature,
                               tf.io.FixedLenSequenceFeature]]
  # training split name
  train_split: str
  # training eval split name
  train_eval_split: str
  # list of infer eval split specs
  infer_eval_splits: Sequence[InferEvalSplit]
  # list of track specs to be used for metrics
  track_specs: Sequence[note_sequences.TrackSpec] = dataclasses.field(
      default_factory=list)

MAESTROV1_CONFIG = DatasetConfig(
    name='maestrov1',
    paths={
        'train':
            'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_ns_wav_train.tfrecord-?????-of-00010',
        'train_subset':
            'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_ns_wav_train.tfrecord-00002-of-00010',
        'validation':
            'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_ns_wav_validation.tfrecord-?????-of-00010',
        'validation_subset':
            'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_ns_wav_validation.tfrecord-0000[06]-of-00010',
        'test':
            'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_ns_wav_test.tfrecord-?????-of-00010'
    },
    features={
        'audio': tf.io.FixedLenFeature([], dtype=tf.string),
        'sequence': tf.io.FixedLenFeature([], dtype=tf.string),
        'id': tf.io.FixedLenFeature([], dtype=tf.string)
    },
    train_split='train',
    train_eval_split='validation_subset',
    infer_eval_splits=[
        InferEvalSplit(name='train', suffix='eval_train_full',
                       include_in_mixture=False),
        InferEvalSplit(name='train_subset', suffix='eval_train'),
        InferEvalSplit(name='validation', suffix='validation_full',
                       include_in_mixture=False),
        InferEvalSplit(name='validation_subset', suffix='validation'),
        InferEvalSplit(name='test', suffix='test', include_in_mixture=False)
    ])


MAESTROV3_CONFIG = DatasetConfig(
    name='maestrov3',
    paths={
        'train':
            'gs://magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0_ns_wav_train.tfrecord-?????-of-00025',
        'train_subset':
            'gs://magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0_ns_wav_train.tfrecord-00004-of-00025',
        'validation':
            'gs://magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0_ns_wav_validation.tfrecord-?????-of-00025',
        'validation_subset':
            'gs://magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0_ns_wav_validation.tfrecord-0002?-of-00025',
        'test':
            'gs://magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0_ns_wav_test.tfrecord-?????-of-00025'
    },
    features={
        'audio': tf.io.FixedLenFeature([], dtype=tf.string),
        'sequence': tf.io.FixedLenFeature([], dtype=tf.string),
        'id': tf.io.FixedLenFeature([], dtype=tf.string)
    },
    train_split='train',
    train_eval_split='validation_subset',
    infer_eval_splits=[
        InferEvalSplit(name='train', suffix='eval_train_full',
                       include_in_mixture=False),
        InferEvalSplit(name='train_subset', suffix='eval_train'),
        InferEvalSplit(name='validation', suffix='validation_full',
                       include_in_mixture=False),
        InferEvalSplit(name='validation_subset', suffix='validation'),
        InferEvalSplit(name='test', suffix='test', include_in_mixture=False)
    ])


GUITARSET_CONFIG = DatasetConfig(
    name='guitarset',
    paths={
        'train':
            'gs://mt3/data/datasets/guitarset/train.tfrecord-?????-of-00019',
        'validation':
            'gs://mt3/data/datasets/guitarset/validation.tfrecord-?????-of-00006',
    },
    features={
        'sequence': tf.io.FixedLenFeature([], dtype=tf.string),
        'audio': tf.io.FixedLenFeature([], dtype=tf.string),
        'velocity_range': tf.io.FixedLenFeature([], dtype=tf.string),
        'id': tf.io.FixedLenFeature([], dtype=tf.string),
    },
    train_split='train',
    train_eval_split='validation',
    infer_eval_splits=[
        InferEvalSplit(name='train', suffix='eval_train'),
        InferEvalSplit(name='validation', suffix='validation'),
    ])


URMP_CONFIG = DatasetConfig(
    name='urmp',
    paths={
        'train': 'gs://mt3/data/datasets/urmp/train.tfrecord',
        'validation': 'gs://mt3/data/datasets/urmp/validation.tfrecord',
    },
    features={
        'id': tf.io.FixedLenFeature([], dtype=tf.string),
        'tracks': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.int64, allow_missing=True),
        'inst_names': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.string, allow_missing=True),
        'audio': tf.io.FixedLenFeature([], dtype=tf.string),
        'sequence': tf.io.FixedLenFeature([], dtype=tf.string),
        'instrument_sequences': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.string, allow_missing=True),
    },
    train_split='train',
    train_eval_split='validation',
    infer_eval_splits=[
        InferEvalSplit(name='train', suffix='eval_train'),
        InferEvalSplit(name='validation', suffix='validation')
    ])


MUSICNET_CONFIG = DatasetConfig(
    name='musicnet',
    paths={
        'train':
            'gs://mt3/data/datasets/musicnet/musicnet-train.tfrecord-?????-of-00036',
        'validation':
            'gs://mt3/data/datasets/musicnet/musicnet-validation.tfrecord-?????-of-00005',
        'test':
            'gs://mt3/data/datasets/musicnet/musicnet-test.tfrecord-?????-of-00003'
    },
    features={
        'id': tf.io.FixedLenFeature([], dtype=tf.string),
        'sample_rate': tf.io.FixedLenFeature([], dtype=tf.float32),
        'audio': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True),
        'sequence': tf.io.FixedLenFeature([], dtype=tf.string)
    },
    train_split='train',
    train_eval_split='validation',
    infer_eval_splits=[
        InferEvalSplit(name='train', suffix='eval_train'),
        InferEvalSplit(name='validation', suffix='validation'),
        InferEvalSplit(name='test', suffix='test', include_in_mixture=False)
    ])


MUSICNET_EM_CONFIG = DatasetConfig(
    name='musicnet_em',
    paths={
        'train':
            'gs://mt3/data/datasets/musicnet_em/train.tfrecord-?????-of-00103',
        'validation':
            'gs://mt3/data/datasets/musicnet_em/validation.tfrecord-?????-of-00005',
        'test':
            'gs://mt3/data/datasets/musicnet_em/test.tfrecord-?????-of-00006'
    },
    features={
        'id': tf.io.FixedLenFeature([], dtype=tf.string),
        'sample_rate': tf.io.FixedLenFeature([], dtype=tf.float32),
        'audio': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True),
        'sequence': tf.io.FixedLenFeature([], dtype=tf.string)
    },
    train_split='train',
    train_eval_split='validation',
    infer_eval_splits=[
        InferEvalSplit(name='train', suffix='eval_train'),
        InferEvalSplit(name='validation', suffix='validation'),
        InferEvalSplit(name='test', suffix='test', include_in_mixture=False)
    ])


CERBERUS4_CONFIG = DatasetConfig(
    name='cerberus4',
    paths={
        'train':
            'gs://mt3/data/datasets/cerberus4/slakh_multi_cerberus_train_bass:drums:guitar:piano.tfrecord-?????-of-00286',
        'train_subset':
            'gs://mt3/data/datasets/cerberus4/slakh_multi_cerberus_train_bass:drums:guitar:piano.tfrecord-00000-of-00286',
        'validation':
            'gs://mt3/data/datasets/cerberus4/slakh_multi_cerberus_validation_bass:drums:guitar:piano.tfrecord-?????-of-00212',
        'validation_subset':
            'gs://mt3/data/datasets/cerberus4/slakh_multi_cerberus_validation_bass:drums:guitar:piano.tfrecord-0000?-of-00212',
        'test':
            'gs://mt3/data/datasets/cerberus4/slakh_multi_cerberus_test_bass:drums:guitar:piano.tfrecord-?????-of-00106'
    },
    features={
        'audio_sample_rate': tf.io.FixedLenFeature([], dtype=tf.int64),
        'inst_names': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.string, allow_missing=True),
        'midi_class': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.int64, allow_missing=True),
        'mix': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True),
        'note_sequences': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.string, allow_missing=True),
        'plugin_name': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.int64, allow_missing=True),
        'program_num': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.int64, allow_missing=True),
        'slakh_class': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.int64, allow_missing=True),
        'src_ids': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.string, allow_missing=True),
        'stems': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True),
        'stems_shape': tf.io.FixedLenFeature([2], dtype=tf.int64),
        'target_type': tf.io.FixedLenFeature([], dtype=tf.string),
        'track_id': tf.io.FixedLenFeature([], dtype=tf.string),
    },
    train_split='train',
    train_eval_split='validation_subset',
    infer_eval_splits=[
        InferEvalSplit(name='train', suffix='eval_train_full',
                       include_in_mixture=False),
        InferEvalSplit(name='train_subset', suffix='eval_train'),
        InferEvalSplit(name='validation', suffix='validation_full',
                       include_in_mixture=False),
        InferEvalSplit(name='validation_subset', suffix='validation'),
        InferEvalSplit(name='test', suffix='test', include_in_mixture=False)
    ],
    track_specs=[
        note_sequences.TrackSpec('bass', program=32),
        note_sequences.TrackSpec('drums', is_drum=True),
        note_sequences.TrackSpec('guitar', program=24),
        note_sequences.TrackSpec('piano', program=0)
    ])


SLAKH_CONFIG = DatasetConfig(
    name='slakh',
    paths={
        'train':
            'gs://mt3/data/datasets/slakh/slakh_multi_full_subsets_10_train_all_inst.tfrecord-?????-of-02307',
        'train_subset':
            'gs://mt3/data/datasets/slakh/slakh_multi_full_subsets_10_train_all_inst.tfrecord-00000-of-02307',
        'validation':
            'gs://mt3/data/datasets/slakh/slakh_multi_full_validation_all_inst.tfrecord-?????-of-00168',
        'validation_subset':
            'gs://mt3/data/datasets/slakh/slakh_multi_full_validation_all_inst.tfrecord-0000?-of-00168',
        'test':
            'gs://mt3/data/datasets/slakh/slakh_multi_full_test_all_inst.tfrecord-?????-of-00109'
    },
    features={
        'audio_sample_rate': tf.io.FixedLenFeature([], dtype=tf.int64),
        'inst_names': tf.io.FixedLenSequenceFeature([], dtype=tf.string,
                                                    allow_missing=True),
        'midi_class': tf.io.FixedLenSequenceFeature([], dtype=tf.int64,
                                                    allow_missing=True),
        'mix': tf.io.FixedLenSequenceFeature([], dtype=tf.float32,
                                             allow_missing=True),
        'note_sequences': tf.io.FixedLenSequenceFeature([], dtype=tf.string,
                                                        allow_missing=True),
        'plugin_name': tf.io.FixedLenSequenceFeature([], dtype=tf.int64,
                                                     allow_missing=True),
        'program_num': tf.io.FixedLenSequenceFeature([], dtype=tf.int64,
                                                     allow_missing=True),
        'slakh_class': tf.io.FixedLenSequenceFeature([], dtype=tf.int64,
                                                     allow_missing=True),
        'src_ids': tf.io.FixedLenSequenceFeature([], dtype=tf.string,
                                                 allow_missing=True),
        'stems': tf.io.FixedLenSequenceFeature([], dtype=tf.float32,
                                               allow_missing=True),
        'stems_shape': tf.io.FixedLenFeature([2], dtype=tf.int64),
        'target_type': tf.io.FixedLenFeature([], dtype=tf.string),
        'track_id': tf.io.FixedLenFeature([], dtype=tf.string),
    },
    train_split='train',
    train_eval_split='validation_subset',
    infer_eval_splits=[
        InferEvalSplit(name='train', suffix='eval_train_full',
                       include_in_mixture=False),
        InferEvalSplit(name='train_subset', suffix='eval_train'),
        InferEvalSplit(name='validation', suffix='validation_full',
                       include_in_mixture=False),
        InferEvalSplit(name='validation_subset', suffix='validation'),
        InferEvalSplit(name='test', suffix='test', include_in_mixture=False)
    ])


