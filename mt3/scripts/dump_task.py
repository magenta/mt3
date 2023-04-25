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

"""Simple debugging utility for printing out task contents."""

import re

from absl import app
from absl import flags

import mt3.tasks  # pylint: disable=unused-import

import seqio
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("task", None, "A registered Task.")
flags.DEFINE_string("task_cache_dir", None, "Directory to use for task cache.")
flags.DEFINE_integer("max_examples", 10,
                     "Maximum number of examples (-1 for no limit).")
flags.DEFINE_string("format_string", "targets = {targets}",
                    "Format for printing examples.")
flags.DEFINE_string("split", "train",
                    "Which split of the dataset, e.g. train or validation.")
flags.DEFINE_integer("sequence_length_inputs", 256,
                     "Sequence length for inputs.")
flags.DEFINE_integer("sequence_length_targets", 1024,
                     "Sequence length for targets.")


def main(_):
  if FLAGS.task_cache_dir:
    seqio.add_global_cache_dirs([FLAGS.task_cache_dir])

  task = seqio.get_mixture_or_task(FLAGS.task)

  ds = task.get_dataset(
      sequence_length={
          "inputs": FLAGS.sequence_length_inputs,
          "targets": FLAGS.sequence_length_targets,
      },
      split=FLAGS.split,
      use_cached=bool(FLAGS.task_cache_dir),
      shuffle=False)

  keys = re.findall(r"{([\w+]+)}", FLAGS.format_string)
  def _example_to_string(ex):
    key_to_string = {}
    for k in keys:
      if k in ex:
        v = ex[k].numpy().tolist()
        key_to_string[k] = task.output_features[k].vocabulary.decode(v)
      else:
        key_to_string[k] = ""
    return FLAGS.format_string.format(**key_to_string)

  for ex in ds.take(FLAGS.max_examples):
    for k, v in ex.items():
      print(f"{k}: {tf.shape(v)}")
    print(_example_to_string(ex))
    print()


if __name__ == "__main__":
  flags.mark_flags_as_required(["task"])

  app.run(main)
