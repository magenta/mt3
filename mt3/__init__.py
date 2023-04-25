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

"""Base module for MT3."""

from mt3 import datasets
from mt3 import event_codec
from mt3 import inference
from mt3 import layers
from mt3 import metrics
from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import run_length_encoding
from mt3 import spectrograms
from mt3 import summaries
from mt3 import tasks
from mt3 import vocabularies

from mt3.version import __version__
