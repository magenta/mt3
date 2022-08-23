# Copyright 2022 The MT3 Authors.
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

"""Install mt3."""

import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'mt3')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

setuptools.setup(
    name='mt3',
    version=__version__,
    description='Multi-Task Multitrack Music Transcription',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/magenta/mt3',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    package_data={
        '': ['*.gin'],
    },
    scripts=[],
    install_requires=[
        'absl-py == 1.1.0',
        'ddsp == 3.4.4',
        'flax == 0.5.2',
        'gin-config == 0.5.0',
        'immutabledict == 2.2.1',
        'librosa == 0.9.2',
        'mir_eval == 0.7',
        'note_seq == 0.0.3',
        'numpy == 1.21.6',
        'pretty_midi == 0.2.9',
        'scikit-learn == 1.0.2',
        'scipy == 1.7.3',
        'seqio == 0.0.8',
        't5 == 0.9.3',
        'tensorflow',
        'tensorflow-datasets == 4.6.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    keywords='music transcription machinelearning audio',
)
