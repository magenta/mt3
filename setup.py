# Copyright 2021 The MT3 Authors.
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

# Lint as: python3
"""Install mt3."""

import setuptools

setuptools.setup(
    name='mt3',
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
        'absl-py',
        'gin-config',
        'immutabledict',
        'librosa',
        'mir_eval',
        'note_seq',
        'numpy',
        'pretty_midi',
        'scikit-learn',
        'scipy',
        'seqio',
        'tensorflow',
        'tensorflow-datasets',
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
