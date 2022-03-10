# MT3: Multi-Task Multitrack Music Transcription

MT3 is a multi-instrument automatic music transcription model that uses the [T5X framework](https://github.com/google-research/t5x). 

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/MT3)

This is not an officially supported Google product.

## Transcribe your own audio

* [Try out Web Demo on Huggingface Spaces 🤗](https://huggingface.co/spaces/akhaliq/MT3)

Use our [colab notebook](mt3/colab/music_transcription_with_transformers.ipynb) to
transcribe audio files of your choosing.  You can use a pretrained checkpoint from
either a) the piano transcription model described in [our ISMIR 2021 paper](https://archives.ismir.net/ismir2021/paper/000030.pdf)
or b) the multi-instrument transcription model described in
[our ICLR 2022 paper](https://openreview.net/pdf?id=iMSjopcOn0p).


## Train a model

For now, we do not (easily) support training.  If you like, you can try to
follow the [T5X training instructions](https://github.com/google-research/t5x#training)
and use one of the tasks defined in [tasks.py](mt3/tasks.py).
