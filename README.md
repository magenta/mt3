# MT3: Multi-Task Multitrack Music Transcription

MT3 is a multi-instrument automatic music transcription model that uses the [T5X framework](https://github.com/google-research/t5x).

This is not an officially supported Google product.

## Transcribe your own audio

Use our [colab notebook](https://colab.research.google.com/github/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb) to
transcribe audio files of your choosing.  You can use a pretrained checkpoint from
either a) the piano transcription model described in [our ISMIR 2021 paper](https://archives.ismir.net/ismir2021/paper/000030.pdf)
or b) the multi-instrument transcription model described in
[our ICLR 2022 paper](https://openreview.net/pdf?id=iMSjopcOn0p).

### Local inference on Apple Silicon (MLX)

On Apple Silicon Macs, you can install the optional `mlx` dependency and use `mt3.mlx_inference.InferenceModel` as a fast drop-in replacement for the JAX inference model using the same pretrained T5X checkpoints:

```bash
pip install -e ".[mlx]"
```

```python
import librosa
import note_seq
from mt3 import mlx_inference

audio, _ = librosa.load('audio.wav', sr=16000, mono=True)
model = mlx_inference.InferenceModel('checkpoints/mt3/', model_type='mt3')
est_ns = model(audio)
note_seq.sequence_proto_to_midi_file(est_ns, 'transcribed.mid')
```


## Train a model

For now, we do not (easily) support training.  If you like, you can try to
follow the [T5X training instructions](https://github.com/google-research/t5x#training)
and use one of the tasks defined in [tasks.py](mt3/tasks.py).
