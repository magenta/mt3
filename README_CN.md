# MT3：一种多乐器音乐转录模型

MT3是一种使用[T5X框架](https://github.com/google-research/t5x)的多乐器自动音乐转换模型。  
这不是Google的正式支持产品。  

## 转录你自己的音频
使用我们的[colab notebook](https://colab.research.google.com/github/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb)来转录你选择的音频文件。
你可以使用我们在[ISMIR 2021 paper](https://archives.ismir.net/ismir2021/paper/000030.pdf)中描述的钢琴转录模型预训练的检查点,或者使用在[ICLR 2022 paper](https://openreview.net/pdf?id=iMSjopcOn0p)中描述的多乐器转录模型预训练的检查点。

## 训练一个模型 
现在,我们提供便捷的模型训练支持。如果你愿意,你可以尝试遵循[T5X训练说明](https://github.com/google-research/t5x#training),并使用[tasks.py](mt3/tasks.py)训练模型。
