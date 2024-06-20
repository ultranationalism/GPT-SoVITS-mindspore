<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
A Powerful Few-shot Voice Conversion and Text-to-Speech WebUI.<br><br>

[**English**](./README.md) | [**中文简体**](./docs/cn/README.md)

</div>

---

## Features:

1. **Zero-shot TTS:** Input a 5-second vocal sample and experience instant text-to-speech conversion.

2. **Few-shot TTS:** Fine-tune the model with just 1 minute of training data for improved voice similarity and realism.

3. **Cross-lingual Support:** Inference in languages different from the training dataset, currently supporting English, Japanese, and Chinese.

4. **WebUI Tools(TODO):** Integrated tools include automatic training set segmentation, Chinese ASR, and text labeling, assisting beginners in creating training datasets and GPT/SoVITS models.

## Installation

### Tested Environments

- Python 3.9, Mindspore 2.2.3, CU116

### Linux

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```
### Install Manually

#### Install Dependences

```bash
pip install -r requirements.txt
```

#### Install FFmpeg

##### Conda Users

```bash
conda install ffmpeg
```

##### Ubuntu/Debian Users

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

## Pretrained Models

You can utilize the model conversion tool `GPT_SoVITS/convert.py` to transform PyTorch model weights into MindSpore model weights.

Download pretrained models from [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) and place them in `GPT_SoVITS/pretrained_models`.

Users in China region can download these two models by entering the links below and clicking "Download a copy"

- [GPT-SoVITS Models](https://www.icloud.com.cn/iclouddrive/056y_Xog_HXpALuVUjscIwTtg#GPT-SoVITS_Models)
