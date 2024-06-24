<div align="center">

<h1>GPT-SoVITS-MindSpore-WebUI</h1>
A Powerful Few-shot Voice Conversion and Text-to-Speech WebUI.<br><br>

[**English**](./README.md) | [**中文简体**](./docs/cn/README.md)

</div>

---

This repo is the implementation of the GPT-SoVITS model in [MindSpore](https://www.mindspore.cn/), reference to the implementation by [RVC-BOSS](https://github.com/RVC-Boss/GPT-SoVITS)

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

You can use the model conversion tool `GPT_SoVITS/convert.py` to transform PyTorch model weights into MindSpore model weights.

```
cd GPT-SoVITS-mindspore
python GPT_SoVITS/convert.py --g_path path_to_your_GPT_model \
--s_path path_to_your_Sovits_model \
```

Download pretrained models from [GPT-SoVITS Models](https://modelscope.cn/models/ultranationalism/GPT-SoVITS-mindspore) and place them in `GPT_SoVITS/pretrained_models`.

## Start Inference

### Launch Webui

You can use a startup script：

```
cd GPT-SoVITS-mindspore
bash launch_webui.sh
```

Or directly launch the Python file:

```
cd GPT-SoVITS-mindspore
python GPT_SoVITS/inference_webui.py
```

### Reference information

Upload a clip for reference audio (**must be 3-10 seconds**) then fill in the **Text for reference audio**, which is basically what does the character say in the audio. Choose the language on the right.

The reference audio is very important as it determines the speed and the emotion of the output. Please try different ones if you did not get your desired output.

### Inference

Fill the **inference text** and set the **inference language**, then click **Start inference**.
