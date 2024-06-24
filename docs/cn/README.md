<div align="center">

<h1>GPT-SoVITS-MindSpore-WebUI</h1>
强大的少样本语音转换与语音合成Web用户界面。<br><br>

[**English**](../../README.md) | [**中文简体**](./README.md) |

</div>

---

这个仓库为GPT-SoVITS模型的[MindSpore](https://www.mindspore.cn/)实现，参考[RVC-BOSS](https://github.com/RVC-Boss/GPT-SoVITS)原仓库的实现

## 功能：

1. **零样本文本到语音（TTS）：** 输入 5 秒的声音样本，即刻体验文本到语音转换。

2. **少样本 TTS：** 仅需 1 分钟的训练数据即可微调模型，提升声音相似度和真实感。

3. **跨语言支持：** 支持与训练数据集不同语言的推理，目前支持英语、日语和中文。

4. **WebUI 工具（即将支持）：** 集成工具包括自动训练集分割、中文自动语音识别(ASR)和文本标注，协助初学者创建训练数据集和 GPT/SoVITS 模型。

## 安装

### 测试通过的环境

- Python 3.9, Mindspore 2.2.3, CU116

_注: numba==0.56.4 需要 python<3.11_

### Linux

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```

### 手动安装

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 安装 FFmpeg

##### Conda 使用者

```bash
conda install ffmpeg
```

##### Ubuntu/Debian 使用者

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

## 预训练模型

你可以使用转换工具 `GPT_SoVITS/convert.py` 将pytorch的模型权重转换为mindspore的模型权重

```
cd GPT-SoVITS-mindspore
python GPT_SoVITS/convert.py --g_path path_to_your_GPT_model \
--s_path path_to_your_Sovits_model \
```

从 [GPT-SoVITS Models](https://modelscope.cn/models/ultranationalism/GPT-SoVITS-mindspore) 下载预训练模型，并将它们放置在 `GPT_SoVITS\pretrained_models` 中。

## 开启推理

### 启动WebUI

你可以使用启动脚本：

```
cd GPT-SoVITS-mindspore
bash launch_webui.sh
```

或者直接用python文件启动:

```
cd GPT-SoVITS-mindspore
python GPT_SoVITS/inference_webui.py
```

### 参考音频

上传一个参考音频片段（**必须为3-10秒**），然后填写**参考音频文本**，这基本上就是角色在音频中说的话。选择右侧的语言。

参考音频非常重要，因为它决定了输出的速度和情感。如果你没有得到你想要的输出，请尝试不同的方法。

### 推理

填写**需要合成的文本**并设置**需要合成的语种**，然后单击**合成语音**。
