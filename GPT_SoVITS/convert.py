import argparse
import copy
import os.path as osp
import re

import torch
import collections
from safetensors.torch import load_file, save_file
from utils import HParams

import mindspore as ms
import json
from mindspore import Tensor

def hparams_to_dict(hparams_obj):
    if isinstance(hparams_obj, HParams):
        return {k: hparams_to_dict(v) if isinstance(v, HParams) else v for k, v in hparams_obj.items()}
    else:
        return hparams_obj

def convert_weight(safetensor, msname):
    model = torch.load(safetensor,map_location="cpu")
    temp=[]
    config={}
    for dtype, cdict in model.items():
        cdict=hparams_to_dict(cdict)
        if dtype=="weight":
            for name,data in cdict.items():
                if '.gamma' in name:
                    name = name.replace('.gamma', '.layer_norm.weight')
                elif '.beta' in name:
                    name = name.replace('.beta', '.layer_norm.bias')
                data = Tensor(data.numpy())
                # 将权重名称和形状作为键值对添加到列表中
                temp.append({"name": name, "data": data})
        elif isinstance(cdict,dict):
            config[dtype]= json.dumps(cdict)
        else:
            print(f'{type(cdict)=}')
            config[dtype]=cdict
            continue

    ms.save_checkpoint(temp, msname,append_dict=config)
    print("convert Stable Diffusion checkpoint(torch) to MindOne Stable Diffusion checkpoint(mindspore) success!")
# 映射表
MAPPING = {
    'dense1.weight': 'linear1.weight',
    'dense1.bias': 'linear1.bias',
    'dense2.weight': 'linear2.weight',
    'dense2.bias': 'linear2.bias',
}

def convert_weight_ckpt(safetensor, msname):
    model = torch.load(safetensor,map_location="cpu")
    temp=[]
    config={}
    for dtype, cdict in model.items():
        cdict=hparams_to_dict(cdict)
        if isinstance(cdict,collections.OrderedDict):
            for name,data in cdict.items():
                if 'model.' in name:
                    name = name.replace('model.', '')
                # 检查名称是否需要替换
                for new_name,old_name in MAPPING.items():
                    if old_name in name:
                        name = name.replace(old_name, new_name)
                        break

                data = Tensor(data.numpy())
                # 将权重名称和形状作为键值对添加到列表中
                temp.append({"name": name, "data": data})
        elif isinstance(cdict,dict):
            config[dtype]= json.dumps(cdict)
        else:
            print(f'{type(cdict)=}')
            config[dtype]=cdict
            continue

    ms.save_checkpoint(temp, msname,append_dict=config)
    print("convert Stable Diffusion checkpoint(torch) to MindOne Stable Diffusion checkpoint(mindspore) success!")


def convert_weight_back(msname, pth):
    # 加载MindOne格式的权重文件
    sd = ms.load_checkpoint(msname)
    # 获取MindOne格式的权重名称列表
    key_ms = list(sd.keys())
    # 复制MindOne格式的权重名称列表，用于修改为torch格式的权重名称
    key_torch = copy.deepcopy(key_ms)
    # 遍历MindOne格式的权重名称列表
    for i in range(len(key_ms)):
        # 如果权重名称中包含归一化层的标识
        if ("norm" in key_ms[i]) or ("ln_" in key_ms[i]) or ("model.diffusion_model.out.0." in key_ms[i]):
            # 如果权重名称中包含gamma，替换为weight
            if "gamma" in key_ms[i]:
                key_torch[i] = key_torch[i][:-5] + "weight"
            # 如果权重名称中包含beta，替换为bias
            if "beta" in key_ms[i]:
                key_torch[i] = key_torch[i][:-4] + "bias"
        # 定义卷积层的权重名称的正则表达式
        pattern1 = r"model\.diffusion_model\.(input_blocks|middle_block|output_blocks)\.[0-8]\.0\.(in_layers|out_layers)\.0\.(gamma|beta)"
        pattern2 = r"model\.diffusion_model\.middle_block\.[02]\.(in_layers|out_layers)\.0\.(gamma|beta)"
        # 如果权重名称符合卷积层的正则表达式
        if re.match(pattern1, key_ms[i]) or re.match(pattern2, key_ms[i]):
            # 如果权重名称中包含gamma，替换为weight
            if "gamma" in key_ms[i]:
                key_torch[i] = key_torch[i][:-5] + "weight"
            # 如果权重名称中包含beta，替换为bias
            if "beta" in key_ms[i]:
                key_torch[i] = key_torch[i][:-4] + "bias"
        # 如果权重名称中包含嵌入层的标识
        if "embedding_table" in key_ms[i]:
            # 替换为embedding.weight
            key_torch[i] = key_torch[i][:-15] + "weight"
        # 如果权重名称中包含文本投影层的标识
        if "conditioner.embedders.1.model.text_projection" in key_ms[i]:
            # 替换为conditioner.embedders.1.model.text_projection.weight
            key_torch[i] = "conditioner.embedders.1.model.text_projection.weight"
    # 创建一个空字典，用于存储转换后的权重数据
    newckpt = {}
    # 遍历MindOne格式和torch格式的权重名称列表
    for i in range(len(key_ms)):
        # 获取MindOne格式和torch格式的权重名称
        kms, kt = key_ms[i], key_torch[i]
        # 将MindOne格式的权重数据转换为torch的Tensor对象
        newckpt[kt] = torch.from_numpy(sd[kms].data.asnumpy())
    newckpt = {k: v.half() for k, v in newckpt.items()}
    # 保存torch格式的权重文件
    save_file(newckpt, pth)
    # 打印转换成功的信息
    print("convert MindOne Stable Diffusion checkpoint(mindspore) to Stable Diffusion checkpoint(torch) success!")

convert_weight_ckpt("/root/GPT-SoVITS/GPT_weights/可莉-e10.ckpt","/root/GPT-SoVITS/GPT_weights/可莉-e10-ms.ckpt")