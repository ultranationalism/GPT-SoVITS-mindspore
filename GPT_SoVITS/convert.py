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

def convert_sovits_weight(safetensor, msname):
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
                temp.append({"name": name, "data": data})
        elif isinstance(cdict,dict):
            config[dtype]= json.dumps(cdict)
        else:
            print(f'{type(cdict)=}')
            config[dtype]=cdict
            continue

    ms.save_checkpoint(temp, msname,append_dict=config)
    print("convert GPT checkpoint(torch) to MindOne Stable Diffusion checkpoint(mindspore) success!")
# 映射表
MAPPING = {
    'dense1.weight': 'linear1.weight',
    'dense1.bias': 'linear1.bias',
    'dense2.weight': 'linear2.weight',
    'dense2.bias': 'linear2.bias',
}

def convert_gpt_weight(gpt_weight, msname):
    model = torch.load(gpt_weight,map_location="cpu")
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
                temp.append({"name": name, "data": data})
        elif isinstance(cdict,dict):
            config[dtype]= json.dumps(cdict)
        else:
            print(f'{type(cdict)=}')
            config[dtype]=cdict
            continue

    ms.save_checkpoint(temp, msname,append_dict=config)
    print("convert Stable Diffusion checkpoint(torch) to MindOne Stable Diffusion checkpoint(mindspore) success!")

convert_gpt_weight("/root/GPT-SoVITS/GPT_weights/可莉-e10.ckpt","/root/GPT-SoVITS/GPT_weights/可莉-e10-ms.ckpt")