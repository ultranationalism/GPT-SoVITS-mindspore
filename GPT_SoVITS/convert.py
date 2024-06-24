import argparse
import os
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

parser = argparse.ArgumentParser()
parser.add_argument('--s_path', type=str, default=None, help='SoVITS模型路径')
parser.add_argument('--g_path', type=str, default=None, help='GPT模型路径')
args = parser.parse_args()

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
            config[dtype]=cdict
            continue

    ms.save_checkpoint(temp, msname,append_dict=config)
    print(f"convert sovits(torch):{safetensor} to sovits(mindspore):{msname} success!")
# 映射表
MAPPING = {
    'dense1.weight': 'linear1.weight',
    'dense1.bias': 'linear1.bias',
    'dense2.weight': 'linear2.weight',
    'dense2.bias': 'linear2.bias',
    "norm1.weight":"norm1.gamma",
    "norm1.bias":"norm1.beta",
    "norm2.weight":"norm2.gamma",
    "norm2.bias":"norm2.beta",
}

def convert_gpt_weight(gpt_weight, msname):
    model = torch.load(gpt_weight,map_location="cpu")
    temp=[]
    config={}
    for dtype, cdict in model.items():
        cdict=hparams_to_dict(cdict)
        if dtype=='weight':
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
            config[dtype]=cdict
            continue

    ms.save_checkpoint(temp, msname,append_dict=config)
    print(f"convert GPT(torch):{gpt_weight} to GPT(mindspore):{msname} success!")

if args.s_path is not None:
    if not os.path.exists('SoVITS_weights'):
        os.makedirs('SoVITS_weights')
    s_filename = os.path.basename(args.s_path)
    convert_sovits_weight(args.s_path, os.path.join('SoVITS_weights', s_filename.replace('.pth', '-ms.ckpt')))
else:
    print("Skip SoVITS model conversion")

if args.g_path is not None:
    if not os.path.exists('GPT_weights'):
        os.makedirs('GPT_weights')
    g_filename = os.path.basename(args.g_path)
    convert_gpt_weight(args.g_path, os.path.join('GPT_weights', g_filename.replace('.ckpt', '-ms.ckpt')))
else:
    print("Skip GPT model conversion")
