import mindspore as ms
import yaml
from safetensors.torch import load_file
from torch import load,Tensor

def save_ckpt_info(ckpt_file, yaml_file):
    # 加载mindspore的ckpt文件
    ckpt_data = ms.load_checkpoint(ckpt_file)
    # 创建一个空字典，用于存储权重信息和张量形状
    ckpt_info = {}
    # 遍历ckpt文件中的权重名称和数据
    for name, data in ckpt_data.items():
        # 获取权重数据的形状
        shape = data.shape
        # 将参数的名称和形状转换为字符串，并去掉空格和逗号
        param_str = str(shape).replace(' ', '')
        # 将权重名称和形状作为键值对添加到字典中
        ckpt_info[name] = param_str
    # 打开yaml文件，如果不存在则创建
    with open(yaml_file, "w") as f:
        # 将字典写入yaml文件
        yaml.dump(ckpt_info, f)
    # 打印保存成功的信息
    print("save ckpt info to yaml file success!")

def save_torch_model_keys_to_yaml(model_file, yaml_file):
    # 使用torch库加载PyTorch模型
    #model = load_file(model_file)
    model=load(model_file)
    # 创建一个空字典，用于存储模型的键
    model_keys = {}
    a=0
    for dtype, cdict in model.items():
        a+=1
        if a==2:break
        for name,data in cdict.items():
            if isinstance(data, Tensor):
                shape = data.shape
                param_str = str(shape).replace(' ', '')
                # 将权重名称和形状作为键值对添加到字典中
                model_keys[dtype+'.'+name] = param_str
            else:
                model_keys[dtype+'.'+name] = str(data)
    # 打开yaml文件，如果不存在则创建
    with open(yaml_file, "w") as f:
        # 将模型的键写入yaml文件
        yaml.dump(model_keys, f, default_flow_style=False)
    # 打印保存成功的信息
    print("模型文件的键已保存到YAML文件中！")

# 调用函数，传入模型文件和目标YAML文件的路径
save_torch_model_keys_to_yaml("/root/GPT-SoVITS/GPT_SoVITS/pretrained_models/s2D488k.pth", "/root/GPT-SoVITS/GPT_SoVITS/tools/mindspore/model_keys.yaml")

#save_ckpt_info("../pangu_low_timestamp-127da122.ckpt","pangu_low_timestamp-127da122.yaml")