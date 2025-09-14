import torch
import torch.nn as nn
from xvlm.models.model_vqa import XVLM



def build_XVLM(config,args):
    model = XVLM(config=config)  # 创建模型实例
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)  # 加载预训练模型
    device = torch.device(args.device)  # 设置设备，如果使用GPU训练，则为'cuda'
    model = model.to(device)  # 将模型发送到设备
    return model



