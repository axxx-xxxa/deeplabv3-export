import torch
import torchvision
import numpy as np
from nets.deeplabv3_plus import DeepLab



num_classes = 21
backbone = "mobilenet"
downsample_factor = 16


model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=False)
model.load_state_dict(torch.load("E:\PycharmProject\work-deeplab\deeplabv3-plus-pytorch-main\model_data\deeplab_mobilenetv2.pth"))#保存的训练模型
model.eval()#切换到eval（）
example = torch.rand(1, 3, 512, 512)#生成一个随机输入维度的输入
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model_data\deeplab_mobilenetv2.pt")