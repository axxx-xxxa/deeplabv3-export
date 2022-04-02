import torch
import torchvision
import numpy as np
import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  

from nets.deeplabv3_plus import DeepLab


if __name__ == '__main__':

    # try:
    #     class_names = get_classes(classes_path)
    #     anchors = get_anchors(anchors_path)
    #     num_classes = len(class_names)
    # except:
    #     class_names = ['bolt','miss','displace']
    #     anchors = 3
    #     num_classes = len(class_names)



    num_classes = 3
    backbone = "mobilenet"
    downsample_factor = 16

    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=False)
    model.load_state_dict(torch.load(
        "../model_data/ep055-loss0.010-val_loss0.013.pth"))  # 保存的训练模型
    model.eval()  # 切换到eval（）
    # model.load_state_dict(torch.load("bolt.pt"))#保存的训练模型
    # model.eval()#切换到eval（）
    example = torch.rand(1, 3, 512, 512)#生成一个随机输入维度的输入
    # traced_m = torch.jit.trace(model, example)
    # f = 'bolt.pt'
    # torch.jit.save(traced_m, f)
    # loaded_m = torch.jit.load(f)
    # print(loaded_m(example))
    # exit()
    # for res in loaded_m(example):
    #     print(res.shape)
    # torch.onnx.export(loaded_m, x, 'model.onnx')
    torch.onnx._export(model, example, '../model_data/ep055-loss0.010-val_loss0.013.onnx', example_outputs=torch.rand((3,)), opset_version=11)
