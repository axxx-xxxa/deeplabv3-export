import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  
import numpy as np
import pycuda.driver as cudadriver
import tensorrt as trt
import torch
import os
import common
from PIL import Image
import cv2
import torchvision
import argparse
from nets.deeplabv3_plus import DeepLab


def PTH_build_ONNX(pth_file_path,num_classes,backbone,downsample_factor):
    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=False)
    model.load_state_dict(torch.load(pth_file_path))
    model.eval() 
    example = torch.rand(1, 3, 512, 512)
    onnx_name = pth_file_path.split(".",1)[0]+'.onnx'
    torch.onnx._export(model, example, onnx_name, example_outputs=torch.rand((3,)), opset_version=11)
    print("pth2onnx finished")
def ONNX_build_engine(onnx_file_path, write_engine=True):

    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    batch_size = 1  # trt推理时最大支持的batchsize
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:
        builder.max_batch_size = batch_size
        config = builder.create_builder_config()
        config.max_workspace_size = common.GiB(2)  
        config.set_flag(trt.BuilderFlag.FP16)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        profile = builder.create_optimization_profile()  # 动态输入时候需要 分别为最小输入、常规输入、最大输入
        # 有几个输入就要写几个profile.set_shape 名字和转onnx的时候要对应
        # tensorrt6以后的版本是支持动态输入的，需要给每个动态输入绑定一个profile，用于指定最小值，常规值和最大值，如果超出这个范围会报异常。
        profile.set_shape("inputs", (1, 3, 240, 240), (1, 3, 480, 480), (1, 3, 960, 960))
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)
        print("Completed creating Engine")
        # 保存engine文件
        engine_name = onnx_file_path.split(".",1)[0]+'.engine'
        if write_engine:
            with open(engine_name, "wb") as f:
                f.write(engine.serialize())
        print("onnx2engine finished")
        return engine

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('app', type=str,help='pth2onnx or onnx2engine')
    parser.add_argument('-n','-num_classes', type=int, default=3, help='num classes')
    parser.add_argument('-b','-backbone', type=str, default='mobilenet', help='backbone')
    parser.add_argument('-d','-downsample_factor', type=int, default=16, help='downsample_factor') 
    parser.add_argument('-p','-pth_path', type=str, help='pth_path or onnx_path') 
    opt = parser.parse_args()
    print(opt)
    if opt.app == 'pth2onnx':
        PTH_build_ONNX(opt.p,opt.n,opt.b,opt.d)
    else:
        ONNX_build_engine(opt.p)   

