# -*- coding: utf-8 -*
import numpy as np
import pycuda.driver as cudadriver
import tensorrt as trt
import torch
import os
import time
import common

from PIL import Image
import cv2
import torchvision

def ONNX_build_engine(onnx_file_path, write_engine=True):
    
    # 通过加载onnx文件，构建engine
    # :param onnx_file_path: onnx文件路径
    # :return: engine
    
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 1、动态输入第一点必须要写的
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
        # 重点
        profile = builder.create_optimization_profile()  # 动态输入时候需要 分别为最小输入、常规输入、最大输入
        # 有几个输入就要写几个profile.set_shape 名字和转onnx的时候要对应
        # tensorrt6以后的版本是支持动态输入的，需要给每个动态输入绑定一个profile，用于指定最小值，常规值和最大值，如果超出这个范围会报异常。
        profile.set_shape("inputs", (1, 3, 240, 240), (1, 3, 480, 480), (1, 3, 960, 960))
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)
        print("Completed creating Engine")
        # 保存engine文件
        if write_engine:
            engine_file_path = '../model_data/ep055-loss0.010-val_loss0.013.engine'
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
        return engine

onnx_file_path = r'../model_data/ep055-loss0.010-val_loss0.013.onnx'
write_engine = True
engine = ONNX_build_engine(onnx_file_path, write_engine)
