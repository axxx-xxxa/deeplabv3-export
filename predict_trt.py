import sys
sys.path.append('../')
import copy
import export.common as common
import cv2
import time
import torch.nn.functional as F
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils.utils import cvtColor, resize_image
from PIL import Image
import torch
import glob
TRT_LOGGER = trt.Logger()


# 用numpy重写softmax
def softmax(out_np, dim):
    s_value = np.exp(out_np) / np.sum(np.exp(out_np), axis=dim, keepdims=True)
    return s_value


class Deeplabv3_engine(object):
    def __init__(self):
        self.engine_path = "model_data/ep055-loss0.010-val_loss0.013.engine"
        self.input_size = [3,512,512]
        self.input_shape = [512,512]
        self.image_size = self.input_size[1:]
        self.engine = self.get_engine()
        self.context = self.engine.create_execution_context()
        self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]

    def get_engine(self):
        # If a serialized engine exists, use it instead of building an engine.
        f = open(self.engine_path, 'rb')
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


    def detect(self, image_src, imgpath,cuda_ctx = pycuda.autoinit.context):
        
        cuda_ctx.push()
        ori_img = copy.deepcopy(image_src)
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        orininal_h  = np.array(ori_img).shape[0]
        orininal_w  = np.array(ori_img).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image1       = cvtColor(Image.open(f"{imgpath}"),)
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image1, (512,512))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#



        IN_IMAGE_H, IN_IMAGE_W = self.image_size

        # Input
        img_in = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        img_in = cv2.resize(img_in, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)

        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)  # (3, 240, 240)
        img_in /= 255.0  # 归一化[0, 1]

        img_in = np.expand_dims(img_in, axis=0)  # (1, 3, 240, 240)

        img_in = np.ascontiguousarray(img_in)
        
        # 动态输入
        self.context.active_optimization_profile = 0
        origin_inputshape = self.context.get_binding_shape(0)
        origin_inputshape[0], origin_inputshape[1], origin_inputshape[2], origin_inputshape[3] = img_in.shape
        self.context.set_binding_shape(0, (origin_inputshape))  # 若每个输入的size不一样，可根据inputs的size更改对应的context中的size

        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine, self.context)
        # Do inference
        inputs[0].host = img_in
        start = time.time()
        trt_outputs = common.do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs,
                                          stream=stream, batch_size=1)
        end = time.time()
        print("inference time = ",end-start)
        if cuda_ctx:
            cuda_ctx.pop()


        pr = trt_outputs[0].reshape(3,512,512)
        pr = torch.tensor(pr)

        pr = F.softmax(pr.permute(1,2,0),dim = -1).numpy()

        #--------------------------------------#
        #   将灰条部分截取掉
        #--------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        #   进行图片的resize
        #---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)
        #------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        #------------------------------------------------#
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(3):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')

        #------------------------------------------------#
        #   将新图片转换成Image的形式
        #------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img))

        #------------------------------------------------#
        #   将新图片和原图片混合
        #------------------------------------------------#
        if 1:
            image = Image.blend(Image.open(f"{imgpath}"),image,0.7)
        
        return image




if __name__ == '__main__':
    image_paths = glob.glob("pole/*")
    # img = cv2.imread("img/street.jpg")
    detect_engine = Deeplabv3_engine()
    for i,imgpath in enumerate(image_paths*3):
        img = cv2.imread(f"{imgpath}")
        output = detect_engine.detect(img,imgpath)
        img_name = imgpath.split("/",1)[1]
        cv2.imwrite(f"pole_res_trt/{i}_{img_name}",np.float32(output))
        # cv2.imwrite("res.jpg",output)