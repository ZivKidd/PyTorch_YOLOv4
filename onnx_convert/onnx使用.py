# -*-coding: utf-8 -*-

import cv2
import numpy as np
from onnxruntime import InferenceSession
# import datetime
import sys
import os

# print(datetime.datetime.now())

def nms(dets, conf_thres=0.001, iou_thres=0.65):
    dets = dets.reshape((dets.shape[1], 6))
    dets = dets[np.where(dets[:, 4] > conf_thres)]
    x1 = dets[:, 0] - dets[:, 2] / 2
    y1 = dets[:, 1] - dets[:, 3] / 2
    x2 = dets[:, 0] + dets[:, 2] / 2
    y2 = dets[:, 1] + dets[:, 3] / 2
    scores = dets[:, 4] * dets[:, 5]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个boundingbox的面积
    order = scores.argsort()[::-1]  # boundingbox的置信度排序
    keep = []  # 用来保存最后留下来的boundingbox
    while order.size > 0:
        i = order[0]  # 置信度最高的boundingbox的index
        keep.append(i)  # 添加本次置信度最高的boundingbox的index

        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留交集小于一定阈值的boundingbox
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return dets[keep]


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        # print("input_name:{}".format(self.input_name))
        # print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        result = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return result

if __name__=='__main__':
    img_path=sys.argv[1]
    # print(img_path)
    # print(sys.argv)
    onnx_model = ONNXModel(os.path.join(os.path.split(sys.argv[0])[0],'test.onnx'))
    # 防止不能读入中文路径的图片
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
    input_size=2016
    # 把最长的边缩到目标尺寸
    scale=input_size/np.max(img.shape)
    img_new=cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

    # 把不足的地方填补灰色
    color=(114,114,114)
    top,bottom=int(round((input_size-int(img.shape[0]*scale))/2 - 0.1)), int(round((input_size-int(img.shape[0]*scale))/2 + 0.1))
    left,right=int(round((input_size-int(img.shape[1]*scale))/2 - 0.1)), int(round((input_size-int(img.shape[1]*scale))/2 + 0.1))
    img_new = cv2.copyMakeBorder(img_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    img_output=img_new.copy()
    img_new = img_new / 255.0
    img_new = np.asarray(img_new, dtype=np.float32)
    img_new=img_new.transpose((2,0,1))
    img_new_reshape=np.reshape(img_new,[1,3,input_size,input_size])

    # 模型预测
    result = onnx_model.forward(img_new_reshape)
    nms_result = nms(result[0])

    # 保存框的中心点
    center=nms_result[:,:2]
    center[:,0]-=left
    center[:,1]-=top
    center/=scale
    center = center[center[:,0].argsort()]
    np.savetxt(os.path.join(os.path.split(sys.argv[1])[0],'corner_center.txt'),center,fmt='%.03f')
    # print()
    # x1 = np.asarray(nms_result[:, 0] - nms_result[:, 2] / 2,dtype=np.int)
    # y1 = np.asarray(nms_result[:, 1] - nms_result[:, 3] / 2,dtype=np.int)
    # x2 = np.asarray(nms_result[:, 0] + nms_result[:, 2] / 2,dtype=np.int)
    # y2 = np.asarray(nms_result[:, 1] + nms_result[:, 3] / 2,dtype=np.int)
    # for i in range(x1.shape[0]):
    #     if(x2[i]>input_size or y2[i]>input_size):
    #         continue
    #     img_output = cv2.rectangle(img=img_output,
    #                         pt1=(x1[i], y1[i]),
    #                         pt2=(x2[i], y2[i]),
    #                         color=(0, 255, 0),
    #                         thickness=3)
    # cv2.imwrite('onnx.png',img_output)

    # print(datetime.datetime.now())

