# -*-coding: utf-8 -*-

import os
# import datetime
import sys

import cv2
import numpy as np
from onnxruntime import InferenceSession
import glob
import json
from scipy.spatial import cKDTree
import tqdm

# print(datetime.datetime.now())

def tag2txt(tag):
    coco128 = []

    f = open(tag, encoding='utf-8')
    text = f.read()
    text = text.split('\n')

    # tiff = tag[:-4] + '.png'
    # tiff = cv2.imread(tiff)
    # tiff = tiff[:, :, :1]

    for t in text:
        if (t == ''):
            continue
        data = json.loads(t)
        col = data['ColIndex']
        row = -1
        for s in data['SegmentInfos']:
            if (s is None):
                # print(tag)
                continue
            if (s['Name'] == 'FB'):
                row = s['RowIndex']
            elif (s['Name'] == 'KP'):
                row = s['RowIndex']
        if (row == -1):
            continue
        # cols.append(col)
        # rows.append(row)
        # row = row / (shape[0] * scale)
        # col = col / (shape[1] * scale)

        # class x_center y_center width height
        coco128.append([col, row])

    if (len(coco128) < 1):
        print(tag)
        return None

    # easydl_json = json.dumps(easydl_json)
    # f = open(os.path.join(r"Z:\subway_scan\coco128",str(ind).zfill(6)+'.txt'), 'w')
    # f.write(easydl_json)
    # f.close()

    # fmt = '%d', '%6f', '%6f', '%6f', '%6f'
    coco128 = np.asarray(coco128)
    return coco128


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


if __name__ == '__main__':
    # onnx_path=sys.argv[1]
    tiffs = glob.glob(r"Z:\subway_scan\guangzhou\origin\*.tiff")
    onnx_model = ONNXModel(r"C:\Users\12037\Desktop\files\PyTorch_YOLOv4\onnx_convert\test.onnx")

    distances=[]
    alls=0
    goods=0
    leaks=0
    wrongs=0
    alls_detect=0

    for tiff in tqdm.tqdm(tiffs):
        tag = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '.tag')
        jpg = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '.jpg')

        intensity_img = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
        depth_img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)

        # 强度图直方图均衡化
        # if (histogram_optimization == 'histogram_optimization_yes'):
        intensity_img = cv2.equalizeHist(intensity_img)

        img = np.zeros((intensity_img.shape[0], intensity_img.shape[1], 3))
        img[:, :, 0] = depth_img[:, :, 0]
        img[:, :, 1] = depth_img[:, :, 0]
        img[:, :, 2] = intensity_img

        input_size = 2016
        scale = input_size / np.max(img.shape)

        img_new = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        # img = cv2.resize(img, (2016, 2016))
        # img_new = np.zeros([1, 3, input_size, input_size])
        # img_new[:,:,:,:]=100
        # img=img[np.newaxis,:,:,:]
        color = (114, 114, 114)
        top, bottom = int(round((input_size - int(img.shape[0] * scale)) / 2 - 0.1)), int(
            round((input_size - int(img.shape[0] * scale)) / 2 + 0.1))
        left, right = int(round((input_size - int(img.shape[1] * scale)) / 2 - 0.1)), int(
            round((input_size - int(img.shape[1] * scale)) / 2 + 0.1))
        img_new = cv2.copyMakeBorder(img_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        img_output = img_new.copy()
        # img_new[0, 0, :, :] = img[:, :, 0]
        # img_new[0, 1, :, :] = img[:, :, 1]
        # img_new[0, 2, :, :] = img[:, :, 2]
        img_new = img_new / 255.0
        img_new = np.asarray(img_new, dtype=np.float32)
        img_new = img_new.transpose((2, 0, 1))
        img_new_reshape = np.reshape(img_new, [1, 3, input_size, input_size])
        result = onnx_model.forward(img_new_reshape)
        nms_result = nms(result[0], conf_thres=0.5)
        center = nms_result[:, :2]
        center[:, 0] -= left
        center[:, 1] -= top
        center /= scale
        # 预测的中心点
        center = center[center[:, 0].argsort()]
        # 实际中心点
        coco128 = tag2txt(tag)


        kdtree = cKDTree(coco128)
        # 每个预测的点到最近的真实点的距离
        dists, inds = kdtree.query(center)
        good=np.where(dists<30)
        distance = dists[good]
        good=len(good[0])
        all=coco128.shape[0]
        leak=all-good
        wrong=dists.shape[0]-good

        distances.extend(distance.tolist())
        # 所有对的
        alls+=all
        # 检测对的
        goods+=good
        # 错检的
        wrongs+=wrong
        # 漏检的
        leaks+=leak
        # 所有检出的
        alls_detect+=dists.shape[0]

        print(np.mean(distances),'所有对的',alls,'检测对的',goods,'错检的',wrongs,'漏检的',leaks,'所有检出的',alls_detect)
