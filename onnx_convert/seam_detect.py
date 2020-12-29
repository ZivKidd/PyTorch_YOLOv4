# -*-coding: utf-8 -*-

# import sys
import argparse
import cv2
import numpy as np
from onnxruntime import InferenceSession
import os
# from skimage import filters
# from skimage.morphology import disk
# from sklearn.cluster import DBSCAN

def nms(dets, conf_thres=0.001, iou_thres=0.65):
    dets = dets[0,:,:6]
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

def distance(a, b, p):
    return np.sum((a - b) ** p) ** (1 / p)


class DBSCAN(object):
    def __init__(self, epsilon, minPts):
        """
        初始化超参数
        :param epsilon: epsilon is the min distance 确定领域的最小距离
        :param minPts: minPts is the min elements. 确定领域的最少元素数目
        """
        self.minPts = minPts
        self.eps = epsilon
        self.X = None
        self.ker = None
        self.k = None
        self.gama = None
        self.omega = None
        self.num = None
        self.dim = None
        self.distMatrix = None
        self.labels = None

    def _update_omega(self):
        """
        确定核心对象的omega集合，
        :return:
        """
        for j1 in range(self.num):
            count = 0
            for j2 in range(self.num):
                dist = distance(self.X[j1, :], self.X[j2, :], 2)
                self.distMatrix[j1, j2] = dist    # 生成距离矩阵
                if dist <= self.eps:
                    count += 1
            if count >= self.minPts:
                self.omega[j1] = 1   # 核心对象的标签设置为1

    def _find_nodes(self):
        """
        找到所有核心对象的密度可达的节点，并将其作为一个聚类簇
        :return:
        """
        while np.sum(self.omega) > 0:         # 当核心对象的数目不为0时，执行下面代码
            index = np.argwhere(self.omega == 1)
            rand = np.random.choice(index.shape[0])
            obj_index = index[rand]        # 随机选择的核心对象的样本编号
            Q = [obj_index]                # 初试化队列Q,只包含随机选择的一个核心对象
            self.gama[obj_index] = -1      # 将未访问的样本集合中的上一步选择的对象的编号设置为-1，表示已选择
            while len(Q) > 0:  # 当初始化队列Q不为空，执行以下代码
                q = Q[0]
                Q = Q[1:]   # 从Q中取出第一个样本q
                dists = self.distMatrix[q, :]
                count = np.argwhere(dists <= self.eps).shape[0]  # 找到该样本的领域中的元素个数
                # print(np.argwhere(dists <= self.eps))
                # print(count)
                # print("*"*19)
                if count >= self.minPts:   # 如果该样本满足领域的个数限制，执行下面代码
                    self.omega[q] = 0
                    delta = []
                    for i in np.argwhere(dists < self.eps):
                        # print(i[-1])
                        if i[-1] in self.gama:
                            Q.append(i[-1])            # 将该样本领域中的元素与未标记的样本取交集，并添加到Q队列中
                            self.gama[i[-1]] = -1      # 将交集中的元素标记为已经访问
            self.labels[self.gama == -1] = self.k  # 生成聚类簇，类编号为k，类中的元素为第一次随机选择的和新对象，所有密度可达的元素
            self.gama[self.gama == -1] = -2   # 将本次循环中访问对象的标签设为-2，表示已访问并且分类。
            self.k += 1  # 类编号+1
            # break

    def fit(self, X):
        # 初始化参数
        self.X = X
        self.k = 1                  # 类编号
        self.num = X.shape[0]       # 样本数
        self.omega = np.zeros(self.num)  # 核心对象的集合
        self.gama = np.arange(0, X.shape[0])  # 未访问的样本集合
        self.distMatrix = np.zeros((self.num, self.num))  # 距离矩阵
        self.labels = np.zeros(X.shape[0])  #类标签
        # 初始化核心对象集合
        self._update_omega()
        # print(np.sum(self.omega))
        # 根据密度可达划分聚类簇
        self._find_nodes()

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

# 图像合成并压缩
def image_compose(tiff,jpg,img_size):
    tiff1 = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
    jpg = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
    # img_shape = tiff1.shape
    # 切掉铁轨部分
    tiff1 = tiff1[int(tiff1.shape[0] * 0.15):int(tiff1.shape[0] * 0.85), :]
    jpg = jpg[int(jpg.shape[0] * 0.15):int(jpg.shape[0] * 0.85), :, 0]

    # 对强度影像预处理
    img = cv2.equalizeHist(tiff1)
    # img = filters.median(img, disk(3))

    # x1 = np.arange(0, img.shape[1])
    # x2 = np.arange(0, img.shape[1] - 1)
    y1 = np.sum(img, axis=0).astype(np.float)
    # y11 = y1.reshape([1, -1])
    # y11 = (y11 - np.min(y11)) / (np.max(y11) - np.min(y11)) * 255

    # y11 = np.tile(y11, [img.shape[0], 1])
    y2 = np.diff(y1)
    y1[0] = 0
    y1[1:] = y2
    y12 = y1.reshape([1, -1])
    y12 = (y12 - np.min(y12)) / (np.max(y12) - np.min(y12)) * 255
    y12 = np.tile(y12, [img.shape[0], 1])

    output = np.zeros([img.shape[0], img.shape[1], 3])
    output[:, :, 2] = img
    output[:, :, 1] = jpg
    output[:, :, 0] = y12

    scale = img_size / np.max(output.shape)
    # output = cv2.resize(output, (img_size, img_size),
    #                     interpolation=cv2.INTER_AREA)
    img_new = cv2.resize(output, (int(output.shape[1] * scale), int(output.shape[0] * scale)))
    # img = cv2.resize(img, (2016, 2016))
    # img_new = np.zeros([1, 3, input_size, input_size])
    # img_new[:,:,:,:]=100
    # img=img[np.newaxis,:,:,:]
    color = (114, 114, 114)
    top, bottom = int(round((img_size - int(img.shape[0] * scale)) / 2 - 0.1)), int(
        round((img_size - int(img.shape[0] * scale)) / 2 + 0.1))
    left, right = int(round((img_size - int(img.shape[1] * scale)) / 2 - 0.1)), int(
        round((img_size - int(img.shape[1] * scale)) / 2 + 0.1))
    img_new = cv2.copyMakeBorder(img_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # img_output = img_new.copy()

    return img_new,left,top,scale

if __name__ == '__main__':
    # onnx_path=sys.argv[1]
    # intensity_img_path = sys.argv[2]
    # depth_img_path = sys.argv[3]
    # txt_path = sys.argv[4]
    #
    # png_path=0
    # if(len(sys.argv)>5):
    #     png_path = sys.argv[5]

    parser = argparse.ArgumentParser(description='Tlsd AI model')
    parser.add_argument('modelPath', type=str, default=None)
    parser.add_argument('folderPath', type=str, default=None)
    parser.add_argument('--outputJPG', type=bool, default=False)
    args = parser.parse_args()

    onnx_path=args.modelPath
    intensity_img_path=os.path.join(args.folderPath,os.path.split(args.folderPath)[1]+'.tiff')
    depth_img_path=os.path.join(args.folderPath,os.path.split(args.folderPath)[1]+'_bolt.jpg')
    txt_path=os.path.join(args.folderPath,os.path.split(args.folderPath)[1]+'_AIResult.txt')

    # intensity_img_path=r"F:\大师  傅\广州（啊   啊）\scan_preview0000\scan_preview0000.tiff"
    # depth_img_path=r"F:\大师  傅\广州（啊   啊）\scan_preview0000\scan_preview0000.jpg"

    input_size=1600
    img_new,left,top,scale=image_compose(intensity_img_path,depth_img_path,input_size)

    img_new = np.asarray(img_new, dtype=np.float32)

    img_new = img_new / 255.0
    img_new = img_new.transpose((2, 0, 1))
    img_new_reshape = np.reshape(img_new, [1, 3, input_size, input_size])
    onnx_model = ONNXModel(onnx_path)
    result = onnx_model.forward(img_new_reshape)
    nms_result = nms(result[0], conf_thres=0.5)
    center = nms_result[:, :2]
    center[:, 0] -= left
    center[:, 1] -= top
    center /= scale
    center[:, 1] += 10200*0.15

    if(center.shape[0]==0):
        np.savetxt(txt_path, center, fmt='%.03f')
        if (args.outputJPG):
            png_path = os.path.join(args.folderPath, os.path.split(args.folderPath)[1] + '_AIResult.jpg')
            img = cv2.imdecode(np.fromfile(intensity_img_path, dtype=np.uint8), -1)
            cv2.imencode('.jpg', img)[1].tofile(png_path)
        # print()

    else:
        # 预测的中心点

        # 聚类
        model = DBSCAN(10, 1)
        model.fit(center)
        yhat =(model.labels).astype(np.int)
        y1=np.zeros((np.max(yhat),2))
        for i in range(np.max(yhat)):
            y1[i]=np.mean(center[np.where(yhat==i+1)],axis=0)
        # y1=[]
        # for i in range(center.sha):
        #     point_now=i
        y1 = y1[y1[:, 0].argsort()]

        np.savetxt(txt_path, y1, fmt='%.03f')

        if (args.outputJPG):
            png_path=os.path.join(args.folderPath,os.path.split(args.folderPath)[1]+'_AIResult.jpg')
            img=cv2.imdecode(np.fromfile(intensity_img_path, dtype=np.uint8), -1)
            # x1 = np.asarray(nms_result[:, 0] - nms_result[:, 2] / 2, dtype=np.int)
            # y1 = np.asarray(nms_result[:, 1] - nms_result[:, 3] / 2, dtype=np.int)
            # x2 = np.asarray(nms_result[:, 0] + nms_result[:, 2] / 2, dtype=np.int)
            # y2 = np.asarray(nms_result[:, 1] + nms_result[:, 3] / 2, dtype=np.int)
            for i in range(y1.shape[0]):
                img = cv2.circle(img=img,
                                    center=(int(y1[i,0]),int(y1[i,1])),
                                 radius=50,
                                    color=(0, 0, 255),
                                    thickness=10)
            cv2.imencode('.jpg', img)[1].tofile(png_path)



