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
import random
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

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
#
# def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
#     """Performs Non-Maximum Suppression (NMS) on inference results
#
#     Returns:
#          detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
#     """
#     # if prediction.dtype is torch.float16:
#     #     prediction = prediction.float()  # to FP32
#
#     nc = prediction[0].shape[1] - 5  # number of classes
#     xc = prediction[..., 4] > conf_thres  # candidates
#
#     # Settings
#     min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
#     max_det = 300  # maximum number of detections per image
#     time_limit = 10.0  # seconds to quit after
#     redundant = True  # require redundant detections
#     multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
#
#     # t = time.time()
#     output = [None] * prediction.shape[0]
#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         x = x[xc[xi]]  # confidence
#
#         # If none remain process next image
#         if not x.shape[0]:
#             continue
#
#         # Compute conf
#         x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
#
#         # Box (center x, center y, width, height) to (x1, y1, x2, y2)
#         box = xywh2xyxy(x[:, :4])
#
#         # Detections matrix nx6 (xyxy, conf, cls)
#         if multi_label:
#             i, j = np.where(x[:, 5:] > conf_thres)
#             x = np.concatenate((box[i], x[i, j + 5, None], j[:, None]), 1)
#         else:  # best class only
#             conf, j = x[:, 5:].max(1, keepdim=True)
#             x = np.concatenate((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
#
#         # Filter by class
#         # if classes:
#         #     x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
#
#         # Apply finite constraint
#         # if not torch.isfinite(x).all():
#         #     x = x[torch.isfinite(x).all(1)]
#
#         # If none remain process next image
#         n = x.shape[0]  # number of boxes
#         if not n:
#             continue
#
#         # Sort by confidence
#         # x = x[x[:, 4].argsort(descending=True)]
#
#         # Batched NMS
#         c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#         boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
#         i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
#         if i.shape[0] > max_det:  # limit detections
#             i = i[:max_det]
#         if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
#             try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#                 iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#                 weights = iou * scores[None]  # box weights
#                 x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
#                 if redundant:
#                     i = i[iou.sum(1) > 1]  # require redundancy
#             except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
#                 print(x, i, x.shape, i.shape)
#                 pass
#
#         output[xi] = x[i]
#         if (time.time() - t) > time_limit:
#             break  # time limit exceeded
#
#     return output

# def nms(dets, conf_thres=0.001, iou_thres=0.65):
#     dets = dets.reshape((dets.shape[1], dets.shape[2]))
#     dets = dets[np.where(dets[:, 4] > conf_thres)]
#     x1 = dets[:, 0] - dets[:, 2] / 2
#     y1 = dets[:, 1] - dets[:, 3] / 2
#     x2 = dets[:, 0] + dets[:, 2] / 2
#     y2 = dets[:, 1] + dets[:, 3] / 2
#     scores = dets[:, 4:5] * dets[:, 5:]
#
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个boundingbox的面积
#     order = scores.argsort()[::-1]  # boundingbox的置信度排序
#     keep = []  # 用来保存最后留下来的boundingbox
#     while order.size > 0:
#         i = order[0]  # 置信度最高的boundingbox的index
#         keep.append(i)  # 添加本次置信度最高的boundingbox的index
#
#         # 当前bbox和剩下bbox之间的交叉区域
#         # 选择大于x1,y1和小于x2,y2的区域
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#
#         # 当前bbox和其他剩下bbox之间交叉区域的面积
#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#
#         # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#         # 保留交集小于一定阈值的boundingbox
#         inds = np.where(ovr <= iou_thres)[0]
#         order = order[inds + 1]
#
#     return dets[keep]

def get_iou(index, best, center_x, center_y, width, height):
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    x11 = np.maximum(x1[best], x1[index[1:]])
    y11 = np.maximum(y1[best], y1[index[1:]])
    x22 = np.minimum(x2[best], x2[index[1:]])
    y22 = np.minimum(y2[best], y2[index[1:]])

    # 如果边框相交, x22 - x11 > 0, 如果边框不相交, w(h)设为0
    w = np.maximum(0, x22 - x11 + 1)
    h = np.maximum(0, y22 - y11 + 1)

    overlaps = w * h

    ious = overlaps / (areas[best] + areas[index[1:]] - overlaps)

    return ious


def nms(dets, conf_thres=0.001, iou_thres=0.3):
    """
    :param dets: numpy矩阵
    :param thresh: iou阈值
    :return:
    """
    dets = dets.reshape((dets.shape[1], dets.shape[2]))
    dets = dets[np.where(dets[:, 4] > conf_thres)]
    if(dets.shape[0]==0):
        return None
    result = []
    nc = dets.shape[1] - 5  # number of classes

    for each in range(nc):
        the_boxes = dets[np.where(dets[:, 5:].argsort()[:, -1] == each)[0].tolist(), :]

        center_x = the_boxes[:, 0]
        center_y = the_boxes[:, 1]
        width = the_boxes[:, 2]
        height = the_boxes[:, 3]
        confidence = the_boxes[:, 4]

        index = confidence.argsort()[::-1]

        keep = []

        while index.size > 0:
            best = index[0]
            keep.append(np.expand_dims(the_boxes[best, :], axis=0))

            ious = get_iou(index, best, center_x, center_y, width, height)

            idx = np.where(ious <= iou_thres)[0]

            index = index[idx + 1]
        if(keep==[]):
            continue
        result.append(np.concatenate(keep, axis=0))

    return np.concatenate(result, axis=0)

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
    tiffs = glob.glob(r"D:\TLSD\data\广州13号线夏园-南岗上行线高速扫描\scan\*\*.tiff")
    onnx_model = ONNXModel(r"D:\TLSD\ai\0114_right_left_ring\last.onnx")

    # random.shuffle(tiffs)
    distances=[]
    alls=0
    goods=0
    leaks=0
    wrongs=0
    alls_detect=0

    for i,tiff in enumerate(tqdm.tqdm(tiffs)):
        tiff=r"D:\TLSD\data\2-8号线广州北延陈家祠-彩虹桥下行\scan\scan_preview_0011\scan_preview_0011.tiff"
        tag = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '.tag')
        jpg = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '_bolt.jpg')

        # if(not(os.path.exists(tag) and os.path.exists(jpg))):
        #     continue
        if(not os.path.exists(jpg)):
            continue

        intensity_img = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
        depth_img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)

        # 强度图直方图均衡化
        # if (histogram_optimization == 'histogram_optimization_yes'):
        # intensity_img = cv2.equalizeHist(intensity_img)

        img=depth_img
        img[:,:,0]=intensity_img
        img[:,:,2]=0

        rail_top = 0.13
        rail_down = 0.87
        img = img[int(img.shape[0] * rail_top):int(img.shape[0] * rail_down), :, :]

        # img1=cv2.imread(r"D:\desktop\files\codes\PycharmProjects\PyTorch_YOLOv4\onnx_convert\001277.png")

        # img = np.zeros((intensity_img.shape[0], intensity_img.shape[1], 3))
        # img[:, :, 0] = depth_img[:, :, 0]
        # img[:, :, 1] = depth_img[:, :, 0]
        # img[:, :, 2] = intensity_img
        # 有时候怎么都检测不出来，可能是opencv读入时三个通道反了
        img=np.flip(img,axis=2)

        # cv2.imwrite('pre/'+str(i)+'_csharp.png',img)
        # continue

        # img=cv2.resize(img,(7000,5000))
        input_size = 1920
        scale = input_size / np.max(img.shape)

        img_new = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),interpolation=cv2.INTER_AREA)
        # img_new = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
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

        # cv2.imwrite('pre/'+str(i)+'.png',img_output)
        cv2.imwrite('pre/'+str(i)+'result.png',img_output)

        # img_new[0, 0, :, :] = img[:, :, 0]
        # img_new[0, 1, :, :] = img[:, :, 1]
        # img_new[0, 2, :, :] = img[:, :, 2]
        img_new = img_new / 255.0
        img_new = np.asarray(img_new, dtype=np.float32)
        img_new = img_new.transpose((2, 0, 1))
        img_new_reshape = np.reshape(img_new, [1, 3, input_size, input_size])
        result = onnx_model.forward(img_new_reshape)
        # result是[center_x, center_y, width, height, confidence, class1_score, class2_score, class3_score...]
        nms_result = nms(result[0])
        if(nms_result is None):
            continue

        # 分类别
        threshold=0.1
        # class_id = nms_result[:, 5:].argsort()[:, -1].tolist()
        nms_result[:, 5:]*= nms_result[:, 4:5]
        nms_result=nms_result[np.where(np.max(nms_result[:,5:],axis=1)>threshold)]
        # nms_result=nms_result[np.where()]
        class_id = nms_result[:, 5:].argsort()[:, -1]
        class_id=np.reshape(class_id,[-1,1])
        nms_result=np.concatenate([xywh2xyxy(nms_result[:,:4]),class_id],axis=1)

        # box=xywh2xyxy(nms_result[:,:4])
        for b in nms_result:

            # color[b[4]]=255
            if(b[4])==0:
                color=(0, 0, 255)
            elif(b[4])==1:
                color=(0, 0, 0)
            else:
                color = (255, 0, 0)

            png = cv2.rectangle(img=img_output,
                                pt1=(int(b[0]), int(b[1])),
                                pt2=(int(b[2]), int(b[3])),
                                color=color,
                                thickness=3)
        cv2.imwrite('pre/'+str(i)+'result.png',img_output)
        # break

        # center = nms_result[:, :2]
        # center[:, 0] -= left
        # center[:, 1] -= top
        # center /= scale
        # # 预测的中心点
        # center = center[center[:, 0].argsort()]
        # # 实际中心点
        # coco128 = tag2txt(tag)
        #
        # print()
        # kdtree = cKDTree(coco128)
        # # 每个预测的点到最近的真实点的距离
        # dists, inds = kdtree.query(center)
        # good=np.where(dists<30)
        # distance = dists[good]
        # good=len(good[0])
        # all=coco128.shape[0]
        # leak=all-good
        # wrong=dists.shape[0]-good
        #
        # distances.extend(distance.tolist())
        # # 所有对的
        # alls+=all
        # # 检测对的
        # goods+=good
        # # 错检的
        # wrongs+=wrong
        # # 漏检的
        # leaks+=leak
        # # 所有检出的
        # alls_detect+=dists.shape[0]
        #
        # print(np.mean(distances),'所有对的',alls,'检测对的',goods,'错检的',wrongs,'漏检的',leaks,'所有检出的',alls_detect)
