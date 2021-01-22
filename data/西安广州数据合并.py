import glob
import os
import tqdm
import shutil
import numpy as np
import cv2
from skimage import io, color,data,filters
from skimage.morphology import disk
import json
import random

def tag2txt(tag, shape, scale, bbox_size=100):
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
        coco128.append([0, col, row, bbox_size, bbox_size])

    if (len(coco128) < 1):
        print(tag)
        return None

    # easydl_json = json.dumps(easydl_json)
    # f = open(os.path.join(r"Z:\subway_scan\coco128",str(ind).zfill(6)+'.txt'), 'w')
    # f.write(easydl_json)
    # f.close()

    # fmt = '%d', '%6f', '%6f', '%6f', '%6f'
    coco128 = np.asarray(coco128).astype(np.float)
    return coco128


# folder = r'/media/sever/zeran/subway_scan/positive1130/origin/*.tiff'
folder_new=r"/media/sever/zeran/subway_scan/guangzhou_xian/synthesis_norail"
tiffs =[]
tiffs.extend(glob.glob(r"/media/sever/zeran/subway_scan/guangzhou/origin/*.tiff")[:700])
tiffs.extend(glob.glob(r"/media/sever/zeran/subway_scan/positive1130/origin/*.tiff")[:700])
random.shuffle(tiffs)
scale=3
for i,tiff in enumerate(tqdm.tqdm(tiffs)):
    # if(i<1908):
    #     continue
    tag = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '.tag')
    jpg = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '.jpg')
    tiff1 = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
    jpg = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
    img_shape=tiff1.shape
    # 切掉铁轨部分
    tiff1=tiff1[int(tiff1.shape[0]*0.15):int(tiff1.shape[0]*0.85),:]
    jpg=jpg[int(jpg.shape[0]*0.15):int(jpg.shape[0]*0.85),:,0]

    # 对强度影像预处理
    img = cv2.equalizeHist(tiff1)
    img = filters.median(img, disk(3))

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
    output[:, :, 0] = img
    output[:, :, 1] = jpg
    output[:, :, 2] = y12

    # tiff1[:, :, 0] = jpg[:, :, 0]
    # tiff1[:, :, 1] = jpg[:, :, 0]
    # tiff1[:, :, 2] = 0
    image_path_new=os.path.join(folder_new,str(i).zfill(6)+ '.png')
    tag_path_new=os.path.join(folder_new,str(i).zfill(6)+'.txt')

    coco128=tag2txt(tag,img_shape,scale)
    coco128[:,2]-=img_shape[0]*0.15
    coco128[:,1]/=output.shape[1]
    coco128[:,3]/=output.shape[1]
    coco128[:,2]/=output.shape[0]
    coco128[:,4]/=output.shape[0]

    # coco128[0,2]=-1
    coco128=coco128[np.logical_and(coco128[:,2]>0,coco128[:,2]<1)]


    # tiff1=cv2.resize(tiff1, (int(tiff1.shape[1] / scale), int(tiff1.shape[0] / scale)), interpolation=cv2.INTER_AREA)
    # cv2.imwrite('1.png', tiff1)
    # break

    fmt = '%d', '%6f', '%6f', '%6f', '%6f'
    # shutil.copy(tag, tag_path_new)
    output = cv2.resize(output, (int(output.shape[1] / scale ), int(output.shape[0] / scale)),
                      interpolation=cv2.INTER_AREA)
    cv2.imwrite(image_path_new, output)
    np.savetxt(tag_path_new,coco128,fmt=fmt)
    # shutil.copy(tag, tag_path_new)

