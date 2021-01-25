import glob
import os
import random

import tqdm
import shutil
import numpy as np
import cv2
import json
import copy
from scipy.interpolate import griddata

def tag2txt(tag):
    coco128 = []

    f = open(tag, encoding='utf-8')
    text = f.read()
    text = text.split('\n')

    for t in text:
        if (t == ''):
            continue
        data = json.loads(t)
        if(data['Valid']==False):
            return None
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
        coco128.append([col, row])

    coco128 = np.asarray(coco128)
    coco128 = coco128[coco128[:, 0].argsort()]
    return coco128

# folder_new=r"E:\result"
# 找到tag文件
tags = []
tags.extend(glob.glob(r"F:\origin\*.tag"))
# random.shuffle(tags)
tags.sort()
print(len(tags))
good_num=0
for ind,tag in enumerate(tqdm.tqdm(tags)):
    # if(ind<1231):
    #     continue
    # print(tag)
    # tag=r"E:\广州数据新\广州8号线北延\逐环标记\广州8号线北延11-5\5-8号线北延鹅掌坦-同德下行\scan\scan_preview_0007\scan_preview_0007.tag"

    # # 根据文件判断短边在左还是右
    # tag_folder=os.path.split(tag)[0]
    # tag_folder=os.path.split(tag_folder)[0]
    # tag_folder=os.path.split(tag_folder)[0]
    # left_shorter = False
    # if(not os.path.exists(os.path.join(tag_folder,'right_shorter.txt'))):
    #     left_shorter=False
    # if( os.path.exists(os.path.join(tag_folder,'left_shorter.txt'))):
    #     continue


    # 图片的长宽
    tiff_path = os.path.join(os.path.split(tag)[0], os.path.split(tag)[1][:-4] + '.tiff')
    jpg_path = os.path.join(os.path.split(tag)[0], os.path.split(tag)[1][:-4] + '.jpg')

    if(not (os.path.exists(tiff_path) and os.path.exists(jpg_path))):
        continue


    tiff_shape = cv2.imdecode(np.fromfile(tiff_path, dtype=np.uint8), -1).shape

    # 读取tag文件
    coco128 = tag2txt(tag)
    wedge=[]
    for i in range(coco128.shape[0]-1):
        y_center = 0.5
        x_center = (coco128[i + 1, 0] + coco128[i, 0]) / 2
        width = (coco128[i + 1, 0] - coco128[i, 0])

        wedge.append([2, x_center, y_center, width, 0.7])

    if (len(wedge) == 0):
        continue
    wedge = np.asarray(wedge)
    if (wedge.size == 5):
        wedge = wedge.reshape([1, 5])
    wedge[:, 1] /= tiff_shape[1]
    wedge[:, 3] /= tiff_shape[1]
    # wedge[:, 2] /= tiff_shape[0]
    # wedge[:, 4] /= tiff_shape[0]

    fmt = '%d', '%6f', '%6f', '%6f', '%6f'
    # np.savetxt(os.path.join(folder_new,str(ind).zfill(6)+'_ring.txt'), wedge, fmt=fmt)
    np.savetxt(tiff_path[:-5]+'_ring.txt', wedge, fmt=fmt)




