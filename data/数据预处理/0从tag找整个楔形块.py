import glob
import os
import random

import tqdm
import shutil
import numpy as np
import cv2
import json
import copy


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

# 找到tag文件
tags=[]
tags.extend(glob.glob(r"G:\*\*\*\*.tag"))
tags.extend(glob.glob(r"G:\*\*\*\*\*.tag"))
tags.extend(glob.glob(r"G:\*\*\*\*\*\*.tag"))
tags.extend(glob.glob(r"G:\*\*\*\*\*\*\*.tag"))
tags.extend(glob.glob(r"G:\*\*\*\*\*\*\*\*.tag"))
tags.extend(glob.glob(r"G:\*\*\*\*\*\*\*\*\*.tag"))
tags.extend(glob.glob(r"G:\*\*\*\*\*\*\*\*\*\*.tag"))
tags.sort()
# random.shuffle(tags)

good_num=0
for tag in tqdm.tqdm(tags):
    # tag=r"E:\广州数据新\广州8号线北延\逐环标记\广州8号线北延11-5\5-8号线北延鹅掌坦-同德下行\scan\scan_preview_0007\scan_preview_0007.tag"

    # 根据文件判断短边在左还是右
    tag_folder=os.path.split(tag)[0]
    tag_folder=os.path.split(tag_folder)[0]
    tag_folder=os.path.split(tag_folder)[0]
    left_shorter = True
    if(os.path.exists(os.path.join(tag_folder,'right_shorter.txt'))):
        left_shorter=False
    elif(not os.path.exists(os.path.join(tag_folder,'left_shorter.txt'))):
        continue

    # 图片的长宽
    tiff = os.path.join(os.path.split(tag)[0], os.path.split(tag)[1][:-4] + '.tiff')
    tiff_shape=cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1).shape

    # 读取tag文件
    coco128 = tag2txt(tag)
    if(coco128.shape[0]<2):
        continue
    wedge=[]
    for i in range(coco128.shape[0]-1):
        # 当图片高5100时，框高200，图片高10200时，框高400
        height = 200
        if (tiff_shape[0] == 10200):
            height = height * 2

        # 计算标记点和图片中间的差别，越边上框越大
        diff = abs(coco128[i,1] - (tiff_shape[0] / 2))
        height *= (diff / (tiff_shape[0] / 2)) + 1
        x_center = (coco128[i + 1, 0] + coco128[i, 0]) / 2
        width = coco128[i + 1, 0] - coco128[i, 0]

        # 根据左和右区分
        if(left_shorter):
            y_center=coco128[i,1]-(height/4)
            wedge.append([1,x_center, y_center,width,height])
        else:
            y_center=coco128[i,1]-(height/2)
            wedge.append([0,x_center, y_center,width,height])

    wedge=np.asarray(wedge)
    wedge[:,1]/=tiff_shape[1]
    wedge[:,3]/=tiff_shape[1]
    wedge[:,2]/=tiff_shape[0]
    wedge[:,4]/=tiff_shape[0]

    fmt = '%d', '%6f', '%6f', '%6f', '%6f'
    np.savetxt(os.path.join(os.path.split(tag)[0], os.path.split(tag)[1][:-4] + '_wedge.txt'), wedge, fmt=fmt)

    png = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
    width = png.shape[1]
    height = png.shape[0]

    for b in wedge:
        png = cv2.rectangle(img=png, pt1=(int(b[1] * width - b[3] * width / 2), int(b[2] * height - b[4] * height / 2)),
                            pt2=(int(b[1] * width + b[3] * width / 2), int(b[2] * height + b[4] * height / 2)),
                            color=(0, 255, 0),
                            thickness=3)

    cv2.imencode('.png', png)[1].tofile(os.path.join(os.path.split(tag)[0], os.path.split(tag)[1][:-4] + '_wedge.png'))
    print(os.path.join(os.path.split(tag)[0], os.path.split(tag)[1][:-4] + '_wedge.png'))

    good_num+=1
print(good_num)

