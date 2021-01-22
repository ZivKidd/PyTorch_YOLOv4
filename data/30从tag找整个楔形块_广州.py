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

# 计算图像横向的对称性
def computeSymmetry(image_rect):
    image_rect_top=image_rect[:int(image_rect.shape[0]/2),:]
    image_rect_down=image_rect[int(image_rect.shape[0]/2):,:]
    image_rect_down=np.flip(image_rect_down, axis=0)

    cv2.imshow('top', image_rect_top)
    cv2.imshow('down', image_rect_down)
    cv2.waitKey(0)

    if(image_rect_down.shape[0]>image_rect_top.shape[0]):
        image_rect_down=image_rect_down[:-1,:]
    diff=np.sum(np.abs(image_rect_down-image_rect_top))/image_rect.size
    return diff

folder_new=r"E:\result"
# 找到tag文件
tags=[]
tags.extend(glob.glob(r"F:\广州数据新\*\*\*.tag"))
tags.extend(glob.glob(r"F:\广州数据新\*\*\*\*.tag"))
tags.extend(glob.glob(r"F:\广州数据新\*\*\scan\*\*.tag"))
tags.extend(glob.glob(r"F:\广州数据新\*\*\*\scan\*\*.tag"))
tags.extend(glob.glob(r"F:\广州数据新\*\*\*\*\scan\*\*.tag"))
tags.extend(glob.glob(r"F:\广州数据新\*\*\*\*\*\scan\*\*.tag"))
# tags.extend(glob.glob(r"F:\广州数据新\*\*\*\*\*\*\scan\*\*.tag"))
# tags.extend(glob.glob(r"F:\广州数据新\*\*\*\*\*\*\*\scan\*\*.tag"))
# tags.extend(glob.glob(r"F:\广州数据新\*\*\*\*\*\*\*\*\scan\*\*.tag"))
# random.shuffle(tags)
tags.sort()
print(len(tags))
good_num=0
for ind,tag in enumerate(tqdm.tqdm(tags)):
    # if(ind<1231):
    #     continue
    print(tag)
    # tag=r"E:\广州数据新\广州8号线北延\逐环标记\广州8号线北延11-5\5-8号线北延鹅掌坦-同德下行\scan\scan_preview_0007\scan_preview_0007.tag"

    # 根据文件判断短边在左还是右
    tag_folder=os.path.split(tag)[0]
    tag_folder=os.path.split(tag_folder)[0]
    tag_folder=os.path.split(tag_folder)[0]
    left_shorter = False
    if(not os.path.exists(os.path.join(tag_folder,'right_shorter.txt'))):
    #     left_shorter=False
    # if( os.path.exists(os.path.join(tag_folder,'left_shorter.txt'))):
        continue


    # 图片的长宽
    tiff = os.path.join(os.path.split(tag)[0], os.path.split(tag)[1][:-4] + '.tiff')
    jpg = os.path.join(os.path.split(tag)[0], os.path.split(tag)[1][:-4] + '_bolt.jpg')

    if(not (os.path.exists(tiff) and os.path.exists(jpg))):
        continue


    tiff_shape = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1).shape
    tiff=cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
    jpg=cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)

    jpg[:, :, 0] = tiff
    jpg[:, :, 2] = 0

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
        height_ori=height

        diff = abs(coco128[i,1] - (tiff_shape[0] / 2))
        height *= (diff / (tiff_shape[0] / 2)) + 1.2
        h=int(height)
        # print(h)

        x1=coco128[i, 0]
        x2=coco128[i+1, 0]
        y1=coco128[i,1]-h
        y2=coco128[i,1]

        rect=jpg[y1:y2,x1:x2,:]
        rect_ori=jpg[y1:y2,x1:x2,:]


        # 先对图像进行预处理
        rect = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)

        # rect = cv2.medianBlur(rect, 7)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # rect = cv2.dilate(rect, kernel)
        _, rect = cv2.threshold(rect, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rect=255-rect

        rect = cv2.medianBlur(rect, 5)
        # cv2.imshow('ori',rect)
        # cv2.waitKey(0)

        # 按从下到上，卷积一个斜线型，找到最大的，就是下边所在
        mean_max1=0
        h_max=0
        for h1 in range(int(h * 0.1), int(h * 0.35), 3):
            x_ind=np.arange(coco128[i+1,0]-coco128[i,0])
            y_ind=(h-np.round((x_ind/(coco128[i+1,0]-coco128[i,0]))*h1)-1).astype(np.int)
            mean=(np.sum(rect[y_ind,x_ind]))/(coco128[i+1,0]-coco128[i,0])
            if(mean>mean_max1):
                mean_max1=mean
                h_max=h1

        # 按从上到下，卷积一个斜线型，找到最大的，就是上边所在
        mean_max2=0
        y_max=0
        # h_max=0
        for y1 in range(0, int(h * 0.3), 3):
            x_ind=np.arange(coco128[i+1,0]-coco128[i,0])
            y_ind=(np.round((x_ind/(coco128[i+1,0]-coco128[i,0]))*h_max)+y1).astype(np.int)
            mean=(np.sum(rect[y_ind,x_ind]))/(coco128[i+1,0]-coco128[i,0])
            if(mean>mean_max2):
                mean_max2=mean
                y_max=y1

        # print(y_max/h,h_max,mean_max1,mean_max2)
        # x1=coco128[i, 0]
        # x2=coco128[i+1, 0]
        # y1=coco128[i,1]-h+y_max
        # y2=coco128[i,1]
        # print(x1,x2,y1,y2)
        # rect1 = jpg[y1:y2, x1:x2,:]
        # cv2.imshow('result',rect1)
        # cv2.waitKey(0)
        if(mean_max2<200 and mean_max1<200):
            continue

        height=h-y_max
        if(height<height_ori):
            height=height_ori
        y_center = (coco128[i,1]-height+coco128[i,1])/2
        x_center=(coco128[i+1,0]+coco128[i,0])/2
        width=(coco128[i+1,0]-coco128[i,0])

        wedge.append([0, x_center, y_center, width, height])

    if(len(wedge)==0):
        continue
    wedge = np.asarray(wedge)
    if (wedge.size == 5):
        wedge = wedge.reshape([1, -1])
    wedge[:, 1] /= tiff_shape[1]
    wedge[:, 3] /= tiff_shape[1]
    wedge[:, 2] /= tiff_shape[0]
    wedge[:, 4] /= tiff_shape[0]

    # 后处理，有些框会比同等高度的框高很多，每个框找同等高度的框，如果别的框比自己低，自己也变低
    for w in wedge:
        x,y,w1,h=w[1:]
        h_others=wedge[np.logical_and(wedge[:,2]>y-0.05,wedge[:,2]<y+0.05)]
        h_min=np.min(h_others[:,4])
        w[4]=h_min
        w[2]+=(h-h_min)/2
        # print()

    fmt = '%d', '%6f', '%6f', '%6f', '%6f'

    cv2.imencode('.png', jpg)[1].tofile(os.path.join(folder_new,str(ind).zfill(6)+'.png'))
    np.savetxt(os.path.join(folder_new,str(ind).zfill(6)+'.txt'), wedge, fmt=fmt)

    # width = jpg.shape[1]
    # height = jpg.shape[0]
    # for b in wedge:
    #     jpg = cv2.rectangle(img=jpg, pt1=(int(b[1] * width - b[3] * width / 2), int(b[2] * height - b[4] * height / 2)),
    #                         pt2=(int(b[1] * width + b[3] * width / 2), int(b[2] * height + b[4] * height / 2)),
    #                         color=(0, 0, 255),
    #                         thickness=3)
        # png = cv2.circle(png, (int(b[1] *width), int(b[2] *height)), 3, (0, 0, 255))
        # print()

    # cv2.imencode('.jpg', jpg)[1].tofile(os.path.join(folder_new,str(ind).zfill(6)+'.jpg'))



