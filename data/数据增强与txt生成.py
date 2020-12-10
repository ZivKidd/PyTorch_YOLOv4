import glob
import os
import random

import tqdm
import shutil
import numpy as np
import cv2
import json
import copy


def tag2txt(tag, shape, scale, bbox_size=200):
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
        row = row / (shape[0] * scale)
        col = col / (shape[1] * scale)

        # class x_center y_center width height
        coco128.append([0, col, row, bbox_size / (shape[1] * scale), bbox_size / (shape[0] * scale)])

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

scale_ori=4
folder = r"Z:\subway_scan\positive1130\synthesis\*.png"
folder_new = r"Z:\subway_scan\positive1130\augmentation"
train_folder=os.path.join(folder_new,'train')
val_folder=os.path.join(folder_new,'val')
test_folder=os.path.join(folder_new,'test')
for f in [train_folder,test_folder,val_folder]:
    if(not os.path.exists(f)):
        os.mkdir(f)
pngs = glob.glob(folder)
pngs.sort()
num=0
# scale = 4
for png in tqdm.tqdm(pngs):
    tag = os.path.join(os.path.split(png)[0], os.path.split(png)[1][:-4] + '.tag')
    # jpg = os.path.join(os.path.split(png)[0], os.path.split(png)[1][:-5] + '.jpg')
    # jpg = cv2.imread(jpg)
    png1 = cv2.imread(png)
    # png1[:, :, 0] = jpg[:, :, 0]
    coco128 = tag2txt(tag, shape=png1.shape, scale=1)
    if (coco128 is None):
        continue
    scale1 = np.around(np.around((np.random.rand(1, 1))[0, 0], decimals=2)+0.2,decimals=2)
    scale_list=[0.5, 0.75, 1, scale1]
    random.shuffle(scale_list)
    for scale in scale_list:
        num+=1
        if (num % 7 < 4):
            image_path_new = os.path.join(train_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.png')
            tag_path_new = os.path.join(train_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.txt')
        elif (num % 7 == 4):
            image_path_new = os.path.join(val_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.png')
            tag_path_new = os.path.join(val_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.txt')
        else:
            image_path_new = os.path.join(test_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.png')
            tag_path_new = os.path.join(test_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.txt')

        if (scale == 1):
            png2 = cv2.resize(png1, (int(png1.shape[1]/scale_ori), int(png1.shape[0]/scale_ori)), interpolation=cv2.INTER_AREA)
            coco1281 = coco128

        else:
            png2 = cv2.resize(png1, (int(png1.shape[1] * scale/scale_ori), int(png1.shape[0]/scale_ori)), interpolation=cv2.INTER_AREA)
            coco1281 = copy.deepcopy(coco128)
            # coco1281[:,1]*=scale
            # coco1281[:, 3] /= scale

        fmt = '%d', '%6f', '%6f', '%6f', '%6f'
        np.savetxt(tag_path_new, coco1281, fmt=fmt)
        cv2.imwrite(image_path_new, png2)
