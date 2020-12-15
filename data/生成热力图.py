import glob
import os
import random

import tqdm
import shutil
import numpy as np
import cv2
import json
import copy


def tag2heatmap(tag, shape, scale, bbox_size=100):
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
        row -= int(shape[0] / 0.7 * 0.15)
        # row = row * scale
        # col = col * scale


        # class x_center y_center width height
        coco128.append([col, row])

    # if (len(coco128) < 1):
    #     print(tag)
    #     return None

    # easydl_json = json.dumps(easydl_json)
    # f = open(os.path.join(r"Z:\subway_scan\coco128",str(ind).zfill(6)+'.txt'), 'w')
    # f.write(easydl_json)
    # f.close()

    # fmt = '%d', '%6f', '%6f', '%6f', '%6f'
    coco128 = np.asarray(coco128,dtype=np.float)
    return coco128

input_size=1440
folder = r"/media/sever/zeran/subway_scan/positive1130/synthesis_norail/*.png"
folder_new = r"/media/sever/zeran/subway_scan/positive1130/heatmap_norail"
# train_folder=os.path.join(folder_new,'train')
# val_folder=os.path.join(folder_new,'val')
# test_folder=os.path.join(folder_new,'test')
# for f in [train_folder,test_folder,val_folder]:
#     if(not os.path.exists(f)):
#         os.mkdir(f)
pngs = glob.glob(folder)
pngs.sort()
num=0
# scale = 4
for png in tqdm.tqdm(pngs):
    tag = os.path.join(os.path.split(png)[0], os.path.split(png)[1][:-4] + '.tag')
    # jpg = os.path.join(os.path.split(png)[0], os.path.split(png)[1][:-5] + '.jpg')
    # jpg = cv2.imread(jpg)
    img = cv2.imread(png)

    # input_size = 2016
    scale = input_size / np.max(img.shape)

    img_new = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    color = (114, 114, 114)
    top, bottom = int(round((input_size - int(img.shape[0] * scale)) / 2 - 0.1)), int(
        round((input_size - int(img.shape[0] * scale)) / 2 + 0.1))
    left, right = int(round((input_size - int(img.shape[1] * scale)) / 2 - 0.1)), int(
        round((input_size - int(img.shape[1] * scale)) / 2 + 0.1))
    img_new = cv2.copyMakeBorder(img_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # png1[:, :, 0] = jpg[:, :, 0]
    coco128 = tag2heatmap(tag, shape=img.shape, scale=scale)
    coco128*=scale
    coco128[:,0]+=left
    coco128[:,1]+=top
    coco128=np.asarray(np.around(coco128),dtype=np.int)
    if (coco128 is None):
        continue

    # output_res_x=png1.shape[1]
    # output_res_y=png1.shape[0]
    sigma = input_size / 64
    # sigma_y = output_res_y / 64
    hms = np.zeros(shape=(input_size, input_size), dtype=np.float32)
    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # sigma = self.sigma
    # for p in keypoints:
    for  pt in coco128:
        try:
            x, y = int(pt[0]), int(pt[1])
            # if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
            #     continue
            ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
            br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

            c, d = max(0, -ul[0]), min(br[0], input_size) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], input_size) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], input_size)
            aa, bb = max(0, ul[1]), min(br[1], input_size)
            hms[aa:bb, cc:dd] = np.maximum(hms[ aa:bb, cc:dd], g[a:b, c:d])
        except:
            continue
    hms*=255
    hms=np.asarray(hms,dtype=np.int)
    cv2.imwrite(os.path.join(folder_new,os.path.split(png)[1]),img_new)
    cv2.imwrite(os.path.join(folder_new,os.path.split(png)[1][:-4]+'_heatmap.png'),hms)
    # print()
    # print()
    # scale1 = np.around(np.around((np.random.rand(1, 1))[0, 0], decimals=2)+0.2,decimals=2)
    # scale_list=[0.25,0.5, 0.75, 1,1.25, 1.5,2]
    # random.shuffle(scale_list)
    # for scale in scale_list:
    #     num+=1
    #     if (num % 7 < 4):
    #         image_path_new = os.path.join(train_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.png')
    #         tag_path_new = os.path.join(train_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.txt')
    #     elif (num % 7 == 4):
    #         image_path_new = os.path.join(val_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.png')
    #         tag_path_new = os.path.join(val_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.txt')
    #     else:
    #         image_path_new = os.path.join(test_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.png')
    #         tag_path_new = os.path.join(test_folder, os.path.split(png)[1][:-4] + '_' + str(scale) + '.txt')
    #
    #     if (scale == 1):
    #         png2 = cv2.resize(png1, (int(png1.shape[1]/scale_ori), int(png1.shape[0]/scale_ori)), interpolation=cv2.INTER_AREA)
    #         coco1281 = coco128
    #
    #     else:
    #         png2 = cv2.resize(png1, (int(png1.shape[1] * scale/scale_ori), int(png1.shape[0]/scale_ori)), interpolation=cv2.INTER_AREA)
    #         coco1281 = copy.deepcopy(coco128)
    #         # coco1281[:,1]*=scale
    #         coco1281[:, 3] /= scale
    #
    #     coco1281[:,0]=int(scale/0.25)
    #
    #     fmt = '%d', '%6f', '%6f', '%6f', '%6f'
    #     np.savetxt(tag_path_new, coco1281, fmt=fmt)
    #     cv2.imwrite(image_path_new, png2)
