import glob
import os
import random

import tqdm
import shutil
import numpy as np
import cv2
import json
import copy


folder = r"/media/sever/zeran/subway_scan/guangzhou/origin/*.tiff"
folder_new=r'/media/sever/zeran/subway_scan/guangzhou/intensity_process'
tiffs = glob.glob(folder)
tiffs.sort()

for tiff in tqdm.tqdm(tiffs):
    img = cv2.imread(tiff)
    img=img[:,:,0]
    name_new=os.path.join(folder_new,os.path.split( tiff)[1])
    # dst=0
    # # 图像归一化
    # fi = img / 255.0
    # # 伽马变换
    # gamma = 1.4
    # out = np.asarray(np.power(fi, gamma)*255,dtype=np.int)
    # dst=cv2.normalize(img, dst, 0, 255, cv2.NORM_L2)
    # cv2.imwrite(name_new,dst)
    # laplation kernel
    # h = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="float32")
    # dst = cv2.filter2D(img, -1, h)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    # dst = clahe.apply(img)
    dst = cv2.equalizeHist(img)
    cv2.imwrite(name_new,dst)

    # break