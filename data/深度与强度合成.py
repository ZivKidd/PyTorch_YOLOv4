import glob
import os
import tqdm
import shutil
import numpy as np
import cv2

folder = r"Z:\subway_scan\positive1130\origin\*.tiff"
folder_new=r"Z:\subway_scan\positive1130\result"
tiffs = glob.glob(folder)
scale=4
for tiff in tqdm.tqdm(tiffs):
    tag = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '.tag')
    jpg = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '.jpg')
    jpg = cv2.imread(jpg)
    tiff1 = cv2.imread(tiff)
    tiff1[:, :, 0] = jpg[:, :, 0]
    image_path_new=os.path.join(folder_new,os.path.split(tiff)[1][:-5] + '.png')
    tag_path_new=os.path.join(folder_new,os.path.split(tiff)[1][:-5] + '.tag')

    tiff1=cv2.resize(tiff1, (int(tiff1.shape[1] / scale), int(tiff1.shape[0] / scale)), interpolation=cv2.INTER_AREA)

    cv2.imwrite(image_path_new, tiff1)
    shutil.copy(tag, tag_path_new)

