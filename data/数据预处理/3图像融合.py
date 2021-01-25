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


folder_new=r"F:\albumentations"
# 找到tag文件
tags = []
tags.extend(glob.glob(r"F:\origin\*.tag"))
# random.shuffle(tags)
tags.sort()
print(len(tags))
good_num=0
for ind,tag in enumerate(tqdm.tqdm(tags)):
    if(ind>21):
        break

    tiff_path = tag[:-4] + '.tiff'
    jpg_path = tag[:-4] + '.jpg'
    txt_path = tag[:-4] + '.txt'

    if(not os.path.exists(txt_path)):
        continue

    tiff=cv2.imdecode(np.fromfile(tiff_path, dtype=np.uint8), -1)
    jpg=cv2.imdecode(np.fromfile(jpg_path, dtype=np.uint8), -1)

    # 缩小图片
    scale=1920/np.max(tiff.shape)
    tiff=cv2.resize(tiff,(int(tiff.shape[1]*scale),int(tiff.shape[0]*scale)),interpolation=cv2.INTER_AREA)
    jpg=cv2.resize(jpg,(int(jpg.shape[1]*scale),int(jpg.shape[0]*scale)),interpolation=cv2.INTER_AREA)

    # 只有强度影像
    png_path_new=os.path.join(folder_new,str(ind).zfill(6)+'_i.png')
    txt_path_new=os.path.join(folder_new,str(ind).zfill(6)+'_i.txt')
    shutil.copy(txt_path,txt_path_new)
    cv2.imencode('.png', tiff)[1].tofile(png_path_new)

    # 强度*2+深度*1
    jpg[:,:,0]=tiff
    jpg[:,:,1]=tiff
    png_path_new=os.path.join(folder_new,str(ind).zfill(6)+'_id.png')
    txt_path_new=os.path.join(folder_new,str(ind).zfill(6)+'_id.txt')
    shutil.copy(txt_path,txt_path_new)
    cv2.imencode('.png', jpg)[1].tofile(png_path_new)



