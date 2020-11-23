import glob
import json
import os

import cv2
import tqdm
import numpy as np

folder = r"Z:\subway_scan\positive\*.tiff"
tiff_files = glob.glob(folder)
# tiff_files=glob.glob(folder+'/*.tiff')
#
tiff_files.sort()
# tiff_files.sort()

scale=4
bbox_size = 200

for i,t in enumerate(tqdm.tqdm(tiff_files)):
    tiff=cv2.imread(t)
    tiff=tiff[:,:,:1]
    # print(np.max(tiff),np.min(tiff))
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    # dst = cv2.filter2D(tiff, -1, kernel=kernel)
    dst = cv2.bilateralFilter(tiff, 20, 150,150)
    cv2.imwrite('1.tiff',dst)
    print()