import glob
import os

import cv2
import numpy as np
import tqdm

folder = r"D:\xuzeran\text\*.tiff"
folder_new = r"D:\xuzeran\text_small"
tiffs = glob.glob(folder)
tiffs.sort()


for i, tiff in enumerate(tqdm.tqdm(tiffs)):

    txt = os.path.splitext(tiff)[0] + '.txt'

    tiff1 = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
    data = np.loadtxt(txt)

    scale=1920/np.max(tiff1.shape)
    transformed_image=cv2.resize(tiff1,(int(tiff1.shape[1]*scale),int(tiff1.shape[0]*scale)),interpolation=cv2.INTER_AREA)

    cv2.imencode('.png', transformed_image)[1].tofile(os.path.join(folder_new,os.path.split(tiff)[1][:-5]+'.png'))
    np.savetxt(os.path.join(folder_new,os.path.split(txt)[1]), data)
