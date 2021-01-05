import glob
import os
import tqdm
import shutil
import numpy as np
import cv2

folder = r"E:\广州数据新\楔形块标记\*.tiff"
folder_new=r'E:\广州数据新\楔形块标记'
tiffs = glob.glob(folder)
tiffs.sort()
# scale=1
for i,tiff in enumerate(tqdm.tqdm(tiffs)):
    # if(i<1908):
    #     continue
    txt = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '.txt')
    jpg = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '.jpg')
    jpg = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
    tiff1 = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
    jpg[:, :, 0] = tiff1
    jpg[:, :, 2] = 0
    image_path_new=os.path.join(folder_new,os.path.split(tiff)[1][:-5] + '.png')
    # tag_path_new=os.path.join(folder_new,os.path.split(tiff)[1][:-5] + '.tag')

    # tiff1=cv2.resize(tiff1, (int(tiff1.shape[1] / scale), int(tiff1.shape[0] / scale)), interpolation=cv2.INTER_AREA)
    # cv2.imwrite('1.png', tiff1)
    # break

    # tiff_clip=tiff1[int(tiff1.shape[0]*0.15):int(tiff1.shape[0]*0.85),:,:]

    # shutil.copy(tag, tag_path_new)
    # cv2.imwrite(image_path_new, tiff_clip)
    # shutil.copy(tag, tag_path_new)
    cv2.imencode('.png', jpg)[1].tofile(image_path_new)

