import glob
import os
import tqdm
import shutil
import numpy as np
import cv2
import copy

# folder = r"E:\*\*.png"
# folder_new = r"E:\norail"
tiffs =[]
tiffs.extend(glob.glob(r"F:\origin\*.tiff"))
# tiffs.extend(glob.glob(r"E:\result_xian\*.png"))
tiffs.sort()
# rail_top = 0.13
# rail_down = 0.87

for i, tiff in enumerate(tqdm.tqdm(tiffs)):
    # if(i<181):
    #     continue
    txt_wedge = os.path.splitext(tiff)[0] + '_wedge.txt'
    txt_ring=os.path.splitext(tiff)[0] + '_ring.txt'
    txt_new = os.path.splitext(tiff)[0] + '.txt'
    # tiff_new = os.path.join(folder_new, str(i).zfill(6)+'.png')

    # tiff1 = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
    if(not os.path.exists(txt_wedge)):
        continue
    data_ring=np.loadtxt(txt_ring)
    data_wedge=np.loadtxt(txt_wedge)
    if(data_wedge.size==5):
        data_wedge=data_wedge.reshape([-1, 5])
    if(data_ring.size==5):
        data_ring=data_ring.reshape([-1,5])

    # data_wedge[:, 2] *= tiff1.shape[0]
    # data_wedge[:, 4] *= tiff1.shape[0]
    # data_wedge[:, 2] -= tiff1.shape[0] * rail_top
    # data_wedge[:, 2] /= tiff1.shape[0] * (rail_down - rail_top)
    # data_wedge[:, 4] /= tiff1.shape[0] * (rail_down - rail_top)

    data_wedge = data_wedge[np.where(data_wedge[:, 2] < 1)[0]]
    data_ring = data_ring[np.where(data_ring[:, 2] < 1)[0]]
    data_ring=data_ring[np.where(data_ring[:,1]>0)[0]]
    data_wedge=data_wedge[np.where(data_wedge[:,1]>0)[0]]


    data_wedge=np.concatenate([data_wedge, data_ring], axis=0)
    np.savetxt(txt_new, data_wedge)

    # tiff_clip = tiff1[int(tiff1.shape[0] * rail_top):int(tiff1.shape[0] * rail_down), :, :]


    # scale=1921/np.max(tiff_clip.shape)
    # tiff_clip=cv2.resize(tiff_clip,(int(tiff_clip.shape[1]*scale),int(tiff_clip.shape[0]*scale)),interpolation=cv2.INTER_AREA)
    #
    #
    # cv2.imencode('.png', tiff_clip)[1].tofile(tiff_new)
