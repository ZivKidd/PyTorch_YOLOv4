import glob
import os
import tqdm
import shutil
import numpy as np
import cv2


folder = r"D:\xuzeran\subway\xiexingkuai_kuang\train\*.png"
folder_new=r"D:\xuzeran\subway\xiexingkuai_kuang\train_norail"
tiffs = glob.glob(folder)
tiffs.sort()
rail_top=0.13
rail_down=0.87
# scale=1
for i,tiff in enumerate(tqdm.tqdm(tiffs)):
    # if(i<137):
    #     continue
    txt = os.path.splitext(tiff)[0]+ '.txt'
    txt_new=os.path.join(folder_new,os.path.split(txt)[1])
    tiff_new=os.path.join(folder_new,os.path.split(tiff)[1])

    tiff1 = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)

    data=np.loadtxt(txt)
    if(data.size==5):
        data=data.reshape([1,-1])
    # data[:,2]*=tiff1.shape[1]
    # data[:,3]*=tiff1.shape[1]
    data[:,2]*=tiff1.shape[0]
    data[:,4]*=tiff1.shape[0]
    data[:, 2] -= tiff1.shape[0]*rail_top
    data[:, 2] /= tiff1.shape[0]*(rail_down-rail_top)
    data[:, 4] /= tiff1.shape[0]*(rail_down-rail_top)

    np.savetxt(txt_new,data)


    tiff_clip=tiff1[int(tiff1.shape[0]*rail_top):int(tiff1.shape[0]*rail_down),:,:]
    cv2.imencode('.png', tiff_clip)[1].tofile(tiff_new)



