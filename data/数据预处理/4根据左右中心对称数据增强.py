import glob
import os
import tqdm
import shutil
import numpy as np
import cv2


folder = r"F:\albumentations\*.png"
# folder_new=r"D:\xuzeran\subway\xiexingkuai_kuang\test_norail"
tiffs = glob.glob(folder)
tiffs.sort()
# rail_top=0.13
# rail_down=0.87
# scale=1
for i,tiff in enumerate(tqdm.tqdm(tiffs)):

    if(i>42):
        break

    txt = os.path.splitext(tiff)[0]+ '.txt'
    txt_new=tiff[:-4]+'_mirror.txt'
    tiff_new=tiff[:-4]+'_mirror.png'
    # tiff_new=os.path.join(folder_new,os.path.split(tiff)[1])

    tiff1 = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)

    data=np.loadtxt(txt)
    if(data.size==5):
        data=data.reshape([-1,5])
    data1=data[np.where(data[:,0]==0)]
    data1[:,0]=1
    data[np.where(data[:, 0] == 0)]=data1
    data[:,1]*=-1
    data[:,1]+=1

    np.savetxt(txt_new,data)


    tiff_clip=np.flip(tiff1,axis=1)
    cv2.imencode('.png', tiff_clip)[1].tofile(tiff_new)
    # break



