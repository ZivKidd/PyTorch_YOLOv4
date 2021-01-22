import glob
import os
import tqdm
import shutil
import numpy as np
import cv2

txts=[]
txts.extend(glob.glob(r"D:\xuzeran\text_small\*\*.txt"))
# txts.extend(glob.glob(r"D:\xuzeran\subway\norail\train_small\*.txt"))

for i,txt in enumerate(tqdm.tqdm(txts)):
    data=np.loadtxt(txt)
    if(data.size==5):
        data=data.reshape([1,5])
    # if(data)
    data=data[np.where(data[:,2]<1)[0]]
    data=data[np.where(data[:,1]>0)[0]]
    # data[:,0]+=2
    np.savetxt(txt, data)




