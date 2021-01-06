import glob
import os
import tqdm
import shutil
import numpy as np
import cv2

txts=[]
txts.extend(glob.glob(r"D:\xuzeran\subway\xiexingkuai_kuang\train_norail\*.txt"))
txts.extend(glob.glob(r"D:\xuzeran\subway\xiexingkuai_kuang\test_norail\*.txt"))

for i,txt in enumerate(tqdm.tqdm(txts)):
    data=np.loadtxt(txt)
    if(data.size==5):
        data=data.reshape([1,-1])
    data=data[np.where(data[:,2]<1)[0]]
    np.savetxt(txt, data)




