import glob
import os
import tqdm
import shutil
import numpy as np

tiffs=[]
tiffs.extend(glob.glob(r"G:\广州数据\*\*\*\scan\*\*.tiff"))
tiffs.extend(glob.glob(r"G:\广州数据\*\*\*\*\scan\*\*.tiff"))

num = 0
txt=[]
folder=r"Z:\subway_scan\guangzhou\origin"
txt_path=os.path.join(folder,'names.txt')
for tiff in tqdm.tqdm(tiffs):
    tag = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '.tag')
    jpg = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '_bolt.jpg')
    if (os.path.exists(tag) and os.path.exists(jpg)):
        num += 1
        shutil.copy(tag, os.path.join(folder,str(num).zfill(6)+'.tag'))
        shutil.copy(tiff, os.path.join(folder,str(num).zfill(6)+'.tiff'))
        shutil.copy(jpg, os.path.join(folder,str(num).zfill(6)+'.jpg'))
        # txt.append([str(num).zfill(6),tiff,tag])
        txt.append([str(num).zfill(6),tiff])

txt=np.asarray(txt,dtype=np.str)
np.savetxt(txt_path,txt,delimiter=';',fmt='%s')
        # print()
print(num)
