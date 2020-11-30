import glob
import os
import tqdm
import shutil
import numpy as np

folder = r"G:\西安数据(正式解算）\*\*\*\scan\*\*.tiff"
tiffs = glob.glob(folder)
num = 0
txt=[]
txt_path=r'Y:\subway\负样本\path.txt'
for tiff in tqdm.tqdm(tiffs):
    tag = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-5] + '.tag')
    if (not os.path.exists(tag)):
        num += 1
        # shutil.copy(tag, os.path.join(r'Y:\subway\负样本',str(num).zfill(6)+'.tag'))
        shutil.copy(tiff, os.path.join(r'Y:\subway\负样本',str(num).zfill(6)+'.tiff'))
        # txt.append([str(num).zfill(6),tiff,tag])
        txt.append([str(num).zfill(6),tiff])

txt=np.asarray(txt,dtype=np.str)
np.savetxt(txt_path,txt,delimiter=';',fmt='%s')
        # print()
print(num)
