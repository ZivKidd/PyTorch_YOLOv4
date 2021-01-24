import glob
import os
import tqdm
import shutil
import numpy as np

# 找到tag文件
wedge_txts=[]
# wedge_txts.extend(glob.glob(r"E:\广州数据新\*\*\*_wedge.txt"))
# wedge_txts.extend(glob.glob(r"E:\广州数据新\*\*\*\*_wedge.txt"))
wedge_txts.extend(glob.glob(r"G:\广州数据新\*\*\*\*\*_wedge.txt"))
wedge_txts.extend(glob.glob(r"G:\广州数据新\*\*\*\*\*\*_wedge.txt"))
wedge_txts.extend(glob.glob(r"G:\广州数据新\*\*\*\*\*\*\*_wedge.txt"))
wedge_txts.extend(glob.glob(r"G:\广州数据新\*\*\*\*\*\*\*\*_wedge.txt"))
# wedge_txts.extend(glob.glob(r"E:\广州数据新\*\*\*\*\*\*\*\*\*_wedge.txt"))
# random.shuffle(tags)

# tiffs=[]
# tiffs.extend(glob.glob(r"G:\广州数据\*\*\*\scan\*\*.tiff"))
# tiffs.extend(glob.glob(r"G:\广州数据\*\*\*\*\scan\*\*.tiff"))

num = 0
# txt=[]
folder=r"E:\广州数据新\楔形块标记"
# txt_path=os.path.join(folder,'names.txt')
for wedge_txt in tqdm.tqdm(wedge_txts):
    tiff = os.path.join(os.path.split(wedge_txt)[0], os.path.split(wedge_txt)[1][:-10] + '.tiff')
    jpg = os.path.join(os.path.split(wedge_txt)[0], os.path.split(wedge_txt)[1][:-10] + '_bolt.jpg')
    if (os.path.exists(tiff) and os.path.exists(jpg)):
        num += 1
        shutil.copy(tiff, os.path.join(folder,str(num).zfill(6)+'.tiff'))
        # shutil.copy(tiff, os.path.join(folder,str(num).zfill(6)+'.tiff'))
        shutil.copy(jpg, os.path.join(folder,str(num).zfill(6)+'.jpg'))
        shutil.copy(wedge_txt, os.path.join(folder,str(num).zfill(6)+'.txt'))
        # txt.append([str(num).zfill(6),tiff,tag])
        # txt.append([str(num).zfill(6),tiff])

# txt=np.asarray(txt,dtype=np.str)
# np.savetxt(txt_path,txt,delimiter=';',fmt='%s')
        # print()
print(num)
