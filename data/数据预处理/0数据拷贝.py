import glob
import os
import tqdm
import shutil
import numpy as np

# tiffs=[]
# tiffs.extend(glob.glob(r"G:\广州数据\*\*\*\scan\*\*.tiff"))
# tiffs.extend(glob.glob(r"G:\广州数据\*\*\*\*\scan\*\*.tiff"))

folder_new = r"F:\origin"
# 找到tag文件
tags = []

tags.extend(glob.glob(r"E:\数据\*\scan\*\*.tag"))
tags.extend(glob.glob(r"E:\数据\*\*\scan\*\*.tag"))
tags.extend(glob.glob(r"E:\数据\*\*\*\scan\*\*.tag"))
tags.extend(glob.glob(r"E:\数据\*\*\*\*\scan\*\*.tag"))
tags.extend(glob.glob(r"E:\数据\*\*\*\*\*\scan\*\*.tag"))
tags.extend(glob.glob(r"E:\数据\*\*\*\*\*\*\scan\*\*.tag"))
tags.extend(glob.glob(r"E:\数据\*\*\*\*\*\*\*\scan\*\*.tag"))
tags.extend(glob.glob(r"E:\数据\*\*\*\*\*\*\*\*\scan\*\*.tag"))
txt=[]
# random.shuffle(tags)
tags.sort()
print(len(tags))
num = 0
for ind, tag in enumerate(tqdm.tqdm(tags)):
    # if(ind<1231):
    #     continue
    # tag=r"E:\广州数据新\广州8号线北延\逐环标记\广州8号线北延11-5\5-8号线北延鹅掌坦-同德下行\scan\scan_preview_0007\scan_preview_0007.tag"

    # 根据文件判断短边在左还是右
    tag_folder = os.path.split(tag)[0]
    tag_folder = os.path.split(tag_folder)[0]
    tag_folder = os.path.split(tag_folder)[0]

    if (not os.path.exists(os.path.join(tag_folder, 'right_shorter.txt'))):
        #     left_shorter=False
        # if( os.path.exists(os.path.join(tag_folder,'left_shorter.txt'))):
        continue

    tiff = os.path.join(tag[:-4] + '.tiff')
    jpg = os.path.join(tag[:-4] + '_bolt.jpg')
    if (os.path.exists(tiff) and os.path.exists(jpg)):
        num += 1
        shutil.copy(tag, os.path.join(folder_new,str(num).zfill(6)+'.tag'))
        shutil.copy(tiff, os.path.join(folder_new,str(num).zfill(6)+'.tiff'))
        shutil.copy(jpg, os.path.join(folder_new,str(num).zfill(6)+'.jpg'))
        # txt.append([str(num).zfill(6),tiff,tag])
        txt.append([str(num).zfill(6),tiff])

txt=np.asarray(txt,dtype=np.str)
np.savetxt(r"F:\origin.txt",txt,delimiter=';',fmt='%s')
        # print()
print(num)
