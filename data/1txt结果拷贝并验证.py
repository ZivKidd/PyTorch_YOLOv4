import glob
import os
import tqdm
import shutil
import numpy as np
import cv2

folder = r"E:\广州数据新\楔形块标记\*.png"
folder_new=r"E:\广州数据新\楔形块标记_结果"
tiffs = glob.glob(folder)
tiffs.sort()
# scale=1
for i,tiff in enumerate(tqdm.tqdm(tiffs)):
    if(i<627):
        continue
    # 读入txt
    txt = os.path.join(os.path.split(tiff)[0], os.path.split(tiff)[1][:-4] + '.txt')
    coco = np.loadtxt(txt)

    if(np.where(coco<0)[0].size>0):
        continue
    if(coco.size==5):
        coco=[coco.tolist()]
    else:
        coco = coco.tolist()

    # 读入图像并画框
    png = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
    width = png.shape[1]
    height = png.shape[0]
    for b in coco:
        png = cv2.rectangle(img=png, pt1=(int(b[1] * width - b[3] * width / 2), int(b[2] * height - b[4] * height / 2)),
                            pt2=(int(b[1] * width + b[3] * width / 2), int(b[2] * height + b[4] * height / 2)),
                            color=(0, 0, 255),
                            thickness=3)

    # 画了框的图像保存
    image_path_new=os.path.join(folder_new,os.path.split(tiff)[1][:-4] + '.jpg')
    cv2.imencode('.jpg', png)[1].tofile(image_path_new)

    shutil.copy(tiff, os.path.join(folder_new,os.path.split(tiff)[1]))
    shutil.copy(txt, os.path.join(folder_new,os.path.split(txt)[1]))
    # shutil.copy(tiff, os.path.join(folder,str(num).zfill(6)+'.tiff'))
    # shutil.copy(jpg, os.path.join(folder, str(num).zfill(6) + '.jpg'))
    # shutil.copy(wedge_txt, os.path.join(folder, str(num).zfill(6) + '.txt'))

