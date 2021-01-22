import glob
import json
import os
import numpy as np
import cv2
import tqdm
png=r"D:\xuzeran\subway\norail\test_small\001258_mirror.png"
coco=png[:-4]+'.txt'
coco=np.loadtxt(coco).tolist()
# bbox_size = 200
png=cv2.imread(png)
width=png.shape[1]
height=png.shape[0]

for b in coco:
    png = cv2.rectangle(img=png, pt1=(int(b[1]*width-b[3]*width/2), int(b[2]*height-b[4]*height/2)),
                         pt2=(int(b[1]*width+b[3]*width/2), int(b[2]*height+b[4]*height/2)), color=(0, 0, 0),
                         thickness=3)
    png = cv2.circle(png, (int(b[1] *width), int(b[2] *height)), 3, (0, 0, 255))
    # print()

cv2.imwrite('rect.png', png)
