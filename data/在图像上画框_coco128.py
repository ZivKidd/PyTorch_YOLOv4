import numpy as np
import cv2

png=r"E:\result_licheng\c_000000.tiff"

coco=png[:-5]+'.txt'

coco=np.loadtxt(coco).tolist()

png=cv2.imdecode(np.fromfile(png, dtype=np.uint8), -1)
width=png.shape[1]
height=png.shape[0]

for b in coco:
    png = cv2.rectangle(img=png, pt1=(int(b[1]*width-b[3]*width/2), int(b[2]*height-b[4]*height/2)),
                         pt2=(int(b[1]*width+b[3]*width/2), int(b[2]*height+b[4]*height/2)), color=(0, 0, 255),
                         thickness=3)

cv2.imencode('.png', png)[1].tofile('rect.png')
