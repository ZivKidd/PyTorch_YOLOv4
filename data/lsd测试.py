import cv2
import numpy as np
import os
# import pylsd
from pylsd.lsd import lsd
fullName = r"D:\TLSD\data\2-8号线广州北延陈家祠-彩虹桥下行\scan\scan_preview_0021\scan_preview_0021_bolt.jpg"
folder, imgName = os.path.split(fullName)
src=cv2.imdecode(np.fromfile(fullName, dtype=np.uint8), -1)
# src = cv2.imread(fullName, cv2.IMREAD_COLOR)
if(len(src.shape)==2):
    src=cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

lines = lsd(gray)
for i in range(lines.shape[0]):
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    width = lines[i, 4]
    cv2.line(src, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))
# cv2.imwrite(os.path.join(folder, 'cv2_' + imgName.split('.')[0] + '.jpg'), src)
cv2.imencode('.png', src)[1].tofile(os.path.join(folder, 'cv2_' + imgName.split('.')[0] + '.png'))