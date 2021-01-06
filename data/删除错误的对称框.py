import glob
import os
import tqdm
import shutil
import numpy as np
import cv2
from symmetry import SymmetryDetector

pngs = []
pngs.extend(glob.glob(r"D:\xuzeran\subway\xiexingkuai_kuang\train_norail\*.png"))
pngs.extend(glob.glob(r"D:\xuzeran\subway\xiexingkuai_kuang\test_norail\*.png"))

for i, png in enumerate(tqdm.tqdm(pngs)):
    # png = r"D:\xuzeran\subway\xiexingkuai_kuang\test_norail\000853.png"

    image = cv2.imdecode(np.fromfile(png, dtype=np.uint8), -1)

    txt = os.path.splitext(png)[0] + '.txt'
    txt = np.loadtxt(txt)
    if (txt.size == 5):
        txt = txt.reshape([1, -1])
    txt[:, 1] *= image.shape[1]
    txt[:, 3] *= image.shape[1]
    txt[:, 2] *= image.shape[0]
    txt[:, 4] *= image.shape[0]
    for t in txt:
        x1 = int(t[1] - t[3] / 2)
        x2 = int(t[1] + t[3] / 2)
        y1 = int(t[2] - t[4] / 2)
        y2 = int(t[2] + t[4] / 2)
        image_rect=image[y1:y2,x1:x2]
        image_rect_ori=image[y1:y2,x1:x2]
        image_rect = cv2.cvtColor(image_rect, cv2.COLOR_BGR2GRAY)
        image_rect = cv2.medianBlur(image_rect, 7)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # 腐蚀图像
        image_rect = cv2.dilate(image_rect, kernel)

        sym_detect = SymmetryDetector(resolution=180, centroid=(-1, -1),
                                               scoreThreshold=0.)
        print(sym_detect.getSymmetry(image_rect, 90))
        # 显示腐蚀后的图像
        # cv2.imshow("Eroded Image", eroded)
        # # 膨胀图像
        # dilated = cv2.dilate(img, kernel)

        # image_rect_left=image_rect[:,:int(image_rect.shape[1]/2)]
        # image_rect_right=image_rect[:,int(image_rect.shape[1]/2):]
        # image_rect_right=np.flip(image_rect_right, axis=1)
        # if(image_rect_right.shape[1]>image_rect_left.shape[1]):
        #     image_rect_right=image_rect_right[:,:-1]
        #
        # diff=np.sum(np.abs(image_rect_right-image_rect_left))/image_rect.size

        # image_rect_gray = cv2.cvtColor(image_rect, cv2.COLOR_BGR2GRAY)
        # # image_rect_gray = cv2.GaussianBlur(image_rect_gray, (5, 5), 0)
        # image_rect_gray = cv2.Canny(image_rect, 50, 150, apertureSize=3)
        # minLineLength = 10
        # maxLineGap = 10
        # lines = cv2.HoughLinesP(image_rect_gray, 1, np.pi / 50, 100, minLineLength, maxLineGap)
        #
        # for x1, y1, x2, y2 in lines[0]:
        #     cv2.line(image_rect_ori, (x1, y1), (x2, y2), (255, 0, 0), 6)
        # # print(diff)

        cv2.imshow('a',image_rect)
        cv2.waitKey(0)
        # print()
    # if(data.size==5):
    #     data=data.reshape([1,-1])
    # data=data[np.where(data[:,2]<1)[0]]
    # np.savetxt(png, data)
