import glob
import os
import random

import tqdm
import shutil
import numpy as np
import cv2
import json
import copy
import matplotlib.pyplot as plt
from skimage import io, color,data,filters
from skimage.morphology import disk
import pywt #小波变换要引入这个包

folder = r"D:\desktop\files\shine\subway_beam\filter_test\*\*.tiff"
# folder_new=r'/media/sever/zeran/subway_scan/guangzhou/intensity_process'
tiffs = glob.glob(folder)
tiffs.sort()


for tiff in tqdm.tqdm(tiffs):
    print(tiff)
    # img = cv2.imread(tiff)
    # img=img[:,:,0]

    # name_new=os.path.join(folder_new,os.path.split( tiff)[1])
    # dst=0
    # # 图像归一化
    # fi = img / 255.0
    # # 伽马变换
    # gamma = 1.4
    # out = np.asarray(np.power(fi, gamma)*255,dtype=np.int)
    # dst=cv2.normalize(img, dst, 0, 255, cv2.NORM_L2)
    # cv2.imwrite(name_new,dst)
    # laplation kernel
    # h = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="float32")
    # dst = cv2.filter2D(img, -1, h)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    # dst = clahe.apply(img)
    # dst = cv2.equalizeHist(img)
    # cv2.imwrite(name_new,dst)

#     # break
#     dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
#
#     fshift = np.fft.fftshift(dft)
# # 设置高通滤波器
#
#     rows, cols = img.shape
#
#     crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
#
#     mask = np.ones((rows, cols, 2), np.uint8)
#
#     size=10
#     mask[crow - size:crow + size, ccol - size:ccol + size] = 0
#  # 掩膜图像和频谱图像乘积
#
#     f = fshift * mask
# # 傅里叶逆变换
#
#     ishift = np.fft.ifftshift(f)
#
#     iimg = cv2.idft(ishift)
#
#     res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
#     blurred = cv2.GaussianBlur(img, (11, 11), 0)
    # canny = cv2.Canny(blurred, 20, 60)
    # result=prewitt(img)
    # blurred = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
    img = img[int(img.shape[0] * 0.15):int(img.shape[0] * 0.85), :]
    # img_ori=copy.deepcopy(img)
    img = cv2.equalizeHist(img)
    img = filters.median(img,disk(3))

    # img = filters.roberts_neg_diag(img)
    # img=img/np.max(img)*255
    # img = cv2.equalizeHist(img)
    #
    x1=np.arange(0,img.shape[1])
    x2=np.arange(0,img.shape[1]-1)
    y1=np.sum(img,axis=0).astype(np.float)
    y11 = y1.reshape([1, -1])
    y11=(y11-np.min(y11))/(np.max(y11)-np.min(y11))*255

    y11=np.tile(y11,[img.shape[0],1])
    y2=np.diff(y1)
    y1[0]=0
    y1[1:]=y2
    y12=y1.reshape([1,-1])
    y12=(y12-np.min(y12))/(np.max(y12)-np.min(y12))*255
    y12=np.tile(y12,[img.shape[0],1])

    output=np.zeros([img.shape[0],img.shape[1],3])
    output[:,:,0]=img
    output[:,:,1]=y11
    output[:,:,2]=y11
    # print()

    # db4 = pywt.Wavelet('db4')
    # coeffs = pywt.wavedec(y2, db4)
    # # 高频系数置零
    # coeffs[len(coeffs) - 1] *= 0
    # coeffs[len(coeffs) - 2] *= 0
    # 重构
    # meta = pywt.waverec(coeffs, db4)
    # y2=np.sum(img_ori,axis=0)
    # # img=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    # # img = np.asarray(img, dtype=np.uint8)
    # # img = cv2.equalizeHist(img)
    #
    # # img=cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
    # plt.plot(x1, y1, 'r--', label='type1')
    # plt.plot(x1, meta, 'b--', label='type1')
    # plt.plot(x2, y2, 'r--', label='type1')
    # # plt.plot(x1, y1, 'bo')
    # plt.show()

    cv2.imencode('.png', output)[1].tofile(tiff[:-5]+'.png')