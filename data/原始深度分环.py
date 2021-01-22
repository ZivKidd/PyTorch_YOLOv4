# -*-coding: utf-8 -*-

import datetime
import glob
import random
import tqdm
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import dask.dataframe as dd
import os
from scipy.optimize import curve_fit

# test function
def function(data, a, b, c):
    x = data[0]
    y = data[1]
    return a * (x**b) * (y**c)

skip_if_png_exists=False
files = glob.glob(r"D:\TLSD\data\*\scan\*\*distance.csv")
files.sort()
# reflection_files = glob.glob(r"D:\TLSD\data\*\scan\*\*reflection.csv")
# random.shuffle(distance_files)
# random.shuffle(reflection_files)
for f in tqdm.tqdm(files):
    f=r"D:\TLSD\data\广州13号线夏园-南岗上行线高速扫描\scan\scan_preview_0035\scan_preview_0035_distance.csv"
    # f=r"D:\TLSD\data\1-6号线西安国际医学中心-仁村站区间右线（013+800）\scan\scan_preview_0030\scan_preview_0030_distance.csv"
    # f=r"D:\TLSD\data\2-8号线广州北延陈家祠-彩虹桥下行\scan\scan_preview_0026\scan_preview_0026_distance.csv"
    print(f)
    # print(datetime.datetime.now())
    try:
        data = pd.read_csv(f, header=None, na_filter=False, sep=',').to_numpy(dtype=np.float)
    except:
        continue
    if(data.shape !=(10200,5000) and data.shape !=(5100,5000)):
        continue
    rows,cols=data.shape
    # data=data.flatten()
    # x=np.arange(0,cols,1).reshape(1,-1)
    # y=np.arange(0,rows,1).reshape(-1,1)
    # x=np.tile(x,[rows,1]).flatten()
    # y=np.tile(y,[1,cols]).flatten()
    # # y=np.arange(0,row.size,1)
    # parameters, covariance = curve_fit(function, [x, y], data)
    diffs=[]
    for i in tqdm.tqdm(range(rows)):
        # if(i%1000!=0):
        #     continue
        row=data[i,:]
        # row=np.sum(data,axis=0)/data.shape[0]
        x=np.arange(0,row.size,1)
        f1 = np.polyfit(x, row, 20)
        p1 = np.poly1d(f1)
        yvals = p1(x)  # 拟合y值
        # means=np.mean(row)
        diff=row-yvals
        # diff.sort()
        per99=np.percentile(diff,99)
        per1=np.percentile(diff,1)
        diff[np.where(diff>per99)]=per99
        diff[np.where(diff<per1)]=per1

        diff *= 255 / (per99 - per1)

        diffs.append(diff.tolist())
        # plot1 = plt.plot(x, row, 's', label='original values')
        # plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
        # plt.title(str(i))
        # plt.show()
    result=np.asarray(diffs)
    # low=np.min(result)
    # high=np.max(result)
    # result[np.where(result<low)]=low
    # result[np.where(result>high)]=high
    # result *= 255 / (high - low)
    # result=255-result
    result=np.asarray(result,dtype=np.uint8)
    # result=result.reshape((result.shape[0],result.shape[1],1))
    # result=cv2.applyColorMap(result,cv2.COLORMAP_JET)
    cv2.imencode('.png', result)[1].tofile(f[:-4] + '_depth.png')
    break

    # per99=np.percentile(data,99)
    # per1=np.percentile(data,1)
    # data[np.where(data>per99)]=per99
    # data[np.where(data<per1)]=per1
    # # data[np.where(data > 5)] = 5
    #
    # meanDepthDarray = np.mean(data, axis=1, keepdims=True)
    #
    # meanDepthDarrayTile = np.tile(meanDepthDarray, data.shape[1])
    #
    # depthDiffDarray = data - meanDepthDarrayTile
    #
    # result = 0.2 * meanDepthDarray + 10 * depthDiffDarray
    # low=np.percentile(result,5)
    # high=np.percentile(result,95)
    # result[np.where(result<low)]=low
    # result[np.where(result>high)]=high
    # result *= 255 / (high - low)
    # result=255-result
    # result=np.asarray(result,dtype=np.uint8)
    # # result=result.reshape((result.shape[0],result.shape[1],1))
    # # result=cv2.applyColorMap(result,cv2.COLORMAP_JET)
    # cv2.imencode('.png', result)[1].tofile(f[:-4] + '_depth.png')
    # break


    # with open(f, 'r') as file:
    #     reader = csv.reader(file, delimiter=',')
    # headers = next(reader)
    #     data = np.asarray(list(reader)).astype(float)
    # print()
    # data = dd.read_csv(f,blocksize=2500e6).values.compute()
    # data=np.loadtxt(fname=f,delimiter=',')
    # data_iter=csv.DictReader(open(f))
    # data = [data for data in data_iter]
    # print()
    # 去除铁轨部分
    # data = data[int(data.shape[0] * 0.15):int(data.shape[0] * 0.85), :]


    # dist_min=0.8
    # dist_max=4.5


    # dist_min = 0.5
    # dist_max = 5.5
    # for axis_inx in [0,1]:
    # axis_inx=1
    #
    # if (os.path.exists(f[:-4] + str(axis_inx)+'.png') and skip_if_png_exists):
    #     continue
    #
    # dist_max = 0.3
    #
    #
    # # data = pd.read_csv(f, header=None, na_filter=False, sep=',').to_numpy(dtype=np.float)
    # # 列间做差
    # data1=np.diff(data,axis=axis_inx)
    # data1=np.abs(data1)
    #
    # data1[np.where(data1 > dist_max)] = 0
    #
    #
    # # plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
    # # plt.show()
    #
    # # data[np.logical_or(data < dist_min, data > dist_max)] = 0
    #
    # # plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
    # # plt.show()
    #
    # # data -= dist_min
    # data1 *= 255 / (dist_max)
    # data1[np.where(data1 < 0)] = 0
    # data1 = data1.astype(np.uint8)
    # # data1=255-data1
    #
    # # plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
    # # plt.show()
    # # dst=0
    # # data = cv2.equalizeHist(data)
    # clahe = cv2.createCLAHE(clipLimit=50, tileGridSize=(50, 50))
    # data1 = clahe.apply(data1)
    #
    # # 每列求和
    # data_col_sum = np.sum(data1,axis=0)
    #
    # x1=np.arange(0,data_col_sum.size)
    # plt.plot(x1, data_col_sum, 'r--')
    # plt.show()
    #
    # # 差分
    # # data1 = np.diff(data1)
    # threshold=np.percentile(data_col_sum,98)
    # rings=np.where(data_col_sum>threshold)
    # print(threshold,rings[0].size/data_col_sum.size)
    #
    # data1_rgb=np.reshape(data1,(data1.shape[0],data1.shape[1],1))
    # data1_rgb=np.tile(data1_rgb,(1,1,3))
    #
    # data1_rgb[:,rings[0]]=[0,0,255]
    #
    # cv2.imencode('.png', data1_rgb)[1].tofile(f[:-4] +'_ring.png')
    # print(f[:-4] + str(axis_inx)+'.png')


    # break

