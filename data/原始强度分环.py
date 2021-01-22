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

skip_if_png_exists=False
files = glob.glob(r"D:\TLSD\data\*\scan\*\*distance.csv")
files.sort()
# reflection_files = glob.glob(r"D:\TLSD\data\*\scan\*\*reflection.csv")
# random.shuffle(distance_files)
# random.shuffle(reflection_files)
for f in tqdm.tqdm(files):
    f=r"D:\TLSD\data\1-6号线西安国际医学中心-仁村站区间右线（013+800）\scan\scan_preview_0028\scan_preview_0028_reflection.csv"
    print(f)
    # print(datetime.datetime.now())
    try:
        data = pd.read_csv(f, header=None, na_filter=False, sep=',').to_numpy(dtype=np.float)
    except:
        continue
    if(data.shape !=(10200,5000) and data.shape !=(5100,5000)):
        continue
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
    data = data[int(data.shape[0] * 0.15):int(data.shape[0] * 0.85), :]


    # dist_min=0.8
    # dist_max=4.5


    # dist_min = 0.5
    # dist_max = 5.5
    # for axis_inx in [0,1]:
    # axis_inx=1

    # if (os.path.exists(f[:-4] + str(axis_inx)+'.png') and skip_if_png_exists):
    #     continue

    # dist_max = 0.3


    # data = pd.read_csv(f, header=None, na_filter=False, sep=',').to_numpy(dtype=np.float)
    # 列间做差
    # data1=np.diff(data,axis=axis_inx)
    # data1=np.abs(data1)

    # data1[np.where(data1 > dist_max)] = 0


    # plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
    # plt.show()

    # data[np.logical_or(data < dist_min, data > dist_max)] = 0

    # plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
    # plt.show()
    dist_min = 10000
    dist_max = 1000000
    data[np.where(data < dist_min)] = 0
    data[np.where(data > dist_max)] = 0

    data *= 255 / (dist_max - dist_min)
    # data -= dist_min
    # data1 *= 255 / (dist_max)
    # data1[np.where(data1 < 0)] = 0
    # data1 = data1.astype(np.uint8)
    # data1=255-data1

    # plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
    # plt.show()
    # dst=0
    # data = cv2.equalizeHist(data)
    # clahe = cv2.createCLAHE(clipLimit=50, tileGridSize=(50, 50))
    # data1 = clahe.apply(data1)

    # 每列求和
    data_col_sum = np.sum(data,axis=0)
    data_col_sum = np.diff(data_col_sum)
    data_col_sum=np.abs(data_col_sum)



    # 差分
    # data1 = np.diff(data1)
    threshold=np.percentile(data_col_sum,98)
    rings=np.where(data_col_sum>threshold)
    print(threshold,rings[0].size/data_col_sum.size)

    x1=np.arange(0,data_col_sum.size)
    plt.plot(x1, data_col_sum, 'r--')
    plt.plot(x1, np.ones(data_col_sum.shape)*threshold, 'b-')
    plt.show()

    data1_rgb=np.reshape(data,(data.shape[0],data.shape[1],1))
    data1_rgb=np.tile(data1_rgb,(1,1,3))

    data1_rgb[:,rings[0]]=[0,0,255]

    cv2.imencode('.png', data1_rgb)[1].tofile(f[:-4] +'_ring.png')
    # print(f[:-4] + str(axis_inx)+'.png')


    break

