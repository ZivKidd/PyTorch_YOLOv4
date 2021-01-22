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

skip_if_png_exists=True
files = glob.glob(r"D:\TLSD\data\*\scan\*\*distance.csv")
files.sort()
# reflection_files = glob.glob(r"D:\TLSD\data\*\scan\*\*reflection.csv")
# random.shuffle(distance_files)
# random.shuffle(reflection_files)
for f in tqdm.tqdm(files):
    # f=r"Z:\subway_scan\origin_data\9-郑州1号线会展中心-黄河南路1（下行，第1趟）\scan\scan_preview_0000\scan_preview_0000_reflection.csv"
    print(f)
    # print(datetime.datetime.now())
    try:
        data = pd.read_csv(f, header=None, na_filter=False, sep=',').to_numpy(dtype=np.float)
    except:
        continue
    if(data.shape !=(10200,5000)):
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

    # dist_min=0.8
    # dist_max=4.5
    if ('reflection' in f):
        if(os.path.exists(f[:-4] + '.png') and skip_if_png_exists):
            continue
        if ('西安' in f):
            dist_min = 10000
            dist_max = 500000
        elif ('广州' in f):
            dist_min = 10000
            dist_max = 1000000
        elif ('郑州' in f):
            dist_min = 10000
            dist_max = 700000

        # data = data[np.where(data > 0)]
        # data = data[np.where(data < 20)]
        # plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
        # plt.show()
        # data = data[int(data.shape[0] * 0.15):int(data.shape[0] * 0.85), :]

        # plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
        # plt.show()

        data[np.where(data < dist_min)] = 0
        data[np.where(data >dist_max)] = dist_max
        data -= dist_min
        data *= 255 / (dist_max - dist_min)
        data[np.where(data < 0)] = 0
        data = data.astype(np.uint8)

        plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
        plt.show()
        # dst=0
        # data = cv2.equalizeHist(data)

        cv2.imencode('.png', data)[1].tofile(f[:-4] + '.png')
        # print(f[:-4] + '.png')
    else:

        # dist_min = 0.5
        # dist_max = 5.5
        for axis_inx in [0,1]:

            if (os.path.exists(f[:-4] + str(axis_inx)+'.png') and skip_if_png_exists):
                continue

            dist_max = 0.3


            # data = pd.read_csv(f, header=None, na_filter=False, sep=',').to_numpy(dtype=np.float)
            data1=np.diff(data,axis=axis_inx)
            data1=np.abs(data1)

            data1[np.where(data1 > dist_max)] = 0

            # data = data[int(data.shape[0] * 0.15):int(data.shape[0] * 0.85), :]

            # plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
            # plt.show()

            # data[np.logical_or(data < dist_min, data > dist_max)] = 0

            # plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
            # plt.show()

            # data -= dist_min
            data1 *= 255 / (dist_max)
            data1[np.where(data1 < 0)] = 0
            data1 = data1.astype(np.uint8)
            data1=255-data1

            # plt.hist(data.reshape([-1]), bins=20, color='red', histtype='stepfilled', alpha=0.75)
            # plt.show()
            # dst=0
            # data = cv2.equalizeHist(data)
            clahe = cv2.createCLAHE(clipLimit=50, tileGridSize=(50, 50))
            data1 = clahe.apply(data1)


            cv2.imencode('.png', data1)[1].tofile(f[:-4] + str(axis_inx)+'.png')
            # print(f[:-4] + str(axis_inx)+'.png')


