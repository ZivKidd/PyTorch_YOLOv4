import glob
import json
import os
import random
import shutil

import cv2
import tqdm

# train val test 按 4 1 2 分

folder = r"E:\norail"
tag_files = glob.glob(folder + '/*.png')
random.shuffle(tag_files)

train_folder=os.path.join(folder,'train')
test_folder=os.path.join(folder,'test')
# val_folder=os.path.join(folder,'val')

for f in [train_folder,test_folder]:
    if(not os.path.exists(f)):
        os.mkdir(f)

for ind,tag in enumerate(tqdm.tqdm(tag_files)):
    tiff = tag[:-4] + '.txt'
    if(ind%7<5):
        shutil.copy(tag,os.path.join(train_folder,os.path.split(tag)[1]))
        shutil.copy(tiff,os.path.join(train_folder,os.path.split(tiff)[1]))
    # elif(ind%7==4):
    #     shutil.copy(tag,os.path.join(val_folder,os.path.split(tag)[1]))
    #     shutil.copy(tiff,os.path.join(val_folder,os.path.split(tiff)[1]))
    else:
        shutil.copy(tag,os.path.join(test_folder,os.path.split(tag)[1]))
        shutil.copy(tiff,os.path.join(test_folder,os.path.split(tiff)[1]))
