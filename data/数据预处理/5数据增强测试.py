import glob
import os

import albumentations as A
import cv2
import numpy as np
import tqdm

folder = r"F:\albumentations\*.png"
# folder_new = r"E:\广州数据新\norail\train_more"
tiffs = glob.glob(folder)
tiffs.sort()

transform = A.Compose([
    # A.RandomBrightnessContrast(always_apply=True),
    # A.IAASharpen(always_apply=True),
    A.RandomGamma(),
    # A.RandomShadow(always_apply=True, shadow_roi=(0, 0, 1, 1), ),
    # A.RandomRain(always_apply=True),
    A.GaussNoise(always_apply=True),
    A.MultiplicativeNoise(always_apply=True),
    # A.RandomFog(always_apply=True),
    # A.GaussianBlur(always_apply=True),
    # A.ColorJitter(always_apply=True),
    # A.OpticalDistortion(always_apply=True,distort_limit=2, shift_limit=0.5)
    A.RandomCrop(1020,500)],
    bbox_params=A.BboxParams(format='yolo'))

for i, tiff in enumerate(tqdm.tqdm(tiffs)):

    if(i>84):
        break
    # if(tiff.contains('_a'))
    # if(i<129):
    #     continue
    txt = os.path.splitext(tiff)[0] + '.txt'
    # txt_new = os.path.join(folder_new, os.path.split(txt)[1])
    # tiff_new = os.path.join(folder_new, os.path.split(tiff)[1])
    tiff1 = cv2.imdecode(np.fromfile(tiff, dtype=np.uint8), -1)
    if(len(tiff1.shape)==2):
        tiff1=np.reshape(tiff1,(tiff1.shape[0],-1,1))
        tiff1=np.tile(tiff1,(1,1,3))
    data = np.loadtxt(txt)
    bbox=np.zeros_like(data)
    bbox[:,:4]=data[:,1:]
    bbox[:,4]=data[:,0]

    nums = int(np.random.random() * 50)
    lines1 = (np.random.random(nums) * tiff1.shape[1]).astype(np.int)
    lines2 = (np.random.random(nums) * tiff1.shape[0]).astype(np.int)
    tiff1[:, lines1, :] = 30
    tiff1[lines2, :, :] = 30

    transformed = transform(image=tiff1,bboxes=bbox)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    # transformed_class_labels = transformed['class_labels']

    # transformed_image[:, :, 2] = 0

    width = np.random.random() *0.7+0.6
    # print(width)
    transformed_image=cv2.resize(transformed_image,(int(transformed_image.shape[1]*width),int(transformed_image.shape[0])))
    # transformed_image=np.flip(transformed_image,axis=2)

    cv2.imencode('.png', transformed_image)[1].tofile(tiff[:-4]+'_a.png')

    data=np.asarray(transformed_bboxes)
    if(data.size==0):
        np.savetxt(txt[:-4] + '_a.txt', data)
        continue
    if(data.size==5):
        data=data.reshape([1,5])

    bbox=np.zeros_like(data)
    bbox[:,1:]=data[:,:4]
    bbox[:,0]=data[:,4]
    np.savetxt(txt[:-4]+'_a.txt', bbox)
