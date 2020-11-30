import glob
import json
import os

import cv2
import tqdm

folder = r"Z:\subway_scan\positive\*.tag"
tag_files = glob.glob(folder)
# tiff_files=glob.glob(folder+'/*.tiff')
#
tag_files.sort()
# tiff_files.sort()

scale=4
bbox_size = 100/scale

for ind,tag in enumerate(tqdm.tqdm(tag_files)):
    easydl_json = {}
    easydl_json['labels'] = []

    # cols = []
    # rows = []
    bbox = []
    f = open(tag, encoding='utf-8')
    text = f.read()
    text = text.split('\n')

    for t in text:
        if (t == ''):
            continue
        data = json.loads(t)
        col = data['ColIndex']
        row = -1
        for s in data['SegmentInfos']:
            if(s is None):
                # print(tag)
                continue
            if (s['Name'] == 'FB'):
                row = s['RowIndex']
            elif (s['Name'] == 'KP'):
                row = s['RowIndex']
        if (row == -1):
            continue
        # cols.append(col)
        # rows.append(row)
        row=row/scale
        col=col/scale

        label={}
        label['y1']=int(row - bbox_size / 2)
        label['x1']=int(col - bbox_size / 2)
        label['y2']=int(row + bbox_size / 2)
        label['x2']=int(col + bbox_size / 2)
        label['name']='beam'

        easydl_json['labels'].append(label)

    if(len(easydl_json['labels'])<5):
        print(tag)
        continue

    easydl_json = json.dumps(easydl_json)
    f = open(os.path.join(r"Z:\subway_scan\easydl\subway",str(ind).zfill(6)+'.json'), 'w')
    f.write(easydl_json)
    f.close()

    tiff = tag[:-4]+'.tiff'
    tiff = cv2.imread(tiff)
    tiff=tiff[:,:,:1]
    tiff=cv2.resize(tiff,(int(tiff.shape[1]/scale),int(tiff.shape[0]/scale)),interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(r"Z:\subway_scan\easydl\subway",str(ind).zfill(6)+'.png'),tiff)
    # print()


