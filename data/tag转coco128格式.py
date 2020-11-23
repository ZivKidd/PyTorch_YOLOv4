import glob
import json
import os

import cv2
import numpy as np
import tqdm

folder = r"/media/sever/data1/xzr/subway/positive"
result_folder='/media/sever/data1/xzr/subway/coco128'
for f in ['train','test','val']:
    folder1=os.path.join(folder,f)
    tag_files = glob.glob(os.path.join(folder1,'*.tag'))
    result_folder1=os.path.join(result_folder,f)

    if(not os.path.exists(result_folder1)):
        os.mkdir(result_folder1)

    tag_files.sort()

    scale = 4
    bbox_size = 200

    for ind, tag in enumerate(tqdm.tqdm(tag_files)):
        coco128 = []
        # easydl_json = {}
        # easydl_json['labels'] = []

        # cols = []
        # rows = []
        bbox = []
        f = open(tag, encoding='utf-8')
        text = f.read()
        text = text.split('\n')

        tiff = tag[:-4] + '.tiff'
        tiff = cv2.imread(tiff)
        tiff = tiff[:, :, :1]

        for t in text:
            if (t == ''):
                continue
            data = json.loads(t)
            col = data['ColIndex']
            row = -1
            for s in data['SegmentInfos']:
                if (s is None):
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
            row = row / tiff.shape[0]
            col = col / tiff.shape[1]

            # class x_center y_center width height
            coco128.append([0, col, row, bbox_size / tiff.shape[1], bbox_size / tiff.shape[0]])

        if (len(coco128) < 5):
            print(tag)
            continue

        # easydl_json = json.dumps(easydl_json)
        # f = open(os.path.join(r"Z:\subway_scan\coco128",str(ind).zfill(6)+'.txt'), 'w')
        # f.write(easydl_json)
        # f.close()

        fmt = '%d', '%6f', '%6f', '%6f', '%6f'
        coco128 = np.asarray(coco128)
        np.savetxt(os.path.join(result_folder1, str(ind).zfill(6) + '.txt'), coco128, fmt=fmt)

        tiff = cv2.resize(tiff, (int(tiff.shape[1] / scale), int(tiff.shape[0] / scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(result_folder1, str(ind).zfill(6) + '.png'), tiff)
        # print()
