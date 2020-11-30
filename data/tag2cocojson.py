import glob
import json
import os

import cv2
import tqdm

folder = r'/media/sever/data1/xzr/efficientdet/data/subway/positive'
tag_files = glob.glob(folder + '/*.tag')
# tiff_files=glob.glob(folder+'/*.tiff')
#
tag_files.sort()
# tiff_files.sort()
bbox_size = 100

coco_json={}
coco_json['images']=[]
coco_json['annotations']=[]

coco_json['categories']=[]
coco_json['categories'].append({'id':1,'name':'corner'})

for ind,tag in enumerate(tqdm.tqdm(tag_files)):
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
                print(tag)
                continue
            if (s['Name'] == 'FB'):
                row = s['RowIndex']
            elif (s['Name'] == 'KP'):
                row = s['RowIndex']
        if (row == -1):
            continue
        # cols.append(col)
        # rows.append(row)
        bbox=[col - bbox_size / 2, row - bbox_size / 2, bbox_size, bbox_size]
        # bbox.append([col - bbox_size / 2, row - bbox_size / 2, bbox_size, bbox_size])

        # 写入标注的属性
        annotation = {}
        annotation['area'] = bbox_size ** 2
        annotation['image_id'] = ind
        annotation['bbox'] = bbox
        annotation['category_id'] = 1
        coco_json['annotations'].append(annotation)

    # 写入图片的属性
    image={}
    image['file_name']=os.path.join(folder, os.path.split(tag)[1][:-4] + '.tiff')
    tiff = cv2.imread(image['file_name'])
    image['height']=tiff.shape[0]
    image['width']=tiff.shape[1]
    image['id']=ind
    coco_json['images'].append(image)

    # json.dump()

coco_json = json.dumps(coco_json)
f = open('/media/sever/data1/xzr/efficientdet/data/subway/coco_json.json', 'w')
f.write(coco_json)
f.close()



    # tiff = os.path.join(folder, os.path.split(tag)[1][:-4] + '.tiff')
    # tiff = cv2.imread(tiff)
    #
    # for b in bbox:
    #     tiff = cv2.rectangle(img=tiff, pt1=(int(b[0]), int(b[1])),
    #                          pt2=(int(b[0] + bbox_size), int(b[1] + bbox_size)), color=(0, 255, 0))
    #     tiff = cv2.circle(tiff, (int(b[0] + bbox_size / 2), int(b[1] + bbox_size / 2)),3, (0, 0, 255))
    #     # print()
    #
    # cv2.imwrite(os.path.join(folder, os.path.split(tag)[1][:-4] + '-rect.tiff'), tiff)

    # print()
