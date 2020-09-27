import os

import numpy as np

from tools.cfg import TXT_PATH_416

LABEL_416 = os.path.join(TXT_PATH_416, 'label.txt')
NPY_PATH = r'E:\VOC2017_416\array'
IMAGES_SIZE=416#图片大小
SEED = 66
anchor_boxes = [
    [126,98],
    [311,277],
    [16,20],
    [67,52],
    [92,180],
    [161,240],
    [28,48],
    [51,109],
    [262,144],
] #[w,h]
LABEL={
    'aeroplane':0,
    'bicycle':1,
    'bird':2,
    'boat':3,
    'bottle':4,
    'bus':5,
    'car':6,
    'cat':7,
    'chair':8,
    'cow':9,
    'diningtable':10,
    'dog':11,
    'horse':12,
    'motorbike':13,
    'person':14,
    'pottedplant':15,
    'sheep':16,
    'sofa':17,
    'train':18,
    'tvmonitor':19,
    }#标签编码


def sort(anchors):
    '''
    对anchor以面积进行排序
    :param anchors: kmeans筛选的anchors
    :return: 以面积排序的索引值
    '''
    anchor_area = np.array(anchors)
    anchor_area = anchor_area[:, 0] * anchor_area[:, 1]
    sorted_anchor_area = np.argsort(anchor_area, axis=0)
    return sorted_anchor_area


sort_index = sort(anchor_boxes)

ANCHOR_BOXES = {
    13: [anchor_boxes[sort_index[-1]], anchor_boxes[sort_index[-2]], anchor_boxes[sort_index[-3]]],
    26: [anchor_boxes[sort_index[-4]], anchor_boxes[sort_index[-5]], anchor_boxes[sort_index[-6]]],
    52: [anchor_boxes[sort_index[-7]], anchor_boxes[sort_index[-8]], anchor_boxes[sort_index[-9]]]
}
ANCHOR_BOXES_AREA={
    13:[ANCHOR_BOXES[13][0][0]*ANCHOR_BOXES[13][0][1],ANCHOR_BOXES[13][1][0]*ANCHOR_BOXES[13][1][1],ANCHOR_BOXES[13][2][0]*ANCHOR_BOXES[13][2][1]],
    26:[ANCHOR_BOXES[26][0][0]*ANCHOR_BOXES[26][0][1],ANCHOR_BOXES[26][1][0]*ANCHOR_BOXES[26][1][1],ANCHOR_BOXES[26][2][0]*ANCHOR_BOXES[26][2][1]],
    52:[ANCHOR_BOXES[52][0][0]*ANCHOR_BOXES[52][0][1],ANCHOR_BOXES[52][1][0]*ANCHOR_BOXES[52][1][1],ANCHOR_BOXES[52][2][0]*ANCHOR_BOXES[52][2][1]],
}
# print(ANCHOR_BOXES)