import math
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from core.cfg import *


class DetectData(Dataset):
    def __init__(self):
        # self.label_path=LABEL_416
        # self.img_path=SQUARE_IMAGE_PATH
        # self.img_path =r'E:\overfit_data\square\images'
        self.img_path = r'E:\VOC2017_gened\images'
        # self.label_path =r'E:\overfit_data\square\label.txt'
        self.label_path = r'E:\VOC2017_416\label.txt'
        # ==========读取txt文档======
        with open(self.label_path, 'r') as f:
            self.text = f.readlines()

        self.data_paths = []
        # ==========读取txt文档======
        for i in range(len(self.text)):
            data_segement = self.text[i].split()  # 将字符串解析为list
            data_name = data_segement[0]  # 第一个元素存的是文件名 ’xxx.jpg'
            self.data_paths.append(os.path.join(self.img_path, data_name))
        # =======数据路径解析========
        # print(self.data_paths)
        # =======数据路径解析========

    def __len__(self):
        return (len(self.data_paths))

    def __getitem__(self, item):
        # ===拿图片数据，注意使用===
        img_data = Image.open(self.data_paths[item])
        # print('第几张图片=>',self.data_paths[item])
        resizer = transforms.Resize((416, 416))
        img_data = resizer(img_data)
        to_tensor = transforms.ToTensor()
        img_data = to_tensor(img_data)
        # ===拿图片数据，注意===

        # ===制作标签=============
        label_segent = self.text[item].split()
        label_segent = label_segent[1:]
        label_segent = list(map(int, label_segent))
        # 取类别和框
        cls = []
        boxes = []
        obj = len(label_segent) // 5
        for i in range(obj):
            # 将所有的类别取出来，并转换为onehot编码
            cls.append(np.array(torch.nn.functional.one_hot(torch.tensor(label_segent[5 * i]), num_classes=20)))
            # 取出所有的boxes
            boxes.append(label_segent[5 * i + 1:5 * i + 5])
        # print('boxes=>',boxes)
        # ===标签赋值=============
        labels = {}
        # 尺度
        for feature_size, anchors in ANCHOR_BOXES.items():
            # 给labels添加key,并初始化值,{13:[13,13,3,25]},13：feature_size,3:3种形状，25:5+20个类
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + len(LABEL)))
        # print(labels)

        # 取出box后分别与9个框比较IOU，将值赋给最大的那个anchor
        for i, box in enumerate(boxes):
            # 取出该box下，anchors的全局最大值
            feature_key = 0
            anchor_order = 0
            cls_order = i
            max_iou = 0
            # print('对图片中的目标 {} 筛选合适的框'.format(i))
            # 对9个anchor进行遍历
            for feature_size, anchors in ANCHOR_BOXES.items():
                for _three, anchor in enumerate(anchors):
                    anchor_area = ANCHOR_BOXES_AREA[feature_size][_three]
                    iou = min(box[2] * box[3], anchor_area) / max(box[2] * box[3], anchor_area)

                    if iou > max_iou:
                        max_iou = iou
                        anchor_order = _three
                        feature_key = feature_size
            #             print('{},max—iou=>{}'.format(_three,max_iou))
            #             print('{},feature_key=>{}'.format(feature_size,feature_key ))
            # print('筛选结束','==='*10)
            #
            # print('best_iou_feature_size=>',feature_key)
            # print('best_box_anchor=>',ANCHOR_BOXES[feature_key][anchor_order])
            # print('anchor_order=>',anchor_order)
            # print('cls=>',cls[cls_order])
            # print('best_iou=>',max_iou)

            # 将iou最大的box放入anchor对应的标签
            cx, cy, w, h = box
            offset_cx, index_x = math.modf(cx / (IMAGES_SIZE / feature_key))
            offset_cy, index_y = math.modf(cy / (IMAGES_SIZE / feature_key))
            # print('idx_x=>',index_x,'off_x=>',offset_cx)
            # print('idx_y=>', index_y, 'off_y=>', offset_cy)
            p_w = w / ANCHOR_BOXES[feature_key][anchor_order][0]
            p_h = h / ANCHOR_BOXES[feature_key][anchor_order][1]
            # print('p_w=>',p_w)
            # print('p_h=>',p_h)
            # exit()
            #     [iou,cy,cy,w,h,cls]->[freature_size,freature_size,3,5+cls_num]
            #     labels[feature_key][int(index_x), int(index_y), anchor_order] = [max_iou, offset_cx, offset_cy, p_w,p_h, *cls[cls_order]]

            labels[feature_key][int(index_x), int(index_y), anchor_order] = [max_iou, offset_cx, offset_cy, np.log(p_h),
                                                                             np.log(p_w), *cls[cls_order]]
        # print('某个label=>',labels[13][9,7,1])

        return img_data, labels[13], labels[26], labels[52]
# import numpy as np
# data=DetectData()
# it=[i for i in range(10)]
# im=np.random.choice(it)
# a=data[im]
