import torch
import torch.nn as nn
from torch.utils.data import Dataset
from core.cfg import *
import os
import time
import math
import numpy as np
class DetectData(Dataset):
    def __init__(self,label_416,npy_path):
        self.label_path=label_416
        self.npy_path=npy_path
        #==========读取txt文档======
        with open(self.label_path,'r') as f:
            self.text=f.readlines()
        # ==========读取txt文档======
        self.data_paths = []
        self.data_buffer=[]
        dir = os.listdir(self.npy_path)
        #=======数据路径解析========
        for i in range(len(self.text)):
            data_segement=self.text[i].split() #将字符串解析为list
            data_name=data_segement[0] #第一个元素存的是文件名 ’xxx.jpg‘
            data_name,_=data_name.split('.')
            data_name=data_name+'.npy'
            self.data_paths.append(os.path.join(self.npy_path,data_name))
        # =======数据路径解析========
        # =======数据内容解析========
        #将内容转换为list，元素为int
            data=data_segement[1:]
            data=list(map(int,data))
            self.data_buffer.append(data)
        # =======数据内容解析========
        # 初始化占用60M左右内存，估计的

    def __len__(self):
        return (len(self.data_buffer))
    def __getitem__(self, item):
        #===拿图片数据，注意使用float16===
        img_data=torch.tensor(np.load(self.data_paths[item])).to(dtype=torch.float16)
        # ===拿图片数据，注意使用float16===
        #===制作标签=============

        label_segent=self.text[item].split()
        label_segent=label_segent[1:]
        label_segent=list(map(int,label_segent))
        #取类别和框
        cls=[]
        boxes=[]
        obj=len(label_segent)//5
        for i in range(obj):
            #将所有的类别取出来，并转换为onehot编码
            cls.append(np.array(torch.nn.functional.one_hot(torch.tensor(label_segent[5*i]),num_classes=20)))
            #取出所有的boxes
            boxes.append(label_segent[5*i+1:5*i+5])
        # print('boxes=>',boxes)
             # ===标签赋值=============
        labels = {}
        #尺度

        for feature_size, anchors in ANCHOR_BOXES.items():
            # 给labels添加key,并初始化值,{13:[13,13,3,25]},13：feature_size,3:3种形状，25:5+20个类
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + len(LABEL)))

        #取出box后分别与9个框比较IOU，将值赋给最大的那个anchor
        for i,box in enumerate(boxes):
            # 取出该box下，anchors的全局最大值
            feature_key = 0
            anchor_order = 0
            cls_order = i
            max_iou = 0

            #对9个anchor进行遍历
            for feature_size, anchors in ANCHOR_BOXES.items():
                for _three,anchor in enumerate(anchors):
                    anchor_area=ANCHOR_BOXES_AREA[feature_size][_three]
                    iou = min(box[2] * box[3], anchor_area) / max(box[2] * box[3], anchor_area)

                    if iou > max_iou:
                        max_iou = iou
                        anchor_order = _three
                        feature_key = feature_size

            # print('best_iou_feature_size=>',feature_key)
            # print('best_box_anchor=>',ANCHOR_BOXES[feature_key][anchor_order])
            # print('anchor_order=>',anchor_order)
            # print('cls=>',cls[cls_order])
            # print('best_iou=>',max_iou)

            #将iou最大的box放入anchor对应的标签
            cx,cy,w,h=box
            offset_cx,index_x=math.modf(cx/(IMAGES_SIZE/feature_key))
            offset_cy, index_y = math.modf(cy / (IMAGES_SIZE / feature_key))
            # print('idx_x=>',index_x,'off_x=>',offset_cx)
            # print('idx_y=>', index_y, 'off_y=>', offset_cy)
            p_w=w/ANCHOR_BOXES[feature_key][anchor_order][0]
            p_h=h/ANCHOR_BOXES[feature_key][anchor_order][1]
            # print('p_w=>',p_w)
            # print('p_h=>',p_h)
        # exit()
        #     [iou,cy,cy,w,h,cls]->[freature_size,freature_size,3,5+cls_num]
        #     labels[feature_key][int(index_x), int(index_y), anchor_order] = [max_iou, offset_cx, offset_cy, p_w,p_h, *cls[cls_order]]

            labels[feature_key][int(index_x),int(index_y),anchor_order]=[max_iou,offset_cx,offset_cy,np.log(p_w),np.log(p_h),*cls[cls_order]]
        print('某个label=>',labels[13][9,7,1])

        return img_data,labels[13],labels[26],labels[52]

# data=DetectData()
# a=data[2]
# print(type(a))