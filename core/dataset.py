import torch
import torch.nn as nn
from torch.utils.data import Dataset
from core.cfg import *
import os
import time
import math
import numpy as np
class DetectData(Dataset):
    def __init__(self):
        self.label_path=LABEL_416
        self.npy_path=NPY_PATH
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
        # img_data=torch.tensor(np.load(self.data_paths[item])).to(dtype=torch.float16)
        # ===拿图片数据，注意使用float16===
        #===制作标签=============
        img_data = torch.tensor(np.load(self.data_paths[item])).to(dtype=torch.float32)

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
        for feature_size,anchors in ANCHOR_BOXES.items():
            #给labels添加key,并初始化值,{13:[13,13,3,25]},13：feature_size,3:3种形状，25:5+20个类
            labels[feature_size]=np.zeros(shape=(feature_size,feature_size,3,5+len(LABEL)))
        #形状
            # 索引到anchor，当前anchor与box做iou
            for _three,anchor in enumerate(anchors):
                # print('anchor=>',anchor)
                #_three代表3种形状
                anchor_area=ANCHOR_BOXES_AREA[feature_size][_three]
                #由目标的坐标位置做索引得到目标值特征图上的位置，在该位置上面放一个当前的anchor的w,h
                #遍历boxes，box与当前anchor做iou，保留最大的值，然后根据当前的box的顺序值得到类别
                best_box=[]
                best_iou=0.
                cls_order=0
                for i,box in enumerate(boxes):
                    iou=min(box[2]*box[3],anchor_area)/max(box[2]*box[3],anchor_area)
                    # print('iou=>',iou)
                    if iou>best_iou:
                        best_iou=iou
                        best_box=box
                        cls_order=i
                # print('best_iou=>',best_iou)
                # print('best_box=>',best_box)
                # print('cls_oder=>',cls_order)
                # print('cls_index=>',cls[cls_order])
                #将iou最大的box放入anchor对应的标签
                cx,cy,w,h=best_box
                offset_cx,index_x=math.modf(cx/(IMAGES_SIZE/feature_size))
                offset_cy, index_y = math.modf(cy / (IMAGES_SIZE / feature_size))
                # print('idx_x=>',index_x,'off_x=>',offset_cx)
                # print('idx_y=>', index_y, 'off_y=>', offset_cy)
                p_w=w/anchor[0]
                p_h=h/anchor[1]
                # print('p_w=>',p_w)
                # print('p_h=>',p_h)
                #[iou,cy,cy,w,h,cls]->[freature_size,freature_size,3,5+cls_num]
                # labels[feature_size][int(index_x), int(index_y), _three] = [iou, offset_cx, offset_cy, p_w,
                #                                                             p_h, *cls[cls_order]]
                labels[feature_size][int(index_x),int(index_y),_three]=[best_iou,offset_cx,offset_cy,np.log(p_w),np.log(p_h),*cls[cls_order]]
                # print(labels[feature_size].shape)
                # print(labels[feature_size][0,0,0])
                # print(labels[feature_size][6,7,0])#[index_cx,index_cy,anchor]
                # print(labels[feature_size][0,0,0])
                # print('three boxes finished'+"=="*20)
            # ===标签赋值=============
        # ===制作标签=============

        return img_data,labels[13],labels[26],labels[52]
#
# data=DetectData()
# a=data[35]
# print(type(a))
