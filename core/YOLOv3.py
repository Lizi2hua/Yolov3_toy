from core.DarkNet53 import DarkNet53
from core.DetectNet import DetectNet
import torch.nn as nn
import torch
from thop import profile

BACKBONE=DarkNet53()
class YOLOv3(nn.Module):
    def __init__(self,num_cls,backbone):
        super(YOLOv3, self).__init__()
        self.num_cls=num_cls
        self.backbone=backbone
        self.head=DetectNet(self.num_cls)

    def forward(self,x):
        _13,_26,_52=self.backbone(x)
        dect_13,dect_26,dect_52=self.head(_13,_26,_52)
        return dect_13,dect_26,dect_52

# model=YOLOv3(20,backbone=BACKBONE)
# data=torch.randn(1,3,416,416)
# _13,_26,_52=model(data)
# print(_13.shape)
# print(_26.shape)
# print(_52.shape)
# flops, params = profile(model, inputs = (data))
# print('flop=>',flops)
# print('param=>',params)
