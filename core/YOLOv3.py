from core.DarkNet53 import DarkNet53
from core.DetectNet import DetectNet
from core.ShuffleNet import ShuffleNet
import torch.nn as nn
import torch
from thop import profile

darknet=DarkNet53()
shufflenet=ShuffleNet()
class YOLOv3(nn.Module):
    def __init__(self,num_cls,use_mode):
        super(YOLOv3, self).__init__()
        self.num_cls=num_cls
        if use_mode=='darknet53':
            self.backbone=darknet
            print('backbone is darknet53')
            self.head=DetectNet(self.num_cls)
        elif use_mode=='shufflenet':
            self.backbone=shufflenet
            self.head=DetectNet(self.num_cls,feature_13_channel=960,feature_26_channel=480,feature_52_channel=240)
            print('backbone is shufflenet')
        else:
            raise ValueError("use_mode should be 'darknet53' or 'shufflenet',not:'{}'".format(use_mode))
    def forward(self,x):
        _13,_26,_52=self.backbone(x)
            # print(_13.shape)
            # print(_26.shape)
            # print(_52.shape)
            # print('done'+'='*20)
        dect_13,dect_26,dect_52=self.head(_13,_26,_52)
        return dect_13,dect_26,dect_52

# model=YOLOv3(20,use_mode='shufflenet')
# data=torch.randn(1,3,416,416)
# _13,_26,_52=model(data)
# print(_13.shape)
# print(_26.shape)
# print(_52.shape)
# flops, params = profile(model, inputs = (data))
# print('flop=>',flops)
# print('param=>',params)
