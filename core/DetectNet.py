import torch.nn as nn
import torch
from core.DarkNet53 import  DarkNet53
from core.DarkNet53 import Conv_BN_Leaky

class ConvSet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvSet, self).__init__()
        self.convs=nn.Sequential(
            Conv_BN_Leaky(in_channels,out_channels,1,1),
            Conv_BN_Leaky(out_channels,in_channels,3,1),

            Conv_BN_Leaky(in_channels,out_channels,1,1),
            Conv_BN_Leaky(out_channels,in_channels,3,1),

            Conv_BN_Leaky(in_channels,out_channels,1,1)
        )
    def forward(self,x):
        return self.convs(x)
class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()

    def forward(self,x):
        return nn.functional.interpolate(x,scale_factor=2,mode='nearest')

class DetectNet(nn.Module):
    def __init__(self,num_cls,feature_13_channel=1024,feature_26_channel=512,feature_52_channel=256):
        super(DetectNet, self).__init__()
        self.out_channel = 3 * (5 + num_cls)
        self.feature_13_channel=feature_13_channel
        self.feature_26_channel=feature_26_channel
        self.feature_52_channel=feature_52_channel
        #13
        self.convset_13=nn.Sequential(
            ConvSet(self.feature_13_channel,512)
        )
        self.dect_13=nn.Sequential(
            Conv_BN_Leaky(512,1024,3,1),
            nn.Conv2d(1024,self.out_channel,1,1)
        )
        # up_26
        self.up_26=nn.Sequential(
            Conv_BN_Leaky(512,256,1,1),
            UpSample()
        )
        #26
        self.convset_26=nn.Sequential(
            ConvSet((256+self.feature_26_channel),256)
        )
        self.dect_26=nn.Sequential(
            Conv_BN_Leaky(256,512,3,1),
            nn.Conv2d(512,self.out_channel,1,1)
        )
        # up_52
        self.up_52=nn.Sequential(
            Conv_BN_Leaky(256,128,1,1),
            UpSample()
        )

        # 52
        self.convset_52=nn.Sequential(
            ConvSet((128+self.feature_52_channel),128)
        )
        self.dect_52=nn.Sequential(
            Conv_BN_Leaky(128,256,3,1),
            nn.Conv2d(256,self.out_channel,1,1)
        )

    def forward(self,h_13,h_26,h_52):
        # 13
        convset13_out=self.convset_13(h_13)
        dect13_out=self.dect_13(convset13_out)

        # 26
        upto_26=self.up_26(convset13_out)
        cat_26=torch.cat((upto_26,h_26),dim=1) #通道层面cat到一起
        convset26_out=self.convset_26(cat_26)
        dect26_out=self.dect_26(convset26_out)

        # 52
        upto_52=self.up_52(convset26_out)
        cat_52=torch.cat((upto_52,h_52),dim=1)
        convset52_out=self.convset_52(cat_52)
        dect52_out=self.dect_52(convset52_out)

        return dect13_out,dect26_out,dect52_out


# dect=DetectNet(20,feature_13_channel=960,feature_26_channel=480,feature_52_channel=240)
# h_13=torch.randn(2,960,13,13)
# h_26=torch.randn(2,480,26,26)
# h_52=torch.randn(2,240,52,52)
# _13,_26,_52=dect(h_13,h_26,h_52)
# print(_13.shape)
# print(_26.shape)
# print(_52.shape)