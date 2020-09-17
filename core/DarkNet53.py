import torch
import torch.nn as nn


class Conv_BN_Leaky(nn.Module):
    '''
    CONV_BN_RELU
    '''

    def __init__(self, in_channels, out_channels, ksize, stride, bias=False):
        super(Conv_BN_Leaky, self).__init__()
        pad = (ksize - 1) // 2
        self.convset = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, ksize, stride, pad, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.convset(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels,n):
        '''

        :param in_channels: 输入通道=输出通道
        :param n: 有几个残差块
        '''
        super(ResBlock, self).__init__()
        self.block=nn.ModuleList()
        for i in range(n):
            resblock_one=nn.ModuleList()
            resblock_one.append(Conv_BN_Leaky(in_channels, in_channels // 2, 1, 1))
            resblock_one.append(Conv_BN_Leaky(in_channels // 2, in_channels, 3, 1))
            self.block.append(resblock_one)

    def forward(self, x):
        for modules in self.block:
            h=x#x是残差
            # print('残差=》',x)
            for res in modules:
                h=res(h)
            x=x+h
            # print("一个残差块的输出=》",x)
        return x


class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleLayer, self).__init__()
        self.downsample = nn.Sequential(
            Conv_BN_Leaky(in_channels, out_channels, 3, 2)
        )

    def forward(self, x):
        return self.downsample(x)


class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.trunk_52=nn.Sequential(
            Conv_BN_Leaky(3,32,3,1),
            DownSampleLayer(32,64),
            ResBlock(64,1),
            DownSampleLayer(64, 128),
            ResBlock(128, 2),
            DownSampleLayer(128, 256),
            ResBlock(256, 8)
        )
        self.trunk_26=nn.Sequential(
            DownSampleLayer(256,512),
            ResBlock(512,8)
        )
        self.trunk_13=nn.Sequential(
            DownSampleLayer(512,1024),
            ResBlock(1024,4)
        )

    def forward(self,x):
        h_52=self.trunk_52(x)
        h_26=self.trunk_26(h_52)
        h_13=self.trunk_13(h_26)
        return h_13,h_26,h_52





# model = DarkNet53()
# # print(model)
# data = torch.randn(1, 3, 416, 416)
# print(data.shape)
# h_52,h_26,h_13= model(data)
# print(h_13.shape)
# print(h_26.shape)
# print(h_52.shape)

