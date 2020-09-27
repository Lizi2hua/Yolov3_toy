import torch
import torch.nn as nn
from collections import OrderedDict

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class conv1x1(nn.Module):
    '''1x1卷积'''

    def __init__(self, inchannel, outchannel, group, relu=True, bias=False):
        super(conv1x1, self).__init__()
        self.relu=relu
        self.group=group
        if self.relu:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels=inchannel, out_channels=outchannel,
                          kernel_size=1, stride=1, bias=bias, groups=self.group),
                nn.BatchNorm2d(outchannel),
                nn.ReLU()
            )
        else:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels=inchannel, out_channels=outchannel,
                          kernel_size=1, stride=1, bias=bias, groups=self.group),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
        if self.relu:
            out=self.conv1x1(x)
            return channel_shuffle(out,self.group)
        return self.conv1x1(x)


class conv3x3(nn.Module):
    '''3x3卷积'''

    def __init__(self, channel, stride, bias=False):
        super(conv3x3, self).__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3
                      , stride=stride, padding=1, groups=channel, bias=bias),
            nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        return self.conv3x3(x)


class ShuffleUnit(nn.Module):
    '''2种ShuffleUnit'''

    def __init__(self,in_channel,out_channel,group,commbine):
        '''commbine='add' or 'cat' '''
        super(ShuffleUnit, self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.bottleneck_channel=out_channel//4
        self.group=group
        self.commbine=commbine
        if self.commbine=='add':
            self.shuffle_unit=self._make_shuffle_unit(stride=1)
        elif self.commbine=='cat':
            self.shuffle_unit =self._make_shuffle_unit(stride=2)
        else:
            raise ValueError("the commbie value is 'add' or 'cat'  ,not '{}' =.=b !".format(self.commbine) )

    def forward(self,x):
        residual=x
        if self.commbine=='add':
            return residual+self.shuffle_unit(x)
        if self.commbine=='cat':
            residual=nn.functional.avg_pool2d(residual,kernel_size=3,stride=2,padding=1)
            out=self.shuffle_unit(x)
            out=torch.cat((residual,out),dim=1)#[N,C,H,W],C cat
            return out

    def _make_shuffle_unit(self,stride):
        shuffle_unit=nn.Sequential(
            conv1x1(self.in_channel,self.bottleneck_channel,self.group),
            conv3x3(self.bottleneck_channel,stride=stride),
            conv1x1(self.bottleneck_channel,self.out_channel,self.group,relu=False)
        )
        return shuffle_unit

class ShuffleNet(nn.Module):
    '''
    此网络是用做YOLOv3的主干网络，为了使得输入416*416的图片，输出52*52，26*26，13*13这3个维度的特征;

    '''
    def __init__(self):
        super(ShuffleNet, self).__init__()
        self.stage_repeat=[-1,3,7,7,3]#stage1,stage2,stage3,stage4,-1自动报错
        #TODO:RuntimeError: Given groups=3, weight of size [60, 80, 1, 1], expected input[1, 264, 104, 104] to have 240 channels, but got 264 channels instead fix
        """判断是cat出了问题，比如24+120=144，但是后面add时候只要120，还有每个unit后面没加激活"""
        self.output_channles=[24,240,408,960,1920]
        self.groups=3
        self.conv1=nn.Conv2d(3,self.output_channles[0],kernel_size=3,stride=2,padding=1,bias=False)
        self.stage1=self._make_stage(1)
        self.stage2=self._make_stage(2)#=>output 52*52 feature map
        self.stage3=self._make_stage(3)#=>output 26*26 feature map
        self.stage4=self._make_stage(4)#=>output 13*13 feature map

    def forward(self,x):
        out=self.conv1(x)
        out=self.stage1(out)
        out52=self.stage2(out)
        out26=self.stage3(out52)
        out13=self.stage4(out26)

        return out52,out26,out13

    def _make_stage(self,stage):
        module=OrderedDict()
        repeat_shuffle=self.stage_repeat[stage]
        stage_name = 'shuffule_unit[{}]'.format(stage)
        if stage==0 or stage>4:
            raise ValueError('stage name shloud be one of [1,2,3,4],but got {}'.format(repeat_shuffle))
        head_module=ShuffleUnit(
            in_channel=self.output_channles[stage-1],
            out_channel=self.output_channles[stage],
            group=self.groups,
            commbine='cat'
        )
        head_name=stage_name+'[downsample]'
        module[head_name]=head_module

        for i in range(repeat_shuffle):
            repeat_name=stage_name+'[shuffle_{}]'.format(i)
            repeat_module=ShuffleUnit(
                in_channel=self.output_channles[stage],
                out_channel=self.output_channles[stage],
                group=self.groups,
                commbine='add'
            )
            module[repeat_name]=repeat_module
        return nn.Sequential(module)


model=ShuffleNet()
data=torch.randn(1,3,416,416)
print(model)
_52,_26,_13=model(data)
print(_52.shape)