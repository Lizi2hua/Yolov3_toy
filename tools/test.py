from tools.cfg import LABEL
import torch.nn as nn
data=['aeroplane','chair','person','sheep']
for key in data:
    print(LABEL[key])
path='G:\\datasets\\VOC2012\\VOCdevkit\\VOC2012\\JPEGImages\\2012_004276.jpg'
a=path.split('\\')
img_name=a[-1]
print(a)
print(img_name)
print('=================')
import numpy as np
import torch
data=np.load(r'E:\VOC2017_416\array\0.npy')
print(data.shape)
data=torch.tensor(data)
print(data)
print(data.dtype)
a=0.6
print('Accuarcy: {:.2f}%'.format(a*100))
# b=[1,2,3,4,5,5,6,7,8,9]
# c=list(map(str,b))
# print(c)

rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
print(input)
print(h0)
print(output)
print(hn)