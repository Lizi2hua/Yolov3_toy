from PIL import Image
import numpy as np
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
# path1=r'E:\VOC2017_gened\images\10483.jpg'
# path2=r'E:\VOC2017_gened\images\100.jpg'
# data=[]
# img=np.array(Image.open(path1))
# img=np.stack(img)
# print(img.shape)
# data.append(list(img))
# img=np.array(Image.open(path2))
# img=np.stack(img)
# print(img.shape)
# data.append(list(img))
# # data=np.stack(data)
# print(data)
# a=torch.randn(3,480,480)
# b=torch.randn(3,500,500)
# # c=torch.cat((a,b))
# # print(c.shape)




#统计图片大小
path=r"E:\VOC2017_gened\images"
dir = os.listdir(path)
W=[]
H=[]
for i in dir:
    img=Image.open(os.path.join(path,i))
    w,h=img.size
    W.append(w)
    H.append(H)
    print(i)
print(W)
plt.hist(W,bins=20,color='steelblue',edgecolor='k')
# plt.hist(H,bins=20,color='steelblue',edgecolor='r')
plt.tick_params(top='off', right='off')
plt.xlabel('宽高')
plt.ylabel('频数')
plt.title('类别统计')
plt.show()

