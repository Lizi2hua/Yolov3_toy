import torch
import numpy as np
from PIL import Image
# a=torch.randn(1,13,13,3,20)
# mask_0=a[...,0]>0
# print(a[mask_0])
# a=torch.randn(1,18,2,2)
# b=torch.randn(1,2,2,3,6)
# # print(a)
# # print(a[...,0])
#
# output=torch.randn(1,75,13,13)
# label=torch.randn(1,13,13,3,25)
#
# def QuatraLoss( output, target):
#     '''
#     4个损失，iou+(cx,cy)+(p_h,p_w)+num_cls，均使用MSELoss
#      :param output: [N,C,H,W],C=(cls_nums+5)*3，网络的输出
#     :param target: [N,H,W,3,C],C=cls_nums+5，标签
#     :param alpha: 正样本放大系数
#     :return:
#         a=torch.randn(1,1,1,3,2)
#         print(a)
#         print(a[...,0])
#     '''
#     output = output.permute(0, 2, 3, 1)  # [N,H,W,C]
#     output = output.reshape(output.size(0), output.size(1), output.size(2), 3,
#                             -1)  # [N,H,W,3,C//3]，C//3->#[cx,cy,p_h,p_w,20cls]
#     # 找出正负样本的索引
#     mask_obj = target[..., 0] > 0  # 使用[3,c//3]这个维度的所有3维度的值
#
#     mask_noobj = target[..., 0] == 0
#
#     # 索引出正例
#     positive = output[mask_obj]
#     positive_label = target[mask_obj]
#     iou=positive[:,0]
#     cx = positive[:, 1]
#     cy=positive[:, 2]
#     p_w=positive[:, 3]
#     p_h=positive[:,4]
#     cls=positive[:,5:]
#     print(cx)
#     print(cls)
#     print(positive_label.shape)
#     print(positive.shape)
#     # iou
# a=[1,2,3,4,5,6]
# print(a[3:4])
# a=torch.tensor([[1.],[2],[3]])
# b=a[...,0]>2
# print('mask=>',b)
# c=torch.nonzero(b)
# print(a[b])
# print(c)
# # a=torch.tensor([1.,2,3])
# # #torch.tensor([1.,2,3])
# # a=torch.tensor([])
# # if a.shape==torch.Size([0]):
# #     print('null')
# b=torch.tensor([0.1,2,3,4,5,6,7])
# print(torch.argmax(b))
# c=[0,0,1,1,3,4]
# print(b[c])
# def ToSquare(image_data):
#     # 得到图片的宽，高
#     w = img_data.width
#     h = img_data.height
#     # 将图片转为array
#     # img_data=np.array(img_data)
#
#     # 以最大值为基准
#     slide = max(w, h)
#     # 用一个全零array来作为填充对象
#     square_img = np.zeros((slide, slide, 3))
#     square_img = np.uint8(square_img)
#
#     square_img = Image.fromarray(square_img)
#
#     # 获取需要填充图片的中心，即原图与需要填充图片同心
#     center_pad = (int(slide / 2), int(slide / 2))
#     # 得到原图在需要填充图片上的坐标
#     xmin, ymin = center_pad[0] - w // 2, center_pad[1] - h // 2
#     xmax, ymax = xmin + w, ymin + h
#     square_img.paste(img_data, (xmin, ymin, xmax, ymax))
#     return square_img
#
# img_data=np.ones((256,128,3),dtype=np.uint8)
# img_data=Image.fromarray(img_data)
# print(img_data.height)
# print(img_data.width)
# img_data=ToSquare(img_data)
# print(img_data.height)
# print(img_data.width)
# a=torch.tensor([[1,2,3,5],[5,2,6,2]])
# print(torch.argmax(a,dim=1))
# a=torch.randn([1,12,416,416])
# b=a.reshape([1,3,4,416,-1])
# b=torch.transpose(b,1,2).contiguous()
#
# print(b.shape)
# import  torchvision.models  as models
# model=models.shufflenet_v2_x1_0(pretrained=False)
#
# print(model)
t=416/13
#原来的坐数据：14 223 188 146 208
#数据集的数据
import math
cx=223
cy=188
w=146
h=208
offx,index_x=math.modf(cx/t)
offy,index_y=math.modf(cy/t)
#变为数据集的格式
vecs=torch.tensor([0.9,offx,offy,h,w,14])
#反算
cx=index_x+vecs[]

# cx=(idxs[:,1].float()+vecs[:,2])*t#图片的x和二维数组的x不同
# cy=(idxs[:,2].float()+vecs[:,1])*t
# w=anchors[a,0]*torch.exp(vecs[:,4])
# h=anchors[a,1]*torch.exp(vecs[:,3])
# # cx = (idxs[:, 1].float() + vecs[:, 1]) * t  # 图片的x和二维数组的x不同
# # cy = (idxs[:, 2].float() + vecs[:, 2]) * t
# # w = anchors[a, 0] * torch.exp(vecs[:, 4])
# # h = anchors[a, 1] * torch.exp(vecs[:, 3])
#
# top_x=cx-w/2
# top_y=cy-h/2
# botton_x=top_x+w
# botton_y=top_y+h
