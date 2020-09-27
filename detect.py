import torch
import torch.nn as nn
from cfg import *
from core.YOLOv3 import YOLOv3
from core.DarkNet53 import DarkNet53
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import time
from core.cfg import *
from  core.nms import NMS
from glob2 import glob
#===========超参数==============
TRESH=0.5
IOU_THRESH=0.05
class Detector():
    def __init__(self):
        BACKBONE=DarkNet53()
        net=YOLOv3(20,BACKBONE)
        self.net=net.cuda()
        cwd=os.path.join(os.getcwd(),SAVE_PATH)
        pt_files=os.listdir(SAVE_PATH)
        pt_files.sort(key=lambda x:os.stat((os.path.join(cwd,x))).st_ctime)
        self.ToTensor=transforms.ToTensor()
        self.Resizer=transforms.Resize((416,416))

        if pt_files:
            target_pt=os.path.join(cwd,pt_files[-1])
            print('加载{}预训练模型文件'.format(target_pt))
            self.net.load_state_dict(torch.load(target_pt))
        else:
            print('未找到预训练模型文件')
        self.net.eval()
    def ToSquare(self,img_data):
        '''
        :param image_data: PIL数据对象
        :return: PIL数据对象
        '''
        # 得到图片的宽，高
        w = img_data.width
        h = img_data.height
        # 将图片转为array
        # img_data=np.array(img_data)

        # 以最大值为基准
        slide = max(w, h)
        # 用一个全零array来作为填充对象
        square_img = np.zeros((slide, slide, 3))
        square_img = np.uint8(square_img)

        square_img = Image.fromarray(square_img)

        # 获取需要填充图片的中心，即原图与需要填充图片同心
        center_pad = (int(slide / 2), int(slide / 2))
        # 得到原图在需要填充图片上的坐标
        xmin, ymin = center_pad[0] - w // 2, center_pad[1] - h // 2
        xmax, ymax = xmin + w, ymin + h
        square_img.paste(img_data, (xmin, ymin, xmax, ymax))
        return square_img

    def __Filter(self,output,thresh):
        '''
        第一次过滤结果，得到网络大于置信度阈值的输出
        :param output: [N,C,H,W],C是我们需要的结果
        :param thresh: anchor置信度阈值
        :return:
        '''
        output=output.permute(0,2,3,1)
        output=output.reshape(output.size(0),output.size(1),output.size(2),3,-1)

        mask=output[...,0]>thresh#布尔值
        idxs=torch.nonzero(mask,as_tuple=False)#索引值
        vecs=output[mask]
        return idxs,vecs
    def __parse(self,idxs,vecs,t,anchors):
        '''
        将索引和向量解析为原图上的坐标
        :param idxs: 索引，用来计算中心点
        :param vecs: 向量，用来计算中心点和框的大小
        :param t: 放大倍数，416/feature_size
        :param anchors: 被调整的anchor
        :return: [[cx,cy,w,h,cls,pic_num]],pic_nums=N,表示第几张图片
        '''
        anchors=torch.Tensor(anchors).cuda()
        n=idxs[:,0]
        a=idxs[:,3]#anchor的索引,[0:2]
        iou=vecs[:,0]#将会用于nms

        cx=(idxs[:,2].float()+vecs[:,1])*t#图片的x和二维数组的x不同
        cy=(idxs[:,1].float()+vecs[:,2])*t
        w=anchors[a,0]*torch.exp(vecs[:,4])
        h=anchors[a,1]*torch.exp(vecs[:,3])
        # cx = (idxs[:, 1].float() + vecs[:, 1]) * t  # 图片的x和二维数组的x不同
        # cy = (idxs[:, 2].float() + vecs[:, 2]) * t
        # w = anchors[a, 0] * torch.exp(vecs[:, 4])
        # h = anchors[a, 1] * torch.exp(vecs[:, 3])

        top_x=cx-w/2
        top_y=cy-h/2
        botton_x=top_x+w
        botton_y=top_y+h

        cls=torch.argmax(vecs[:,5:],dim=1)

        return torch.stack((iou,top_x,top_y,botton_x,botton_y,cls,n.float()),dim=1)



    def __call__(self, img_data):
        '''
        :param img_data: 是原始的读取之后的图片数据，必须是PIL对象
        :return:
        '''
        start_time=time.time()
        #传入PIL对象，得到3个结果
        img_data=self.ToSquare(img_data)
        square_img=self.Resizer(img_data)
        square_img=self.ToTensor(square_img)
        square_img = torch.unsqueeze(square_img, dim=0)#[C,H,W]->[N,C,H,W]
        square_img=square_img.cuda()
        dect_13,dect_26,dect_52=self.net(square_img)

        #经过解析得到原图上面的bndboxes
        boxes=torch.tensor([]).cuda()
        idx_13,vec_13=self.__Filter(dect_13,TRESH)
        if idx_13.shape[0]!=0:
            boxes_13 = self.__parse(idx_13, vec_13, 416 / 13, ANCHOR_BOXES[13])
            boxes = torch.cat((boxes, boxes_13), dim=0)

        idx_26, vec_26 = self.__Filter(dect_26, TRESH)
        if idx_26.shape[0]!=0:
            boxes_26=self.__parse(idx_26,vec_26,416/26,ANCHOR_BOXES[26])
            boxes = torch.cat((boxes, boxes_26), dim=0)

        idx_52,vec_52=self.__Filter(dect_52,TRESH)
        if idx_52.shape[0]!=0:
            boxes_52=self.__parse(idx_52,vec_52,416/26,ANCHOR_BOXES[52])
            boxes = torch.cat((boxes, boxes_52), dim=0)
        end_time=time.time()

        if boxes.shape[0]!=0:
            cost=(end_time-start_time)
            return NMS(boxes,IOU_THRESH),cost
        if boxes.shape[0]==0:
            cost = (end_time - start_time)
            return torch.tensor([]),cost



# dect=Detector()
# files=r'E:\overfit_data\square\images'
# dir=glob(os.path.join(files,'*.jpg'))
# for i in range(100):
#     img_file=dir[i]
#     # print(img_file)
#     img_data=Image.open(img_file)
#     boxes,cost=dect(img_data)
#     print(boxes)
#     print(cost)


