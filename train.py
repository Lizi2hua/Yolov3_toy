import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from core.DarkNet53 import DarkNet53
from core.YOLOv3 import YOLOv3
# from core.dataset_v3 import DetectData
from core.dataset_v3 import DetectData
import torch.optim as optim
from cfg import *
import os
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy

BATCH_SIZE=1
# BACKBONE=DarkNet53()

class Train():
    def __init__(self):
        # self.dataset=DetectData(LABEL_416,NPY_PATH)
        self.dataset = DetectData()
        # self.net=YOLOv3(20,BACKBONE).cuda()
        self.net = YOLOv3(20, use_mode='shufflenet').cuda()
        self.summary=SummaryWriter(LOG_DIR)
        self.opt=optim.Adam(self.net.parameters(),lr=1e-3)
        #判断save path是否存在，如果存在，判断是否有存储的pt文件并加载最后一轮pt文件
        self.cwd=os.path.join(os.getcwd(),SAVE_PATH)

        if  os.path.exists(SAVE_PATH):
            files=os.listdir(SAVE_PATH)
            #对文件以创建时间排序
            files.sort(key=lambda  x:os.stat(os.path.join(self.cwd,x)).st_ctime)
            if files :
                # 加载最后保存的训练文件
                print('加载=>',os.path.join(self.cwd,files[-1]))
                self.net.load_state_dict(torch.load(os.path.join(self.cwd,files[-1])))
            else:
                pass
        else:
            os.mkdir(SAVE_PATH)
        #==========================================



    # def loss_fn(self,output, target, alpha):
    #     # 损失函数定义
    #     conf_loss_fn = torch.nn.BCEWithLogitsLoss()  # 定义置信度损失函数
    #     center_loss_fn = torch.nn.BCEWithLogitsLoss()  # 定义中心点损失函数
    #     wh_loss_fn = torch.nn.MSELoss()  # 宽高损失
    #     cls_loss_fn = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失
    #     output = output.permute(0, 2, 3, 1)
    #     output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    #
    #     target = target.cuda()
    #
    #     # 负样本的时候只需要计算置信度损失
    #     mask_noobj = target[..., 0] <= 0.1
    #     output_noobj, target_noobj = output[mask_noobj], target[mask_noobj]
    #     loss_noobj = conf_loss_fn(output_noobj[:, 0], target_noobj[:, 0])
    #
    #     mask_obj = target[..., 0] > 0.1
    #     output_obj, target_obj = output[mask_obj], target[mask_obj]
    #     if output_obj.size(0) > 0:
    #         loss_obj_conf = conf_loss_fn(output_obj[:, 0], target_obj[:, 0])
    #         loss_obj_center = center_loss_fn(output_obj[:, 1:3], target_obj[:, 1:3])
    #         loss_obj_wh = wh_loss_fn(output_obj[:, 3:5], target_obj[:, 3:5])
    #         loss_obj_cls = cls_loss_fn(output_obj[:, 5:], target_obj[:, 5:].long())
    #         loss_obj = loss_obj_conf + loss_obj_center + loss_obj_wh + loss_obj_cls
    #
    #         return alpha * loss_obj + (1 - alpha) * loss_noobj
    #     else:
    #         return loss_noobj

    def QuatraLoss(self,output,target,alpha):
        '''
        4个损失，iou+(cx,cy)+(p_h,p_w)+num_cls，均使用MSELoss
         :param output: [N,C,H,W],C=(cls_nums+5)*3，网络的输出
        :param target: [N,H,W,3,C],C=cls_nums+5，标签
        :param alpha: 正样本放大系数
        :return:
            a=torch.randn(1,1,1,3,2)
            print(a)
            print(a[...,0])
        # '''

        # CELoss=nn.CrossEntropyLoss()
        # NLLLoss=nn.NLLLoss()
        output = output.permute(0, 2, 3, 1)  # [N,H,W,C]
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)  # [N,H,W,3,C//3]，C//3->#[cx,cy,p_h,p_w,20cls]
        # 找出正负样本的索引
        mask_obj = target[..., 0] > 0.1#使用[3,c//3]这个维度的所有3维度的值
        mask_noobj = target[..., 0] <= 0.1

        #索引出正例
        positive=output[mask_obj]
        positive_label=target[mask_obj]

        # 索引出负例
        negetive = output[mask_noobj]
        negetive_label = target[mask_noobj]
        # print(negetive_label.shape)
        # output
        n_iou = negetive[:, 0]
        # target
        n_iou_label = negetive_label[:, 0]
        negetive_loss = (torch.mean((n_iou_label - n_iou) ** 2))

        if positive.shape[0]>0:
            # print(positive.shape)
            #output
            iou=positive[:,0]
            c=positive[:,1:3]
            p_hw=positive[:,3:5]
            cls=positive[:,5:]
            #target
            iou_label=positive_label[:,0]
            c_label=positive_label[:,1:3]
            p_hw_label=positive_label[:,3:5]
            cls_label=positive_label[:,5:]

            #MSE LOSS
            positive_loss=(torch.mean((iou_label-iou)**2)
                           +torch.mean((c_label-c)**2)
                           +torch.mean((p_hw_label-p_hw)**2)
                           +torch.mean((cls_label-cls)**2)
                           )
            #TODO:系数问题，不加似乎拟合慢
            # return (1-alpha)*positive_loss+alpha*negetive_loss
            return  positive_loss + negetive_loss
        #为空的话只需计算iou损失
        else:
            return   negetive_loss

    def __call__(self):
        dataloader=DataLoader(self.dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True,num_workers=0)
        cont_eopch = 0
        while True :
            print('一个EPOCH开始')
            start_time=time.time()
            jiont_loss_mean=[]
            loss_13_mean=[]
            loss_26_mean=[]
            loss_52_mean=[]

            for i,(img,label_13,label_26,label_52) in enumerate(dataloader):
                img=img.cuda()
                out_13,out_26,out_52=self.net(img)
                label_13=label_13.cuda()
                label_26=label_26.cuda()
                label_52=label_52.cuda()

                loss_13=self.QuatraLoss(out_13,label_13,0.9)
                loss_26 = self.QuatraLoss(out_26, label_26,0.9)
                loss_52 = self.QuatraLoss(out_52, label_52, 0.9)

                loss_13_mean.append(loss_13.item())
                loss_26_mean.append(loss_26.item())
                loss_52_mean.append(loss_52.item())

                loss=loss_13+loss_26+loss_52
                jiont_loss_mean.append(loss.item())


                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


            cont_eopch+=1
            end_time=time.time()
            print('一个EPOCH花了{}'.format(end_time - start_time))

            mean_loss=torch.mean(torch.tensor(jiont_loss_mean))
            print(mean_loss)
            loss_13_mean=torch.mean(torch.tensor(loss_13_mean))
            loss_26_mean = torch.mean(torch.tensor(loss_26_mean))
            loss_52_mean = torch.mean(torch.tensor(loss_52_mean))

            self.summary.add_scalar('joint_loss',mean_loss,cont_eopch)
            self.summary.add_scalar('loss_13',loss_13_mean,cont_eopch)
            self.summary.add_scalar('loss_26', loss_26_mean,cont_eopch)
            self.summary.add_scalar('loss_52', loss_52_mean,cont_eopch)
            save_model_path=os.path.join(self.cwd,'shuffle_1pic_version.pt'.format(cont_eopch))
            # save_model_path = os.path.join(self.cwd, 'all_version_{}.pt'.format(cont_eopch))
            torch.save(self.net.state_dict(),save_model_path)

trainer=Train()
trainer()
