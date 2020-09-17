import torch
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from core.DarkNet53 import DarkNet53
from core.YOLOv3 import YOLOv3
from core.dataset_v2 import DetectData
from cfg import *
import os


BACKBONE=DarkNet53()
LOG_DIR='logs'
SAVE_PATH='saved_models'
class Train():
    def __init__(self):
        self.dataset=DetectData(LABEL_416,NPY_PATH)
        self.net=YOLOv3(20,BACKBONE)
        self.summary=SummaryWriter(LOG_DIR)
        #判断save path是否存在，如果存在，判断是否有存储的pt文件并加载最后一轮pt文件
        if  os.path.exists(SAVE_PATH):
            files=os.listdir(SAVE_PATH)
            print(files)
            if files :
                print('加载=>',files)
                #根据创建时间排序
                # files=
                # load_file=
            else:
                print('空')
        else:
            os.mkdir(SAVE_PATH)


trainer=Train()
trainer