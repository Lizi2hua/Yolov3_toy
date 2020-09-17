import torch
import torch.nn as nn
import time
import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
# path=r"E:\VOC2017_gened\images"
path=r'F:\lab\数据压缩测试'
BATCH_SIZE=30
EPOCH=10

class FaceDatav2(Dataset):
    def __init__(self):
        start_time=time.time()
        super(FaceDatav2, self).__init__()
        # print(dir)
        dir=os.listdir(path)
        self.img_data_buffer=[]
        data_dir=[]
        for i in dir:
            data_dir.append(os.path.join(path,str(i)))
        print('加载数据到内存。。。')
        cont=0
        for i in data_dir:
            cont+=1
            print('加载{}进内存, {}/{}'.format(i, cont, len(data_dir)))
            data=np.load(i)
            self.img_data_buffer.append(data)
        print('加载成功。。。')
        self.img_data_buffer=torch.tensor(self.img_data_buffer)
        print(self.img_data_buffer.shape)



dataset=FaceDatav2()
data=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,drop_last=True)
start_time=time.time()
for epoch in tqdm(range(EPOCH)):
    for i,(data_,label_) in enumerate(data):
        data=data_.cuda()
        label=label_.cuda()
end_time=time.time()
print('%3.f'%(end_time-start_time))