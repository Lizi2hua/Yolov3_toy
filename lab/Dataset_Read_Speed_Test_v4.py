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
import matplotlib.pyplot as plt
path=r'C:\Users\liewei\Desktop\face_data\alldata'
BATCH_SIZE=30
EPOCH=200

class FaceDatav2(Dataset):
    def __init__(self):
        start_time=time.time()
        super(FaceDatav2, self).__init__()
        # print(dir)
        dir=os.listdir(path)
        self.img_data_buffer=[]
        self.data_dir=[]
        for i in dir:
            self.data_dir.append(os.path.join(path,str(i)))
    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, item):
        data_path=self.data_dir[item]
        data=torch.tensor(np.load(data_path))
        label=data_path.split('\\')[6]
        label=label.split('.')[0]
        label=torch.tensor(int(label))

        return data,label




dataset=FaceDatav2()
data=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,drop_last=True)
epoch_times=[]
start_time_load=time.time()
for epoch in tqdm(range(EPOCH)):
    start_time=time.time()
    for datax,label_ in data:
        # print(datax,label_)
        dataxx=datax.cuda()
        label=label_.cuda()
    end_time=time.time()
    epoch_time=end_time-start_time
    epoch_times.append(epoch_time)
end_time_load=time.time()
print(epoch_times)
x=range(EPOCH)
plt.plot(x,epoch_times)
print('%.3f'%(end_time_load-start_time_load))
plt.show()