import torch
import torch.nn as nn
import time
import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
path=r"E:\CASIAWebFaceData"
BATCH_SIZE=30
EPOCH=1

class FaceDatav2(Dataset):
    def __init__(self,path):
        start_time=time.time()
        super(FaceDatav2, self).__init__()
        self.data_path=path
        dir = os.listdir(self.data_path)
        img_dir = []
        self.img_data=[]
        # print(dir)
        for i in dir:
            img_dir.append(os.path.join(self.data_path, str(i)))
        # print(img_dir)
        # 将数据读入内存,以list[n,h,w,c]，然后get_item以索引形式获取数据
        self.img_paths = []
        for i in img_dir:
            img_path = os.listdir(str(i))
            for j in img_path:
                self.img_paths.append(os.path.join(str(i), str(j)))
                img=Image.open(os.path.join(str(i), str(j)))
                img = np.array(img)
                self.img_data.append(img)
        self.img_data=np.stack(self.img_data)
        end_time=time.time()
        print("数据集初始化完成!花了\033[1;31m%.3fs\033[0m"%(end_time-start_time))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img_path=self.img_paths[item]
        label = img_path.split("\\")[2]
        label=torch.tensor(int(label))
        img=self.img_data[item]
        resizer=transforms.Resize((160,160))
        img=resizer(img)
        to_tensor=transforms.ToTensor()
        img_data=to_tensor(img)

        return img_data,label
dataset=FaceDatav2(path)
data=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,drop_last=True)
start_time=time.time()
for epoch in tqdm(range(EPOCH)):
    for i,(data_,label_) in enumerate(data):
        data=data_.cuda()
        label=label_.cuda()
end_time=time.time()
print('%3.f'%(end_time-start_time))
