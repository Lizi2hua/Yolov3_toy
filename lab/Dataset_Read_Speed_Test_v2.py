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
path=r'C:\Users\liewei\Desktop\face_data\all'
BATCH_SIZE=30
EPOCH=10

class FaceDatav2(Dataset):
    def __init__(self,path):
        start_time=time.time()
        super(FaceDatav2, self).__init__()
        self.data_path=path
        dir = os.listdir(self.data_path)
        img_dir = []
        self.img_data=[]
        self.img_paths=[]
        # print(dir)
        for i in dir:
            img_dir.append(os.path.join(self.data_path, str(i)))
        # print(img_dir)
        # 将数据读入内存,以list[n,h,w,c]，然后get_item以索引形式获取数据
        print('将数据加载入内存...')
        self.img_tensors=torch.tensor([])
        self.buffer=[]
        cont=0
        for i in img_dir:
            cont+=1
            print('加载{}进内存, {}/{}'.format(i,cont,len(img_dir)))
            self.img_paths.append(os.path.join(str(i), i))
            img=Image.open(os.path.join(str(i), i))
            resizer = transforms.Resize((500, 500))
            img = resizer(img)
            to_tensor = transforms.ToTensor()
            self.img_tensor = to_tensor(img)
            self.img_tensors=torch.cat((self.img_tensors,self.img_tensor))
            # self.buffer.append(self.img_tensor)
        # self.img_tensors=torch.tensor(self.buffer)
        self.img_tensors=self.img_tensors.reshape(len(img_dir),3,500,500)
        print(self.img_tensors.shape)
        end_time=time.time()
        data_array=np.array(self.img_data)
        print('数据的形状：{}'.format(data_array.shape))
        print("数据集初始化完成!花了\033[1;31m%.3fs\033[0m"%(end_time-start_time))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img_path=self.img_paths[item]
        label = img_path.split("\\")[3]
        label=label.split('.')[0]
        label=torch.tensor(int(label))
        img_data=self.img_tensors[item]


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
