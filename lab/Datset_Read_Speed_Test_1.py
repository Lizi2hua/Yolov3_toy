import torch
import torch.nn as nn
import time
import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from tqdm import tqdm
path=r"E:\CASIAWebFaceData"
BATCH_SIZE=30
EPOCH=1
class FaceData(Dataset):
    def __init__(self,path):
        start_time=time.time()
        super(FaceData, self).__init__()
        self.data_path=path
        dir = os.listdir(self.data_path)
        img_dir = []
        # print(dir)
        for i in dir:
            img_dir.append(os.path.join(self.data_path, str(i)))
        # print(img_dir)
        self.img_paths = []
        for i in img_dir:
            img_path = os.listdir(str(i))
            for j in img_path:
                self.img_paths.append(os.path.join(str(i), str(j)))
        end_time=time.time()
        print("数据集初始化完成!花了\033[1;31m%.3fs\033[0m"%(end_time-start_time))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img_path=self.img_paths[item]
        # print(img_path)
        # label=img_path.split("\\")[5]
        label = img_path.split("\\")[2]
        # print(label)
        label=torch.tensor(int(label))
        img=Image.open(img_path)
        resizer=transforms.Resize((160,160))
        img=resizer(img)
        to_tensor=transforms.ToTensor()
        img_data=to_tensor(img)

        return img_data,label
dataset=FaceData(path)
data=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,drop_last=True)
start_time=time.time()
for epoch in tqdm(range(EPOCH)):
    for i,(data,label) in enumerate(data):
        data=data.cuda()
        label=label.cuda()
end_time=time.time()
print('%3.f'%(end_time-start_time))

