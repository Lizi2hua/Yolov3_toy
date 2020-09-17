import torch
import torch.nn as nn
import time
import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
# path=r"E:\VOC2017_gened\images"
path=r'C:\Users\liewei\Desktop\face_data\all'
BATCH_SIZE=30
EPOCH=200
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
            self.img_paths.append(os.path.join(path, i))
        end_time=time.time()
        # print(self.img_paths)
        # exit()
        print("数据集初始化完成!花了\033[1;31m%.3fs\033[0m"%(end_time-start_time))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img_path=self.img_paths[item]
        # print(img_path)
        label=img_path.split("\\")[6]
        # print(label)
        # label = img_path.split("\\")[3]
        # print(label)
        label=label.split('.')[0]
        # print(label)
        label=torch.tensor(int(label))

        # exit()
        img=Image.open(img_path)
        resizer=transforms.Resize((160,160))
        img=resizer(img)
        to_tensor=transforms.ToTensor()
        img_data=to_tensor(img)

        return img_data,label

#测试读取数据的时间
dataset=FaceData(path)
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

