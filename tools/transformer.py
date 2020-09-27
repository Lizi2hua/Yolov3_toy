from tools.cfg import *
from PIL import Image
import  os
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
import numpy as np
import json
import time

#该类是将图片转换为416*416大小然后存为array，npy文件，然后对原来的坐标进行调整

class transformer():
    def __init__(self):
        self.img_dir=SQUARE_IMAGE_PATH
        self.label_path=os.path.join(TXT_PATH,'label.txt')
        self.label_path_416 = os.path.join(TXT_PATH_416, 'label.txt')
        self.array_path=IMAGES_TO_ARRAY_PATH
        print("原来的图片路径:{} 标签路径:{}".format(self.img_dir,self.label_path))

    def __call__(self, *args, **kwargs):
        #读取标注文件
        print('打开正方形图片标注txt：{}'.format(self.label_path))
        with open(self.label_path,'r') as f:
            text=f.readlines()#[str,str,str]
        #由于python读取的顺序和存储的顺序不同,必须使用存的顺序
        files=[]
        for i in range(len(text)):
            target_label = text[i]
            target_label = target_label.split()  # [str,str,...]
            # 图片名称
            img_name = target_label[0]
            files.append(img_name)
        print('创建{}*{}图片和标注txt文件路径:{}'.format(IMAGE_SIZE,IMAGE_SIZE,self.label_path_416))
        with open(self.label_path_416,'w') as f:
            for i,img in tqdm(enumerate(files)):
                # print('存{}图片为array'.format(img))
                img_data=Image.open(os.path.join(self.img_dir,img))
                img_data_copy=img_data
                # ==========图像变为416*416==============
                resizer = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
                img_data = resizer(img_data)
                to_tensor = transforms.ToTensor()
                img_data = to_tensor(img_data)
                # print(img_data)
                # exit()
                #将原本的float32转换为float16
                # img_data = to_tensor(img_data).to(dtype=torch.float16)
                #float32
                img_data = to_tensor(img_data) #30G左右空间
                # print(img_data.dtype)
                # exit()
                img_data = np.array(img_data)
                np.save(os.path.join(self.array_path,"{}.npy".format(i)),img_data)
                time.sleep(0.01)

                # ==========图像变为416*416==============

                #=========数据变为416*416==============
                #获得图片的边长
                img_data_copy=np.array(img_data_copy)
                slide,_,_=img_data_copy.shape
                '''
                坐标变换思路：用416除以原图边长得到缩放系数，原坐标乘以缩放系数就是在416图片上的坐标
                '''
                factor=416/slide
                # print('factor',factor)
                #标注信息
                target_label=text[i]
                target_label = target_label.split()  # [str,str,...]
                #图片名称
                img_name = target_label[0]  # str
                # print("人工矫正：转化{}======={}标注坐标中".format(img,img_name))
                # 获得图片的坐标信息
                object_and_cord = target_label[1:]
                # 将list里面的字符串变为float类型
                object_and_cord = list(map(float, object_and_cord))
                # print(object_and_cord)
                #总共有几个目标
                object_num=len(object_and_cord)//5
                # print(object_num)
                #将所有的位置坐标组装为一个[N]的list
                line=img_name+' '
                for j in range(object_num):
                    start=5*j+1
                    end=5*j+5
                    obj=5*j
                    segment=object_and_cord[start:end]
                    # print(segment,'1')
                    segment=np.array(segment)
                    # print(segment,'2')
                    #乘缩放因子
                    segment=segment*factor
                    # print(segment,'3')
                    #四舍五入取整
                    segment=segment.round()
                    # print(segment,'4')
                    #变为int
                    segment=list(map(int,segment))
                    # print(segment,'5')

                    #组装为一条str
                    line=line+str(int(object_and_cord[obj]))+' '
                    line=line+str(segment[0])+' '
                    line=line+str(segment[1])+' '
                    line=line+str(segment[2])+' '
                    line=line+str(segment[3])+' '
                #一条结束
                line=line+'\n'
                # print(line)
                f.write(line)
                # =========数据变为416*416==============

# trans=transformer()
# trans()

