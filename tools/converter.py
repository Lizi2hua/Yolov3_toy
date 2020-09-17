# 将图片转换为正方形
import os
import numpy as np
from PIL import Image
from tools.cfg import *
from tools.xml_parser import parser
from tqdm import tqdm
import time

class Converter():
    '''
    将数据进行调整并写入txt文件
    '''
    def __init__(self,parser):
        #传入初始化了的parser
        print('解析xml...')
        self.image_file_path, self.size, self.object_in_image, self.coord = parser()
        print('解析完成！')
        self.image_path = IMAGES_PATH
        self.save_path = SQUARE_IMAGE_PATH
        self.txt_path = TXT_PATH
        self.label=LABEL
    def convert_to_square(self, image, suffix):
        '''

        :param image: 某张图片的相对路径
        :param suffix: 图片的后缀名，即编号
        :return: 坐标的偏移量offset_x,offset_y
        将一张图片转换为一张正方形图片，非原图的部分用0填充。
        '''
        img_data = Image.open(image)
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
        # 偏移为xmin,ymin

        square_img.paste(img_data, (xmin, ymin, xmax, ymax))
        # square_img.show()
        square_img.save(os.path.join(self.save_path, '{}.jpg'.format(suffix)))

        offset_x = xmin
        offset_y = ymin
        return offset_x, offset_y
    # 坐标调整
    def CorrdChanger(self):
        '''
        用遍历的方法将原来数据中的坐标调整为图片变成正方形的坐标，该方法会调用
        #convert_to_square#方法，所以会存图片
        :param coords: 原数据中的所有坐标
        :param img_files:图片的绝对路径
        '''
        print("调整坐标...")
        cx_cy_w_h=[]
        for i, img in enumerate(self.image_file_path):
            cxx_cyy_ww_hh=[]
            # 将图片转化为正方形，并存到相对应的路径
            offset_x, offset_y = self.convert_to_square(img, i)
            # 取出该图片对应标签的坐标,coords[i]
            # 遍历目标，修改原坐标
            for j ,_,in enumerate(self.coord[i]):

                #[4]<-[xmin,ymin,xmax,ymax]
                self.coord[i][j][0] = self.coord[i][j][0] + offset_x
                self.coord[i][j][1] = self.coord[i][j][1] + +offset_y
                self.coord[i][j][2] = self.coord[i][j][2] + offset_x
                self.coord[i][j][3] = self.coord[i][j][3] + offset_y
                #计算标注框的中心，宽高
                w=self.coord[i][j][2]-self.coord[i][j][0]
                h=self.coord[i][j][3]-self.coord[i][j][1]
                cx=self.coord[i][j][0]+w//2
                cy=self.coord[i][j][1]+h//2
                cxx_cyy_ww_hh.append([cx,cy,w,h])
            cx_cy_w_h.append(cxx_cyy_ww_hh)

        return cx_cy_w_h
    def cls_encoder(self):
        '''
        将类别名编码为数字
        :param objects: [[['person']]]
        ->objects[[[14]]]
        '''
        # 遍历
        print('类别编码...')
        for N in range(len(self.object_in_image)):
            for n, key in enumerate(self.object_in_image[N]):
                self.object_in_image[N][n] = self.label[key]


    def __call__(self, *args, **kwargs):
        bndboxes=self.CorrdChanger()
        self.cls_encoder()
        # print('objects->',self.object_in_image)
        # print('bndbox->',bndboxes)
        '''
        万事具备，只等着写入txt了，各个数据的形状：
        ->bndbox=[N,n,4]
        ->object_in_image=[N,n]
        ->文件名称[N]
        '''

        txt_file = os.path.join(self.txt_path, 'label.txt')
        # with 上下文管理器，操作完成后自动关闭文件
        print("开始写入。。。")
        with open(txt_file,'w') as f:
             #取值，组合为[filename object cx cy w h ...\n]形式
            #以迭代次数为文件名（迭代次数和转换为正方形的图片名称一一对应）
            for i in range(len(self.image_file_path)):
                data=''
                file_name='{}.jpg'.format(i)
                data=data+file_name+' '
                for j,ob in enumerate(self.object_in_image[i]):
                    obj=str(ob)
                    box=bndboxes[i][j]
                    #将list变为str
                    box=str(box[0])+' '+str(box[1])+' '+str(box[2])+' '+str(box[3])+' '
                    data=data+obj+' '+box
                data=data+'\n'
                f.write(data)
        print('写入完毕!')



