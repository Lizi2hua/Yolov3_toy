from detect import Detector
from PIL import  Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob2 import glob
from cfg import LABEL_Parser
from PIL import Image
import cv2
from detect import Detector
import numpy as np


def ToSquare(img_data):
    '''
    :param image_data: PIL数据对象
    :return: PIL数据对象
    '''
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
    square_img.paste(img_data, (xmin, ymin, xmax, ymax))
    return square_img

def draw_rect(img_path,boxes):
    '''
    :param img_path: 图片的路径，建议绝对路径
    :param label_data: 数据的格式最少的二维的
    :return:
    '''
    '''这个是在原图上画的图，但是网络出的结果是在416图上的结果，所以应该将原图变为416*416'''
    img=Image.open(img_path)
    square_img=ToSquare(img)
    square_img=np.asarray(square_img)#将pil对象转换为array->cv2
    #将图片变为416*416
    square_img=cv2.resize(square_img,(416,416))
    square_img=cv2.cvtColor(square_img,cv2.COLOR_BGR2RGB)

    for box in boxes:
        cls = LABEL_Parser[box[4]]
        top=(box[1],box[2])
        botton=(box[3],box[4])
        cv2.rectangle(square_img,top,botton,(0,0,255))
        cv2.putText(square_img,cls,top,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow(img_path,square_img)
    cv2.waitKey(0)
if __name__ == '__main__':
    dect=Detector()
    Image_Path = r'E:\overfit_data\square\images'
    # Image_Path = r'G:\datasets\VOC2012\VOCdevkit\VOC2012\JPEGImages'
    Images=glob(os.path.join(Image_Path,'*.jpg'))
    # Images = [os.path.join(Image_Path, '2007_000027.jpg')]
    # boxes = [[199-225//2,200-87//2,199+225,200+87,0 ]]
    # draw_rect(Images, boxes)
    for img in Images:
        img_data=Image.open(img)
        boxes,cost_time=dect(img_data)
        draw_rect(img,boxes)
        print('cost {}'.format(cost_time))
    cv2.destroyAllWindows()


