from detect import Detector
from PIL import  Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob2 import glob
from cfg import LABEL_Parser
def draw_rect(img_path,boxes):
    '''
    :param img_path: 图片的路径，建议绝对路径
    :param label_data: 数据的格式最少的二维的
    :return:
    '''
    img=plt.imread(img_path)
    fig,ax=plt.subplots(1)
    ax.imshow(img)

    for box in boxes:
        top=(box[0],box[1])
        w=box[2]-box[0]
        h=box[3]-box[1]
        rect=patches.Rectangle(top,w,h,linewidth=1,edgecolor='r',fill=False)
        anno=LABEL_Parser[box[4]]
        print(anno)
        plt.annotate(s=anno,xy=top,color='r')
        ax.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    dect=Detector()
    Image_Path = r'E:\overfit_data\square\images'
    Images=glob(os.path.join(Image_Path,'*.jpg'))
    # Images = [os.path.join(Image_Path, '100.jpg')]

    for img in Images:
        img_data=Image.open(img)
        boxes,cost_time=dect(img_data)
        draw_rect(img,boxes)



