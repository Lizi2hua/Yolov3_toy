import matplotlib.pyplot as plt
import matplotlib.patches as pathes
from tools.xml_parser import parser
import time
parse=parser()
# path,size,OBJECTS,cord=parse()
#############
path=[r'E:\VOC2017_gened\images\0.jpg',r'E:\VOC2017_gened\images\1.jpg',r'E:\VOC2017_gened\images\2.jpg']
OBJECTS=[[14], [0, 0, 14, 14], [0, 0, 0]]
cord=[[[268, 226, 175, 250]], [[239, 240, 271, 105], [165, 215, 64, 35], [204, 314, 18, 49], [35, 323, 18, 49]], [[254, 252, 490, 156], [451, 280, 61, 26], [368, 272, 86, 35]]]
for i,image in enumerate(path):
    #读取图片
    img=plt.imread(image)
    fig,ax=plt.subplots(1)
    ax.imshow(img)

    #读取该图片中的目标类别
    objects=OBJECTS[i]
    #遍历目标类别
    for j,object in enumerate(objects):
        #获取该目标对应的框的坐标
        box=cord[i][j]
        w = box[2]
        h = box[3]
        top=(box[0]-w//2,box[1]-h//2)
        rect=pathes.Rectangle(top,w,h,linewidth=1,edgecolor='r',fill=False)
        plt.annotate(object,top,color='r')
        plt.title(image)
        ax.add_patch(rect)
    plt.show()
