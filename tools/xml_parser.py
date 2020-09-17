import xml.etree.cElementTree as ET
import glob2 as glob
import tools.cfg as cfg
import time
from tqdm import tqdm
import os
class parser():
    """
    返回：文件绝对路径[N->str]，图片大小[N,3->int]，图片中的对象[N,n->str],图片中对象的坐标[N,n,4->int]
    N：文件数量
    n: 目标数量
    """
    def __init__(self):
        self.anno_path=cfg.ANNOTATIONS_PATH
        self.xml_path=glob.glob('{}/*xml'.format(self.anno_path))
        self.images_path=cfg.IMAGES_PATH
        self.image_file_path=[]
        self.image_names=[]
        self.image_sizes=[]
        self.object_in_image=[]
        self.all_cords=[]
    def __call__(self, *args, **kwargs):
        for xml_file in tqdm(self.xml_path):
        #===============调试用==================#
        # for xml_file in tqdm(self.xml_path[0:100]):
        # ===============调试用==================#
            tree=ET.parse(xml_file)
            # print('*'*20)
            #得到图片的基本信息#
            image_number=tree.findtext('./filename')
            image_height=tree.findtext("./size/height")
            image_width=tree.findtext("./size/width")
            image_channel=tree.findtext("./size/depth")
            image_size=[int(image_height),int(image_width),int(image_channel)]
            # print("iamge_size->",image_size)
            #将图片信息加到图像信息收集list中
            self.image_sizes.append(image_size)
            #收集图片的文件路径
            self.image_file_path.append(os.path.join(self.images_path,image_number))
            #################

            #得到图片中的目标信息#
            objects=[]#目标
            cords=[]#坐标,[objects,4],objects表示有多少类
            for obj in tree.iter('object'):
                #获取目标名字
                object_name=obj.findtext('name')
                objects.append(object_name)
                ###########

                #获取目标位置信息
                xmin=int(obj.findtext("bndbox/xmin"))
                ymin=int(float(obj.findtext("bndbox/ymin")))
                # ymin = int(obj.findtext("bndbox/ymin"))
                '''
                将会有一个报错：76%|███████▌  | 13025/17125 [00:04<00:01, 3158.52it/s]
                ValueError: invalid literal for int() with base 10: '45.70000076293945'\
                故修改为ymin=int(float())
                '''
                xmax = int(obj.findtext("bndbox/xmax"))
                ymax = int(obj.findtext("bndbox/ymax"))
                cord=[xmin,ymin,xmax,ymax]
                cords.append(cord)

            # print("object->",objects)
            # print("object cord->",cords)
            ##################
            #将目标信息加到目标收集list中
            self.object_in_image.append(objects)
            #将坐标信息加到坐标收集list中
            self.all_cords.append(cords)
        #     print('*'*20)
        # print(self.image_sizes)
        # print(type(self.image_sizes))
        # print(self.object_in_image)
        # print(type(self.object_in_image))
        # print(self.all_cords)
        # print(type(self.all_cords))
        # print(self.image_file_path)
        # print(type(self.image_file_path))
    # exit()
        return self.image_file_path,self.image_sizes,self.object_in_image,self.all_cords

