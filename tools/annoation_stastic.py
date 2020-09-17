import matplotlib.pyplot as plt
from tools.xml_parser import parser
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

parse=parser()
path,size,OBJECTS,cord=parse()

#统计类别
objects=[]
for i,object in enumerate(OBJECTS):
    for cls in range(len(object)):
        objects.append(OBJECTS[i][cls])
plt.hist(objects,bins=20,color='steelblue',edgecolor='k')
plt.tick_params(top='off', right='off')
plt.xlabel('类别')
plt.ylabel('频数')
plt.title('类别统计')
plt.show()



