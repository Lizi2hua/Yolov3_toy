import torch
import numpy as np
from core.cfg import *
from tqdm import tqdm
import matplotlib.pylab as plt

# SEED=np.arange(0,150,3)

def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def iou(box,clusters):
    '''
    计算box与k cluster的IoU
    :param box:数据中所有的宽，高->[2]
    :param clusters:聚类框->[k,2]，k指聚类框个数
    :return:iou->[k]
    $$IoU=\frac{Insert}{Union}=\frac{min(boxes,clusters)}{boxes+clusters-min(boxes,clusters)}$$
    '''
    x=np.minimum(box[0],clusters[:,0])#[k]
    y=np.minimum(box[1],clusters[:,1])#[k]
    #校验异常情况：box没有w,h=0
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    #计算IOU
    insert=x*y
    box_area=box[0]*box[1]
    clusters_area=clusters[:,0]*clusters[:,1]
    iou =insert/(box_area+clusters_area-insert)
    return iou

def kmeans(boxes,k,seed,dist=np.median):
    '''
    使用IoU进行kmeans 聚类
    :param boxes: [N,2]
    :param k: [k]
    :param seed：随机种子
    :param dist:求距离均值函数
    :return: 聚类结果[k,2]
    '''
    rows=boxes.shape[0]
    distances=np.empty((rows,k))
    last_clusters = np.zeros((rows,))
    np.random.seed(seed)
    # the Forgy method will fail if the whole array contains the same rows
    # 初始化k个聚类中心（从原始数据集中随机选择k个）
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            # 定义的距离度量公式：d(box,centroid)=1-IOU(box,centroid)。到聚类中心的距离越小越好，
            # 但IOU值是越大越好，所以使用 1 - IOU，这样就保证距离越小，IOU值越大。
            # 计算所有的boxes和clusters的值（row，k）
            distances[row] = 1 - iou(boxes[row], clusters)
            #print(distances)
        # 将标注框分配给“距离”最近的聚类中心（也就是这里代码就是选出（对于每一个box）距离最小的那个聚类中心）。
        nearest_clusters = np.argmin(distances, axis=1)
        # 直到聚类中心改变量为0（也就是聚类中心不变了）。
        if (last_clusters == nearest_clusters).all():
            break
        # 计算每个群的中心（这里把每一个类的中位数作为新的聚类中心）
        for cluster in range(k):
            #这一句是把所有的boxes分到k堆数据中,比较别扭，就是分好了k堆数据，每堆求它的中位数作为新的点
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters

if __name__ == '__main__':
    #读取文件
    path=LABEL_416
    with open(path,'r') as f:
        text=f.readlines()
    # print(len(text))
    boxes=[]#append [w,h]
    print('============解析txt文件============')
    for index in range(len(text)):
        #得到由元素为字符串的列表
        list_str=text[index].split()#['17104.jpg', '14', '380', '329', '44', '138', '14', '269', '305', '26', '66']
        #取出图片名称之外的数据
        img_name=list_str[0]
        data=list_str[1:]#['14', '380', '329', '44', '138', '14', '269', '305', '26', '66']
        #得到当前数据中含有的目标个数
        obj_nums=len(data)//5
        print('{}中有{}个框'.format(img_name,obj_nums))
        #从框中取出w,h
        for l in range(obj_nums):
            w=int(data[5*l+3])
            h=int(data[5*l+4])
            box=[w,h]
            boxes.append(box)
    boxes=np.array(boxes)
    print(boxes.shape)
    print('============keans,随机种子筛选============')
    scores=[]
    max_score=0.
    best_seed=0.
    for seed in [SEED]:
        clusters=kmeans(boxes=boxes,k=9,seed=seed)
        print(clusters)
        score=avg_iou(boxes,clusters)
        if score>max_score:
            max_score=score
            best_seed=seed
        print('Accuarcy: {:.2f}%'.format(score*100))
        scores.append(score)
    print(max_score)
    print(best_seed)
    plt.plot(SEED,scores)
    plt.show()
