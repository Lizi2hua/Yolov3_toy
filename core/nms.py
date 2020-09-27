#boxes:[topx,topy,bottonx,bottony,cls,pic_num]
import torch
from torchvision.ops import nms

def NMS(boxes,thresh):
        '''
        输入boxes,输出nms后的boxes
        :param boxes: [iou,cx,cy,w,h,cls,pic_num]
        :param thresh: 阈值，iou越小，得到的框越少
        :return:坐标数据为int,类型为list
        '''
        corrd=boxes[:,1:5]
        scores=boxes[:,0]
        idx=nms(corrd,scores,thresh)
        boxes=boxes[idx]
        new_boxes=[]
        for box in boxes:
                new_boxes.append(list(map(int,box[1:])))
        return new_boxes

# test_data=torch.tensor([[  0.8668, 200.3518, 102.0343, 320.0775, 261.0364,  14.0000,   0.0000],
#         [  0.8919, 155.5276, 286.4453, 247.2570, 372.1382,  14.0000,   0.0000],
#         [  0.8740, 179.3462, 291.6580, 244.4833, 363.7092,   6.0000,   0.0000],
#         [  0.9035, 162.7664, 298.8277, 230.1461, 383.3662,  14.0000,   0.0000],
#         [  0.8409, 194.6018,  39.3520, 291.2624,  95.6809,  14.0000,   0.0000],
#         [  0.8119, 182.4503,  54.6932, 288.9676,  96.6587,  14.0000,   0.0000],
#         [  0.8014, 150.7256, 285.3645, 256.5946, 362.3152,  14.0000,   0.0000],
#         [  0.8066, 142.5499, 293.1692, 248.8327, 370.2477,  14.0000,   0.0000],
#         [  0.8377, 151.6082, 293.4248, 255.3940, 369.8005,  14.0000,   0.0000],
#         [  0.8066, 143.7897, 299.8250, 247.1461, 379.2415,  14.0000,   0.0000]],)
# box=NMS(test_data,0.5)
# print(box)