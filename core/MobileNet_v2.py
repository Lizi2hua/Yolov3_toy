import torch.nn as nn
from torch.nn import modules
from torchvision.models import  mobilenet_v2
model=mobilenet_v2(pretrained=False)
print(model)