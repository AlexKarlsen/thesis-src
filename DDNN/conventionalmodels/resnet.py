from __future__ import print_function
import torch

import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

from time import perf_counter # uses the most precise clock on OS 

class ResNet(nn.Module):
    def __init__(self, out_channels, pretrained=True):
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.clf = nn.Linear(2048, out_channels)

    def forward(self, x):
        batch = x.shape[0]
        time_start = perf_counter()
        x = self.model(x)
        pred = self.clf(x.view(batch,-1))
        time_end = perf_counter()
        return pred, time_end - time_start

if __name__ == "__main__":
    import PIL
    import numpy as np
    from torch.autograd import Variable
    img = PIL.Image.open('test_images/turtoise.JPEG')

    loader = transforms.Compose([transforms.Scale((224,224)), transforms.ToTensor()])
    
    img = loader(img)
    img = Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    model = ResNet(out_channels=1000)
    model.eval()
    pred, time = model(img)

    p = F.softmax(pred)
    p, l = torch.topk(pred, k=5)
    print(p)
    print(l)
