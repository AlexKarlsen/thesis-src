"""DenseNet"""

from __future__ import print_function
from time import perf_counter # uses the most precise clock on OS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class DenseNet(nn.Module):
    def __init__(self, out_channels, pretrained=True):
        super(DenseNet, self).__init__()
        self.model = models.densenet121(pretrained=pretrained)
        self.clf = nn.Linear(1024, out_channels)

        # freeze base
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch = x.shape[0]
        time_start = perf_counter()
        x = self.model(x)
        pred = self.clf(x.view(batch,-1))
        time_end = perf_counter()
        return pred, time_end - time_start

if __name__ == "__main__":
    import PIL
    from torch.autograd import Variable
    img = PIL.Image.open('test_images/turtoise.JPEG')
    loader = transforms.Compose([transforms.Scale((224,224)), transforms.ToTensor()])
    img = loader(img)
    img = Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    model = DenseNet(out_channels=1000)
    model.eval()
    pred, time = model(img)

    p = F.softmax(pred)
    p, l = torch.topk(pred, k=5)
    print(p)
    print(l)