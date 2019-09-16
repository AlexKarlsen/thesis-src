from __future__ import print_function
import torch

import torch.nn as nn
from torchsummary import summary

from torchvision import models as models

class DResNet(nn.Module):
    def __init__(self, out_channels):
        super(DResNet, self).__init__()

        self.branches = 4

        self.model = models.resnet50(pretrained=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.clf1 = nn.Linear(256, out_channels)
        self.clf2 = nn.Linear(512, out_channels)
        self.clf3 = nn.Linear(1024, out_channels)
        self.clf4 = nn.Linear(2048, out_channels)

        for params in self.model.parameters():
            params.requires_grad = False

    def forward(self, x):
        batch = x.shape[0]
        predictions = []

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        p = self.pool(x)
        p = self.clf1(p.view(batch,-1))
        predictions.append(p)

        x = self.model.layer2(x)
        p = self.pool(x)
        p = self.clf2(p.view(batch,-1))
        predictions.append(p)

        x = self.model.layer3(x)
        p = self.pool(x)
        p = self.clf3(p.view(batch,-1))
        predictions.append(p)

        x = self.model.layer4(x)
        p = self.pool(x)
        p = self.clf4(p.view(batch,-1))
        predictions.append(p)
        
        return predictions