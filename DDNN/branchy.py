from __future__ import print_function
import torch

import torch.nn as nn
from torchsummary import summary

from torchvision import models as models

class BranchyNet:
    def __init__(self, in_channels, out_channels):
        super(BranchyNet, self).__init__()

        self.mainbranch = models.resnet152(pretrained=False)
        self.exit1branch = exit1(self.mainbranch, out_channels)
        self.exit2branch = exit2(self.mainbranch, out_channels)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, out_channels)


    def forward(self, x):
        batch = x.shape[0]
        predictions = []

        p, h = self.exit1branch.forward(x)
        predictions.append(p)

        p, h = self.exit2branch.forward(h)
        predictions.append(p)

        x = self.mainbranch.layer3[24:36].forward(h)
        x = self.mainbranch.layer4(x)

        x = self.mainbranch.pool(x)
        p = self.classifier(x.view(batch,-1))

        predictions.append(p)
        
        return predictions


class exit2:
    def __init__(self, main_model, out_channels):
        super(exit2, self).__init__()

        self.resnet101 = models.resnet101(pretrained=False)

        self.mainbranch = main_model.layer3[7:23]
        self.exit2branch = self.resnet101.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, out_channels)

    def forward(self, x):
        batch = x.shape[0]

        h = self.mainbranch(x)
        h = self.exit2branch
        x = self.pool(h)
        
        return h, self.classifier(x.view(batch,-1))

class exit1:
    def __init__(self, main_model, out_channels):
        super(exit1, self).__init__() 

        self.main_model = main_model

        self.resnet50 = models.resnet50(pretrained=False)
        self.exit1branch = nn.Sequential(
            self.resnet50.layer3,
            self.resnet50.layer4
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, out_channels)

    def forward(self, x):
        batch = x.shape[0]

        h = self.main_model.conv1(x)
        h = self.main_model.bn1(h)
        h = self.main_model.relu(h)
        h = self.main_model.maxpool(h)
        h = self.main_model.layer1(h)
        h = self.main_model.layer2(h)
        x = self.exit1branch(h)
        x = self.pool(x)

        return h, self.classifier(x.view(batch,-1)) 


model = BranchyNet(3,20)