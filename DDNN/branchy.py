from __future__ import print_function
import torch

import torch.nn as nn
from torchsummary import summary

from torchvision import models as models

class BranchyNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BranchyNet, self).__init__()

        self.mainbranch = models.resnet50(pretrained=True)
        self.exit1branch = exit1(self.mainbranch, out_channels)
        self.exit2branch = exit2(self.mainbranch, out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.base = nn.Sequential(
            self.mainbranch,
            self.exit1branch,
            self.exit2branch,
            self.pool
        )
        self.classifier = nn.Linear(2048, out_channels)

        for param in self.mainbranch.parameters():
            param.requires_grad = False


    def forward(self, x):
        batch = x.shape[0]
        predictions = []

        h, p = self.exit1branch.forward(x)
        predictions.append(p)

        h, p = self.exit2branch.forward(h)
        predictions.append(p)

        x = self.mainbranch.layer3(h)
        x = self.mainbranch.layer4(x)

        x = self.pool(x)
        p = self.classifier(x.view(batch,-1))

        predictions.append(p)
        
        return predictions


class exit2(nn.Module):
    def __init__(self, main_model, out_channels):
        super(exit2, self).__init__()

        resnet101layer3 = models.resnet101(pretrained=True).layer3
        resnet101layer4 = models.resnet101(pretrained=True).layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, out_channels)

        self.mainbranch = main_model.layer2

        self.exit2branch = nn.Sequential(
            resnet101layer3,
            resnet101layer4,
            self.pool
        )

        # for param in self.mainbranch.parameters():
        #     param.requires_grad = False

        for param in self.exit2branch.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch = x.shape[0]

        h = self.mainbranch(x)
        x = self.exit2branch(h)
        
        return h, self.classifier(x.view(batch,-1))

class exit1(nn.Module):
    def __init__(self, main_model, out_channels):
        super(exit1, self).__init__() 

        # main branch is handled in sub-branches to easily extend to distributed network spiltting
        self.main_model = main_model

        resnet50layer2 = models.resnet50(pretrained=True).layer2
        resnet50layer3 = models.resnet50(pretrained=True).layer3
        resnet50layer4 = models.resnet50(pretrained=True).layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, out_channels)

        self.exit1branch = nn.Sequential(
            resnet50layer2,
            resnet50layer3,
            resnet50layer4,
            self.pool
        )

        # for param in self.main_model.parameters():
        #     param.requires_grad = False
        
        for param in self.exit1branch.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch = x.shape[0]

        h = self.main_model.conv1(x)
        h = self.main_model.bn1(h)
        h = self.main_model.relu(h)
        h = self.main_model.maxpool(h)
        h = self.main_model.layer1(h)
        x = self.exit1branch(h)

        return h, self.classifier(x.view(batch,-1)) 
