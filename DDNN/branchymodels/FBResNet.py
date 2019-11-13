from __future__ import print_function
import torch

import torch.nn as nn
from torchvision import models

from time import perf_counter # uses the most precise clock on OS 

class FBResNet(nn.Module):
    def __init__(self ,out_channels, pretrained=True):
        super(FBResNet, self).__init__()

        self.branches = 4

        self.model = models.resnet101(pretrained=pretrained)

        for param in self.model.parameters():
            param.requires_grad = False

        self.conv1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.maxpool # according to article, maxpool belong to conv2_x, however to simplify code using exit it is moved to conv1
        )

        self.exit1 = Exit(self.model.layer1, 256, out_channels)
        self.exit2 = Exit(self.model.layer2, 512, out_channels)
        self.exit3 = Exit(self.model.layer3, 1024, out_channels)
        self.exit4 = Exit(self.model.layer4, 2048, out_channels)

    def forward(self, x):
        predictions = []
        timings = []

        time_start = perf_counter()
        x = self.conv1(x) # this must always be called before exit

        p, x = self.exit1(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)
        

        p, x = self.exit2(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)
        

        p, x = self.exit3(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)

        p, x = self.exit4(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)
        
        return predictions, timings

class Exit(nn.Module):
    def __init__(self, base_model, output_size, out_channels):
        super(Exit, self).__init__()

        self.base = base_model
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Linear(output_size, out_channels)

    def forward(self, x):
        batch = x.shape[0]

        x = self.base(x)
        p = self.pool(x)
        p = self.clf(p.view(batch,-1))

        return p, x