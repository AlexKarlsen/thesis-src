from __future__ import print_function
import torch

import torch.nn as nn
from torchvision import models

from time import perf_counter # uses the most precise clock on OS 

class BResNet(nn.Module):
    def __init__(self ,out_channels, pretrained=True):
        super(BResNet, self).__init__()

        self.branches = 4

        model = models.resnet101(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.maxpool # according to article, maxpool belong to conv2_x, however to simplify code using exit it is moved to conv1
        )

        # self.blocks = nn.ModuleList()
        # self.classifier = nn.ModuleList()

        # channels = [256, 512, 1024, 2048]

        
        self.exit1 = Exit(model.layer1, 256, out_channels)
        self.exit2 = Exit(model.layer2, 512, out_channels)
        self.exit3 = Exit(model.layer3, 1024, out_channels)
        self.exit4 = Exit(model.layer4, 2048, out_channels)

        del model

        # exits = [self.exit1,self.exit2,self.exit3,self.exit4]
        # clfs = [self.exit1.clf,self.exit2.clf,self.exit3.clf,self.exit4.clf]
        # for e, clf in zip(exits,clfs):
        #     self.blocks.append(e)
        #     self.classifier.append(clf)

    def forward(self, x):
        # res = []
        # for i in range(self.branches):
        #     p, x = self.blocks[i](x)
        #     res.append(self.classifier[i](x))
        # return res
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
