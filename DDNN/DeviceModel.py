from __future__ import print_function
import torch

import torch.nn as nn

from torchvision import models as models

class DeviceModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeviceModel, self).__init__()

        # getting the MobileNetV2 network with pretrained weights
        self.model = models.mobilenet_v2(pretrained=True)

        # only the network base
        self.model = self.model.features

        # add pooling and new classifier 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, out_channels)

        # freeze base
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch = x.shape[0]
        h = self.model(x)
        x = self.pool(h)
        return h, self.classifier(x.view(batch,-1))
