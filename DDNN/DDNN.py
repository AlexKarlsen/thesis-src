from __future__ import print_function
import torch

import torch.nn as nn

from torchvision import models as models

from DeviceModel import DeviceModel

class DDNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_devices):
        super(DDNN, self).__init__()
        self.num_devices = num_devices
        self.device_models = []
        for _ in range(num_devices):
            self.device_models.append(DeviceModel(in_channels, out_channels))
        self.device_models = nn.ModuleList(self.device_models)

        cloud_input_channels = 1280*num_devices
        self.cloud_model = models.resnet50(pretrained=False)
        # changing input layer to accept the additional channels
        self.cloud_model.conv1 = nn.Conv2d(cloud_input_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # removing fc
        self.cloud_model = nn.Sequential(*(list(self.cloud_model.children())[:-1]))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, out_channels)

    def forward(self, x):
        B = x.shape[0]

        # feature and predictions from devices
        hs, predictions = [], []
        for i, device_model in enumerate(self.device_models):
            h, prediction = device_model(x)
            hs.append(h)
            predictions.append(prediction)

        # concatenating the features into a single tensor
        h = torch.cat(hs, dim=1)

        # run cloud network
        h = self.cloud_model(h)
        h = self.pool(h)
        prediction = self.classifier(h.view(B, -1))

        # add prediction to list
        predictions.append(prediction)
        return predictions
