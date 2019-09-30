from __future__ import print_function
import torch

import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F

from time import perf_counter # uses the most precise clock on OS 

class BDenseNet(nn.Module):
    def __init__(self, out_channels, pretrained=True):
        super(BDenseNet, self).__init__()

        self.branches = 4

        self.model = models.densenet121(pretrained=pretrained, num_classes=out_channels)

        exit1 = nn.Sequential(
            self.model.features.conv0,
            self.model.features.norm0,
            self.model.features.relu0,
            self.model.features.pool0,
            self.model.features.denseblock1
        )

        self.exit1 = Exit(exit1, self.model.features.transition1, 256, out_channels)
        self.exit2 = Exit(self.model.features.denseblock2, self.model.features.transition2, 512, out_channels)
        self.exit3 = Exit(self.model.features.denseblock3, self.model.features.transition3, 1024, out_channels)

    def forward(self, x):
        predictions = []
        timings = []

        time_start = perf_counter()

        p, x = self.exit1(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)
        

        p, x = self.exit2(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)
        

        p, x = self.exit3(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)

        x = self.model.features.denseblock4(x)
        x = self.model.features.norm5(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        p = self.model.classifier(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)
        
        return predictions, timings

class Exit(nn.Module):
    def __init__(self, base_model, transistion_layer, output_size, out_channels):
        super(Exit, self).__init__()

        self.base = base_model
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.transistion_layer = transistion_layer
        self.clf = nn.Linear(output_size, out_channels)

    def forward(self, x):
        batch = x.shape[0]

        x = self.base(x)
        p = self.pool(x)
        p = self.clf(p.view(batch,-1))
        x = self.transistion_layer(x)

        return p, x

if __name__ == "__main__":
    import PIL
    import numpy as np
    img = PIL.Image.open('test_images/turtoise.JPEG')

    loader = transforms.Compose([transforms.Scale((224,224)), transforms.ToTensor()])
    
    img = loader(img)
    img = Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    model = BDenseNet(out_channels=1000)
    model.eval()
    pred, time = model(img)

    for p in pred:
        p = F.softmax(p)
        p, l = torch.topk(p, k=5)
        print(p)
        print(l)
    # sm = torch.nn.Softmax()
    # probabilities = sm(p, ) 
    # print(probabilities) 