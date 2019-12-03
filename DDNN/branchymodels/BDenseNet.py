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

        model = models.densenet121(pretrained=pretrained)     
        exit1 = nn.Sequential(
            model.features.conv0,
            model.features.norm0,
            model.features.relu0,
            model.features.pool0,
            model.features.denseblock1
        )

        self.exit1 = Exit(exit1, 256, out_channels)
        self.exit2 = Exit(model.features.denseblock2, 512, out_channels)
        self.exit3 = Exit(model.features.denseblock3, 1024, out_channels)
        self.exit4 = Exit4(model.features.denseblock4, model.features.norm5, out_channels)

        self.transistion1 = model.features.transition1
        self.transistion2 = model.features.transition2
        self.transistion3 = model.features.transition3     

        del model  

    def forward(self, x):
        predictions = []
        timings = []

        time_start = perf_counter()

        p, x = self.exit1(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)
        x = self.transistion1(x)

        p, x = self.exit2(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)
        x = self.transistion2(x)
        
        p, x = self.exit3(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)
        x = self.transistion3(x)

        p = self.exit4(x)
        timings.append((perf_counter()-time_start)*1000)
        predictions.append(p)
        
        return predictions, timings

class Exit4(nn.Module):
    def __init__(self, denseblock4, norm5, out_channels):
        super(Exit4, self).__init__()
        self.model = nn.Sequential(
            denseblock4,
            norm5
        )
        self.clf = nn.Linear(1024, out_channels)

    def forward(self, x):
        x = self.model(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        p = self.clf(x)

        return p

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

if __name__ == "__main__":
    import PIL
    import numpy as np
    img = PIL.Image.open('test_images/turtoise.JPEG')

    loader = transforms.Compose([transforms.Scale((224,224)), transforms.ToTensor()])
    
    img = loader(img)
    img = Variable(img, requires_grad=False)
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