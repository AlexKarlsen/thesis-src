import torch.nn.functional as F

from time import perf_counter

class predictor:
    def __init__(self, model, data, model_type):
        self.nExits = 4
        if model_type == 'b-densenet' or model_type == 'b-resnet':
            self.exits = [model.exit1, model.exit2, model.exit3, model.exit4]
            if model_type == 'b-densenet':
                self.transistions = [model.transistion1, model.transistion2, model.transistion3]
            elif model_type == 'b-resnet':
                self.conv1 = model.conv1
        elif model_type == 'msdnet':
            self.exits = model.blocks
            self.model = model
        self.counter = 0
        self.data = data
        self.model_type = model_type

    def __next__(self):
        if self.model_type == 'b-densenet':
            if self.counter == 0:
                p, self.data = self.exits[self.counter](self.data)
                self.counter += 1
                return p
            elif self.counter == 3:
                self.data = self.transistions[self.counter-1](self.data)
                p = self.exits[self.counter](self.data)
                return p
            else:
                self.data = self.transistions[self.counter-1](self.data)
                p, self.data = self.exits[self.counter](self.data)
                self.counter += 1
                return p
        elif self.model_type == 'b-resnet':
            if self.counter == 0:
                self.data = self.conv1(self.data)
            p, self.data = self.exits[self.counter](self.data)
            self.counter += 1
            return p
        elif self.model_type == 'msdnet':
            self.data = self.model.blocks[self.counter](self.data)
            p = self.model.classifier[self.counter](self.data)
            self.counter += 1
            return p
        else:
            p = self.model(self.data)
            return p