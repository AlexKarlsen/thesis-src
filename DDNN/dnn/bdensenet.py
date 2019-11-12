import torch.nn.functional as F

from time import perf_counter

class predictor:
    def __init__(self, model, data):
        self.nExits = 4
        self.exits = [model.exit1, model.exit2, model.exit3, model.exit4]
        self.transistions = [model.transistion1, model.transistion2, model.transistion3]
        self.counter = 0
        self.data = data

    def __next__(self):
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