import os
from PIL import Image
import json

from torch.utils.data import Dataset
from torchvision import transforms


class MiniImageNet(Dataset):

    def __init__(self, dataset_root, setname, transform):
        csv_path = json.loads(os.path.join(dataset_root, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(dataset_root, setname, wnid, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
