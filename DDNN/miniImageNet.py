import os
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MiniImageNet(Dataset):

    def __init__(self, dataset_root, setname, n_classes, transform):
        self.root = dataset_root
        text_file = open(os.path.join(dataset_root, 'imagenet_1000.txt'), "r")
        lines = text_file.read().split('\n')
        classes = lines[:n_classes]

        self.wnid_to_classes = self._load_meta_file()[0]

        data = []
        label = []
        lb = 0

        self.classes = []

        for wnid in classes:
            images_in_class = os.path.join(dataset_root, setname, wnid)
            for image in os.listdir(images_in_class):
                data.append(os.path.join(dataset_root, setname, wnid, image))
                self.classes.append(wnid)
                label.append(lb)
            lb += 1

        self.data = data
        self.label = label

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    @property
    def meta_file(self):
        return os.path.join(self.root, 'meta.bin')

    def _load_meta_file(self):
        try:
            return torch.load(self.meta_file)
        except:
            raise Exception('file not found')
