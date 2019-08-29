import os

from torchvision import datasets, transforms
import torch
from util import Partition
import numpy as np

def get_dataset(dataset_root, dataset, batch_size, is_cuda=True):
    if dataset == 'voc':
        train, test, train_loader, test_loader = get_voc(dataset_root, batch_size, is_cuda)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))

    return train, train_loader, test, test_loader


def get_voc(dataset_root, batch_size, is_cuda=True, valid_size=.2):
    kwargs = {'num_workers': 4, 'pin_memory': True} if is_cuda else {}
    data_dir = 'voc'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_root, data_dir, x), data_transforms[x]) for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, drop_last=False, **kwargs) for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    return image_datasets['train'], image_datasets['test'], dataloaders['train'], dataloaders['test']