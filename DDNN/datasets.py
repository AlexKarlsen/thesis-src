import os

from torchvision import datasets, transforms
import torch
from util import Partition
import numpy as np

from imgaug import augmenters as iaa
import imgaug as ia

def get_dataset(dataset_root, dataset, batch_size, is_cuda=True):
    if dataset == 'voc':
        train, test, train_loader, test_loader = get_voc(dataset_root, batch_size, is_cuda)
    elif dataset_root == 'imagenet':
        train, test, train_loader, test_loader = get_imagenet(dataset_root, batch_size, is_cuda)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))

    return train, train_loader, test, test_loader

class ImgAugTransform:
  def __init__(self):
    #sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    self.aug = iaa.Sequential([
        iaa.Resize((224, 224)),
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-20, 20), mode='symmetric'),
        iaa.Sometimes(0.25,
                      iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                 iaa.CoarseDropout(0.1, size_percent=0.5)])),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
    ])
      
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)

def get_voc(dataset_root, batch_size, is_cuda=True):
    kwargs = {'num_workers': 4, 'pin_memory': True} if is_cuda else {}
    data_dir = 'voc'

    data_transforms = {
        'train': transforms.Compose([
            ImgAugTransform(),
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
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    # class_names = image_datasets['train'].classes
    
    return image_datasets['train'], image_datasets['test'], dataloaders['train'], dataloaders['test']

def get_imagenet(dataset_root, batch_size, is_cuda=True):
    kwargs = {'num_workers': 4, 'pin_memory': True} if is_cuda else {}
    data_dir = 'imagenet'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            ImgAugTransform(),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_root, data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, drop_last=False, **kwargs) for x in ['train', 'val']}

    return image_datasets['train'], image_datasets['val'], dataloaders['train'], dataloaders['val']
        