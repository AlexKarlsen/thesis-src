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
    elif dataset == 'imagenet':
        train, test, train_loader, test_loader = get_imagenet(dataset_root, batch_size, is_cuda)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))

    return train, train_loader, test, test_loader

class ImgAugTransform:
  def __init__(self):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    self.aug = iaa.Sequential(
                [
                    iaa.Resize((224,224)),
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5), # horizontally flip 50% of all images
                    # crop images by -5% to 10% of their height/width
                    sometimes(iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=ia.ALL,
                        pad_cval=(0, 255)
                    )),
                    sometimes(iaa.Affine(
                        resize={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45), # rotate by -45 to +45 degrees
                        shear=(-16, 16), # shear by -16 to +16 degrees
                        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                        mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )),
                    # execute 0 to 5 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.SomeOf((0, 5),
                        [
                            #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                            ]),
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                            #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                            # search either for all edges or for directed edges,
                            # blend the result with the original image using a blobby mask
                            iaa.SimplexNoiseAlpha(iaa.OneOf([
                                iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                            ])),
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                            iaa.OneOf([
                                iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                            ]),
                            #iaa.Invert(0.05, per_channel=True), # invert color channels
                            iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                            iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                            # either change the brightness of the whole image (sometimes
                            # per channel) or change the brightness of subareas
                            iaa.OneOf([
                                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                iaa.FrequencyNoiseAlpha(
                                    exponent=(-4, 0),
                                    first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                    second=iaa.ContrastNormalization((0.5, 2.0))
                                )
                            ]),
                            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                            iaa.Grayscale(alpha=(0.0, 1.0)),
                            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                            # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                            # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                        ],
                        random_order=True
                    )
                ],
                random_order=True
            )
      
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

    image_datasets = {x: datasets.ImageNet(os.path.join(dataset_root, data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, drop_last=False, **kwargs) for x in ['train', 'val']}

    return image_datasets['train'], image_datasets['val'], dataloaders['train'], dataloaders['val']
        