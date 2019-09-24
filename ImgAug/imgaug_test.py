
import torch
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import argparse
from augmentations import Augmentation


class Augmenter():
    def __init__(self, level, img_size):
        # format plotting: remove ticks
        mpl.rcParams['xtick.labelsize'] = 0
        mpl.rcParams['ytick.labelsize'] = 0

        mpl.rcParams['xtick.major.size'] = 0
        mpl.rcParams['ytick.major.size'] = 0

        mpl.rcParams['figure.autolayout'] = True
        mpl.rcParams['figure.figsize'] = (32,32)
        mpl.rcParams['figure.dpi'] = 600

        self.augmentation = Augmentation(level, img_size)
        

    def load_images(self, path):
        images = []
        for img in path:
            # load image and convert to numpy
            img = Image.open(img)
            img = np.array(img)
            images.append(img)
        return images

    def show_augmentation(self, images, cols=8, rows=8):
        self.augmentation.aug.show_grid(images, cols=cols, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Augmentation tester')
    parser.add_argument('--level', default='light', help='Level of augmentation')
    parser.add_argument('--img_size', default=(224, 224), type=tuple, help='(height, width)')
    parser.add_argument('--dataset_root', default='images', help='path to images')
    parser.add_argument('--n_images', default=1, type=int, help='number of images')
    args = parser.parse_args()

    augmenter = Augmenter(args.level, args.img_size)
    img_names = os.listdir(args.dataset_root)
    images = []
    for img in img_names:
        images.append(os.path.join(args.dataset_root, img))
    images = augmenter.load_images(images)
    augmenter.show_augmentation(images[0:args.n_images])

