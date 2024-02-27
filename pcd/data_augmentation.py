import math
import numbers
import random
import warnings
from copy import deepcopy
from PIL import Image, ImageFilter, ImageOps

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class ToRGB:
    def __call__(self, x):
        return x.convert("RGB")


class Solarization:
    def __call__(self, x):
        return ImageOps.solarize(x)


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def byol_transform(image_size=224, min_scale=0.08, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if isinstance(min_scale, numbers.Number):
        min_scale = [min_scale, min_scale]
    else:
        assert isinstance(min_scale, (list, tuple)), "min_scale must be a number or list or tuple"
    transform_q = transforms.Compose([
        ToRGB(),
        transforms.RandomResizedCrop(image_size, scale=(min_scale[0], 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_k = transforms.Compose([
        ToRGB(),
        transforms.RandomResizedCrop(image_size, scale=(min_scale[1], 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
        transforms.RandomApply([Solarization()], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return [transform_q, transform_k]


def typical_imagenet_transform(train=True, image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if train:
        transform = transforms.Compose(
            [
                ToRGB(),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                ToRGB(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transform


class AsymmetricTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        image1 = self.transform1(x)
        image2 = self.transform2(x)
        return [image1, image2]