import random

import numpy as np
import torch
import torchvision.transforms as tt
from torchvision import transforms as T
from torchvision.transforms import functional as F


def parse_transforms(transforms, num_img_channels, img_size):
    parsed_transforms = []
    if transforms is not None:
        transforms = "" if transforms is None else transforms
        parsed_transforms = eval(transforms)
        print(f"Parsed Transforms: {parsed_transforms}")

    transforms = []
    for t in parsed_transforms:
        if t == "ToTensor":
            transforms.append(PILToTensor())
        elif t == "Resize":
            transforms.append(Resize(img_size))
        elif t == "CenterCrop":
            transforms.append(CenterCrop(img_size))
        elif t == "RandomCrop":
            transforms.append(RandomCrop(img_size))
        elif t == "RandomResizedCrop":
            transforms.append(
                RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(1, 1))
            )
        elif t == "RandomRotation":
            transforms.append(RandomRotation(degrees=10))
        elif t == "ColorJitter":
            transforms.append(ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
        elif t == "Normalize":
            transforms.append(
                Normalize(num_img_channels * [0.5], num_img_channels * [0.5])
            )
    return Compose(transforms)


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Transform:
    def __repr__(self):
        attr_str = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attr_str})"

    __str__ = __repr__


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        transform_str = "\n    ".join([repr(t) for t in self.transforms])
        return f"Compose([\n    {transform_str}\n])"


class RandomResize(Transform):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size, antialias=True)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(Transform):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(Transform):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class RandomRotation(Transform):
    def __init__(self, degrees):
        if isinstance(degrees, (int, float)):
            self.degrees = (-np.abs(degrees), np.abs(degrees))
        else:
            assert len(degrees) == 2, "degrees should be a number or a pair of numbers"
            self.degrees = degrees

    def __call__(self, image, target):
        angle = random.uniform(*self.degrees)
        image = F.rotate(image, angle)
        target = F.rotate(target, angle, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomResizedCrop(Transform):
    def __init__(self, size, scale=(0.8, 1.0), ratio=(1, 1)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, target):
        crop_params = T.RandomResizedCrop.get_params(
            image, scale=self.scale, ratio=self.ratio
        )
        image = F.resized_crop(image, *crop_params, size=self.size)
        target = F.resized_crop(
            target,
            *crop_params,
            size=self.size,
            interpolation=T.InterpolationMode.NEAREST,
        )
        return image, target


class Resize(Transform):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class CenterCrop(Transform):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor(Transform):
    def __call__(self, image, target):
        image = F.pil_to_tensor(image).float()
        target = torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0).float()
        return image, target


class ColorJitter(Transform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        image = F.adjust_brightness(image, self.brightness)
        image = F.adjust_contrast(image, self.contrast)
        image = F.adjust_saturation(image, self.saturation)
        image = F.adjust_hue(image, self.hue)
        return image, target


class ToDtype(Transform):
    def __init__(self, dtype, scale=False):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, image, target):
        if not self.scale:
            return image.to(dtype=self.dtype), target
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
