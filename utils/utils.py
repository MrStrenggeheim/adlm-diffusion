import glob
import os
import time

import numpy as np
import torch
from PIL import Image


def make_grid(images, rows, cols):
    """Make a RBG PIL Image grid of images"""
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def tv_loss(x):
    """Total variation loss for spatial regularization."""
    diff_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    diff_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    return (torch.mean(diff_x) + torch.mean(diff_y)) / 2


def get_last_checkpoint(log_dir):
    """Get .ckpt file with the latest timestamp in dir (recursive)."""
    ckpts = sorted(
        glob.glob(os.path.join(log_dir, "**/checkpoints/*.ckpt"), recursive=True)
    )
    return ckpts[-1] if ckpts else None


def calc_dataset_var(dataset):
    """Calculate the variance of the training images."""
    all_images = []
    for img, _ in dataset:
        img = img / 255.0
        all_images.append(img.flatten())
    all_images = np.concatenate(all_images)
    return np.var(all_images)


def readable_timestamp():
    return time.ctime().replace("  ", " ").replace(" ", "_").replace(":", "_").lower()
