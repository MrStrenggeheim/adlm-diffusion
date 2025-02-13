import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tt


def parse_transforms(transforms, num_img_channels, img_size):
    transform_img = []
    transform_seg = []
    transform_raw = []
    transforms = "" if transforms is None else transforms
    parsed_transforms = eval(transforms)
    print(f"Parsed Transforms: {parsed_transforms}")

    for t in parsed_transforms:
        if t == "ToTensor":
            transform_img.append(tt.ToTensor())
            transform_seg.append(tt.ToTensor())
            transform_raw.append(tt.ToTensor())
        elif t == "Resize":
            transform_img.append(tt.Resize(img_size))
            transform_seg.append(tt.Resize(img_size))
            transform_raw.append(tt.Resize(img_size, interpolation=Image.NEAREST))
        elif t == "CenterCrop":
            transform_img.append(tt.CenterCrop(img_size))
            transform_seg.append(tt.CenterCrop(img_size))
            transform_raw.append(tt.CenterCrop(img_size))
        elif t == "ColorJitter":
            transform_img.append(
                tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            )
        elif t == "Normalize":
            transform_img.append(
                tt.Normalize(num_img_channels * [0.5], num_img_channels * [0.5])
            )
            transform_seg.append(
                tt.Normalize(num_img_channels * [0.5], num_img_channels * [0.5])
            )

    return (
        tt.Compose(transform_img),
        tt.Compose(transform_seg),
        tt.Compose(transform_raw),
    )


class AmosDataset(Dataset):
    """
    Amos dataset class for PyTorch.
    """

    def __init__(
        self,
        img_dir,
        seg_dir,
        split: Literal["train", "val", "test"],
        num_img_channels,  # expect 1 or 3
        img_size,
        transforms,
        index_range=None,
        slice_range=None,
        only_labeled=False,
        img_name_filter=None,
        load_images_as_np_arrays=False,  # if True, load img and seg as tensor. expect .pt, using torch.load
    ):
        self.split = split
        self.images_folder = os.path.join(img_dir, split)
        self.labels_folder = os.path.join(seg_dir, split)

        print(f"Loading Amos {split} data")
        print(f"Images folder: {self.images_folder}")
        print(f"Labels folder: {self.labels_folder}")

        # load images, do recursive search for all images in multiple folders
        images_list = []
        labels_list = []
        for root, _, files in os.walk(self.images_folder, followlinks=True):
            print("Including images from", os.path.relpath(root, self.images_folder))
            for file in files:
                images_list.append(
                    os.path.join(os.path.relpath(root, self.images_folder), file)
                )
        for root, _, files in os.walk(self.labels_folder, followlinks=True):
            print("Including labels from", os.path.relpath(root, self.labels_folder))
            for file in files:
                labels_list.append(
                    os.path.join(os.path.relpath(root, self.labels_folder), file)
                )
        images_list = sorted(images_list)
        labels_list = sorted(labels_list)
        # Assume that the images and labels are named the same
        # items_list = list(set(images_list).intersection(set(labels_list)))
        images_df = pd.DataFrame(images_list, columns=["image"])
        labels_df = pd.DataFrame(labels_list, columns=["label"])
        print(f"{len(images_df)} images and {len(labels_df)} labels")
        images_df = images_df[images_df["image"].isin(labels_df["label"])].reset_index()
        labels_df = labels_df[labels_df["label"].isin(images_df["image"])].reset_index()
        print(f"{len(images_df)} images and {len(labels_df)} labels after intersection")

        # filter image not in range
        if index_range:
            assert len(index_range) == 2, "index_range must be a list of two integers"
            index_range = range(index_range[0], index_range[1] + 1)
            index_mask = images_df["image"].apply(
                lambda x: self._filter_filename(x, index_range, filter_type="index")
            )
        else:
            index_mask = [True] * len(images_df)
        # filter slice not in range
        if slice_range:
            assert len(slice_range) == 2, "slice_range must be a list of two integers"
            slice_range = range(slice_range[0], slice_range[1] + 1)
            slice_mask = images_df["image"].apply(
                lambda x: self._filter_filename(x, slice_range, filter_type="slice")
            )
        else:
            slice_mask = [True] * len(images_df)

        combined_mask = np.logical_and(index_mask, slice_mask)
        images_df = images_df[combined_mask]
        labels_df = labels_df[combined_mask]

        # filter if not at least one pixel is labeled. NOT RECOMMENDED
        if only_labeled:
            print(f"Filtering only labeled images ...")
            label_mask = labels_df["label"].apply(
                lambda label: np.array(Image.open(self.labels_folder + label)).sum() > 0
            )
            images_df = images_df[label_mask]
            labels_df = labels_df[label_mask]

        print(f"{len(images_df)} images and {len(labels_df)} labels after masking")

        assert len(images_df) == len(
            labels_df
        ), "Number of images and labels do not match"

        print(
            f"{len(set(images_df["image"]).intersection(set(labels_df["label"])))} images and labels after intersection"
        )
        self.dataset = pd.merge(images_df, labels_df, left_on="image", right_on="label")

        if img_name_filter is not None:
            print(f"Filtering images by name ...")
            self.dataset = self.dataset[
                self.dataset["image"].isin(img_name_filter)
            ].reset_index(drop=True)

        print(f"Loaded {len(self.dataset)} {split} images")

        self.load_images_as_np_arrays = load_images_as_np_arrays
        self.num_img_channels = num_img_channels
        self.img_size = img_size

        self.transform_img, self.transform_seg, self.transform_raw = parse_transforms(
            transforms, num_img_channels, img_size
        )

        print(
            f"""
            Transform img: {self.transform_img},
            Transform seg: {self.transform_seg},
            index_range: {index_range}, 
            slice_range: {slice_range}, 
            only_labeled: {only_labeled}"""
        )

    def _filter_filename(self, filename, range, filter_type="index"):
        """
        Filters filenames and keeps only those with indices in the given range.
        Assumes the filename format is: "amos_XXXX_sYYY.png" (XXXX is the index)
        """
        # Extract the index part
        try:
            if filter_type == "index":
                index = int(filename.split("_")[1])  # Extract the XXXX part
            elif filter_type == "slice":
                index = int(filename.split("s")[1])  # Extract the YYY part
            else:
                raise ValueError("filter_type must be either 'index' or 'slice'")
            return index in range
        except (IndexError, ValueError):
            return False  # Skip files that don't match the format

    def __getitem__(self, index):
        """
        Returns the image and label at the given index.
        """
        img_path = os.path.join(self.images_folder, self.dataset["image"][index])
        seg_path = os.path.join(self.labels_folder, self.dataset["label"][index])

        if self.load_images_as_np_arrays:
            img = torch.load(img_path)
            label = torch.load(seg_path)
        else:
            img = Image.open(img_path)
            label = Image.open(seg_path)

        if self.transform_img:
            img_transform = self.transform_img(img)
        if self.transform_seg:
            label_transform = self.transform_seg(label)

        if self.split == "test":
            label_raw = self.transform_raw(label)
            return {
                "images": img_transform,
                "images_target": label_transform,
                "images_target_raw": label_raw,
            }

        return {"images": img_transform, "images_target": label_transform}

    def __len__(self):
        return len(self.dataset)
