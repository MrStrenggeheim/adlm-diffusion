import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        img_dir,
        transforms,
    ):
        self.images_folder = img_dir
        print(f"Images folder: {self.images_folder}")

        # load images, do recursive search for all images in multiple folders
        images_list = []
        for root, _, files in os.walk(self.images_folder, followlinks=True):
            print("Including images from", os.path.relpath(root, self.images_folder))
            for file in files:
                images_list.append(
                    os.path.join(os.path.relpath(root, self.images_folder), file)
                )
        images_list = sorted(images_list)

        self.dataset = pd.DataFrame(images_list, columns=["image"])
        self.transforms = transforms

        print(f"Loaded {len(self.dataset)} images")
        print(f"Transforms: {self.transforms}")
        print("=" * 130)

    def __getitem__(self, index):
        """
        Returns the image and label at the given index.
        """
        img_path = os.path.join(self.images_folder, self.dataset["image"][index])
        img = Image.open(img_path)

        if self.transforms:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.dataset)
