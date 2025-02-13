import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

DATASET_FOLDER = "/vol/aimspace/projects/practical_WS2425/diffusion/data/amos_slices"
SPLIT_FOLDER = ["labelsTr", "labelsVa"]

IDX_RANGE = range(0, 500)  # filter for MRI
SLC_RANGE = range(0, 1000)  # should include all

MIN_PXL_MASK = None  # n pixels not to be background
MIN_SEG_CLS = None  # min number of unique seg classes, including background


def filter_name(filename, path, is_mask=True):
    # assume name format amos_xxxx_syyy.png
    r = re.compile(r"amos_(\d+)_s(\d+).png")
    idx, slc = r.match(filename).groups()
    idx, slc = int(idx), int(slc)

    if idx not in IDX_RANGE or slc not in SLC_RANGE:
        return False

    # if is mask check for appropriate args
    if is_mask:
        if MIN_PXL_MASK is not None or MIN_SEG_CLS is not None:
            mask = Image.open(path)
            mask_np = np.array(mask)

            if MIN_PXL_MASK is not None:
                if len(mask_np[mask_np != 0]) < MIN_PXL_MASK:
                    return False

            if MIN_SEG_CLS is not None:
                if len(np.unique(mask_np)) < MIN_SEG_CLS:
                    return False

    if not is_mask:
        # check for other
        pass

    return True


def process_file(file, split_folder):
    try:
        if filter_name(file, os.path.join(split_folder, file), True):
            return file
    except Exception as e:
        print(f"Error: {e}")
    return None


def filter_all():
    # assuming same naming in image and mask !!
    filtered_file_names = []

    with ThreadPoolExecutor() as executor:
        for split in SPLIT_FOLDER:
            split_folder = os.path.join(DATASET_FOLDER, split)
            file_names = os.listdir(split_folder)
            file_names.sort()

            futures = [
                executor.submit(process_file, file, split_folder) for file in file_names
            ]

            for future in tqdm(futures, desc=split):
                result = future.result()
                if result:
                    filtered_file_names.append(result)

    return filtered_file_names


if __name__ == "__main__":
    filtered_file_names = filter_all()
    # save to csv
    with open("filtered_files.csv", "w") as f:
        for name in filtered_file_names:
            f.write(name + "\n")
