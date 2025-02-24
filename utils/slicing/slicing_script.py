import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from TPTBox import NII
from tqdm import tqdm

PROJECT_FOLDER = "/vol/miltank/projects/practical_WS2425/diffusion/data/"
SLICING_AXIS = ["axial", "coronal", "sagittal"]
ORIENTATION = ("I", "P", "R")  # from top left corner
RESCALE = (1, 1, 1)
AX_IDX = {
    "axial": max("".join(ORIENTATION).find("S"), "".join(ORIENTATION).find("I")),
    "coronal": max("".join(ORIENTATION).find("A"), "".join(ORIENTATION).find("P")),
    "sagittal": max("".join(ORIENTATION).find("L"), "".join(ORIENTATION).find("R")),
}
CLIPPING_RANGE = (-256, 256)
SCALE_LABELS = None  # give range, labels will be scaled to e.g. (0, 255)
CT_IDX_RANGE = range(0, 500)

# give input names with ending / and output names without /
MAPPING = [
    # {"in": "amos22/imagesTr/", "out": "amos_slices/imagesTr", "seg": False},
    # {"in": "amos22/imagesVa/", "out": "amos_slices/imagesVa", "seg": False},
    # {"in": "amos22/imagesTs/", "out": "amos_slices/imagesTs", "seg": False},
    # {"in": "amos_robert/labelsTr/", "out": "amos_robert_slices/labelsTr", "seg": True},
    # {"in": "amos_robert/labelsVa/", "out": "amos_robert_slices/labelsVa", "seg": True},
    # {"in": "amos_robert/labelsTs/", "out": "amos_robert_slices/labelsTs", "seg": True},
    {"in": "amos22/labelsVa/", "out": "amos_slices/labelsVa", "seg": True},
]


def is_ct(file_name):
    return int(file_name.split("_")[1].split(".")[0]) in CT_IDX_RANGE


def load_nii(file_name, nii_folder, is_seg):
    nii = NII.load(nii_folder + file_name, seg=is_seg)
    if RESCALE:
        nii.rescale(RESCALE, verbose=False, inplace=True)
    nii.reorient(ORIENTATION, verbose=False, inplace=True)
    return nii.get_array()


def safe_normalize(array, new_min=0, new_max=1, percentile: int = None):
    old_min = array.min()
    old_max = array.max()
    if old_min == old_max:
        return array
    if percentile:
        old_min = np.percentile(array, percentile)
        old_max = np.percentile(array, 100 - percentile)
        array = np.clip(array, old_min, old_max)
    return (array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def normalize_nii(nii_np, file_name, is_seg):
    if not is_seg:
        if is_ct(file_name):
            # clip only if is CT
            nii_np = np.clip(nii_np, CLIPPING_RANGE[0], CLIPPING_RANGE[1])
        nii_np = safe_normalize(nii_np, 0, 255, percentile=1)
    if is_seg and SCALE_LABELS:
        nii_np = safe_normalize(nii_np, 0, 255)
    return nii_np


def save_slice(slice, i, slice_folder, file_name, is_seg):
    pil = Image.fromarray(slice.astype(np.uint8), "L")
    slice_string = "s" + str(i).zfill(3)
    path = slice_folder + file_name.replace(".nii.gz", "") + "_" + slice_string + ".png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil.save(path)


def get_slice(np_array, axis, idx):
    if axis == "axial":
        return np_array[idx, :, :]
    elif axis == "coronal":
        return np_array[:, idx, :]
    elif axis == "sagittal":
        return np_array[:, :, idx]


def slice_nii(file_name, nii_folder, slice_folder, is_seg):
    nii_np = load_nii(file_name, nii_folder, is_seg=is_seg)
    nii_np = normalize_nii(nii_np, file_name, is_seg=is_seg)

    for axis in SLICING_AXIS:
        # print(file_name, axis)
        for i in range(nii_np.shape[AX_IDX[axis]]):
            slice = get_slice(nii_np, axis, i)
            save_slice(slice, i, slice_folder + f"_{axis}/", file_name, is_seg)


if __name__ == "__main__":

    for entry in MAPPING:
        nii_folder = PROJECT_FOLDER + entry["in"]
        slice_folder = PROJECT_FOLDER + entry["out"]
        is_seg = entry["seg"]

        files = os.listdir(nii_folder)
        files = sorted(files)
        print(f"Processing {nii_folder} with {len(files)} files")

        # for test run:
        # slice_nii(files[1], nii_folder, slice_folder, is_seg)

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(
                    slice_nii,
                    file_name,
                    nii_folder,
                    slice_folder,
                    is_seg,
                )
                for file_name in files
            ]

            # progress
            with tqdm(total=len(files)) as pbar:
                for future in as_completed(futures):
                    pbar.update(1)

    print("Done")
