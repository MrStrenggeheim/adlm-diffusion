# %%
import json
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image
from TPTBox import NII

# mri = Image.open(
#     "/vol/miltank/projects/practical_WS2425/diffusion/code/segmentation-guided-diffusion/output/ddim-amos_mri_all_axis-256-1-concat-segguided/samples/0003-1202_orig.png"
# )
# ct = Image.open(
#     "/vol/miltank/projects/practical_WS2425/diffusion/code/segmentation-guided-diffusion/output/ddim-amos_ct_all_axis-256-1-concat-segguided/samples/0000-0000_orig.png"
# )

# mri_np = np.array(mri).flatten()  # / np.max(np.array(mri))
# ct_np = np.array(ct).flatten()  # / np.max(np.array(ct))

# plt.hist(mri_np, bins=256, range=(0, 255), density=True, color="r", alpha=0.7)
# plt.hist(ct_np, bins=256, range=(0, 255), density=True, color="b", alpha=0.7)

# # plt cut y axis
# plt.ylim(0, 0.03)
# # plt.ylim(0, 10)

# %%
## Load Data
project_folder = "/vol/miltank/projects/practical_WS2425/diffusion/"
img_nii_dir = project_folder + "data/amos22/"
seg_nii_dir = project_folder + "data/amos_robert/"
seg_nii_dir = project_folder + "data/amos22/"
with open(project_folder + "data/amos22/dataset.json") as f:
    info = json.load(f)
info.keys()
info["labels"]
labels_dict = {
    "0": "background, Hintergrund",
    "1": "spleen, Milz",
    "2": "right kidney, rechte Niere",
    "3": "left kidney, linke Niere",
    "4": "gall bladder, Gallenblase",
    "5": "esophagus, Speiseröhre",
    "6": "liver, Leber",
    "7": "stomach, Magen",
    "8": "arota, Aorta",
    "9": "postcava, Hohlvene",
    "10": "pancreas, Bauchspeicheldrüse",
    "11": "right adrenal gland, rechte Nebenniere",
    "12": "left adrenal gland, linke Nebenniere",
    "13": "duodenum, Zwölffingerdarm",
    "14": "bladder, Blase",
    "15": "prostate/uterus, Prostata/Gebärmutter",
}
# labels_dict = json.loads(open("label_info.json").read())["german"]
# nii_idx = 549
# split="Va"
# slice_idx=100
# slice_direction = 'axial'


ct_idx = 1
mri_idx = 510
split = "Tr"

ct_files = sorted(os.listdir(img_nii_dir + "/images" + split))[:1]
mri_files = sorted(os.listdir(img_nii_dir + "/images" + split))[-1:]

# %%
ct = [
    NII.load(img_nii_dir + "/images" + split + "/" + file, seg=False)
    .get_array()
    .flatten()
    for file in ct_files
]
mri = [
    NII.load(img_nii_dir + "/images" + split + "/" + file, seg=False)
    .get_array()
    .flatten()
    for file in mri_files
]

# %%
ct_np = np.concatenate(ct)
mri_np = np.concatenate(mri)

ct_np.shape, mri_np.shape
# %%
import seaborn as sns

ct_np_sample = ct_np[np.random.choice(ct_np.shape[0], 10000, replace=False)]
mri_np_sample = mri_np[np.random.choice(mri_np.shape[0], 10000, replace=False)]

dist = {"ct": ct_np_sample, "mri": mri_np_sample}
# %%
sns.set_style("white")
k = sns.kdeplot(data=dist, fill=True)
k.set(xlim=(-1024, 1024), ylim=(0, 0.003))


# %%

plt.hist(ct_np, bins=256, range=(-1024, 1024), density=True, color="r", alpha=0.7)
plt.hist(mri_np, bins=256, range=(-1024, 1024), density=True, color="b", alpha=0.7)
plt.ylim(0, 0.007)

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

# plot the same data on both Axes
ax1.hist(ct_np, bins=256, range=(-1024, 1024), density=True, color="r", alpha=0.7)
ax1.hist(mri_np, bins=256, range=(-1024, 1024), density=True, color="b", alpha=0.7)
ax2.hist(ct_np, bins=256, range=(-1024, 1024), density=True, color="r", alpha=0.7)
ax2.hist(mri_np, bins=256, range=(-1024, 1024), density=True, color="b", alpha=0.7)

# zoom-in / limit the view to different portions of the data
ax1.set_ylim(0.002, 0.05)  # outliers only
ax2.set_ylim(0, 0.002)  # most of the data

# hide the spines between ax1 and ax2
ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)


plt.show()

# %%
import SimpleITK as sitk

img = sitk.ReadImage(
    img_nii_dir + "/images" + split + "/" + mri_files[-1],
    outputPixelType=sitk.sitkFloat32,
)
img_np = sitk.GetArrayFromImage(img)
img_np = np.clip(img_np, 0, 1024)
plt.imshow(img_np[50], cmap="gray")
# np.unique(img_np)
# %%
corrector = sitk.N4BiasFieldCorrectionImageFilter()
img_cor = corrector.Execute(img)
# %%

img_before = sitk.GetArrayFromImage(img)
img_after = sitk.GetArrayFromImage(img_cor)

# show images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_before[0], cmap="gray", vmin=0)
plt.title("Before")
plt.subplot(1, 2, 2)
plt.imshow(img_after[0], cmap="gray", vmin=0)
plt.title("After")

# show histograms
plt.figure(figsize=(10, 5))
plt.hist(
    img_before.flatten(),
    bins=256,
    range=(0, 500),
    density=True,
    color="r",
    alpha=0.7,
)
plt.hist(
    img_after.flatten(),
    bins=256,
    range=(0, 500),
    density=True,
    color="b",
    alpha=0.7,
)
plt.ylim(0, 0.007)


import matplotlib.pyplot as plt

# %%
import numpy as np

# Create bins spanning the range (-256, 256)
bins = np.linspace(-256, 256, 512)

# Generate synthetic data to mimic typical MRI intensity distributions

# T1-weighted: bimodal (e.g., dark CSF and bright white matter)
t1_data = np.concatenate(
    [
        np.random.normal(-50, 20, 1000),  # lower intensity peak (CSF)
        np.random.normal(150, 30, 1500),  # higher intensity peak (white matter)
    ]
)

# T2-weighted: shifted to higher intensities due to bright CSF
t2_data = np.random.normal(150, 30, 2500)

# Proton Density: narrower, centered distribution
pd_data = np.random.normal(50, 10, 2500)

# FLAIR: similar to T2 but with suppressed CSF signal (less extreme high values)
flair_data = np.concatenate(
    [
        np.random.normal(100, 20, 1500),  # tissue signal
        np.random.normal(50, 20, 500),  # residual lower intensities
    ]
)

# STIR: fat suppression; assume two peaks for demonstration
stir_data = np.concatenate(
    [
        np.random.normal(0, 20, 1000),  # suppressed fat
        np.random.normal(80, 20, 1500),  # other tissue
    ]
)

# Plot the histograms
plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.hist(t1_data, bins=bins, color="blue", alpha=0.7)
plt.title("T1-weighted")

plt.subplot(232)
plt.hist(t2_data, bins=bins, color="red", alpha=0.7)
plt.title("T2-weighted")

plt.subplot(233)
plt.hist(pd_data, bins=bins, color="green", alpha=0.7)
plt.title("Proton Density")

plt.subplot(234)
plt.hist(flair_data, bins=bins, color="purple", alpha=0.7)
plt.title("FLAIR")

plt.subplot(235)
plt.hist(stir_data, bins=bins, color="orange", alpha=0.7)
plt.title("STIR")

plt.tight_layout()
plt.show()


# %%
img = Image.open(
    "/vol/miltank/projects/practical_WS2425/diffusion/data/amos_slices/imagesTr_axial/amos_0570_s210.png"
)
plt.imshow(img, cmap="gray")


# %%
img = Image.open(
    "/vol/miltank/projects/practical_WS2425/diffusion/data/amos_slices/imagesTr_axial/amos_0571_s003.png"
)
# plt.hist(np.array(img).flatten(), bins=256, range=(0, 255), density=True, color="r", alpha=0.7)
# plt.imshow(img, cmap="gray", vmin=10, vmax=255)
# img

# get 1 and 99 percentile
np_img = np.array(img)
one_percentile = np.percentile(np_img, 1)
ninety_nine_percentile = np.percentile(np_img, 99)
one_percentile, ninety_nine_percentile

# normalize image
np_img = (np_img - one_percentile) / (ninety_nine_percentile - one_percentile)
np_img = np.clip(np_img, 0, 1)
plt.imshow(np_img, cmap="gray")
