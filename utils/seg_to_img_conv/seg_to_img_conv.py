# %%
import argparse
import json
import os

import matplotlib as mpl
import numpy as np
import TPTBox as tpt
from matplotlib import pyplot as plt
from PIL import Image

# from TPTBox.mesh3D.mesh_colors import get_color_by_label

# add path as argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--normalize", action="store_true")
args = parser.parse_args()

images = [
    f
    for f in os.listdir(args.path)
    if f.endswith(".png") and "seg" in f and not "_rgb" in f
]
print(f"Converting {len(images)} images")
colors = json.load(
    open("/vol/miltank/projects/practical_WS2425/diffusion/code/eda/label_info.json")
)["colors"]
colors = list(map(tuple, colors))


def convert_image(file):
    img = np.array(Image.open(os.path.join(args.path, file)).convert("L"))
    if any(img.flatten() > 72):
        print(f"Image {file} has values > 255. Normalizing")
        args.normalize = True
    if args.normalize:
        img = np.rint(img / 255 * 72).astype(np.uint8)
    img = np.array(np.vectorize(lambda x: colors[x])(img))
    img = img.astype(np.uint8).transpose(1, 2, 0)
    img = Image.fromarray(img, "RGB")
    img.save(os.path.join(args.path, file.replace(".png", "_rgb.png")))


for img in images:
    convert_image(img)
