import argparse
import json
import os

import numpy as np
from PIL import Image

# from TPTBox.mesh3D.mesh_colors import get_color_by_label

LABEL_INFO = "/vol/miltank/projects/practical_WS2425/diffusion/code/eda/label_info.json"


def convert_imagages_at(path, colors, normalize):
    images = [
        f
        for f in os.listdir(path)
        if f.endswith(".png") and "seg" in f and not "_rgb" in f
    ]
    print(f"Converting {len(images)} images at {path}")
    for img in images:
        img_path = os.path.join(path, img)
        img_conv = convert_image(img_path, colors, normalize)
        img_conv.save(img_path.replace(".png", "_rgb.png"))


def convert_image(img_path, colors, normalize):
    img = np.array(Image.open(img_path).convert("L"))
    if any(img.flatten() > 72):
        print(f"Image {img_path} has values > 255. Normalizing")
        normalize = True
    if normalize:
        img = np.rint(img / 255 * 72).astype(np.uint8)
    img = np.array(np.vectorize(lambda x: colors[x])(img))
    img = img.astype(np.uint8).transpose(1, 2, 0)
    img = Image.fromarray(img, "RGB")
    return img


def main():
    # add path as argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "-r", action="store_true", help="Recursively walk through the path"
    )
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    colors = json.load(open(LABEL_INFO))["colors"]
    colors = list(map(tuple, colors))

    if args.r:
        for root, _, _ in os.walk(args.path):
            convert_imagages_at(root, colors, args.normalize)
    else:
        convert_imagages_at(args.path, colors, args.normalize)


if __name__ == "__main__":
    main()
