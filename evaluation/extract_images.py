"""This helper script extracts images in a grid into separate images."""

# %%
import os
import re

from PIL import Image


def get_images_from_grid(grid, grid_layout, img_size, padding=0):
    images = []
    for i in range(grid_layout[0]):
        for j in range(grid_layout[1]):
            x = i * (img_size + padding) + padding
            y = j * (img_size + padding) + padding
            images.append(grid.crop((x, y, x + img_size, y + img_size)))
    return images


model_folder = "/vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/ddim_amos_ct_allseg/256_40/"
grid_layout = (4, 4)
img_size = 256
all_folder = model_folder + "all"
sample_folder = model_folder + "img_gen"
origin_folder = model_folder + "img"
condit_folder = model_folder + "seg"
os.makedirs(sample_folder, exist_ok=True)
os.makedirs(origin_folder, exist_ok=True)
os.makedirs(condit_folder, exist_ok=True)
files = os.listdir(all_folder)
r = re.compile(r"\d+-\d+.png")
files = [f for f in files if r.match(f)]
files.sort()
len(files)
for file in files:
    sample_grid = Image.open(os.path.join(all_folder, file))
    origin_grid = Image.open(
        os.path.join(all_folder, file.replace(".png", "_orig.png"))
    )
    condit_grid = Image.open(
        os.path.join(all_folder, file.replace(".png", "_cond.png"))
    )

    # split images
    sample_images = get_images_from_grid(sample_grid, grid_layout, img_size)
    origin_images = get_images_from_grid(origin_grid, grid_layout, img_size, padding=2)
    condit_images = get_images_from_grid(condit_grid, grid_layout, img_size, padding=2)

    # save images
    for i in range(grid_layout[0] * grid_layout[1]):
        sample_images[i].save(
            os.path.join(sample_folder, file.replace(".png", f"_sample_{i}.png"))
        )
        origin_images[i].save(
            os.path.join(origin_folder, file.replace(".png", f"_origin_{i}.png"))
        )
        condit_images[i].save(
            os.path.join(condit_folder, file.replace(".png", f"_condit_{i}.png"))
        )
