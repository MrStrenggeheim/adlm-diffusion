# %%
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as tt
from model import SegmentationModel
from PIL import Image
from skimage import io

# %%

parser = argparse.ArgumentParser(description="Inference script for AMOS segmentator")
parser.add_argument(
    "--input_dir", type=str, required=True, help="Path to the input directory"
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Path to the output directory"
)
parser.add_argument(
    "--ckpt_path", type=str, required=True, help="Path to the checkpoint file"
)
parser.add_argument(
    "--img_size", type=int, default=256, help="Size of the input images"
)
args = parser.parse_args()
# args = argparse.Namespace(
#     input_dir="/vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/amos_ct_all_axis/256_40/img",
#     output_dir="/vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/amos_ct_all_axis/256_40/seg_pred",
#     ckpt_path="/vol/miltank/projects/practical_WS2425/diffusion/code/amos_segmentator/logs/final/version_6/checkpoints/amos_segmentator_checkpoint-epoch=04-val_loss=0.17.ckpt",
#     img_size=256,
# )

device = "cuda" if torch.cuda.is_available() else "cpu"

pl.seed_everything(0)

print(f"Loading model from {args.ckpt_path}")
model = SegmentationModel.load_from_checkpoint(args.ckpt_path).to(device)

# %%
# path = "/vol/miltank/projects/practical_WS2425/diffusion/code/evaluation/input/ddim_amos_ct_allseg/256_40/all/0040-0000"
# a = path + "_cond.png"
# b = path + "_orig.png"
# c = path + ".png"
# a, b, c = Image.open(a), Image.open(b), Image.open(c)
# a, b, c = a.convert("L"), b.convert("L"), c.convert("L")
# a.size, b.size, c.size
# %%

# Load the input data
transforms = tt.Compose(
    [
        tt.ToTensor(),
        tt.Resize(args.img_size),
        tt.Normalize(mean=[0.5], std=[0.5]),
    ]
)
print(f"Loading images from {args.input_dir}")
image_files = os.listdir(args.input_dir)
images = [Image.open(os.path.join(args.input_dir, f)).convert("L") for f in image_files]
print(f"Loaded {len(images)} images")
print(f"shape: {images[0].size}")

# %%
images[0]

# %%

preds = []
# Perform inference
for img in images:
    img = transforms(img).to(device)
    img = img.unsqueeze(0)
    pred = model(img)
    pred = pred.squeeze(0)
    pred = pred.detach().cpu().numpy()
    pred = np.argmax(pred, axis=0, keepdims=False)
    preds.append(pred)


# %%
# Save the predictions
print(f"Saving predictions to {args.output_dir}")
os.makedirs(args.output_dir, exist_ok=True)
for pred, image_file in zip(preds, image_files):
    Image.fromarray(pred.astype(np.uint8)).save(
        os.path.join(args.output_dir, f"{image_file}"),
    )


# %%
