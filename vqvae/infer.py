import os

# add .. to path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import yaml
from main import VQVAETrainingModule
from torchvision import transforms
from tqdm import tqdm
from utils.amos import AmosDataset

from utils.utils import load_amos

# sys.path.append("..")

# change dir to parent
# os.chdir(os.path.dirname(os.getcwd()))


# Load configuration from YAML
img_config = os.getcwd() + "/config.yaml"
seg_config = os.getcwd() + "/config_labels.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")
with open(img_config, "r") as f:
    img_config = yaml.safe_load(f)
with open(seg_config, "r") as f:
    seg_config = yaml.safe_load(f)
image_vqvae_ckpt = "/vol/aimspace/projects/practical_WS2425/diffusion/code/vqvae/runs/vqvae/vqvae-32x32x4-amos-images/checkpoints/last.ckpt"
label_vqvae_ckpt = "/vol/aimspace/projects/practical_WS2425/diffusion/code/vqvae/runs/vqvae_labels/vqvae-32x32x4-amos-labels/checkpoints/last.ckpt"

seg_vae = VQVAETrainingModule.load_from_checkpoint(label_vqvae_ckpt).to(device).eval()
img_vae = VQVAETrainingModule.load_from_checkpoint(image_vqvae_ckpt).to(device).eval()
img_encoder = img_vae.model.encoder
seg_encoder = seg_vae.model.encoder
img_preqconv = img_vae.model.pre_quantization_conv
seg_preqconv = seg_vae.model.pre_quantization_conv
img_vecquant = img_vae.model.vector_quantization
seg_vecquant = seg_vae.model.vector_quantization
img_embedder = torch.nn.Sequential(img_encoder, img_preqconv, img_vecquant)
seg_embedder = torch.nn.Sequential(seg_encoder, seg_preqconv, seg_vecquant)

in_dir = "/vol/aimspace/projects/practical_WS2425/diffusion/data/amos_robert_slices/"
out_dir = (
    "/vol/aimspace/projects/practical_WS2425/diffusion/data/amos_robert_embeddings/"
)
splits = ["train", "val"]


def embed_batch(batch):
    img, seg, names = batch["image"], batch["label"], batch["name"]
    img = img.to(device)
    seg = seg.to(device)
    img_embedding = img_embedder(img)
    seg_embedding = seg_embedder(seg)

    for i, s, n in zip(img_embedding, seg_embedding, names):
        torch.save(
            i.cpu().detach(),
            os.path.join(out_dir, "images", split, f"{n}.pt"),
        )
        torch.save(
            s.cpu().detach(),
            os.path.join(out_dir, "labels", split, f"{n}.pt"),
        )


if __name__ == "__main__":

    for split in splits:
        print(split)
        ds = AmosDataset(
            os.path.join(in_dir),
            split,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(512),
                    transforms.CenterCrop(512),
                    transforms.Normalize(0.5, 0.5),
                ]
            ),
            index_range=(0, 500),
            slice_range=None,
            only_labeled=False,
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

        print(f"Embedding {len(dl)} batches")
        for batch in tqdm(dl):
            embed_batch(batch)
        # with ThreadPoolExecutor(max_workers=32) as executor:
        #     with tqdm(total=len(dl)) as pbar:
        #         for batch in dl:
        #             future = executor.submit(embed_batch, batch)
        #             future.result()  # Wait for the task to finish
        #             pbar.update(1)
        # with ThreadPoolExecutor(max_workers=1) as executor:
        #     futures = [executor.submit(embed_batch, batch) for batch in dl]

        #     # progress
        #     with tqdm(total=len(dl)) as pbar:
        #         for future in as_completed(futures):
        #             pbar.update(1)

    print("Done")
