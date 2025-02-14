import argparse
import os
from itertools import batched

import diffusers
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as tt
from eval import SegGuidedDDIMPipeline, SegGuidedDDPMPipeline
from PIL import Image
from skimage import io
from torch import nn
from training import TrainingConfig

from utils.amos import AmosDataset, parse_transforms
from utils.utils import make_grid


def main(
    img_dir,
    seg_dir,
    output_dir,
    ckpt_path,
    num_eval_batches,
    num_preds_per_seg,
    img_size,  # used for transform
    num_img_channels,  # 1, 3, or load as array
    load_images_as_np_arrays,  # if True, load img and seg as tensor. expect .pt, using torch.load
    dataset,  # name
    img_name_filter,  # list with allowed image names for custom filtering (slices, indexes, min_amount of labels)
    img_type,  # CT or MRI
    index_range,
    model_type,  # DDPM DDIM
    transforms,
    segmentation_guided,
    segmentation_ingestion_mode,  # concat, add, mul, replace
    segmentation_channel_mode,  # single or multi
    num_segmentation_classes,
    train_batch_size,
    eval_batch_size,
    num_epochs=50,
    resume=False,
    use_ablated_segmentations=False,
    eval_shuffle_dataloader=True,
    # arguments only used in eval
    eval_mask_removal=False,
    eval_blank_mask=False,
    eval_sample_size=1000,
):
    pl.seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("running on {}".format(device))

    config = TrainingConfig(
        image_size=img_size,
        dataset=dataset,
        segmentation_guided=segmentation_guided,
        segmentation_ingestion_mode=segmentation_ingestion_mode,
        segmentation_channel_mode=segmentation_channel_mode,
        num_segmentation_classes=num_segmentation_classes,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs,
        output_dir=output_dir,
        model_type=model_type,
        resume=resume,
        use_ablated_segmentations=use_ablated_segmentations,
    )
    # _, transforms = parse_transforms(transforms, num_img_channels, img_size)

    print(f"Loading model from {ckpt_path}")
    in_channels = num_img_channels
    if config.segmentation_guided:
        assert config.num_segmentation_classes is not None
        assert (
            config.num_segmentation_classes > 1
        ), "must have at least 2 segmentation classes (INCLUDING background)"
        if config.segmentation_channel_mode == "single":
            if segmentation_ingestion_mode == "concat":
                in_channels += 1
        elif config.segmentation_channel_mode == "multi":
            raise NotImplementedError("multi-channel segmentation not implemented yet")
    model = diffusers.UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=in_channels,  # the number of input channels, 3 for RGB images
        out_channels=num_img_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(
            128,
            128,
            256,
            256,
            512,
            512,
        ),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    model = model.from_pretrained(ckpt_path, use_safetensors=True)
    model = nn.DataParallel(model)
    model.to(device)

    # define noise scheduler
    if model_type == "DDPM":
        noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    elif model_type == "DDIM":
        noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)

    if config.model_type == "DDPM":
        if config.segmentation_guided:
            pipeline = SegGuidedDDPMPipeline(
                unet=model.module,
                scheduler=noise_scheduler,
                eval_dataloader=None,
                external_config=config,
            )
        else:
            pipeline = diffusers.DDPMPipeline(
                unet=model.module, scheduler=noise_scheduler
            )
    elif config.model_type == "DDIM":
        if config.segmentation_guided:
            pipeline = SegGuidedDDIMPipeline(
                unet=model.module,
                scheduler=noise_scheduler,
                eval_dataloader=None,
                external_config=config,
            )
        else:
            pipeline = diffusers.DDIMPipeline(
                unet=model.module, scheduler=noise_scheduler
            )

        # set images to include
    if img_type == "CT":
        index_range = (0, 500)
    elif img_type == "MRI":
        index_range = (500, 600)
    else:
        index_range = index_range

    dataset = AmosDataset(
        img_dir,
        seg_dir,
        "test",
        num_img_channels=num_img_channels,
        img_size=img_size,
        transforms=transforms,
        index_range=index_range,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=eval_batch_size, shuffle=True, num_workers=4
    )
    os.makedirs(os.path.join(output_dir, "img_gen"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "img"), exist_ok=True)

    print(
        f"Starting inference of {num_eval_batches} batches of {eval_batch_size} images"
    )

    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= num_eval_batches:
            break
        # print([(k, batch[k].shape) for k in batch.keys()])
        # print([np.unique(batch[k].numpy()) for k in batch.keys()])
        images_list = []
        for i in range(num_preds_per_seg):
            print(f"Generating pass {i} for batch {batch_idx}")
            if config.segmentation_guided:
                images = pipeline(
                    batch_size=eval_batch_size,
                    seg_batch=batch,
                ).images
            else:
                images = pipeline(
                    batch_size=eval_batch_size,
                ).images
            images_list.append(images)
        # list of list PIL Image (list of batches)
        # (N, B, C, H, W) -> (B, N, C, H, W)
        images_list = list(zip(*images_list))

        for i, pred in enumerate(images_list):  # for each batch
            grid = make_grid(pred, 1, num_preds_per_seg)
            grid.save(
                os.path.join(output_dir, "img_gen", f"{batch_idx}_{i}_pred.png"),
            )
        for i, seg in enumerate(batch["images_target_raw"]):
            tt.ToPILImage()(seg).save(
                os.path.join(output_dir, "seg", f"{batch_idx}_{i}_seg.png"),
            )
        for i, img in enumerate(batch["images"]):
            torchvision.utils.save_image(
                img,
                os.path.join(output_dir, "img", f"{batch_idx}_{i}_img.png"),
                normalize=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for Seg Guided Diffusion"
    )
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--seg_dir", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the unet folder",
    )
    parser.add_argument(
        "--num_eval_batches",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_preds_per_seg",
        type=int,
        default=1,
    )
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_img_channels", type=int, default=1)
    parser.add_argument("--load_images_as_np_arrays", action="store_true")
    parser.add_argument("--dataset", type=str, default="breast_mri")
    parser.add_argument("--img_name_filter", type=str, default=None)
    parser.add_argument("--img_type", type=str, default=None)
    parser.add_argument("--index_range", type=int, nargs=2, default=None)
    parser.add_argument("--model_type", type=str, default="DDPM")
    parser.add_argument("--transforms", type=str, default=None)
    parser.add_argument(
        "--segmentation_guided",
        action="store_true",
        help="use segmentation guided training/sampling?",
    )
    parser.add_argument(
        "--segmentation_ingestion_mode",
        type=str,
        default="concat",
        help="concat, add, mul, replace",
    )
    parser.add_argument(
        "--segmentation_channel_mode",
        type=str,
        default="single",
        help="single == all segmentations in one channel, multi == each segmentation in its own channel",
    )
    parser.add_argument(
        "--num_segmentation_classes",
        type=int,
        default=None,
        help="number of segmentation classes, including background",
    )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from last saved model",
    )

    # novel options
    parser.add_argument(
        "--use_ablated_segmentations",
        action="store_true",
        help="use mask ablated training and any evaluation? sometimes randomly remove class(es) from mask during training and sampling.",
    )

    # other options
    parser.add_argument(
        "--eval_noshuffle_dataloader",
        action="store_true",
        help="if true, don't shuffle the eval dataloader",
    )

    # args only used in eval
    parser.add_argument(
        "--eval_mask_removal",
        action="store_true",
        help="if true, evaluate gradually removing anatomies from mask and re-sampling",
    )
    parser.add_argument(
        "--eval_blank_mask",
        action="store_true",
        help="if true, evaluate sampling conditioned on blank (zeros) masks",
    )
    parser.add_argument(
        "--eval_sample_size",
        type=int,
        default=1000,
        help="number of images to sample when using eval_many mode",
    )

    args = parser.parse_args()

    main(
        args.img_dir,
        args.seg_dir,
        args.output_dir,
        args.ckpt_path,
        args.num_eval_batches,
        args.num_preds_per_seg,
        args.img_size,
        args.num_img_channels,
        args.load_images_as_np_arrays,
        args.dataset,
        args.img_name_filter,
        args.img_type,
        args.index_range,
        args.model_type,
        args.transforms,
        args.segmentation_guided,
        args.segmentation_ingestion_mode,
        args.segmentation_channel_mode,
        args.num_segmentation_classes,
        args.train_batch_size,
        args.eval_batch_size,
        args.num_epochs,
        args.resume,
        args.use_ablated_segmentations,
        not args.eval_noshuffle_dataloader,
        # args only used in eval
        args.eval_mask_removal,
        args.eval_blank_mask,
        args.eval_sample_size,
    )
