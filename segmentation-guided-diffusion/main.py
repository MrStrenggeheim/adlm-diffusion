import os
from argparse import ArgumentParser

# HF imports
import diffusers
import numpy as np
import pandas as pd

# torch imports
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from eval import evaluate_generation, evaluate_sample_many
from torch import nn

# custom imports
from training import TrainingConfig, train_loop

from utils.amos import AmosDataset


def main(
    mode,  # train val
    img_size,  # used for transform
    num_img_channels,  # 1, 3, or load as array
    load_images_as_np_arrays,  # if True, load img and seg as tensor. expect .pt, using torch.load
    dataset,  # name
    img_dir,
    seg_dir,
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
    # image name filter
    if img_name_filter is not None:
        img_name_filter = pd.read_csv(img_name_filter, header=None)[0].tolist()

    # set images to include
    if img_type == "CT":
        index_range = (0, 500)
    elif img_type == "MRI":
        index_range = (500, 600)
    else:
        index_range = index_range

    # GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on {}".format(device))

    # load config
    output_dir = "output/{}-{}-{}-{}-{}".format(
        model_type.lower(),
        dataset,
        img_size,
        num_img_channels,
        segmentation_ingestion_mode,
    )  # the model namy locally and on the HF Hub
    if segmentation_guided:
        output_dir += "-segguided"
        assert (
            seg_dir is not None
        ), "must provide segmentation directory for segmentation guided training/sampling"

    if use_ablated_segmentations or eval_mask_removal or eval_blank_mask:
        output_dir += "-ablated"

    print("output dir: {}".format(output_dir))

    if mode == "train":
        evalset_name = "val"
        assert img_dir is not None, "must provide image directory for training"
    elif "eval" in mode:
        evalset_name = "test"

    print("using evaluation set: {}".format(evalset_name))

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

    dataset_train = AmosDataset(
        img_dir,
        seg_dir,
        "train",
        num_img_channels=num_img_channels,
        img_size=img_size,
        transforms=transforms,
        index_range=index_range,
        img_name_filter=img_name_filter,
        load_images_as_np_arrays=load_images_as_np_arrays,
    )
    dataset_eval = AmosDataset(
        img_dir,
        seg_dir,
        "val",
        num_img_channels=num_img_channels,
        img_size=img_size,
        transforms=transforms,
        index_range=index_range,
        img_name_filter=img_name_filter,
        load_images_as_np_arrays=load_images_as_np_arrays,
    )

    if (img_dir is None) and (not segmentation_guided):
        train_dataloader = None
        # just make placeholder dataloaders to iterate through when sampling from uncond model
        eval_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.zeros(
                    config.eval_batch_size,
                    num_img_channels,
                    config.image_size,
                    config.image_size,
                )
            ),
            batch_size=config.eval_batch_size,
            shuffle=eval_shuffle_dataloader,
            num_workers=16,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=16,
            prefetch_factor=16,
        )

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=config.eval_batch_size,
            shuffle=eval_shuffle_dataloader,
            num_workers=16,
            prefetch_factor=16,
        )

    # define the model
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

    if (mode == "train" and resume) or "eval" in mode:
        if os.path.exists(config.output_dir):
            epoch_folders = [f for f in os.listdir(config.output_dir) if "epoch" in f]
            if len(epoch_folders) > 0:
                last_ckpt_folder = sorted(epoch_folders)[-1]
                config.start_epoch = int(last_ckpt_folder.split("_")[-2]) + 1
                if mode == "train":
                    print(
                        "resuming from last model from folder: {}".format(
                            last_ckpt_folder
                        )
                    )
                elif "eval" in mode:
                    print(
                        "loading saved model from folder: {}".format(last_ckpt_folder)
                    )

                model = model.from_pretrained(
                    os.path.join(config.output_dir, last_ckpt_folder, "unet"),
                    use_safetensors=True,
                )
            else:
                print(f"no saved model found in {config.output_dir}")
        else:
            print(f"folder not found: {config.output_dir}")

    model = nn.DataParallel(model)
    model.to(device)

    # define noise scheduler
    if model_type == "DDPM":
        noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    elif model_type == "DDIM":
        noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)

    if mode == "train":
        # training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )

        # train
        train_loop(
            config,
            model,
            noise_scheduler,
            optimizer,
            train_dataloader,
            eval_dataloader,
            lr_scheduler,
            device=device,
        )
    elif mode == "eval":
        """
        default eval behavior:
        evaluate image generation or translation (if for conditional model, either evaluate naive class conditioning but not CFG,
        or with CFG),
        possibly conditioned on masks.

        has various options.
        """
        evaluate_generation(
            config,
            model,
            noise_scheduler,
            eval_dataloader,
            eval_mask_removal=eval_mask_removal,
            eval_blank_mask=eval_blank_mask,
            device=device,
        )

    elif mode == "eval_many":
        """
        generate many images and save them to a directory, saved individually
        """
        evaluate_sample_many(
            eval_sample_size,
            config,
            model,
            noise_scheduler,
            eval_dataloader,
            device=device,
        )

    else:
        raise ValueError('mode "{}" not supported.'.format(mode))


if __name__ == "__main__":
    # parse args:
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_img_channels", type=int, default=1)
    parser.add_argument("--load_images_as_np_arrays", action="store_true")
    parser.add_argument("--dataset", type=str, default="breast_mri")
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--seg_dir", type=str, default=None)
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
        args.mode,
        args.img_size,
        args.num_img_channels,
        args.load_images_as_np_arrays,
        args.dataset,
        args.img_dir,
        args.seg_dir,
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
