"""
training utils
"""

import math
import os
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
from eval import (
    SegGuidedDDIMPipeline,
    SegGuidedDDPMPipeline,
    add_segmentations_to_noise,
    evaluate,
)
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


@dataclass
class TrainingConfig:
    model_type: str = "DDPM"
    image_size: int = 256  # the generated image resolution
    train_batch_size: int = 32
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_epochs: int = 200
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5  # was 1e-4
    lr_warmup_steps: int = 500
    save_model_epochs: int | float = (
        0.25  # save model every n epochs or every n% of total epochs
    )
    mixed_precision: str = (
        "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    )
    output_dir: str = None

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = (
        True  # overwrite the old model when re-running the notebook
    )
    seed: int = 0

    # custom options
    segmentation_guided: bool = False
    segmentation_ingestion_mode: str = "concat"
    segmentation_channel_mode: str = "single"
    num_segmentation_classes: int = None  # INCLUDING background
    use_ablated_segmentations: bool = False
    dataset: str = "breast_mri"
    resume: int = None
    start_epoch: int = 0
    cfg_p_uncond: float = 0.2  # p_uncond in classifier-free guidance paper
    cfg_weight: float = 0.3  # w in the paper
    trans_noise_level: float = (
        0.5  # ratio of time step t to noise trans_start_images to total T before denoising in translation. e.g. value of 0.5 means t = 500 for default T = 1000.
    )
    use_cfg_for_eval_conditioning: bool = (
        True  # whether to use classifier-free guidance for or just naive class conditioning for main sampling loop
    )
    cfg_maskguidance_condmodel_only: bool = (
        True  # if using mask guidance AND cfg, only give mask to conditional network
    )
    # ^ this is because giving mask to both uncond and cond model make class guidance not work
    # (see "Classifier-free guidance resolution weighting." in ControlNet paper)


def train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    eval_dataloader,
    lr_scheduler,
    device="cuda",
):
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.

    global_step = 0

    # logging
    run_name = "{}-{}-{}".format(
        config.model_type.lower(), config.dataset, config.image_size
    )
    if config.segmentation_guided:
        run_name += "-segguided"
    writer = SummaryWriter(comment=run_name)

    # for loading segs to condition on:
    eval_dataloader = iter(eval_dataloader)

    # Now you train the model
    for epoch in range(config.start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            clean_images = clean_images.to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if config.segmentation_guided:
                noisy_images = add_segmentations_to_noise(
                    noisy_images, batch, config, device
                )

            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            writer.add_scalar("loss", loss.detach().item(), global_step)

            progress_bar.set_postfix(**logs)
            global_step += 1

            if step % 500 == 0:
                eval_and_save(
                    config, model, noise_scheduler, eval_dataloader, epoch, step
                )

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        eval_and_save(config, model, noise_scheduler, eval_dataloader, epoch, step)


def eval_and_save(config, model, noise_scheduler, eval_dataloader, epoch, step):
    if config.model_type == "DDPM":
        if config.segmentation_guided:
            pipeline = SegGuidedDDPMPipeline(
                unet=model.module,
                scheduler=noise_scheduler,
                eval_dataloader=eval_dataloader,
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
                eval_dataloader=eval_dataloader,
                external_config=config,
            )
        else:
            pipeline = diffusers.DDIMPipeline(
                unet=model.module, scheduler=noise_scheduler
            )

    model.eval()

    if config.segmentation_guided:
        seg_batch = next(eval_dataloader)
        evaluate(config, epoch, pipeline, seg_batch, step=step)
    else:
        evaluate(config, epoch, pipeline, step=step)

    # exclude non serializable objects
    del pipeline.config.eval_dataloader
    del pipeline.config.external_config

    pipeline.save_pretrained(
        os.path.join(config.output_dir, f"epoch_{epoch:04d}_{step:04d}"),
        safe_serialization=True,
    )
    # save only last 3
    ckpt_list = sorted([f for f in os.listdir(config.output_dir) if "epoch" in f])
    if len(ckpt_list) > 3:
        for ckpt in ckpt_list[:-3]:
            shutil.rmtree(os.path.join(config.output_dir, ckpt), ignore_errors=True)
