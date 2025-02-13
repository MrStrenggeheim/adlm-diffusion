import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from monai.losses import DiceLoss, FocalLoss
from monai.networks.nets import UNet
from torchvision.utils import make_grid

# cutom imports from ../utils
sys.path.append("..")
from dataclasses import dataclass
from typing import Optional

from utils.utils import tv_loss


@dataclass
class SegmentationModelConfig:
    include_bg: bool = False
    model_type: str = "unet"  # Valid values: "unet", "pretrained"
    lr: float = 1e-4
    tv_weight: float = 0.05
    focal_weight: float = 1.0
    weight_decay: float = 1e-2

    def from_args(args):
        return SegmentationModelConfig(
            **{
                k: v
                for k, v in vars(args).items()
                if k in SegmentationModelConfig.__dataclass_fields__
            }
        )


class SegmentationModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.lr = args.lr
        self.tv_weight = args.tv_weight
        self.focal_weight = args.focal_weight
        self.weight_decay = args.weight_decay
        self.dice_loss = DiceLoss(
            include_background=args.include_bg,
            to_onehot_y=True,
            softmax=True,
            reduction="mean",
        )
        self.focal_loss = FocalLoss(
            include_background=args.include_bg,
            to_onehot_y=True,
            use_softmax=True,
            reduction="mean",
        )
        self.tv_loss = tv_loss

        if args.model_type == "pretrained":
            self.model = models.segmentation.fcn_resnet50(pretrained=True)
            self.model.classifier[4] = nn.Conv2d(512, 73, kernel_size=1)
        else:
            self.model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=73,
                channels=(16, 32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2, 2),
                kernel_size=3,
                # dropout=0.1,
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # cosine scheduling, Tmax = number of epochs to reach min lr
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        return (
            self.model(x)["out"]
            if isinstance(self.model, models.segmentation.FCN)
            else self.model(x)
        )

    def _loss_calc(self, stage, outputs, targets):
        focal_loss = self.focal_loss(outputs, targets)
        dice_loss = self.dice_loss(outputs, targets)
        tv_loss = self.tv_loss(outputs)
        loss = dice_loss + focal_loss * self.focal_weight + tv_loss * self.tv_weight
        self.log(f"{stage}_dice_loss", dice_loss, prog_bar=True)
        self.log(f"{stage}_focal_loss", focal_loss, prog_bar=True)
        self.log(f"{stage}_tv_loss", tv_loss, prog_bar=True)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        images, targets = (batch["images"], batch["images_target"])
        outputs = self(images)
        return self._loss_calc("train", outputs, targets)

    def validation_step(self, batch, batch_idx):
        images, targets = batch["images"], batch["images_target"]
        outputs = self(images)
        loss = self._loss_calc("val", outputs, targets)

        # also save images
        if batch_idx == 0:
            self.log("lr", self.lr_schedulers().get_last_lr()[0])
            exp = self.trainer.logger.experiment
            # convert onehot back to class idx
            outputs = torch.argmax(outputs, dim=1, keepdim=True).float()
            imgage_grid = self._make_grid(images)
            target_grid = self._make_grid(targets.float())
            output_grid = self._make_grid(outputs)
            exp.add_image("input", imgage_grid, self.global_step)
            exp.add_image("target", target_grid, self.global_step)
            exp.add_image("output", output_grid, self.global_step)
        return loss

    def _make_grid(self, x):
        return make_grid(x[:16], normalize=True, scale_each=False, nrow=4)
