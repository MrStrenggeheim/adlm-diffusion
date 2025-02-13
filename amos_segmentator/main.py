import argparse
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

# cutom imports from ../utils
sys.path.append("..")

from model import SegmentationModel, SegmentationModelConfig

from utils.dataset.amos import AMOSDataModule
from utils.utils import get_last_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="amos_segmentator")
    parser.add_argument("--log_dir", type=str, default="tb_logs")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--seg_dir", type=str, required=True)
    parser.add_argument("--transforms", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--include_bg", type=bool, default=False)
    parser.add_argument(
        "--model_type", type=str, choices=["unet", "pretrained"], default="unet"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tv_weight", type=float, default=0.05)
    parser.add_argument("--focal_weight", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=100)
    args = parser.parse_args()

    # tech setup
    pl.seed_everything(0)
    torch.set_float32_matmul_precision("high")  # allow tensor cores

    dataset_module = AMOSDataModule(
        args.img_dir,
        args.seg_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        transforms=args.transforms,
        num_workers=args.num_workers,
    )

    logger = TensorBoardLogger(args.log_dir, name=args.exp_name)
    logger.log_hyperparams(vars(args))

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="amos_segmentator_checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        save_last=True,
    )

    cfg = SegmentationModelConfig.from_args(args)
    model = SegmentationModel(cfg)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=16,
        val_check_interval=0.25,
        # check_val_every_n_epoch=10,
        limit_val_batches=0.25,
        callbacks=[checkpoint_callback],
        fast_dev_run=False,
    )

    summary(model, (1, args.img_size, args.img_size), depth=10)

    last_ckpt = None
    if args.resume:
        last_ckpt = get_last_checkpoint(os.path.join(args.log_dir, args.exp_name))
        if last_ckpt:
            print(f"Resuming from checkpoint: {last_ckpt}")
        else:
            print("No checkpoint found. Training from scratch.")
    trainer.fit(model, dataset_module, ckpt_path=last_ckpt)


# archive #####################
# torch.save(model.state_dict(), "amos_segmentator.pth")
# self.entropy_loss = nn.CrossEntropyLoss(
#     # wight bg less
#     # weight=torch.from_numpy(np.array([0.1] + [1.0] * 72)).float()
# )
# targets_onehot = (
#     torch.nn.functional.one_hot(targets.long().squeeze(), num_classes=73)
#     .permute(0, 3, 1, 2)
#     .float()
# )
# )
# )
# )
# )
# )
# )
# )
