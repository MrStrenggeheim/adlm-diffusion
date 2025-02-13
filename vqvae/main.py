import argparse
import os
import random
import sys
import time
from collections import deque

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml
from models.vqvae import VQVAE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

# cutom imports from ../utils
sys.path.append("..")
from utils.dataset.amos import AmosDataset


# Define PyTorch Lightning Module
class VQVAETrainingModule(pl.LightningModule):
    def __init__(self, config):
        super(VQVAETrainingModule, self).__init__()
        self.save_hyperparameters(config)
        self.model = VQVAE(
            config["model"]["n_channel"],
            config["model"]["n_hiddens"],
            config["model"]["n_residual_hiddens"],
            config["model"]["n_residual_layers"],
            config["model"]["n_embeddings"],
            config["model"]["embedding_dim"],
            config["model"]["beta"],
        )
        self.losses: dict[str, deque] = {}
        self.train_on = "label" if config["dataset"]["train_on_labels"] else "image"

        summary(self.model, (config["model"]["n_channel"], 512, 512))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.train_on]
        embedding_loss, x_hat, perplexity = self.model(x)
        recon_loss = F.mse_loss(x_hat, x)
        loss = recon_loss + embedding_loss

        # Track losses
        train_loss = {"loss": loss, "recon_loss": recon_loss, "perplexity": perplexity}
        for key, val in train_loss.items():
            if key not in self.losses:
                self.losses[key] = deque(maxlen=300)
            self.losses[key].append(val.item())

        # Log metrics
        self.log_dict(
            {key: sum(val) / len(val) for key, val in self.losses.items()},
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self):
        pass  # No need to log reconstructions here, it is handled in validation

    def validation_step(self, batch, batch_idx):
        x = batch[self.train_on]
        embedding_loss, x_hat, perplexity = self.model(x)
        recon_loss = F.mse_loss(x_hat, x)
        loss = recon_loss + embedding_loss

        # Log validation metrics
        val_loss = {
            "val_loss": loss,
            "val_recon_loss": recon_loss,
            "val_perplexity": perplexity,
        }
        self.log_dict(
            {f"{key}": val.item() for key, val in val_loss.items()},
            sync_dist=True,
            prog_bar=True,
        )

        # Log reconstruction images to TensorBoard after each validation
        if batch_idx == 0:  # Log only the first batch
            exp = self.logger.experiment
            exp.add_image(
                "Validation_Reconstruction",
                self.reconstruct_images(batch),
                self.current_epoch,
            )

        return loss

    def reconstruct_images(self, batch):
        # Get sample reconstruction images
        x = batch[self.train_on]
        x = x.to(self.device)
        _, x_hat, _ = self.model(x)
        grid = vutils.make_grid(torch.cat([x[:8], x_hat[:8]]), nrow=8, normalize=True)
        return grid

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["training"]["learning_rate"],
            amsgrad=True,
        )


# Train the model
if __name__ == "__main__":
    # add_argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Load configuration from YAML
    CONFIG_PATH = os.getcwd() + "/" + args.config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # set seed
    seed = config["training"]["seed"]
    pl.seed_everything(seed)

    dataset_train = AmosDataset(
        config["dataset"]["img_dir"],
        config["dataset"]["seg_dir"],
        "train",
        num_img_channels=config["dataset"]["num_img_channels"],
        img_size=config["dataset"]["img_size"],
        transforms=config["dataset"]["transforms"],
    )
    dataset_eval = AmosDataset(
        config["dataset"]["img_dir"],
        config["dataset"]["seg_dir"],
        "val",
        num_img_channels=config["dataset"]["num_img_channels"],
        img_size=config["dataset"]["img_size"],
    )
    training_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    logger = TensorBoardLogger(
        config["logging"]["log_dir"], name=config["logging"]["experiment_name"]
    )
    logger.log_hyperparams(config)

    # Set up PyTorch Lightning Trainer with Early Stopping and Model Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename=f"{config['logging']['experiment_name']}-{logger.version}"
        + "-{epoch:02d}-{val_loss:.6f}",
        save_top_k=2,
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config["training"]["early_stopping_patience"],
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        callbacks=[
            checkpoint_callback,
            # early_stopping_callback
        ],
        log_every_n_steps=config["logging"]["log_interval"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,  # if torch.cuda.is_available() else None,
        logger=logger,
        val_check_interval=config["training"]["validation_interval"],
        fast_dev_run=False,
    )

    print(f"Using device {"gpu" if torch.cuda.is_available() else "cpu"}")

    vqvae_module = VQVAETrainingModule(config)

    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.fit(
            vqvae_module,
            training_loader,
            validation_loader,
            ckpt_path=args.resume,
        )
    else:
        print("Starting new training run")
        trainer.fit(vqvae_module, training_loader, validation_loader)
