"""
Modern Face Swap Model Training Script
Uses PyTorch with ONNX export capability
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm


class FaceSwapEncoder(nn.Module):
    """Encoder network for face swap"""

    def __init__(self, input_channels: int = 3, latent_dim: int = 512):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Flatten and FC layers
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1024 * 8 * 8, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, latent_dim),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class FaceSwapDecoder(nn.Module):
    """Decoder network for face swap"""

    def __init__(self, latent_dim: int = 512, output_channels: int = 3):
        super().__init__()

        # FC to reshape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024 * 8 * 8),
            nn.BatchNorm1d(1024 * 8 * 8),
            nn.ReLU(inplace=True),
        )

        self.reshape = lambda x: x.view(-1, 1024, 8, 8)

        # Deconvolutional layers
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final conv to get output channels
        self.final = nn.Sequential(
            nn.Conv2d(32, output_channels, 7, padding=3), nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.final(x)
        return x


class FaceSwapAutoencoder(nn.Module):
    """Complete face swap autoencoder"""

    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.encoder_A = FaceSwapEncoder(latent_dim=latent_dim)
        self.encoder_B = FaceSwapEncoder(latent_dim=latent_dim)
        self.decoder_A = FaceSwapDecoder(latent_dim=latent_dim)
        self.decoder_B = FaceSwapDecoder(latent_dim=latent_dim)

    def forward(self, x, person="A"):
        if person == "A":
            latent = self.encoder_A(x)
            output = self.decoder_A(latent)
        else:
            latent = self.encoder_B(x)
            output = self.decoder_B(latent)
        return output, latent

    def swap(self, x, source="A", target="B"):
        """Swap faces from source to target"""
        if source == "A":
            latent = self.encoder_A(x)
            output = self.decoder_B(latent)
        else:
            latent = self.encoder_B(x)
            output = self.decoder_A(latent)
        return output


class FaceDataset(Dataset):
    """Dataset for face images"""

    def __init__(self, root_dir: str, person: str, transform=None):
        self.root_dir = Path(root_dir) / person
        self.transform = transform
        self.images = list(self.root_dir.glob("*.jpg")) + list(
            self.root_dir.glob("*.png")
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


class FaceSwapTrainer:
    """Trainer for face swap model"""

    def __init__(self, model: nn.Module, device: torch.device, config: Dict[str, Any]):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # Optimizers
        self.optimizer_A = optim.Adam(
            [
                {"params": model.encoder_A.parameters()},
                {"params": model.decoder_A.parameters()},
            ],
            lr=config["learning_rate"],
            betas=(0.5, 0.999),
        )

        self.optimizer_B = optim.Adam(
            [
                {"params": model.encoder_B.parameters()},
                {"params": model.decoder_B.parameters()},
            ],
            lr=config["learning_rate"],
            betas=(0.5, 0.999),
        )

        # Loss functions
        self.criterion_recon = nn.L1Loss()
        self.criterion_perceptual = PerceptualLoss().to(device)

        # Logging
        self.writer = SummaryWriter(config["log_dir"])
        if config.get("use_wandb", False):
            wandb.init(project="ooblex-face-swap", config=config)

    def train_epoch(
        self, dataloader_A: DataLoader, dataloader_B: DataLoader, epoch: int
    ):
        """Train one epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(
            zip(dataloader_A, dataloader_B),
            total=min(len(dataloader_A), len(dataloader_B)),
        )
        for i, (batch_A, batch_B) in enumerate(pbar):
            batch_A = batch_A.to(self.device)
            batch_B = batch_B.to(self.device)

            # Train A -> A
            self.optimizer_A.zero_grad()
            output_A, latent_A = self.model(batch_A, person="A")
            loss_A_recon = self.criterion_recon(output_A, batch_A)
            loss_A_perceptual = self.criterion_perceptual(output_A, batch_A)
            loss_A = loss_A_recon + 0.1 * loss_A_perceptual
            loss_A.backward()
            self.optimizer_A.step()

            # Train B -> B
            self.optimizer_B.zero_grad()
            output_B, latent_B = self.model(batch_B, person="B")
            loss_B_recon = self.criterion_recon(output_B, batch_B)
            loss_B_perceptual = self.criterion_perceptual(output_B, batch_B)
            loss_B = loss_B_recon + 0.1 * loss_B_perceptual
            loss_B.backward()
            self.optimizer_B.step()

            # Total loss
            loss = (loss_A + loss_B) / 2
            total_loss += loss.item()

            # Update progress bar
            pbar.set_description(f"Epoch {epoch} - Loss: {loss.item():.4f}")

            # Log metrics
            if i % 100 == 0:
                self.writer.add_scalar(
                    "Loss/A", loss_A.item(), epoch * len(dataloader_A) + i
                )
                self.writer.add_scalar(
                    "Loss/B", loss_B.item(), epoch * len(dataloader_B) + i
                )

                if self.config.get("use_wandb", False):
                    wandb.log(
                        {
                            "loss_A": loss_A.item(),
                            "loss_B": loss_B.item(),
                            "epoch": epoch,
                        }
                    )

        return total_loss / len(pbar)

    def validate(self, dataloader_A: DataLoader, dataloader_B: DataLoader):
        """Validate model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_A, batch_B in zip(dataloader_A, dataloader_B):
                batch_A = batch_A.to(self.device)
                batch_B = batch_B.to(self.device)

                # Validate A -> B swap
                output_AB = self.model.swap(batch_A, source="A", target="B")
                loss_AB = self.criterion_recon(output_AB, batch_B)

                # Validate B -> A swap
                output_BA = self.model.swap(batch_B, source="B", target="A")
                loss_BA = self.criterion_recon(output_BA, batch_A)

                total_loss += (loss_AB + loss_BA).item() / 2

        return total_loss / min(len(dataloader_A), len(dataloader_B))

    def save_checkpoint(self, epoch: int, loss: float, path: str):
        """Save model checkpoint"""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_A_state_dict": self.optimizer_A.state_dict(),
                "optimizer_B_state_dict": self.optimizer_B.state_dict(),
                "loss": loss,
                "config": self.config,
            },
            path,
        )

    def export_onnx(self, output_path: str):
        """Export model to ONNX format"""
        self.model.eval()

        # Export encoder A
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        torch.onnx.export(
            self.model.encoder_A,
            dummy_input,
            f"{output_path}/encoder_A.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["latent"],
            dynamic_axes={"input": {0: "batch_size"}, "latent": {0: "batch_size"}},
        )

        # Export decoder B (for A->B swap)
        dummy_latent = torch.randn(1, 512).to(self.device)
        torch.onnx.export(
            self.model.decoder_B,
            dummy_latent,
            f"{output_path}/decoder_B.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["latent"],
            output_names=["output"],
            dynamic_axes={"latent": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        print(f"Models exported to {output_path}")


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""

    def __init__(self):
        super().__init__()
        from torchvision import models

        vgg = models.vgg19(pretrained=True).features

        # Extract specific layers
        self.layers = nn.ModuleList(
            [
                vgg[:4],  # relu1_2
                vgg[4:9],  # relu2_2
                vgg[9:18],  # relu3_4
            ]
        )

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0

        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += nn.functional.l1_loss(x, y)

        return loss


def main():
    parser = argparse.ArgumentParser(description="Train Face Swap Model")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to face dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./checkpoints", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument(
        "--export-onnx", action="store_true", help="Export to ONNX after training"
    )

    args = parser.parse_args()

    # Configuration
    config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "log_dir": f"{args.output_dir}/logs",
        "use_wandb": args.use_wandb,
    }

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Datasets
    dataset_A = FaceDataset(args.data_dir, "A", transform=transform)
    dataset_B = FaceDataset(args.data_dir, "B", transform=transform)

    # Dataloaders
    dataloader_A = DataLoader(
        dataset_A, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    dataloader_B = DataLoader(
        dataset_B, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Model
    model = FaceSwapAutoencoder(latent_dim=512)

    # Trainer
    trainer = FaceSwapTrainer(model, device, config)

    # Training loop
    best_loss = float("inf")
    for epoch in range(args.epochs):
        # Train
        train_loss = trainer.train_epoch(dataloader_A, dataloader_B, epoch)

        # Validate
        val_loss = trainer.validate(dataloader_A, dataloader_B)

        print(
            f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            trainer.save_checkpoint(
                epoch, val_loss, f"{args.output_dir}/best_model.pth"
            )

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                epoch, val_loss, f"{args.output_dir}/checkpoint_epoch_{epoch+1}.pth"
            )

    # Export to ONNX
    if args.export_onnx:
        onnx_dir = f"{args.output_dir}/onnx"
        os.makedirs(onnx_dir, exist_ok=True)
        trainer.export_onnx(onnx_dir)

    print("Training completed!")


if __name__ == "__main__":
    main()
