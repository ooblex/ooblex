"""
Style Transfer Model Training Script
Fast neural style transfer using PyTorch
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


class ConvBlock(nn.Module):
    """Convolutional block with instance normalization"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block for style transfer"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, 3, 1, 1),
            ConvBlock(channels, channels, 3, 1, 1, activation=False),
        )

    def forward(self, x):
        return x + self.block(x)


class StyleTransferNet(nn.Module):
    """Fast neural style transfer network"""

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(3, 32, 9, 1, 4),
            ConvBlock(32, 64, 3, 2, 1),
            ConvBlock(64, 128, 3, 2, 1),
        )

        # Residual blocks
        self.residual = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 9, 1, 4),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return x


class StyleDataset(Dataset):
    """Dataset for style transfer training"""

    def __init__(self, content_dir, transform=None):
        self.content_dir = Path(content_dir)
        self.images = list(self.content_dir.glob("*.jpg")) + list(
            self.content_dir.glob("*.png")
        )
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def train_style_transfer(args):
    """Train style transfer model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = StyleDataset(args.content_dir, transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Model
    model = StyleTransferNet().to(device)

    # Load style image
    style_img = Image.open(args.style_image).convert("RGB")
    style_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    style_img = style_transform(style_img).unsqueeze(0).to(device)

    # Loss network (VGG)
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    # Extract style features
    style_features = extract_features(style_img, vgg)
    style_gram = {
        layer: gram_matrix(features) for layer, features in style_features.items()
    }

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(batch)

            # Content loss
            content_features = extract_features(batch, vgg)
            output_features = extract_features(output, vgg)
            content_loss = nn.functional.mse_loss(
                output_features["relu2_2"], content_features["relu2_2"]
            )

            # Style loss
            style_loss = 0
            for layer in style_gram:
                output_gram = gram_matrix(output_features[layer])
                style_loss += nn.functional.mse_loss(
                    output_gram, style_gram[layer].expand_as(output_gram)
                )

            # Total loss
            loss = args.content_weight * content_loss + args.style_weight * style_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss/len(dataloader):.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "style": args.style_image,
                },
                f"{args.output_dir}/style_transfer_epoch_{epoch+1}.pth",
            )

    # Save final model
    torch.save(
        {"model_state_dict": model.state_dict(), "style": args.style_image},
        f"{args.output_dir}/style_transfer_best.pth",
    )


def extract_features(x, model):
    """Extract features from VGG"""
    features = {}
    layers = {
        "3": "relu1_2",
        "8": "relu2_2",
        "17": "relu3_3",
        "26": "relu4_3",
        "35": "relu5_3",
    }

    for name, module in model._modules.items():
        x = module(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(x):
    """Calculate Gram matrix"""
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-dir", type=str, required=True)
    parser.add_argument("--style-image", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=100000.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_style_transfer(args)
