"""Küçük evrişimli autoencoder (yol haritası Adım 8)."""
from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """B×1×64×64 standardize log mel → aynı boyutta yeniden oluşturma.

    Encoder : Conv2d 1→16→32→64 (k=3, s=2, p=1) + ReLU      → B×64×8×8
    Darboğaz: Flatten → Linear(4096→bottleneck)             → B×bottleneck
    Decoder : Linear(bottleneck→4096) + ReLU → B×64×8×8
              ConvTranspose2d 64→32→16→1 (k=4, s=2, p=1); ara katmanlarda ReLU
    Çıkış   : B×1×64×64, son katmanda aktivasyon YOK (standardize girdi ± değer taşır)
    """

    def __init__(self, n_mels: int = 64, frames: int = 64, bottleneck: int = 128):
        super().__init__()
        if n_mels % 8 or frames % 8:
            raise ValueError("n_mels ve frames 8'e bölünebilmeli (3 adet stride-2 katman).")
        self.n_mels, self.frames = n_mels, frames
        self.enc_shape = (64, n_mels // 8, frames // 8)
        flat = 64 * (n_mels // 8) * (frames // 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.to_latent = nn.Linear(flat, bottleneck)
        self.from_latent = nn.Sequential(nn.Linear(bottleneck, flat), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        h = self.from_latent(z).view(-1, *self.enc_shape)
        return self.decoder(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
