"""
Decoder from shared representation to 2D seismic-like section (SEG-Y 2D).
"""

import torch
import torch.nn as nn
from typing import Optional


class SectionDecoder(nn.Module):
    """
    Decodes encoded features + optional processing residual into 2D section (B, 1, H, W).
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        out_channels: int = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
