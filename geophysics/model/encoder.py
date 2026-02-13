"""
Multi-modal encoder for seismic, magnetics, and gravity.
Uses availability mask so the model adapts when some inputs are missing.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class MultiModalEncoder(nn.Module):
    """
    Encodes seismic, magnetics, and gravity 2D sections into a shared representation.
    Inputs are (B, n_traces, n_samples). Mask (B, 3) indicates [seismic, magnetics, gravity] available.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        n_layers: int = 4,
        max_freq_bands: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        # Per-modality 2D conv stacks (then fused)
        self.seismic_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 5, padding=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
        )
        self.magnetics_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 5, padding=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
        )
        self.gravity_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 5, padding=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
        )
        # Mask embedding: learn how to weight missing modalities
        self.mask_embed = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, base_channels * 2 * 3),
        )
        # Fuse: 3 * (base*2) -> base*4
        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels * 2 * 3 + base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
        )
        self._out_channels = base_channels * 4

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(
        self,
        seismic: torch.Tensor,
        magnetics: torch.Tensor,
        gravity: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        seismic, magnetics, gravity: (B, H, W). mask: (B, 3).
        Returns (B, C, H, W) encoded features.
        """
        B, H, W = seismic.shape
        device = seismic.device
        # (B, 1, H, W)
        s = seismic.unsqueeze(1)
        m = magnetics.unsqueeze(1)
        g = gravity.unsqueeze(1)
        # Encode each modality
        es = self.seismic_conv(s)
        em = self.magnetics_conv(m)
        eg = self.gravity_conv(g)
        # Weight by availability so missing modality doesn't add noise
        mask_expand = mask.view(B, 3, 1, 1)
        es = es * mask_expand[:, 0:1]
        em = em * mask_expand[:, 1:2]
        eg = eg * mask_expand[:, 2:3]
        # Mask embedding to inform fusion
        mask_feat = self.mask_embed(mask)
        mask_feat = mask_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)
        fused = torch.cat([es, em, eg, mask_feat], dim=1)
        return self.fuse(fused)
