"""
Adaptable pipeline network: multi-modal (seismic, magnetics, gravity) -> SEG-Y 2D.
Uses the BS 11/00 processing sequence when data and tools are available;
otherwise relies on the NN to produce the section. Tool availability is configurable.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple

from .encoder import MultiModalEncoder
from .decoder import SectionDecoder
from ..config import ProcessingConfig


class GeophysicsPipelineNet(nn.Module):
    """
    Neural network that analyses seismic, magnetics, and gravity to produce 2D SEG-Y-like section.
    - When seismic + processing tools are available: runs deterministic sequence and blends with NN output.
    - When data or tools are missing: uses only NN output (adaptable).
    """

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        base_channels: int = 32,
        encoder_layers: int = 4,
        out_channels: int = 1,
        use_processing_branch: bool = True,
        blend_learned: bool = True,
    ):
        super().__init__()
        self.config = config or ProcessingConfig()
        self.out_channels = out_channels
        self.use_processing_branch = use_processing_branch
        self.blend_learned = blend_learned

        self.encoder = MultiModalEncoder(
            in_channels=3,
            base_channels=base_channels,
            n_layers=encoder_layers,
        )
        self.decoder = SectionDecoder(
            in_channels=self.encoder.out_channels,
            base_channels=base_channels,
            out_channels=out_channels,
        )
        # Learnable blend: when processing branch is present, how much to use vs NN
        self.blend = nn.Parameter(torch.tensor(0.5))

    def _run_processing_sequence_numpy(
        self,
        seismic_np: np.ndarray,
        dt_ms: float,
        dx_m: float = 25.0,
        velocity_rms: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Run deterministic processing sequence; returns None if not run."""
        if not self.use_processing_branch:
            return None
        if not self.config.get_tool_available("bandpass") and not self.config.get_tool_available("kirchhoff_migration"):
            return None
        from ..processing.sequence import run_processing_sequence
        try:
            out = run_processing_sequence(
                seismic_np,
                self.config,
                dt_ms=dt_ms,
                dx_m=dx_m,
                velocity_rms=velocity_rms,
            )
            return out
        except Exception:
            return None

    def forward(
        self,
        seismic: torch.Tensor,
        magnetics: torch.Tensor,
        gravity: torch.Tensor,
        mask: torch.Tensor,
        processing_seismic: Optional[torch.Tensor] = None,
        dt_ms: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        seismic, magnetics, gravity: (B, H, W). mask: (B, 3) [has_seismic, has_magnetics, has_gravity].
        processing_seismic: optional (B, H, W) from external sequence (e.g. run on CPU).
        Returns dict with "section" (B, 1, H, W) and optionally "processing_section".
        """
        B, H, W = seismic.shape
        device = seismic.device
        dt_ms = dt_ms or 4.0

        # Encode multi-modal input (mask tells which modalities are present)
        feats = self.encoder(seismic, magnetics, gravity, mask)
        # Decode to 2D section
        section_nn = self.decoder(feats)

        out = {"section": section_nn, "section_nn": section_nn}

        if processing_seismic is not None and self.use_processing_branch:
            # Blend processing result with NN output (learnable weight)
            alpha = torch.sigmoid(self.blend)
            section = alpha * section_nn + (1 - alpha) * processing_seismic.unsqueeze(1)
            out["section"] = section
            out["processing_section"] = processing_seismic.unsqueeze(1)
        elif self.blend_learned:
            out["section"] = section_nn

        return out

    def forward_with_processing(
        self,
        seismic: torch.Tensor,
        magnetics: torch.Tensor,
        gravity: torch.Tensor,
        mask: torch.Tensor,
        dt_ms: float = 4.0,
        dx_m: float = 25.0,
        velocity_rms: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward and optionally compute processing branch from seismic numpy (on CPU).
        Use this when you have seismic and want to apply the full sequence; processing runs in numpy.
        """
        processing_seismic = None
        if self.use_processing_branch and mask[:, 0].any().item():
            # Run sequence for samples that have seismic
            batch = seismic.shape[0]
            processed_list = []
            for b in range(batch):
                if mask[b, 0].item() > 0.5:
                    np_seis = seismic[b].detach().cpu().numpy()
                    proc = self._run_processing_sequence_numpy(np_seis, dt_ms, dx_m, velocity_rms)
                    if proc is not None:
                        processed_list.append(torch.from_numpy(proc).float().to(seismic.device))
                    else:
                        processed_list.append(seismic[b])
                else:
                    processed_list.append(seismic[b])
            if processed_list:
                processing_seismic = torch.stack(processed_list, dim=0)
        return self.forward(
            seismic, magnetics, gravity, mask,
            processing_seismic=processing_seismic,
            dt_ms=dt_ms,
        )

    def to_segy_2d(
        self,
        section: torch.Tensor,
        dt_ms: float = 4.0,
    ) -> np.ndarray:
        """Convert model output (B, 1, H, W) to numpy (H, W) for first batch item for SEG-Y write."""
        x = section[0, 0].detach().cpu().numpy()
        return x
