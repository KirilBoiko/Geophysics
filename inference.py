#!/usr/bin/env python3
"""
Run inference and export 2D SEG-Y.
Usage:
  python inference.py --checkpoint checkpoint.pt --seismic line1.sgy --out line1_out.sgy
  python inference.py --checkpoint checkpoint.pt --magnetics mag.npy --gravity grav.npy --out out.sgy
  python inference.py --checkpoint checkpoint.pt --seismic line1.sgy --magnetics mag.npy --out line1_out.sgy
Any subset of seismic/magnetics/gravity can be provided; model adapts.
"""

import argparse
from pathlib import Path

import torch
import numpy as np

from geophysics.config import ProcessingConfig
from geophysics.data.loaders import make_batch
from geophysics.model.pipeline_net import GeophysicsPipelineNet
from geophysics.io_segy import write_segy_2d, read_segy_2d


def main():
    parser = argparse.ArgumentParser(description="Inference and SEG-Y export")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--seismic", type=str, default=None)
    parser.add_argument("--magnetics", type=str, default=None)
    parser.add_argument("--gravity", type=str, default=None)
    parser.add_argument("--out", type=str, default="output.sgy", help="Output SEG-Y path")
    parser.add_argument("--target_shape", type=str, default=None, help="H,W for output (optional)")
    parser.add_argument("--dt_ms", type=float, default=4.0)
    parser.add_argument("--no_processing", action="store_true", help="Skip deterministic processing branch")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", ProcessingConfig())
    else:
        state = ckpt
        config = ProcessingConfig()

    model = GeophysicsPipelineNet(
        config=config,
        use_processing_branch=not args.no_processing,
        blend_learned=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    target_shape = None
    if args.target_shape:
        target_shape = tuple(int(x) for x in args.target_shape.split(","))

    batch = make_batch(
        seismic_path=args.seismic,
        magnetics_path=args.magnetics,
        gravity_path=args.gravity,
        target_shape=target_shape,
    )
    if batch.n_traces == 0:
        batch.n_traces = 128
        batch.n_samples = 512
        batch.has_seismic = True
        batch.seismic = np.random.randn(batch.n_traces, batch.n_samples).astype(np.float32) * 0.01

    tensors = batch.to_tensor_dict()
    seismic = tensors["seismic"].unsqueeze(0)
    magnetics = tensors["magnetics"].unsqueeze(0)
    gravity = tensors["gravity"].unsqueeze(0)
    mask = tensors["mask"].unsqueeze(0)

    with torch.no_grad():
        out = model.forward_with_processing(
            seismic, magnetics, gravity, mask,
            dt_ms=args.dt_ms or batch.dt_ms,
            dx_m=25.0,
        )
    section = out["section"]
    data_2d = model.to_segy_2d(section, dt_ms=args.dt_ms or batch.dt_ms)

    write_segy_2d(args.out, data_2d, dt_ms=args.dt_ms or batch.dt_ms)
    print(f"Written 2D SEG-Y: {args.out} shape={data_2d.shape}")


if __name__ == "__main__":
    main()
