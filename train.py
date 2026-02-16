#!/usr/bin/env python3
"""
Train the geophysics pipeline network.
Usage:
  python train.py --seismic_dir /path/to/segy --target_dir /path/to/target_segy
  python train.py --seismic_dir /path --magnetics_dir /path --gravity_dir /path --target_dir /path
When some modalities are missing for a sample, mask is set accordingly; the model learns to adapt.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from geophysics.config import ProcessingConfig
from geophysics.data.loaders import make_batch, load_seismic, MultiModalBatch
from geophysics.model.pipeline_net import GeophysicsPipelineNet
from geophysics.io_segy import read_segy_2d, write_segy_2d


class GeophysicsDataset(Dataset):
    """Dataset of multi-modal inputs and target 2D section (SEG-Y)."""

    def __init__(
        self,
        seismic_paths=None,
        magnetics_paths=None,
        gravity_paths=None,
        target_paths=None,
        target_shape=(128, 512),
    ):
        self.target_shape = target_shape
        self.samples = []
        seismic_paths = seismic_paths or []
        magnetics_paths = magnetics_paths or []
        gravity_paths = gravity_paths or []
        target_paths = target_paths or []
        n = max(len(seismic_paths), len(magnetics_paths), len(gravity_paths), len(target_paths), 1)
        for i in range(n):
            self.samples.append({
                "seismic": seismic_paths[i] if i < len(seismic_paths) else None,
                "magnetics": magnetics_paths[i] if i < len(magnetics_paths) else None,
                "gravity": gravity_paths[i] if i < len(gravity_paths) else None,
                "target": target_paths[i] if i < len(target_paths) else None,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        batch = make_batch(
            seismic_path=s["seismic"],
            magnetics_path=s["magnetics"],
            gravity_path=s["gravity"],
            target_shape=self.target_shape,
        )
        if batch.n_traces == 0:
            batch.n_traces, batch.n_samples = self.target_shape[0], self.target_shape[1]
        tensors = batch.to_tensor_dict()
        target = torch.zeros(self.target_shape[0], self.target_shape[1], dtype=torch.float32)
        if s["target"] and Path(s["target"]).exists():
            try:
                data, _ = read_segy_2d(s["target"])
                from scipy.ndimage import zoom
                if data.shape != self.target_shape:
                    data = zoom(data, (self.target_shape[0] / data.shape[0], self.target_shape[1] / data.shape[1]), order=1)
                target = torch.from_numpy(data.astype(np.float32))
            except Exception:
                pass
        tensors["target"] = target
        return tensors


def collect_paths(dir_path, ext=".sgy"):
    if not dir_path or not os.path.isdir(dir_path):
        return []
    return sorted(Path(dir_path).glob(f"*{ext}")) + sorted(Path(dir_path).glob("*.segy"))


def main():
    parser = argparse.ArgumentParser(description="Train GeophysicsPipelineNet")
    parser.add_argument("--seismic_dir", type=str, default=None, help="Directory of seismic SEG-Y")
    parser.add_argument("--magnetics_dir", type=str, default=None)
    parser.add_argument("--gravity_dir", type=str, default=None)
    parser.add_argument("--target_dir", type=str, default=None, help="Target 2D SEG-Y (ground truth)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target_shape", type=str, default="128,512", help="H,W")
    parser.add_argument("--out", type=str, default="checkpoint.pt")
    parser.add_argument("--no_processing_branch", action="store_true", help="Do not use deterministic processing branch")
    args = parser.parse_args()

    target_shape = tuple(int(x) for x in args.target_shape.split(","))
    seismic_paths = [str(p) for p in collect_paths(args.seismic_dir)]
    magnetics_paths = [str(p) for p in collect_paths(args.magnetics_dir, ".npy")] or [str(p) for p in collect_paths(args.magnetics_dir, ".txt")]
    gravity_paths = [str(p) for p in collect_paths(args.gravity_dir, ".npy")] or [str(p) for p in collect_paths(args.gravity_dir, ".txt")]
    target_paths = [str(p) for p in collect_paths(args.target_dir)]
    if not seismic_paths and not magnetics_paths and not gravity_paths:
        # Dummy dataset for testing
        seismic_paths = [None]
        magnetics_paths = [None]
        gravity_paths = [None]
        target_paths = [None]

    n_samples = max(1, len(seismic_paths), len(magnetics_paths), len(gravity_paths), len(target_paths))
    def pad(lst, n, fill=None):
        return (lst or [fill]) + [fill] * max(0, n - len(lst or [fill]))
    dataset = GeophysicsDataset(
        seismic_paths=pad(seismic_paths, n_samples),
        magnetics_paths=pad(magnetics_paths, n_samples),
        gravity_paths=pad(gravity_paths, n_samples),
        target_paths=pad(target_paths, n_samples),
        target_shape=target_shape,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    config = ProcessingConfig()
    model = GeophysicsPipelineNet(
        config=config,
        use_processing_branch=not args.no_processing_branch,
        blend_learned=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Device configuration: Check for M3 Pro GPU (mps), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using M3 Pro GPU (MPS) for training!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA) for training!")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            seismic = batch["seismic"].to(device)
            magnetics = batch["magnetics"].to(device)
            gravity = batch["gravity"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device).unsqueeze(1)

            out = model.forward_with_processing(
                seismic, magnetics, gravity, mask,
                dt_ms=4.0,
                dx_m=25.0,
            )
            pred = out["section"]
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/len(loader):.6f}")

    torch.save({"model": model.state_dict(), "config": config}, args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
