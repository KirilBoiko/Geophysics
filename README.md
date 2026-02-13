# Geophysics multi-modal to SEG-Y 2D

Neural network that analyses **seismic**, **magnetics**, and **gravity** data to produce **2D SEG-Y** sections. It follows a BS 11/00–style processing sequence (Kirchhoff migration, band-pass filters, AGC, F-K filter, time-variant scaling, etc.) and is **adaptable**: it still produces an output when some data or processing tools are missing.

## Features

- **Multi-modal input**: Seismic (SEG-D/SEG-Y), magnetics, and gravity; any subset can be provided.
- **Processing sequence** (from the 28-step workflow):
  - Resample, bulk statics, trace edit, TAR, band-pass (4–8–60–70 Hz and time-variant)
  - AGC, top muting, F-K fan filter, deconvolution, Radon (stubs)
  - Kirchhoff time migration (max dip 45°, 0.95× RMS velocity, max 70 Hz)
  - Time-variant band-pass and time-variant scaling, header static, SEG-Y output
- **Tool availability**: Each step can be turned on/off in `ProcessingConfig.tool_available`; the model learns to rely on the NN when a tool is unavailable.
- **Output**: 2D section suitable for SEG-Y export.

## Setup

```bash
cd /Users/kirilboiko/Geophysics
pip install -r requirements.txt
```

## Usage

### Training

Train on directories of seismic/magnetics/gravity and target SEG-Y:

```bash
python train.py --seismic_dir /path/to/segy --target_dir /path/to/target_segy --epochs 50 --out checkpoint.pt
python train.py --seismic_dir /path --magnetics_dir /path --gravity_dir /path --target_dir /path --out checkpoint.pt
```

With only magnetics and gravity (no seismic), use `--no_processing_branch` so the model uses only the NN path.

### Inference and SEG-Y export

Run the model and write 2D SEG-Y:

```bash
python inference.py --checkpoint checkpoint.pt --seismic line1.sgy --out line1_out.sgy
python inference.py --checkpoint checkpoint.pt --magnetics mag.npy --gravity grav.npy --out out.sgy
python inference.py --checkpoint checkpoint.pt --seismic line1.sgy --magnetics mag.npy --out line1_out.sgy
```

Any combination of `--seismic`, `--magnetics`, `--gravity` is valid; missing inputs are zero-filled and the mask tells the model to adapt.

### Programmatic use

```python
from geophysics.config import ProcessingConfig, DEFAULT_CONFIG
from geophysics.data.loaders import make_batch
from geophysics.model.pipeline_net import GeophysicsPipelineNet
from geophysics.io_segy import write_segy_2d
import torch

# Optional: disable specific tools so the model must adapt
config = ProcessingConfig()
config.tool_available["kirchhoff_migration"] = False  # e.g. when migration is unavailable

model = GeophysicsPipelineNet(config=config)
batch = make_batch(seismic_path="line.sgy", target_shape=(128, 512))
tensors = batch.to_tensor_dict()
seismic = tensors["seismic"].unsqueeze(0)
magnetics = tensors["magnetics"].unsqueeze(0)
gravity = tensors["gravity"].unsqueeze(0)
mask = tensors["mask"].unsqueeze(0)

with torch.no_grad():
    out = model.forward_with_processing(seismic, magnetics, gravity, mask, dt_ms=4.0, dx_m=25.0)
section = out["section"]
write_segy_2d("output.sgy", model.to_segy_2d(section), dt_ms=4.0)
```

## Project layout

- `geophysics/config.py` – Processing sequence parameters and `tool_available` flags.
- `geophysics/data/loaders.py` – Load seismic/magnetics/gravity and build batches with masks.
- `geophysics/processing/` – Filters (band-pass, time-variant band-pass, AGC, F-K), Kirchhoff time migration, and `run_processing_sequence()`.
- `geophysics/model/` – Multi-modal encoder, section decoder, and `GeophysicsPipelineNet` (optional processing branch + learned blend).
- `geophysics/io_segy.py` – Read/write 2D SEG-Y.
- `train.py` – Training script.
- `inference.py` – Inference and SEG-Y export.

## Adaptability

- **Missing data**: Pass only the modalities you have; set paths for the rest to `None`. The mask is set accordingly and the encoder weights missing modalities to zero.
- **Missing tools**: Set `config.tool_available["kirchhoff_migration"] = False` (or any other step) to skip that step; the deterministic branch then does less and the NN carries the rest.
- **No seismic at all**: Use `--no_processing_branch` in training/inference so the output is purely NN-driven from magnetics/gravity.
