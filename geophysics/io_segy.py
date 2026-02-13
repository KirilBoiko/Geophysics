"""
Read/write 2D SEG-Y for model input and output.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    import segyio
    HAS_SEGYIO = True
except ImportError:
    HAS_SEGYIO = False


def write_segy_2d(
    path: str,
    data: np.ndarray,
    dt_ms: float = 4.0,
    n_traces: Optional[int] = None,
    n_samples: Optional[int] = None,
) -> None:
    """
    Write 2D section to SEG-Y. data: (n_traces, n_samples).
    """
    if not HAS_SEGYIO:
        raise ImportError("segyio is required for write_segy_2d. Install with: pip install segyio")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ntr, ns = data.shape
    if n_traces is not None:
        ntr = n_traces
    if n_samples is not None:
        ns = n_samples
    data = np.asarray(data, dtype=np.float32)
    if data.shape != (ntr, ns):
        from scipy.ndimage import zoom
        data = zoom(data, (ntr / data.shape[0], ns / data.shape[1]), order=1)
    dt_micros = int(dt_ms * 1000)
    spec = segyio.spec()
    spec.format = 1
    spec.sorting = 2
    spec.samples = list(range(0, ns * dt_micros, dt_micros))
    spec.tracecount = ntr
    with segyio.create(str(path), spec) as f:
        for i in range(ntr):
            f.trace[i] = data[i]
        f.bin[segyio.su.dt] = dt_micros
    return None


def read_segy_2d(path: str) -> Tuple[np.ndarray, float]:
    """Read 2D SEG-Y; return (data, dt_ms)."""
    if not HAS_SEGYIO:
        raise ImportError("segyio is required for read_segy_2d. Install with: pip install segyio")
    with segyio.open(path, "r", ignore_geometry=True) as f:
        ntr = len(f.trace)
        ns = len(f.samples)
        dt_ms = (f.samples[1] - f.samples[0]) / 1000.0 if ns > 1 else 4.0
        data = np.stack([f.trace[i] for i in range(ntr)], axis=0)
    return data.astype(np.float64), dt_ms
