"""
Data loaders for seismic, magnetics, and gravity.
Each loader returns data + availability mask so the model can adapt when data is missing.
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclass
class MultiModalBatch:
    """Batch of multi-modal geophysics data with availability flags."""

    seismic: Optional[np.ndarray] = None   # (n_traces, n_samples) or (n_traces, n_samples, n_chan)
    magnetics: Optional[np.ndarray] = None  # (n_profiles,) or (n_traces, n_samples) 2D line
    gravity: Optional[np.ndarray] = None   # (n_profiles,) or (n_traces, n_samples) 2D line

    # Masks: True = data present and valid
    has_seismic: bool = False
    has_magnetics: bool = False
    has_gravity: bool = False

    # Optional metadata for SEG-Y export
    dt_ms: float = 4.0
    n_traces: int = 0
    n_samples: int = 0

    def to_tensor_dict(self):
        """Return dict suitable for model input; missing data as zeros + mask."""
        import torch
        n_t = self.n_traces or 1
        n_s = self.n_samples or 1
        out = {}
        if self.has_seismic and self.seismic is not None:
            s = self.seismic
            if s.ndim == 3:
                s = s.mean(axis=-1)  # collapse channels to 2D
            out["seismic"] = torch.from_numpy(s.astype(np.float32))
        else:
            out["seismic"] = torch.zeros((n_t, n_s), dtype=torch.float32)
        if self.has_magnetics and self.magnetics is not None:
            m = self.magnetics
            if m.ndim == 1:
                m = np.broadcast_to(m[:, None], (m.size, n_s))
            out["magnetics"] = torch.from_numpy(m.astype(np.float32))
        else:
            out["magnetics"] = torch.zeros((n_t, n_s), dtype=torch.float32)
        if self.has_gravity and self.gravity is not None:
            g = self.gravity
            if g.ndim == 1:
                g = np.broadcast_to(g[:, None], (g.size, n_s))
            out["gravity"] = torch.from_numpy(g.astype(np.float32))
        else:
            out["gravity"] = torch.zeros((n_t, n_s), dtype=torch.float32)
        out["mask"] = torch.tensor(
            [self.has_seismic, self.has_magnetics, self.has_gravity],
            dtype=torch.float32
        )
        return out


def load_seismic(
    path: Optional[Union[str, Path]] = None,
    array: Optional[np.ndarray] = None,
    dt_ms: float = 4.0,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Load seismic 2D section from SEG-Y/SEG-D or from array.
    Returns (data, dt_ms). data is (n_traces, n_samples).
    """
    if array is not None:
        return array, dt_ms
    if path is None:
        return None, dt_ms
    path = Path(path)
    if not path.exists():
        return None, dt_ms
    try:
        import segyio
        with segyio.open(str(path), "r", ignore_geometry=True) as f:
            n_traces = len(f.trace)
            n_samples = len(f.samples)
            dt_ms = (f.samples[1] - f.samples[0]) / 1000.0 if n_samples > 1 else 4.0
            data = np.stack([f.trace[i] for i in range(n_traces)], axis=0).astype(np.float64)
        return data, dt_ms
    except Exception:
        return None, dt_ms


def load_magnetics(
    path: Optional[Union[str, Path]] = None,
    array: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Load magnetics 2D line or 1D profile.
    Returns (n_traces,) or (n_traces, n_samples). None if unavailable.
    """
    if array is not None:
        return np.asarray(array, dtype=np.float64)
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = np.loadtxt(path)
        if data.ndim == 1:
            return data
        return data[:, 1] if data.shape[1] > 1 else data.ravel()
    except Exception:
        try:
            return np.load(path)
        except Exception:
            return None


def load_gravity(
    path: Optional[Union[str, Path]] = None,
    array: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Load gravity 2D line or 1D profile.
    Returns (n_traces,) or (n_traces, n_samples). None if unavailable.
    """
    if array is not None:
        return np.asarray(array, dtype=np.float64)
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = np.loadtxt(path)
        if data.ndim == 1:
            return data
        return data[:, 1] if data.shape[1] > 1 else data.ravel()
    except Exception:
        try:
            return np.load(path)
        except Exception:
            return None


def make_batch(
    seismic_path: Optional[Union[str, Path]] = None,
    magnetics_path: Optional[Union[str, Path]] = None,
    gravity_path: Optional[Union[str, Path]] = None,
    seismic_array: Optional[np.ndarray] = None,
    magnetics_array: Optional[np.ndarray] = None,
    gravity_array: Optional[np.ndarray] = None,
    target_shape: Optional[Tuple[int, int]] = None,
) -> MultiModalBatch:
    """
    Build a MultiModalBatch from paths or arrays. Reshape to target_shape (n_traces, n_samples) if given.
    """
    seismic, dt_ms = load_seismic(seismic_path, seismic_array)
    magnetics = load_magnetics(magnetics_path, magnetics_array)
    gravity = load_gravity(gravity_path, gravity_array)

    n_traces, n_samples = 0, 0
    if seismic is not None:
        if seismic.ndim == 3:
            seismic = seismic.mean(axis=-1)
        n_traces, n_samples = seismic.shape
    if magnetics is not None and n_traces == 0:
        n_traces = magnetics.shape[0]
        n_samples = magnetics.shape[1] if magnetics.ndim > 1 else 1
    if gravity is not None and n_traces == 0:
        n_traces = gravity.shape[0]
        n_samples = gravity.shape[1] if gravity.ndim > 1 else 1

    if target_shape:
        from scipy.ndimage import zoom
        nt, ns = target_shape
        if seismic is not None and (n_traces, n_samples) != (nt, ns):
            zoom_factors = (nt / seismic.shape[0], ns / seismic.shape[1])
            seismic = zoom(seismic, zoom_factors, order=1)
        n_traces, n_samples = nt, ns
        if magnetics is not None:
            if magnetics.ndim == 1:
                magnetics = np.interp(
                    np.linspace(0, magnetics.size - 1, nt),
                    np.arange(magnetics.size),
                    magnetics
                )
                magnetics = np.broadcast_to(magnetics[:, None], (nt, ns))
            else:
                magnetics = zoom(magnetics, (nt / magnetics.shape[0], ns / magnetics.shape[1]), order=1)
        if gravity is not None:
            if gravity.ndim == 1:
                gravity = np.interp(
                    np.linspace(0, gravity.size - 1, nt),
                    np.arange(gravity.size),
                    gravity
                )
                gravity = np.broadcast_to(gravity[:, None], (nt, ns))
            else:
                gravity = zoom(gravity, (nt / gravity.shape[0], ns / gravity.shape[1]), order=1)

    return MultiModalBatch(
        seismic=seismic,
        magnetics=magnetics,
        gravity=gravity,
        has_seismic=seismic is not None,
        has_magnetics=magnetics is not None,
        has_gravity=gravity is not None,
        dt_ms=dt_ms,
        n_traces=n_traces or (target_shape[0] if target_shape else 0),
        n_samples=n_samples or (target_shape[1] if target_shape else 0),
    )
