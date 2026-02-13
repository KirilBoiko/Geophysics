"""
Kirchhoff time migration for 2D post-stack data.
When tool is unavailable, returns None so the model can use a learned alternative.
"""

import numpy as np
from typing import Optional, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def kirchhoff_time_migration(
    data: np.ndarray,
    dt_ms: float,
    dx_m: float,
    velocity_rms: np.ndarray,
    max_dip_deg: float = 45.0,
    rms_smooth: bool = True,
    max_freq_hz: float = 70.0,
) -> np.ndarray:
    """
    2D Kirchhoff time migration (post-stack).
    data: (n_traces, n_samples). velocity_rms: (n_samples,) or scalar, in m/s.
    """
    ntr, ns = data.shape
    dt_s = dt_ms / 1000.0
    t = np.arange(ns, dtype=np.float64) * dt_s
    if np.isscalar(velocity_rms) or velocity_rms.size == 1:
        v = np.full(ns, float(velocity_rms))
    else:
        v = np.asarray(velocity_rms, dtype=np.float64).ravel()[:ns]
        if v.size < ns:
            v = np.resize(v, ns)
    if rms_smooth:
        from scipy.ndimage import uniform_filter1d
        v = uniform_filter1d(v, size=max(5, ns // 20), mode="nearest")
    # Migration: for each output (x0, t0), sum input along diffraction hyperbola
    # t^2 = t0^2 + (2*x/v)^2  =>  t = sqrt(t0^2 + (2*dx/v)^2)
    out = np.zeros_like(data)
    max_dip_rad = np.deg2rad(max_dip_deg)
    half_aperature = int(np.ceil(ntr / 2))
    for i in range(ntr):
        for it0 in range(ns):
            t0 = t[it0]
            v0 = v[it0]
            if v0 <= 0:
                continue
            val = 0.0
            count = 0
            for j in range(max(0, i - half_aperature), min(ntr, i + half_aperature + 1)):
                dx = abs((j - i) * dx_m)
                # travel time from (j,t) to (i,t0): t = sqrt(t0^2 + (2*dx/v)^2)
                t_mig = np.sqrt(t0**2 + (2.0 * dx / v0) ** 2)
                it = int(round(t_mig / dt_s))
                if it < 0 or it >= ns:
                    continue
                # dip limit: skip if slope too steep
                if t0 > 1e-6:
                    slope = 2 * dx / (v0 * t0)
                    if np.arctan(slope) > max_dip_rad:
                        continue
                val += data[j, it]
                count += 1
            if count > 0:
                out[i, it0] = val / count
    return out


def kirchhoff_time_migration_torch(
    data: "torch.Tensor",
    dt_ms: float,
    dx_m: float,
    velocity_rms: "torch.Tensor",
    max_dip_deg: float = 45.0,
    max_freq_hz: float = 70.0,
) -> "torch.Tensor":
    """
    Differentiable-ish Kirchhoff migration in PyTorch (simplified: no full aperture).
    For training, gradient flows through interpolation.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for kirchhoff_time_migration_torch")
    ntr, ns = data.shape
    device = data.device
    dt_s = dt_ms / 1000.0
    t = torch.arange(ns, dtype=data.dtype, device=device) * dt_s
    v = velocity_rms
    if v.dim() == 0:
        v = v.expand(ns)
    v = v[:ns].reshape(-1)
    if v.size(0) < ns:
        v = torch.nn.functional.pad(v, (0, ns - v.size(0)), value=v[-1])
    # Simplified: small aperture sum for efficiency
    half = min(31, ntr // 2)
    out = torch.zeros_like(data)
    for di in range(-half, half + 1):
        j = torch.arange(ntr, device=device) + di
        j = torch.clamp(j, 0, ntr - 1)
        dx = torch.abs(di * dx_m)
        # t_mig = sqrt(t0^2 + (2*dx/v)^2)
        t_mig = torch.sqrt(t**2 + (2.0 * dx / v) ** 2)
        it_float = t_mig / dt_s
        it0 = torch.clamp(it_float.long(), 0, ns - 2)
        w = it_float - it0.float()
        val = (1 - w) * data[j, it0] + w * data[j, it0 + 1]
        out += val
    out /= (2 * half + 1)
    return out
