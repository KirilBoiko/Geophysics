"""
Band-pass filters, AGC, and time-variant operations.
Compatible with numpy and torch for use in pipeline.
"""

import numpy as np
from typing import Tuple, List, Union, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _design_bandpass(
    low: Tuple[float, float],
    high: Tuple[float, float],
    dt_s: float,
    ntaps: int = 101,
) -> np.ndarray:
    """Design a band-pass FIR (low and high are (f1, f2) taper in Hz)."""
    from scipy.signal import firwin
    fs = 1.0 / dt_s
    nyq = fs / 2.0
    if high[1] >= nyq * 0.99:
        # low-pass only
        return firwin(ntaps, high[0] / nyq, pass_zero=False)
    return firwin(
        ntaps,
        [low[0] / nyq, low[1] / nyq, high[0] / nyq, high[1] / nyq],
        pass_zero=False,
    )


def bandpass_filter(
    data: np.ndarray,
    dt_ms: float,
    low_hz: Tuple[float, float] = (4.0, 8.0),
    high_hz: Tuple[float, float] = (60.0, 70.0),
    axis: int = -1,
) -> np.ndarray:
    """
    Apply band-pass filter along axis (default: time axis).
    data: (..., n_time)
    """
    dt_s = dt_ms / 1000.0
    kernel = _design_bandpass(low_hz, high_hz, dt_s)
    from scipy.signal import convolve
    out = np.apply_along_axis(
        lambda x: convolve(x, kernel, mode="same"),
        axis,
        np.asarray(data, dtype=np.float64),
    )
    return out.astype(data.dtype)


def time_variant_bandpass(
    data: np.ndarray,
    dt_ms: float,
    time_gates: List[Tuple[float, float, Tuple[float, float], Tuple[float, float]]],
    axis: int = -1,
) -> np.ndarray:
    """
    Apply different band-pass in time windows.
    time_gates: list of (t_start_ms, t_end_ms, low_hz_pair, high_hz_pair).
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[axis]
    t = np.arange(n, dtype=np.float64) * dt_ms
    out = np.zeros_like(data)
    for t0, t1, low, high in time_gates:
        mask = (t >= t0) & (t <= t1)
        if not np.any(mask):
            continue
        kernel = _design_bandpass(low, high, dt_ms / 1000.0)
        from scipy.signal import convolve
        sl = [slice(None)] * data.ndim
        sl[axis] = mask
        segment = data[tuple(sl)].copy()
        if axis == -1:
            out[..., mask] = np.apply_along_axis(
                lambda x: convolve(x, kernel, mode="same"),
                axis,
                segment,
            )
        else:
            out[tuple(sl)] = np.apply_along_axis(
                lambda x: convolve(x, kernel, mode="same"),
                axis,
                segment,
            )
    # Fill gaps with original
    covered = np.zeros(n, dtype=bool)
    for t0, t1, _, _ in time_gates:
        covered |= (t >= t0) & (t <= t1)
    if not np.all(covered):
        out[..., ~covered] = data[..., ~covered]
    return out


def agc(
    data: Union[np.ndarray, "torch.Tensor"],
    window_ms: float,
    dt_ms: float,
    axis: int = -1,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Automatic gain control: normalize by sliding-window RMS.
    window_ms: operator length in ms.
    """
    window_samples = max(1, int(round(window_ms / dt_ms)))
    is_torch = HAS_TORCH and isinstance(data, torch.Tensor)
    if is_torch:
        x = data
        if axis != -1:
            x = x.transpose(axis, -1)
        pad = window_samples // 2
        x2 = x * x
        kernel = torch.ones(1, 1, window_samples, device=x.device, dtype=x.dtype) / window_samples
        if x.dim() == 2:
            rms = torch.sqrt(
                torch.nn.functional.conv1d(
                    x2.unsqueeze(0).unsqueeze(0),
                    kernel,
                    padding=pad,
                ).squeeze()
            )
        else:
            rms = torch.sqrt(
                torch.nn.functional.conv1d(
                    x2.unsqueeze(0),
                    kernel,
                    padding=pad,
                ).squeeze(0)
            )
        rms = torch.clamp(rms, min=1e-8)
        out = x / rms
        if axis != -1:
            out = out.transpose(axis, -1)
        return out
    x = np.asarray(data, dtype=np.float64)
    from scipy.ndimage import uniform_filter1d
    rms = np.sqrt(uniform_filter1d(x * x, size=window_samples, axis=axis, mode="nearest"))
    rms = np.clip(rms, 1e-8, None)
    return (x / rms).astype(data.dtype)


def fk_filter_fan(
    data: np.ndarray,
    dx_m: float,
    dt_ms: float,
    vel_min: float,
    vel_max: float,
    freq_range: Tuple[float, float] = (5.0, 50.0),
) -> np.ndarray:
    """
    Simple F-K fan filter (2D): keep energy in pass velocity range.
    data: (n_traces, n_samples). dx_m: trace spacing; dt_ms: sample interval.
    """
    from scipy.fft import fft2, ifft2, fftfreq
    ntr, ns = data.shape
    dt_s = dt_ms / 1000.0
    f = fftfreq(ns, dt_s)
    k = fftfreq(ntr, dx_m)
    # FFT2 layout: data is (ntr, ns) -> first dim k (trace), second dim f (time)
    K_2d, F_2d = np.meshgrid(k, f, indexing="ij")
    K_2d = np.broadcast_to(K_2d, (ntr, ns))
    F_2d = np.broadcast_to(F_2d, (ntr, ns))
    # Slope = f/k = v => v = f/k. Pass |v| in [vel_min, vel_max]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(np.abs(K_2d) > 1e-12, F_2d / K_2d, np.inf)
    mask = (np.abs(v) >= vel_min) & (np.abs(v) <= vel_max)
    mask &= (np.abs(F_2d) >= freq_range[0]) & (np.abs(F_2d) <= freq_range[1])
    mask |= (np.abs(F_2d) < freq_range[0])
    spec = fft2(data)
    spec[~mask] *= 0.0
    return np.real(ifft2(spec)).astype(data.dtype)
