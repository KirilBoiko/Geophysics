"""
Run the BS 11/00-style processing sequence with optional steps.
When a tool is unavailable, the step is skipped and the previous result is passed through;
the caller (or NN) can fill in later.
"""

import numpy as np
from typing import Optional, Dict, Any
from ..config import ProcessingConfig


def run_processing_sequence(
    seismic: np.ndarray,
    config: ProcessingConfig,
    dt_ms: Optional[float] = None,
    dx_m: float = 25.0,
    velocity_rms: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply the 28-step sequence where tools are available.
    seismic: (n_traces, n_samples). Returns processed section; skips steps when tool_available is False.
    """
    from .filters import bandpass_filter, time_variant_bandpass, agc, fk_filter_fan
    from .migration import kirchhoff_time_migration

    dt_ms = dt_ms or config.resample_dt_ms
    data = np.asarray(seismic, dtype=np.float64)
    ntr, ns = data.shape

    # Step 2: Resample (if input is at raw rate and we want coarser sampling)
    if config.get_tool_available("resample") and dt_ms == config.input_dt_ms and config.input_dt_ms != config.resample_dt_ms:
        if config.input_dt_ms < config.resample_dt_ms:
            from scipy.signal import resample
            new_ns = int(ns * config.input_dt_ms / config.resample_dt_ms)
            data = resample(data, new_ns, axis=1)
            ns = new_ns
            dt_ms = config.resample_dt_ms

    # Step 3: Bulk statics (to sea level) - skip or zero for simplicity
    if config.get_tool_available("bulk_statics") and config.bulk_statics_to_sea_level:
        pass  # Would apply time shift; leave as-is for generic case

    # Step 4: Trace edit - skip (noise edit typically manual)
    if config.get_tool_available("trace_edit"):
        pass

    # Step 5: TAR - spherical divergence
    if config.get_tool_available("tar") and config.spherical_divergence:
        t = np.arange(ns, dtype=np.float64) * (dt_ms / 1000.0)
        t = np.maximum(t, 0.001)
        data = data * t  # simple divergence correction

    # Step 6: Band-pass 4-8-60-70 Hz
    if config.get_tool_available("bandpass"):
        data = bandpass_filter(
            data, dt_ms,
            low_hz=config.bandpass_low,
            high_hz=config.bandpass_high,
            axis=1,
        )

    # Step 7: Velocity analysis - we don't modify data; use provided velocity_rms later
    # Step 8: AGC 600 ms
    if config.get_tool_available("agc"):
        data = agc(data, config.agc_operator_length_ms, dt_ms, axis=1)

    # Step 9: Top muting - zero first few samples
    if config.get_tool_available("muting"):
        mute_samples = min(ns // 10, int(200 / dt_ms))
        data[:, :mute_samples] = 0.0

    # Step 10: F-K filter
    if config.get_tool_available("fk_filter"):
        data = fk_filter_fan(
            data, dx_m, dt_ms,
            config.fk_vel_min, config.fk_vel_max,
            config.fk_freq_range,
        )

    # Step 11: Adaptive deconvolution - skip (complex); use predictive later
    if config.get_tool_available("deconvolution"):
        pass

    # Step 12: Top muting again
    if config.get_tool_available("muting"):
        mute_samples = min(ns // 10, int(200 / dt_ms))
        data[:, :mute_samples] = 0.0

    # Step 13: Radon - skip (would need velocity panels)
    if config.get_tool_available("radon"):
        pass

    # Step 14: Remove AGC - we don't track AGC state; skip
    # Step 15-16: DMO - skip for 2D simplified
    if config.get_tool_available("dmo"):
        pass
    # Step 17-18: Velocity for stack, NMO + muting - assume already stacked input
    if config.get_tool_available("nmo_stack"):
        pass
    # Step 19: CDP stack - assume input is already stacked
    # Step 20: AGC 600 ms
    if config.get_tool_available("agc"):
        data = agc(data, config.agc_post_stack_length_ms, dt_ms, axis=1)
    # Step 21: Predictive deconvolution - skip (operator-based)
    # Step 22: F-X decon - skip
    if config.get_tool_available("deconvolution"):
        pass

    # Step 23: Kirchhoff time migration
    if config.get_tool_available("kirchhoff_migration"):
        v_rms = velocity_rms
        if v_rms is None:
            v_rms = 2500.0  # default m/s
        data = kirchhoff_time_migration(
            data, dt_ms, dx_m,
            v_rms,
            max_dip_deg=config.migration_max_dip_deg,
            rms_smooth=True,
            max_freq_hz=config.migration_max_freq_hz,
        )

    # Step 24: Time-variant band-pass
    if config.get_tool_available("time_variant_bandpass") and config.time_variant_bandpass:
        data = time_variant_bandpass(
            data, dt_ms,
            [(t0, t1, low, high) for t0, t1, low, high in config.time_variant_bandpass],
            axis=1,
        )

    # Step 25-26: Time-variant scaling
    if config.get_tool_available("time_variant_scaling") and config.time_variant_gain:
        t = np.arange(ns, dtype=np.float64) * dt_ms
        gain = np.ones(ns, dtype=np.float64)
        for t0, t1, g in config.time_variant_gain:
            mask = (t >= t0) & (t <= t1)
            gain[mask] = g
        data = data * gain

    # Step 27: Header static - time shift (here we could shift samples)
    if config.get_tool_available("header_static") and config.header_static_ms != 0:
        shift = int(round(config.header_static_ms / dt_ms))
        if shift > 0:
            data = np.roll(data, shift, axis=1)
            data[:, :shift] = 0.0
        elif shift < 0:
            data = np.roll(data, shift, axis=1)
            data[:, shift:] = 0.0

    return data
