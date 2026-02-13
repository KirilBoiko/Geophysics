"""
Processing sequence configuration (BS 11/00 Survey style).
Parameters from the 28-step sequence; all steps are optional when data/tools unavailable.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ProcessingConfig:
    """Config for the processing sequence. Steps can be disabled via tool_available."""

    # Input (step 1)
    input_dt_ms: float = 2.0
    input_duration_ms: float = 7100.0
    input_channels: int = 120

    # Step 2: Resample
    resample_dt_ms: float = 4.0

    # Step 3: Bulk statics
    bulk_statics_to_sea_level: bool = True

    # Step 5: TAR
    spherical_divergence: bool = True

    # Step 6: Band-pass (first pass)
    bandpass_low: Tuple[float, float] = (4.0, 8.0)   # Hz (low taper)
    bandpass_high: Tuple[float, float] = (60.0, 70.0)  # Hz (high taper)

    # Step 7: Velocity analysis
    velocity_analysis_step_km: float = 2.0

    # Step 8: AGC
    agc_operator_length_ms: float = 600.0

    # Step 10: F-K filter
    fk_vel_min: float = 1440.0   # m/s
    fk_vel_max: float = 2740.0
    fk_freq_range: Tuple[float, float] = (5.0, 50.0)

    # Step 11: Adaptive deconvolution
    deconv_operator_length_ms: float = 160.0
    deconv_predict_distance_ms: float = 32.0
    deconv_adapt_rate: float = 0.2

    # Step 15/17: Velocity for DMO/stack
    velocity_step_km: float = 2.0

    # Step 20: AGC (post-stack)
    agc_post_stack_length_ms: float = 600.0

    # Step 21: Predictive deconvolution
    pred_deconv_operator_ms: float = 240.0
    pred_deconv_predict_ms: float = 32.0
    pred_deconv_white_noise: float = 1.0

    # Step 22: F-X decon
    fx_window_length: int = 11
    fx_filter_samples: int = 7

    # Step 23: Kirchhoff time migration
    migration_max_dip_deg: float = 45.0
    migration_velocity_rms_factor: float = 0.95
    migration_max_freq_hz: float = 70.0

    # Step 24: Time-variant band-pass (time gates in ms)
    time_variant_bandpass: List[Tuple[float, float, Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=lambda: [
            (0.0, 1200.0, (7.0, 13.0), (50.0, 70.0)),
            (1500.0, 3000.0, (5.0, 10.0), (40.0, 50.0)),
            (3500.0, 6000.0, (4.0, 8.0), (25.0, 35.0)),
        ]
    )

    # Step 26: User gain values (time ms, gain)
    time_variant_gain: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 3300.0, 1.0), (4000.0, 6600.0, 0.4)]
    )

    # Step 27: Header static
    header_static_ms: float = 10.0

    # Which tools are available (model can still produce output when False)
    tool_available: dict = field(default_factory=lambda: {
        "resample": True,
        "bulk_statics": True,
        "trace_edit": True,
        "tar": True,
        "bandpass": True,
        "velocity_analysis": True,
        "agc": True,
        "muting": True,
        "fk_filter": True,
        "deconvolution": True,
        "radon": True,
        "dmo": True,
        "nmo_stack": True,
        "kirchhoff_migration": True,
        "time_variant_bandpass": True,
        "time_variant_scaling": True,
        "header_static": True,
    })

    def get_tool_available(self, name: str) -> bool:
        return self.tool_available.get(name, False)


DEFAULT_CONFIG = ProcessingConfig()
