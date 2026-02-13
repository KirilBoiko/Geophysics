"""Geophysics NN: multi-modal seismic/magnetics/gravity to SEG-Y 2D with adaptable processing."""

from .config import ProcessingConfig, DEFAULT_CONFIG

def __getattr__(name):
    if name == "GeophysicsPipelineNet":
        from .model.pipeline_net import GeophysicsPipelineNet
        return GeophysicsPipelineNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ProcessingConfig", "DEFAULT_CONFIG", "GeophysicsPipelineNet"]
