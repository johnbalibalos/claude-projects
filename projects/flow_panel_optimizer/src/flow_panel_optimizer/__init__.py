"""Flow Panel Optimizer - Calculate spectral similarity metrics for flow cytometry panel design."""

__version__ = "0.1.0"

from flow_panel_optimizer.models.spectrum import Spectrum
from flow_panel_optimizer.models.fluorophore import Fluorophore
from flow_panel_optimizer.models.panel import Panel

__all__ = ["Spectrum", "Fluorophore", "Panel", "__version__"]
