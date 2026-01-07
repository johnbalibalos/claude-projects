"""Spectrum data model for emission/excitation spectra."""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass
class Spectrum:
    """Emission or excitation spectrum.

    Attributes:
        wavelengths: Array of wavelength values in nm.
        intensities: Array of intensity values (normalized 0-1).
        spectrum_type: Type of spectrum - 'emission', 'excitation', or 'absorption'.
        source: Data source - 'fpbase', 'cytek', 'manual', etc.
    """

    wavelengths: np.ndarray
    intensities: np.ndarray
    spectrum_type: Literal["emission", "excitation", "absorption"]
    source: str = "unknown"

    def __post_init__(self):
        """Validate and normalize spectrum data."""
        self.wavelengths = np.asarray(self.wavelengths, dtype=np.float64)
        self.intensities = np.asarray(self.intensities, dtype=np.float64)

        if len(self.wavelengths) != len(self.intensities):
            raise ValueError(
                f"Wavelengths and intensities must have same length. "
                f"Got {len(self.wavelengths)} and {len(self.intensities)}"
            )

        if len(self.wavelengths) == 0:
            raise ValueError("Spectrum must have at least one data point")

    def normalize(self) -> "Spectrum":
        """Return a new Spectrum with intensities normalized to 0-1 range."""
        max_val = np.max(self.intensities)
        if max_val > 0:
            normalized = self.intensities / max_val
        else:
            normalized = self.intensities
        return Spectrum(
            wavelengths=self.wavelengths.copy(),
            intensities=normalized,
            spectrum_type=self.spectrum_type,
            source=self.source,
        )

    def interpolate(self, target_wavelengths: np.ndarray) -> "Spectrum":
        """Interpolate spectrum to match target wavelength grid.

        Args:
            target_wavelengths: Array of wavelengths to interpolate to.

        Returns:
            New Spectrum with interpolated values at target wavelengths.
        """
        interpolated = np.interp(
            target_wavelengths,
            self.wavelengths,
            self.intensities,
            left=0.0,
            right=0.0,
        )
        return Spectrum(
            wavelengths=target_wavelengths.copy(),
            intensities=interpolated,
            spectrum_type=self.spectrum_type,
            source=self.source,
        )

    def get_peak_wavelength(self) -> float:
        """Get wavelength of maximum intensity."""
        peak_idx = np.argmax(self.intensities)
        return float(self.wavelengths[peak_idx])

    def get_fwhm(self) -> float:
        """Calculate full width at half maximum (FWHM).

        Returns:
            FWHM in nm, or 0 if cannot be calculated.
        """
        half_max = np.max(self.intensities) / 2
        above_half = self.intensities >= half_max

        if not np.any(above_half):
            return 0.0

        indices = np.where(above_half)[0]
        if len(indices) < 2:
            return 0.0

        return float(self.wavelengths[indices[-1]] - self.wavelengths[indices[0]])

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "wavelengths": self.wavelengths.tolist(),
            "intensities": self.intensities.tolist(),
            "spectrum_type": self.spectrum_type,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Spectrum":
        """Create Spectrum from dictionary."""
        return cls(
            wavelengths=np.array(data["wavelengths"]),
            intensities=np.array(data["intensities"]),
            spectrum_type=data["spectrum_type"],
            source=data.get("source", "unknown"),
        )

    def __len__(self) -> int:
        """Return number of data points."""
        return len(self.wavelengths)
