"""Fluorophore data model."""

from dataclasses import dataclass, field
from typing import Optional

from flow_panel_optimizer.models.spectrum import Spectrum


@dataclass
class Fluorophore:
    """A fluorescent dye or protein used in flow cytometry.

    Attributes:
        name: Primary name of the fluorophore (e.g., 'PE', 'FITC', 'BV421').
        aliases: Alternative names (e.g., ['Phycoerythrin', 'R-PE']).
        excitation_max: Peak excitation wavelength in nm.
        emission_max: Peak emission wavelength in nm.
        emission_spectrum: Full emission spectrum data.
        excitation_spectrum: Full excitation spectrum data.
        stain_index: Brightness metric (higher = brighter signal).
        extinction_coefficient: Molar extinction coefficient.
        quantum_yield: Fluorescence quantum yield (0-1).
        primary_laser: Optimal excitation laser (e.g., 'violet', 'blue', 'yellow-green', 'red').
        source: Data source (e.g., 'fpbase', 'cytek', 'manual').
    """

    name: str
    aliases: list[str] = field(default_factory=list)
    excitation_max: Optional[float] = None
    emission_max: Optional[float] = None
    emission_spectrum: Optional[Spectrum] = None
    excitation_spectrum: Optional[Spectrum] = None
    stain_index: Optional[float] = None
    extinction_coefficient: Optional[float] = None
    quantum_yield: Optional[float] = None
    primary_laser: Optional[str] = None
    source: str = "unknown"

    def __post_init__(self):
        """Validate fluorophore data."""
        if not self.name:
            raise ValueError("Fluorophore name cannot be empty")

    def get_emission_peak(self) -> Optional[float]:
        """Get emission peak from spectrum or stored value."""
        if self.emission_spectrum is not None:
            return self.emission_spectrum.get_peak_wavelength()
        return self.emission_max

    def get_excitation_peak(self) -> Optional[float]:
        """Get excitation peak from spectrum or stored value."""
        if self.excitation_spectrum is not None:
            return self.excitation_spectrum.get_peak_wavelength()
        return self.excitation_max

    def has_emission_spectrum(self) -> bool:
        """Check if full emission spectrum is available."""
        return self.emission_spectrum is not None

    def get_laser_line(self) -> Optional[str]:
        """Get the recommended laser line based on excitation peak.

        Returns:
            Laser name or None if excitation data unavailable.
        """
        if self.primary_laser:
            return self.primary_laser

        ex_peak = self.get_excitation_peak()
        if ex_peak is None:
            return None

        # Standard flow cytometry laser lines
        if 395 <= ex_peak <= 420:
            return "violet"  # 405 nm
        elif 470 <= ex_peak <= 500:
            return "blue"  # 488 nm
        elif 530 <= ex_peak <= 570:
            return "yellow-green"  # 561 nm
        elif 620 <= ex_peak <= 650:
            return "red"  # 633/640 nm
        elif 340 <= ex_peak <= 365:
            return "UV"  # 355 nm
        else:
            return "unknown"

    def matches_name(self, query: str) -> bool:
        """Check if query matches this fluorophore's name or aliases.

        Args:
            query: Name to search for (case-insensitive).

        Returns:
            True if query matches name or any alias.
        """
        query_lower = query.lower().strip()
        if self.name.lower() == query_lower:
            return True
        return any(alias.lower() == query_lower for alias in self.aliases)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = {
            "name": self.name,
            "aliases": self.aliases,
            "excitation_max": self.excitation_max,
            "emission_max": self.emission_max,
            "stain_index": self.stain_index,
            "extinction_coefficient": self.extinction_coefficient,
            "quantum_yield": self.quantum_yield,
            "primary_laser": self.primary_laser,
            "source": self.source,
        }
        if self.emission_spectrum:
            data["emission_spectrum"] = self.emission_spectrum.to_dict()
        if self.excitation_spectrum:
            data["excitation_spectrum"] = self.excitation_spectrum.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Fluorophore":
        """Create Fluorophore from dictionary."""
        emission_spectrum = None
        excitation_spectrum = None

        if "emission_spectrum" in data and data["emission_spectrum"]:
            emission_spectrum = Spectrum.from_dict(data["emission_spectrum"])
        if "excitation_spectrum" in data and data["excitation_spectrum"]:
            excitation_spectrum = Spectrum.from_dict(data["excitation_spectrum"])

        return cls(
            name=data["name"],
            aliases=data.get("aliases", []),
            excitation_max=data.get("excitation_max"),
            emission_max=data.get("emission_max"),
            emission_spectrum=emission_spectrum,
            excitation_spectrum=excitation_spectrum,
            stain_index=data.get("stain_index"),
            extinction_coefficient=data.get("extinction_coefficient"),
            quantum_yield=data.get("quantum_yield"),
            primary_laser=data.get("primary_laser"),
            source=data.get("source", "unknown"),
        )

    def __hash__(self):
        """Hash based on name for use in sets/dicts."""
        return hash(self.name)

    def __eq__(self, other):
        """Equality based on name."""
        if isinstance(other, Fluorophore):
            return self.name == other.name
        return False
