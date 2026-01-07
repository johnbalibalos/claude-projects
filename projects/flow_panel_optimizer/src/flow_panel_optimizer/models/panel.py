"""Panel data model for flow cytometry panel definitions."""

from dataclasses import dataclass, field
from typing import Optional

from flow_panel_optimizer.models.fluorophore import Fluorophore


@dataclass
class Panel:
    """A flow cytometry panel definition.

    Attributes:
        name: Panel identifier (e.g., 'OMIP-069', 'Custom Panel 1').
        fluorophores: List of fluorophores in the panel.
        instrument: Target instrument (e.g., 'Cytek Aurora 5-laser').
        description: Human-readable description of the panel.
        reference: Publication or source reference.
        markers: Optional dict mapping fluorophore name to cell marker (e.g., {'PE': 'CD4'}).
    """

    name: str
    fluorophores: list[Fluorophore] = field(default_factory=list)
    instrument: Optional[str] = None
    description: Optional[str] = None
    reference: Optional[str] = None
    markers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate panel data."""
        if not self.name:
            raise ValueError("Panel name cannot be empty")

    def add_fluorophore(self, fluorophore: Fluorophore, marker: Optional[str] = None):
        """Add a fluorophore to the panel.

        Args:
            fluorophore: Fluorophore to add.
            marker: Optional cell marker this fluorophore is conjugated to.
        """
        if fluorophore not in self.fluorophores:
            self.fluorophores.append(fluorophore)
            if marker:
                self.markers[fluorophore.name] = marker

    def remove_fluorophore(self, name: str) -> bool:
        """Remove a fluorophore by name.

        Args:
            name: Name of fluorophore to remove.

        Returns:
            True if removed, False if not found.
        """
        for i, f in enumerate(self.fluorophores):
            if f.name == name:
                self.fluorophores.pop(i)
                self.markers.pop(name, None)
                return True
        return False

    def get_fluorophore(self, name: str) -> Optional[Fluorophore]:
        """Get a fluorophore by name or alias.

        Args:
            name: Name or alias to search for.

        Returns:
            Fluorophore if found, None otherwise.
        """
        for f in self.fluorophores:
            if f.matches_name(name):
                return f
        return None

    def get_fluorophore_names(self) -> list[str]:
        """Get list of all fluorophore names in the panel."""
        return [f.name for f in self.fluorophores]

    def size(self) -> int:
        """Get number of fluorophores in panel."""
        return len(self.fluorophores)

    def has_fluorophore(self, name: str) -> bool:
        """Check if panel contains a fluorophore."""
        return self.get_fluorophore(name) is not None

    def get_fluorophores_by_laser(self) -> dict[str, list[Fluorophore]]:
        """Group fluorophores by their primary laser.

        Returns:
            Dict mapping laser name to list of fluorophores.
        """
        by_laser: dict[str, list[Fluorophore]] = {}
        for f in self.fluorophores:
            laser = f.get_laser_line() or "unknown"
            if laser not in by_laser:
                by_laser[laser] = []
            by_laser[laser].append(f)
        return by_laser

    def get_marker(self, fluorophore_name: str) -> Optional[str]:
        """Get the marker associated with a fluorophore."""
        return self.markers.get(fluorophore_name)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "fluorophores": [f.to_dict() for f in self.fluorophores],
            "instrument": self.instrument,
            "description": self.description,
            "reference": self.reference,
            "markers": self.markers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Panel":
        """Create Panel from dictionary."""
        fluorophores = [Fluorophore.from_dict(f) for f in data.get("fluorophores", [])]
        return cls(
            name=data["name"],
            fluorophores=fluorophores,
            instrument=data.get("instrument"),
            description=data.get("description"),
            reference=data.get("reference"),
            markers=data.get("markers", {}),
        )

    def __len__(self) -> int:
        """Return number of fluorophores."""
        return len(self.fluorophores)

    def __iter__(self):
        """Iterate over fluorophores."""
        return iter(self.fluorophores)

    def __contains__(self, item):
        """Check if fluorophore in panel."""
        if isinstance(item, Fluorophore):
            return item in self.fluorophores
        if isinstance(item, str):
            return self.has_fluorophore(item)
        return False
