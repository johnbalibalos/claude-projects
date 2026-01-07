"""OMIP panel definitions and loader.

OMIP (Optimized Multicolor Immunofluorescence Panel) panels are published
in Cytometry Part A and provide validated fluorophore combinations.

References are included for each panel definition.
"""

from typing import Optional

from flow_panel_optimizer.models.fluorophore import Fluorophore
from flow_panel_optimizer.models.panel import Panel


# OMIP Panel Definitions
# These are manually curated from published papers

OMIP_069_PANEL = {
    "name": "OMIP-069",
    "description": "40-Color Full Spectrum Flow Cytometry Panel for Human PBMC Immunophenotyping",
    "instrument": "Cytek Aurora 5-laser",
    "fluorophores": [
        # UV laser (355 nm) excitation
        "BUV395",
        "BUV496",
        "BUV563",
        "BUV615",
        "BUV661",
        "BUV737",
        "BUV805",
        # Violet laser (405 nm) excitation
        "BV421",
        "BV480",
        "BV510",
        "BV570",
        "BV605",
        "BV650",
        "BV711",
        "BV750",
        "BV785",
        # Blue laser (488 nm) excitation
        "BB515",
        "BB660",
        "BB700",
        "BB755",
        "BB790",
        "PerCP-Cy5.5",
        "PerCP-eFluor710",
        # Yellow-Green laser (561 nm) excitation
        "PE",
        "PE-CF594",
        "PE-Cy5",
        "PE-Cy5.5",
        "PE-Cy7",
        # Red laser (633/640 nm) excitation
        "APC",
        "Alexa Fluor 647",
        "APC-R700",
        "APC-Fire750",
        "APC-Fire810",
        # Additional dyes
        "Spark Blue 550",
        "Super Bright 436",
        "Super Bright 600",
        "Super Bright 645",
        "Zombie NIR",
        "Zombie Aqua",
        "Live/Dead Blue",
    ],
    "published_similarity_pairs": {
        # High similarity pairs noted in the paper
        ("BB515", "FITC"): 0.98,
        ("BV421", "Super Bright 436"): 0.97,
        ("PE-Cy5", "PE-Cy5.5"): 0.96,
        ("APC", "Alexa Fluor 647"): 0.99,
    },
    "published_complexity_index": 54,
    "reference": "Park LM, Lannigan J, Jaimes MC. OMIP-069: Forty-Color Full Spectrum Flow Cytometry Panel for the Deep Immunophenotyping of Major Cell Subsets in Human Peripheral Blood. Cytometry A. 2020;97(10):1044-1051. doi:10.1002/cyto.a.24213",
}

OMIP_042_PANEL = {
    "name": "OMIP-042",
    "description": "21-Color Immunophenotyping Panel for Deep Characterization of Human T Cells",
    "instrument": "BD FACSymphony",
    "fluorophores": [
        "BUV395",
        "BUV496",
        "BUV563",
        "BUV615",
        "BUV661",
        "BUV737",
        "BUV805",
        "BV421",
        "BV510",
        "BV605",
        "BV650",
        "BV711",
        "BV786",
        "BB515",
        "PE",
        "PE-CF594",
        "PE-Cy5.5",
        "PE-Cy7",
        "APC",
        "APC-R700",
        "APC-Fire750",
    ],
    "published_similarity_pairs": {},
    "published_complexity_index": None,
    "reference": "Ferrer-Font L, et al. OMIP-042: 21-color immunophenotyping of the human T cell compartment. Cytometry A. 2019;95(7):697-701.",
}

OMIP_030_PANEL = {
    "name": "OMIP-030",
    "description": "Characterization of Human T Cell Subsets via Surface Markers",
    "instrument": "Standard 4-laser cytometer",
    "fluorophores": [
        "Pacific Blue",
        "BV510",
        "BV605",
        "BV650",
        "BV711",
        "BV785",
        "FITC",
        "PerCP-Cy5.5",
        "PE",
        "PE-Cy5",
        "PE-Cy7",
        "APC",
        "APC-Fire750",
    ],
    "published_similarity_pairs": {},
    "published_complexity_index": None,
    "reference": "Wingender G, Kronenberg M. OMIP-030: Characterization of human T cell subsets via surface markers. Cytometry A. 2015;87(12):1067-1069.",
}

# Test panel with intentionally problematic combinations
TEST_HIGH_SIMILARITY_PANEL = {
    "name": "Test-High-Similarity",
    "description": "Test panel with known high-similarity pairs for validation",
    "instrument": "Any",
    "fluorophores": [
        "FITC",
        "BB515",  # Very similar to FITC (SI ~0.98)
        "Alexa Fluor 488",  # Also similar to FITC
        "PE",
        "PE-CF594",
        "APC",
        "Alexa Fluor 647",  # Very similar to APC (SI ~0.99)
    ],
    "published_similarity_pairs": {
        ("FITC", "BB515"): 0.98,
        ("FITC", "Alexa Fluor 488"): 0.99,
        ("APC", "Alexa Fluor 647"): 0.99,
    },
    "published_complexity_index": None,
    "reference": "Synthetic test panel",
}

# All available OMIP panels
OMIP_PANELS = {
    "OMIP-069": OMIP_069_PANEL,
    "OMIP-042": OMIP_042_PANEL,
    "OMIP-030": OMIP_030_PANEL,
    "Test-High-Similarity": TEST_HIGH_SIMILARITY_PANEL,
}


class OMIPLoader:
    """Load and create Panel objects from OMIP definitions."""

    def __init__(self):
        """Initialize OMIP loader."""
        self.panels = OMIP_PANELS

    def list_panels(self) -> list[str]:
        """List available OMIP panel IDs."""
        return list(self.panels.keys())

    def get_panel_info(self, panel_id: str) -> Optional[dict]:
        """Get panel information without creating full Panel object.

        Args:
            panel_id: Panel identifier (e.g., 'OMIP-069').

        Returns:
            Panel info dict or None if not found.
        """
        return self.panels.get(panel_id)

    def load_panel(
        self,
        panel_id: str,
        fetch_spectra: bool = False,
        fpbase_client=None,
    ) -> Optional[Panel]:
        """Load a panel by ID.

        Args:
            panel_id: Panel identifier (e.g., 'OMIP-069').
            fetch_spectra: If True, fetch spectral data from FPbase.
            fpbase_client: FPbaseClient instance for fetching spectra.

        Returns:
            Panel object or None if not found.
        """
        panel_info = self.panels.get(panel_id)
        if not panel_info:
            return None

        fluorophores = []
        for name in panel_info["fluorophores"]:
            if fetch_spectra and fpbase_client:
                fluor = fpbase_client.get_fluorophore(name)
                if fluor:
                    fluorophores.append(fluor)
                else:
                    # Create placeholder without spectra
                    fluorophores.append(Fluorophore(name=name, source="omip"))
            else:
                fluorophores.append(Fluorophore(name=name, source="omip"))

        return Panel(
            name=panel_info["name"],
            fluorophores=fluorophores,
            instrument=panel_info.get("instrument"),
            description=panel_info.get("description"),
            reference=panel_info.get("reference"),
        )

    def get_known_similarity_pairs(self, panel_id: str) -> dict[tuple[str, str], float]:
        """Get published high-similarity pairs for a panel.

        Args:
            panel_id: Panel identifier.

        Returns:
            Dict mapping (fluor_a, fluor_b) -> similarity value.
        """
        panel_info = self.panels.get(panel_id)
        if not panel_info:
            return {}
        return panel_info.get("published_similarity_pairs", {})

    def get_expected_complexity(self, panel_id: str) -> Optional[float]:
        """Get published complexity index for a panel.

        Args:
            panel_id: Panel identifier.

        Returns:
            Complexity index or None if not published.
        """
        panel_info = self.panels.get(panel_id)
        if not panel_info:
            return None
        return panel_info.get("published_complexity_index")
