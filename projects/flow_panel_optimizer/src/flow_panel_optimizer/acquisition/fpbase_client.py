"""FPbase GraphQL API client for fluorophore spectra."""

import json
from typing import Optional
import numpy as np
import requests

from flow_panel_optimizer.models.spectrum import Spectrum
from flow_panel_optimizer.models.fluorophore import Fluorophore


class FPbaseClient:
    """Client for fetching fluorophore data from FPbase GraphQL API.

    FPbase (https://www.fpbase.org/) is a database of fluorescent proteins
    and dyes with spectral data. This client uses the public GraphQL API.

    Note: FPbase focuses on microscopy fluorophores. Many flow cytometry
    tandem dyes may have incomplete coverage.
    """

    GRAPHQL_URL = "https://www.fpbase.org/graphql/"

    # Common flow cytometry dyes with their FPbase IDs
    FLOW_DYES = {
        "FITC": "fitc",
        "PE": "pe",
        "APC": "apc",
        "PerCP": "percp",
        "Alexa Fluor 488": "alexa-fluor-488",
        "Alexa Fluor 647": "alexa-fluor-647",
        "Alexa Fluor 700": "alexa-fluor-700",
        "Pacific Blue": "pacific-blue",
        "PE-Cy5": "pe-cy5",
        "PE-Cy7": "pe-cy7",
        "APC-Cy7": "apc-cy7",
        "BV421": "brilliant-violet-421",
        "BV510": "brilliant-violet-510",
        "BV605": "brilliant-violet-605",
        "BV650": "brilliant-violet-650",
        "BV711": "brilliant-violet-711",
        "BV750": "brilliant-violet-750",
        "BV785": "brilliant-violet-785",
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize FPbase client.

        Args:
            cache_dir: Optional directory for caching responses.
        """
        self.cache_dir = cache_dir
        self._cache: dict[str, dict] = {}

    def _query(self, query: str, variables: Optional[dict] = None) -> dict:
        """Execute a GraphQL query.

        Args:
            query: GraphQL query string.
            variables: Optional query variables.

        Returns:
            Query response data.

        Raises:
            RuntimeError: If query fails.
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            response = requests.post(
                self.GRAPHQL_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                raise RuntimeError(f"GraphQL errors: {result['errors']}")

            return result.get("data", {})

        except requests.RequestException as e:
            raise RuntimeError(f"FPbase API request failed: {e}")

    def get_fluorophore(self, name: str) -> Optional[Fluorophore]:
        """Fetch fluorophore data by name or slug.

        Args:
            name: Fluorophore name or FPbase slug.

        Returns:
            Fluorophore with spectral data, or None if not found.
        """
        # Try to find the FPbase slug
        slug = self.FLOW_DYES.get(name, name.lower().replace(" ", "-"))

        query = """
        query GetFluorophore($slug: String!) {
            dye(slug: $slug) {
                name
                slug
                exMax
                emMax
                extCoeff
                qy
                spectra {
                    subtype
                    data
                }
            }
        }
        """

        try:
            data = self._query(query, {"slug": slug})
            dye_data = data.get("dye")

            if not dye_data:
                # Try as a protein instead
                return self._get_protein(slug)

            return self._parse_fluorophore(dye_data, "dye")

        except RuntimeError:
            return None

    def _get_protein(self, slug: str) -> Optional[Fluorophore]:
        """Fetch fluorescent protein data.

        Args:
            slug: FPbase slug for the protein.

        Returns:
            Fluorophore with spectral data, or None if not found.
        """
        query = """
        query GetProtein($slug: String!) {
            protein(slug: $slug) {
                name
                slug
                defaultState {
                    exMax
                    emMax
                    extCoeff
                    qy
                    spectra {
                        subtype
                        data
                    }
                }
            }
        }
        """

        try:
            data = self._query(query, {"slug": slug})
            protein_data = data.get("protein")

            if not protein_data or not protein_data.get("defaultState"):
                return None

            state = protein_data["defaultState"]
            return self._parse_fluorophore(
                {
                    "name": protein_data["name"],
                    "slug": protein_data["slug"],
                    "exMax": state.get("exMax"),
                    "emMax": state.get("emMax"),
                    "extCoeff": state.get("extCoeff"),
                    "qy": state.get("qy"),
                    "spectra": state.get("spectra", []),
                },
                "protein",
            )

        except RuntimeError:
            return None

    def _parse_fluorophore(self, data: dict, source_type: str) -> Fluorophore:
        """Parse API response into Fluorophore object.

        Args:
            data: API response data.
            source_type: 'dye' or 'protein'.

        Returns:
            Populated Fluorophore object.
        """
        emission_spectrum = None
        excitation_spectrum = None

        for spectrum in data.get("spectra", []):
            subtype = spectrum.get("subtype", "").upper()
            spec_data = spectrum.get("data")

            if not spec_data:
                continue

            # Parse spectrum data - format is [[wavelength, intensity], ...]
            if isinstance(spec_data, str):
                spec_data = json.loads(spec_data)

            wavelengths = np.array([point[0] for point in spec_data])
            intensities = np.array([point[1] for point in spec_data])

            if subtype in ("EM", "EMISSION"):
                emission_spectrum = Spectrum(
                    wavelengths=wavelengths,
                    intensities=intensities,
                    spectrum_type="emission",
                    source="fpbase",
                )
            elif subtype in ("EX", "EXCITATION", "AB", "ABSORPTION"):
                excitation_spectrum = Spectrum(
                    wavelengths=wavelengths,
                    intensities=intensities,
                    spectrum_type="excitation",
                    source="fpbase",
                )

        return Fluorophore(
            name=data["name"],
            aliases=[data.get("slug", "")],
            excitation_max=data.get("exMax"),
            emission_max=data.get("emMax"),
            emission_spectrum=emission_spectrum,
            excitation_spectrum=excitation_spectrum,
            extinction_coefficient=data.get("extCoeff"),
            quantum_yield=data.get("qy"),
            source="fpbase",
        )

    def search_fluorophores(self, query: str, limit: int = 20) -> list[dict]:
        """Search for fluorophores by name.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of matching fluorophore info dicts.
        """
        gql_query = """
        query SearchFluorophores($query: String!) {
            dyes(name_Icontains: $query, first: 20) {
                edges {
                    node {
                        name
                        slug
                        exMax
                        emMax
                    }
                }
            }
        }
        """

        try:
            data = self._query(gql_query, {"query": query})
            edges = data.get("dyes", {}).get("edges", [])
            return [
                {
                    "name": edge["node"]["name"],
                    "slug": edge["node"]["slug"],
                    "excitation_max": edge["node"].get("exMax"),
                    "emission_max": edge["node"].get("emMax"),
                }
                for edge in edges[:limit]
            ]
        except RuntimeError:
            return []

    def get_multiple(self, names: list[str]) -> dict[str, Fluorophore]:
        """Fetch multiple fluorophores.

        Args:
            names: List of fluorophore names.

        Returns:
            Dict mapping name to Fluorophore (only found ones).
        """
        result = {}
        for name in names:
            fluor = self.get_fluorophore(name)
            if fluor:
                result[name] = fluor
        return result
