"""Cytek PDF extraction for fluorochrome spectral data."""

import re
from pathlib import Path
from typing import Optional
import numpy as np

# pdfplumber is imported lazily in __init__ to avoid import errors
# when the cryptography/pdfminer dependencies have issues
pdfplumber = None


def _import_pdfplumber():
    """Lazily import pdfplumber to avoid import-time failures."""
    global pdfplumber
    if pdfplumber is None:
        try:
            import pdfplumber as _pdfplumber
            pdfplumber = _pdfplumber
        except Exception as e:
            raise ImportError(
                f"pdfplumber could not be imported: {e}. "
                "Install with: pip install pdfplumber"
            )
    return pdfplumber


class CytekPDFExtractor:
    """Extract spectral data from Cytek fluorochrome guide PDFs.

    Cytek publishes publicly available fluorochrome guides containing:
    - Spread matrices (NxN heatmaps)
    - Stain index values
    - Spectrum signatures (bar charts per detector channel)

    PDF URLs (publicly available):
        4-laser VBYGR: N9_20018_Rev._A_4L_VBYGR_Fluor_Guide.pdf
        3-laser VBR: N9_20019_Rev._A_3L_VBR_Fluor_Guide.pdf
        2-laser BR: N9_20020_Rev._A_2L_BR_Fluor_Guide.pdf
        2-laser VB: N9_20021_Rev._A_2L_VB_FLuor_Guide.pdf
    """

    # Cytek detector channel names for different configurations
    DETECTOR_CHANNELS = {
        "5-laser": {
            "UV": ["UV1", "UV2", "UV3", "UV4", "UV5", "UV6", "UV7", "UV8", "UV9", "UV10",
                   "UV11", "UV12", "UV13", "UV14", "UV15", "UV16"],
            "V": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                  "V11", "V12", "V13", "V14", "V15", "V16"],
            "B": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10",
                  "B11", "B12", "B13", "B14"],
            "YG": ["YG1", "YG2", "YG3", "YG4", "YG5", "YG6", "YG7", "YG8", "YG9", "YG10"],
            "R": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
        },
        "4-laser": {
            "V": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                  "V11", "V12", "V13", "V14", "V15", "V16"],
            "B": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10",
                  "B11", "B12", "B13", "B14"],
            "YG": ["YG1", "YG2", "YG3", "YG4", "YG5", "YG6", "YG7", "YG8", "YG9", "YG10"],
            "R": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
        },
    }

    def __init__(self):
        """Initialize Cytek PDF extractor."""
        # Lazily import pdfplumber to avoid import-time failures
        _import_pdfplumber()

    def extract_spread_matrix(self, pdf_path: Path) -> dict:
        """Extract NxN spread matrix from Cytek PDF.

        The spread matrix shows how much spillover spreading occurs between
        each pair of fluorophores. Values are typically on a 0-100 scale.

        Args:
            pdf_path: Path to Cytek fluorochrome guide PDF.

        Returns:
            dict with keys:
                - fluorophores: list of fluorophore names
                - matrix: np.ndarray of spread values (0-100 scale)
                - page_number: which page the matrix was found on

        Raises:
            FileNotFoundError: If PDF not found.
            ValueError: If matrix cannot be extracted.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        fluorophores = []
        matrix_data = []
        page_found = -1

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Look for tables that might be spread matrices
                tables = page.extract_tables()

                for table in tables:
                    if self._is_spread_matrix_table(table):
                        fluorophores, matrix_data = self._parse_spread_table(table)
                        page_found = page_num + 1
                        break

                if fluorophores:
                    break

        if not fluorophores:
            raise ValueError(
                f"Could not extract spread matrix from {pdf_path}. "
                "The PDF format may have changed or matrix not found."
            )

        return {
            "fluorophores": fluorophores,
            "matrix": np.array(matrix_data),
            "page_number": page_found,
        }

    def _is_spread_matrix_table(self, table: list[list]) -> bool:
        """Check if a table looks like a spread matrix.

        Args:
            table: Extracted table data.

        Returns:
            True if table appears to be a spread matrix.
        """
        if not table or len(table) < 5:
            return False

        # Check for numeric values in expected range
        numeric_count = 0
        for row in table[1:]:  # Skip header
            for cell in row[1:]:  # Skip row label
                if cell:
                    try:
                        val = float(str(cell).replace(",", ""))
                        if 0 <= val <= 100:
                            numeric_count += 1
                    except (ValueError, TypeError):
                        pass

        # Need substantial numeric content
        return numeric_count > 10

    def _parse_spread_table(self, table: list[list]) -> tuple[list[str], list[list[float]]]:
        """Parse a spread matrix table.

        Args:
            table: Extracted table data.

        Returns:
            Tuple of (fluorophore_names, matrix_values).
        """
        # First row should be header with fluorophore names
        header = [str(cell).strip() if cell else "" for cell in table[0]]

        # Filter out empty headers and find fluorophore names
        fluorophores = [h for h in header[1:] if h and not h.isdigit()]

        matrix = []
        for row in table[1:]:
            if not row or not row[0]:
                continue

            row_name = str(row[0]).strip()
            if row_name in fluorophores or self._looks_like_fluorophore(row_name):
                row_values = []
                for cell in row[1:len(fluorophores) + 1]:
                    try:
                        val = float(str(cell).replace(",", "")) if cell else 0.0
                        row_values.append(val)
                    except (ValueError, TypeError):
                        row_values.append(0.0)

                if row_values:
                    matrix.append(row_values)

        return fluorophores, matrix

    def _looks_like_fluorophore(self, name: str) -> bool:
        """Check if a string looks like a fluorophore name."""
        # Common patterns in fluorophore names
        patterns = [
            r"^BV\d+",
            r"^BUV\d+",
            r"^BB\d+",
            r"^PE",
            r"^APC",
            r"^FITC",
            r"^PerCP",
            r"^Alexa",
            r"Cy\d+",
        ]
        return any(re.match(p, name, re.IGNORECASE) for p in patterns)

    def extract_stain_indices(self, pdf_path: Path) -> dict[str, float]:
        """Extract stain index values per fluorophore.

        Stain index is a measure of fluorophore brightness that accounts
        for both signal intensity and spread.

        Args:
            pdf_path: Path to Cytek fluorochrome guide PDF.

        Returns:
            dict mapping fluorophore name -> stain index (float).

        Raises:
            FileNotFoundError: If PDF not found.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        stain_indices = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""

                # Look for stain index patterns like "PE: 245" or "PE (245)"
                # Also look in tables
                tables = page.extract_tables()

                for table in tables:
                    if self._is_stain_index_table(table):
                        stain_indices.update(self._parse_stain_index_table(table))

        return stain_indices

    def _is_stain_index_table(self, table: list[list]) -> bool:
        """Check if a table contains stain index data."""
        if not table or len(table) < 2:
            return False

        # Look for "Stain Index" or "SI" in header
        header_text = " ".join(str(cell) for cell in table[0] if cell).lower()
        return "stain" in header_text or "si" in header_text.split()

    def _parse_stain_index_table(self, table: list[list]) -> dict[str, float]:
        """Parse stain index values from a table."""
        result = {}

        # Find the stain index column
        header = table[0]
        si_col = -1
        for i, cell in enumerate(header):
            if cell and "stain" in str(cell).lower():
                si_col = i
                break

        if si_col < 0:
            return result

        for row in table[1:]:
            if len(row) <= si_col:
                continue

            name = str(row[0]).strip() if row[0] else ""
            if not name or not self._looks_like_fluorophore(name):
                continue

            try:
                value = float(str(row[si_col]).replace(",", ""))
                result[name] = value
            except (ValueError, TypeError):
                pass

        return result

    def extract_spectrum_signatures(
        self, pdf_path: Path
    ) -> dict[str, dict[str, list]]:
        """Extract spectrum signatures from bar charts.

        Note: This is complex as spectral data is often in images/charts
        rather than tables. This method attempts to extract tabular data
        that represents spectrum signatures.

        Args:
            pdf_path: Path to Cytek fluorochrome guide PDF.

        Returns:
            dict mapping fluorophore name -> {
                'channels': ['V1', 'V2', ...],
                'values': [mfi_v1, mfi_v2, ...]
            }

        Raises:
            FileNotFoundError: If PDF not found.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        signatures = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()

                for table in tables:
                    if self._is_signature_table(table):
                        signatures.update(self._parse_signature_table(table))

        return signatures

    def _is_signature_table(self, table: list[list]) -> bool:
        """Check if a table contains spectrum signature data."""
        if not table or len(table) < 2:
            return False

        # Look for channel names in header (V1, B1, etc.)
        header_text = " ".join(str(cell) for cell in table[0] if cell)
        channel_patterns = [r"V\d", r"B\d", r"YG\d", r"R\d", r"UV\d"]

        return any(re.search(p, header_text) for p in channel_patterns)

    def _parse_signature_table(
        self, table: list[list]
    ) -> dict[str, dict[str, list]]:
        """Parse spectrum signatures from a table."""
        result = {}

        if not table:
            return result

        # Header contains channel names
        channels = []
        for cell in table[0][1:]:
            if cell:
                channel = str(cell).strip()
                if re.match(r"^(V|B|YG|R|UV)\d+", channel):
                    channels.append(channel)

        if not channels:
            return result

        for row in table[1:]:
            if not row or not row[0]:
                continue

            name = str(row[0]).strip()
            if not self._looks_like_fluorophore(name):
                continue

            values = []
            for cell in row[1:len(channels) + 1]:
                try:
                    val = float(str(cell).replace(",", "")) if cell else 0.0
                    values.append(val)
                except (ValueError, TypeError):
                    values.append(0.0)

            if values:
                result[name] = {"channels": channels, "values": values}

        return result


def download_cytek_pdfs(output_dir: Path) -> list[Path]:
    """Download Cytek fluorochrome guide PDFs.

    Args:
        output_dir: Directory to save PDFs.

    Returns:
        List of downloaded file paths.

    Note:
        These are publicly available PDFs from Cytek's website.
    """
    import requests

    urls = {
        "4L_VBYGR": "https://welcome.cytekbio.com/hubfs/Website%20Downloadable%20Content/Data%20Sheets/Fluorochrome%20Guides/N9_20018_Rev._A_4L_VBYGR_Fluor_Guide.pdf",
        "3L_VBR": "https://welcome.cytekbio.com/hubfs/Website%20Downloadable%20Content/Data%20Sheets/Fluorochrome%20Guides/N9_20019_Rev._A_3L_VBR_Fluor_Guide.pdf",
        "2L_BR": "https://welcome.cytekbio.com/hubfs/Website%20Downloadable%20Content/Data%20Sheets/Fluorochrome%20Guides/N9_20020_Rev._A_2L_BR_Fluor_Guide.pdf",
        "2L_VB": "https://welcome.cytekbio.com/hubfs/Website%20Downloadable%20Content/Data%20Sheets/Fluorochrome%20Guides/N9_20021_Rev._A_2L_VB_FLuor_Guide.pdf",
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for name, url in urls.items():
        output_path = output_dir / f"cytek_{name}.pdf"
        if output_path.exists():
            downloaded.append(output_path)
            continue

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            downloaded.append(output_path)
        except requests.RequestException as e:
            print(f"Failed to download {name}: {e}")

    return downloaded
