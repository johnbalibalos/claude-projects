"""OMIP validation for comparing calculated metrics against published values.

This module validates that our metric calculations match published OMIP
panel data, providing confidence in the accuracy of the implementation.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from flow_panel_optimizer.acquisition.omip_loader import OMIPLoader, OMIP_PANELS
from flow_panel_optimizer.spectral.similarity import (
    build_similarity_matrix,
    find_high_similarity_pairs,
)
from flow_panel_optimizer.spectral.complexity import complexity_index
from flow_panel_optimizer.spectral.spreading import build_spreading_matrix


@dataclass
class ValidationResult:
    """Result of validating calculations against published data."""

    panel_name: str
    passed: bool
    similarity_tests: list[dict] = field(default_factory=list)
    complexity_test: Optional[dict] = None
    overall_notes: list[str] = field(default_factory=list)


class OMIPValidator:
    """Validate metric calculations against published OMIP data.

    This validator compares our calculated values against:
    1. Published similarity pairs from OMIP papers
    2. Published complexity indices
    3. Expected high-risk pairs based on panel composition
    """

    def __init__(self, tolerance: float = 0.05):
        """Initialize OMIP validator.

        Args:
            tolerance: Acceptable difference between calculated and
                published values (default 0.05 = 5%).
        """
        self.tolerance = tolerance
        self.loader = OMIPLoader()

    def validate_panel(
        self,
        panel_id: str,
        spectra: Optional[dict[str, np.ndarray]] = None,
        stain_indices: Optional[dict[str, float]] = None,
    ) -> ValidationResult:
        """Validate calculations for a specific OMIP panel.

        Args:
            panel_id: OMIP panel identifier (e.g., 'OMIP-069').
            spectra: Optional dict mapping fluorophore name -> emission spectrum.
                If not provided, cannot validate similarity pairs.
            stain_indices: Optional dict mapping fluorophore name -> stain index.

        Returns:
            ValidationResult with detailed test outcomes.
        """
        panel_info = self.loader.get_panel_info(panel_id)
        if not panel_info:
            return ValidationResult(
                panel_name=panel_id,
                passed=False,
                overall_notes=[f"Unknown panel: {panel_id}"],
            )

        result = ValidationResult(panel_name=panel_id, passed=True)

        # Test 1: Validate known similarity pairs
        known_pairs = self.loader.get_known_similarity_pairs(panel_id)
        if known_pairs and spectra:
            for (fluor_a, fluor_b), expected_sim in known_pairs.items():
                if fluor_a in spectra and fluor_b in spectra:
                    from flow_panel_optimizer.spectral.similarity import cosine_similarity

                    calculated_sim = cosine_similarity(
                        spectra[fluor_a], spectra[fluor_b]
                    )

                    diff = abs(calculated_sim - expected_sim)
                    test_passed = diff <= self.tolerance

                    result.similarity_tests.append({
                        "fluor_a": fluor_a,
                        "fluor_b": fluor_b,
                        "expected": expected_sim,
                        "calculated": round(calculated_sim, 4),
                        "difference": round(diff, 4),
                        "passed": test_passed,
                    })

                    if not test_passed:
                        result.passed = False
                else:
                    result.similarity_tests.append({
                        "fluor_a": fluor_a,
                        "fluor_b": fluor_b,
                        "expected": expected_sim,
                        "calculated": None,
                        "difference": None,
                        "passed": None,
                        "note": "Spectrum data not available",
                    })

        # Test 2: Validate complexity index
        expected_ci = self.loader.get_expected_complexity(panel_id)
        if expected_ci is not None and spectra:
            names, sim_matrix = build_similarity_matrix(spectra)
            calculated_ci = complexity_index(sim_matrix)

            ci_diff = abs(calculated_ci - expected_ci)
            # Use percentage tolerance for complexity (10%)
            ci_tolerance = expected_ci * 0.10
            ci_passed = ci_diff <= ci_tolerance

            result.complexity_test = {
                "expected": expected_ci,
                "calculated": calculated_ci,
                "difference": round(ci_diff, 1),
                "tolerance": round(ci_tolerance, 1),
                "passed": ci_passed,
            }

            if not ci_passed:
                result.passed = False
                result.overall_notes.append(
                    f"Complexity index mismatch: expected {expected_ci}, "
                    f"got {calculated_ci}"
                )

        # Add summary notes
        if not result.similarity_tests:
            result.overall_notes.append("No similarity pairs to validate")
        else:
            passed_sim = sum(1 for t in result.similarity_tests if t.get("passed"))
            total_sim = len([t for t in result.similarity_tests if t.get("passed") is not None])
            result.overall_notes.append(
                f"Similarity tests: {passed_sim}/{total_sim} passed"
            )

        if result.complexity_test:
            if result.complexity_test["passed"]:
                result.overall_notes.append("Complexity index validated")
            else:
                result.overall_notes.append("Complexity index validation failed")
        else:
            result.overall_notes.append("No complexity index to validate")

        return result

    def validate_all_panels(
        self,
        spectra_by_panel: Optional[dict[str, dict[str, np.ndarray]]] = None,
    ) -> dict[str, ValidationResult]:
        """Validate all available OMIP panels.

        Args:
            spectra_by_panel: Optional dict mapping panel_id -> spectra dict.

        Returns:
            Dict mapping panel_id -> ValidationResult.
        """
        results = {}
        for panel_id in self.loader.list_panels():
            spectra = None
            if spectra_by_panel and panel_id in spectra_by_panel:
                spectra = spectra_by_panel[panel_id]
            results[panel_id] = self.validate_panel(panel_id, spectra)
        return results

    def generate_report(self, results: dict[str, ValidationResult]) -> str:
        """Generate human-readable validation report.

        Args:
            results: Dict of validation results from validate_all_panels().

        Returns:
            Formatted report string.
        """
        lines = ["=" * 60, "OMIP Validation Report", "=" * 60, ""]

        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)

        lines.append(f"Overall: {passed_count}/{total_count} panels passed")
        lines.append("")

        for panel_id, result in results.items():
            status = "✓ PASS" if result.passed else "✗ FAIL"
            lines.append(f"{panel_id}: {status}")

            # Similarity tests
            if result.similarity_tests:
                lines.append("  Similarity pairs:")
                for test in result.similarity_tests:
                    if test.get("passed") is None:
                        lines.append(
                            f"    {test['fluor_a']} - {test['fluor_b']}: "
                            f"SKIPPED ({test.get('note', 'no data')})"
                        )
                    else:
                        status_str = "✓" if test["passed"] else "✗"
                        lines.append(
                            f"    {test['fluor_a']} - {test['fluor_b']}: "
                            f"{status_str} expected={test['expected']:.3f}, "
                            f"calc={test['calculated']:.3f}"
                        )

            # Complexity test
            if result.complexity_test:
                test = result.complexity_test
                status_str = "✓" if test["passed"] else "✗"
                lines.append(
                    f"  Complexity: {status_str} expected={test['expected']}, "
                    f"calc={test['calculated']:.1f}"
                )

            # Notes
            if result.overall_notes:
                lines.append(f"  Notes: {'; '.join(result.overall_notes)}")

            lines.append("")

        return "\n".join(lines)


def create_synthetic_test_spectra() -> dict[str, np.ndarray]:
    """Create synthetic spectra for testing.

    This generates approximate spectra based on known emission peaks
    for common flow cytometry fluorophores.

    Returns:
        Dict mapping fluorophore name -> emission spectrum array.
    """
    wavelengths = np.linspace(400, 800, 100)

    def gaussian_spectrum(peak: float, width: float = 30) -> np.ndarray:
        """Generate Gaussian emission spectrum."""
        return np.exp(-((wavelengths - peak) ** 2) / (2 * width ** 2))

    # Approximate emission peaks for common fluorophores
    spectra = {
        "FITC": gaussian_spectrum(520, 25),
        "BB515": gaussian_spectrum(515, 25),  # Similar to FITC
        "Alexa Fluor 488": gaussian_spectrum(519, 24),
        "PE": gaussian_spectrum(575, 30),
        "PE-CF594": gaussian_spectrum(610, 28),
        "PE-Cy5": gaussian_spectrum(670, 35),
        "PE-Cy5.5": gaussian_spectrum(695, 35),
        "PE-Cy7": gaussian_spectrum(775, 40),
        "PerCP": gaussian_spectrum(675, 30),
        "PerCP-Cy5.5": gaussian_spectrum(695, 35),
        "APC": gaussian_spectrum(660, 25),
        "Alexa Fluor 647": gaussian_spectrum(665, 25),  # Similar to APC
        "APC-R700": gaussian_spectrum(719, 30),
        "APC-Fire750": gaussian_spectrum(787, 35),
        "APC-Fire810": gaussian_spectrum(810, 35),
        "BV421": gaussian_spectrum(421, 20),
        "BV480": gaussian_spectrum(478, 22),
        "BV510": gaussian_spectrum(510, 25),
        "BV570": gaussian_spectrum(570, 25),
        "BV605": gaussian_spectrum(602, 25),
        "BV650": gaussian_spectrum(645, 25),
        "BV711": gaussian_spectrum(711, 28),
        "BV750": gaussian_spectrum(750, 30),
        "BV785": gaussian_spectrum(785, 32),
        "BUV395": gaussian_spectrum(395, 18),
        "BUV496": gaussian_spectrum(496, 22),
        "BUV563": gaussian_spectrum(563, 25),
        "BUV615": gaussian_spectrum(615, 28),
        "BUV661": gaussian_spectrum(661, 28),
        "BUV737": gaussian_spectrum(737, 30),
        "BUV805": gaussian_spectrum(805, 35),
        "Pacific Blue": gaussian_spectrum(455, 22),
        "Super Bright 436": gaussian_spectrum(436, 20),  # Similar to BV421
    }

    # Normalize all spectra
    for name in spectra:
        spectra[name] = spectra[name] / np.max(spectra[name])

    return spectra
