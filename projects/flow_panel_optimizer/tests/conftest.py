"""Pytest fixtures for flow_panel_optimizer tests."""

import pytest
import numpy as np

from flow_panel_optimizer.models.spectrum import Spectrum
from flow_panel_optimizer.models.fluorophore import Fluorophore
from flow_panel_optimizer.models.panel import Panel


@pytest.fixture
def sample_wavelengths():
    """Standard wavelength grid for testing."""
    return np.linspace(400, 800, 100)


@pytest.fixture
def sample_emission_spectrum(sample_wavelengths):
    """Sample Gaussian emission spectrum."""
    peak = 520
    width = 30
    intensities = np.exp(-((sample_wavelengths - peak) ** 2) / (2 * width ** 2))
    return Spectrum(
        wavelengths=sample_wavelengths,
        intensities=intensities,
        spectrum_type="emission",
        source="test",
    )


@pytest.fixture
def fitc_like_spectrum(sample_wavelengths):
    """FITC-like emission spectrum (peak ~520nm)."""
    intensities = np.exp(-((sample_wavelengths - 520) ** 2) / (2 * 25 ** 2))
    return Spectrum(
        wavelengths=sample_wavelengths,
        intensities=intensities,
        spectrum_type="emission",
        source="test",
    )


@pytest.fixture
def pe_like_spectrum(sample_wavelengths):
    """PE-like emission spectrum (peak ~575nm)."""
    intensities = np.exp(-((sample_wavelengths - 575) ** 2) / (2 * 30 ** 2))
    return Spectrum(
        wavelengths=sample_wavelengths,
        intensities=intensities,
        spectrum_type="emission",
        source="test",
    )


@pytest.fixture
def apc_like_spectrum(sample_wavelengths):
    """APC-like emission spectrum (peak ~660nm)."""
    intensities = np.exp(-((sample_wavelengths - 660) ** 2) / (2 * 25 ** 2))
    return Spectrum(
        wavelengths=sample_wavelengths,
        intensities=intensities,
        spectrum_type="emission",
        source="test",
    )


@pytest.fixture
def sample_fluorophore(fitc_like_spectrum):
    """Sample fluorophore with spectra."""
    return Fluorophore(
        name="Test-FITC",
        aliases=["test-fitc", "FITC-test"],
        excitation_max=495,
        emission_max=520,
        emission_spectrum=fitc_like_spectrum,
        stain_index=150.0,
        source="test",
    )


@pytest.fixture
def sample_panel():
    """Sample panel with basic fluorophores."""
    return Panel(
        name="Test Panel",
        fluorophores=[
            Fluorophore(name="FITC", emission_max=520, source="test"),
            Fluorophore(name="PE", emission_max=575, source="test"),
            Fluorophore(name="APC", emission_max=660, source="test"),
        ],
        instrument="Test Cytometer",
        description="Test panel for unit tests",
    )


@pytest.fixture
def high_similarity_spectra(sample_wavelengths):
    """Two highly similar spectra for testing similarity detection."""
    spec_a = np.exp(-((sample_wavelengths - 520) ** 2) / (2 * 25 ** 2))
    spec_b = np.exp(-((sample_wavelengths - 515) ** 2) / (2 * 25 ** 2))  # 5nm shift
    return spec_a, spec_b


@pytest.fixture
def low_similarity_spectra(sample_wavelengths):
    """Two dissimilar spectra for testing."""
    spec_a = np.exp(-((sample_wavelengths - 450) ** 2) / (2 * 20 ** 2))  # Blue
    spec_b = np.exp(-((sample_wavelengths - 700) ** 2) / (2 * 25 ** 2))  # Red
    return spec_a, spec_b


@pytest.fixture
def test_spectra_dict(sample_wavelengths):
    """Dict of test spectra for matrix operations."""
    return {
        "FITC": np.exp(-((sample_wavelengths - 520) ** 2) / (2 * 25 ** 2)),
        "BB515": np.exp(-((sample_wavelengths - 515) ** 2) / (2 * 25 ** 2)),
        "PE": np.exp(-((sample_wavelengths - 575) ** 2) / (2 * 30 ** 2)),
        "APC": np.exp(-((sample_wavelengths - 660) ** 2) / (2 * 25 ** 2)),
    }
