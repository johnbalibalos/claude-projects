# Flow Panel Optimizer

A Python CLI tool for calculating spectral similarity metrics for flow cytometry panel design.

## Features

- **Cosine Similarity Matrix**: Calculate pairwise spectral similarity between fluorophores
- **Complexity Index**: Measure overall panel interference (Cytek-style metric)
- **Spillover Spreading Matrix**: Estimate theoretical signal spreading
- **Consensus Analysis**: Check if all three metrics agree on risk assessment
- **OMIP Validation**: Compare calculations against published OMIP panels

## Installation

```bash
cd flow_panel_optimizer
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
# Calculate similarity between fluorophores
flow-panel similarity PE FITC APC BV421

# Check panel complexity
flow-panel complexity PE FITC APC BV421 PE-Cy5 APC-Cy7

# Run full consensus analysis
flow-panel consensus PE FITC APC BV421

# Validate against OMIP-069 (40-color panel)
flow-panel validate-omip OMIP-069
```

## Available Commands

| Command | Description |
|---------|-------------|
| `similarity` | Calculate cosine similarity matrix |
| `complexity` | Calculate complexity index |
| `spreading` | Calculate theoretical spreading matrix |
| `consensus` | Run all metrics and check consensus |
| `validate-omip` | Validate against published OMIP |
| `list-panels` | List available OMIP panels |
| `list-fluorophores` | List available fluorophores |

## Output Formats

All commands support multiple output formats:
- `--format table` (default): Human-readable table
- `--format json`: JSON for programmatic use
- `--format csv`: CSV for spreadsheet import

Save results to file with `--output results.json`.

## Metrics

### Cosine Similarity

Measures the angle between two emission spectra vectors. Higher values indicate more similar spectra.

| Score | Risk Level |
|-------|------------|
| ≥ 0.98 | Critical |
| ≥ 0.95 | High |
| ≥ 0.90 | Moderate |
| ≥ 0.80 | Low |
| < 0.80 | Minimal |

### Complexity Index

Measures cumulative interference from all high-similarity pairs. Lower is better.

For reference, OMIP-069 (a well-designed 40-color panel) has a complexity index of 54.

### Spillover Spreading

Theoretical estimate of signal spread between channels. Based on Nguyen et al. (2013).

**Note**: This is an approximation. Actual SSM requires single-stained controls and instrument data.

## Testing

```bash
pytest
pytest -v  # Verbose output
pytest --cov=flow_panel_optimizer  # With coverage
```

## Limitations

### Theoretical vs. Actual Calculations
- **Spectral data is approximated**: Uses Gaussian fits to emission peaks, not real measured spectra from FPbase or manufacturer data
- **Spreading matrix is theoretical only**: Actual SSM requires single-stained controls and instrument-specific measurements
- **No excitation spectra considered**: Real overlap depends on both excitation and emission

### Validation Gaps
- **Limited ground truth**: Complexity index is not a published standard metric with validated thresholds
- **No instrument validation**: Calculations not validated against real cytometer outputs or spreading matrices
- **OMIP comparison is incomplete**: Only covers a subset of published panels

### Scope Limitations
- **No antibody considerations**: Does not account for antibody availability, cross-reactivity, or clone performance
- **No co-expression modeling**: Assumes markers can be treated independently (not always true)
- **Single instrument assumption**: Does not model differences between instrument configurations

### Known Issues
- Fluorophore database is incomplete (primarily common cytometry dyes)
- Some tandem dyes (e.g., PE-Cy7) degrade over time - not modeled
- Autofluorescence not considered

## References

1. Park LM et al. OMIP-069: Forty-Color Full Spectrum Flow Cytometry Panel. Cytometry A. 2020;97(10):1044-1051.

2. Nguyen R et al. Quantifying spillover spreading for comparing instrument performance. Cytometry A. 2013;83(3):306-315.

3. FlowJo Documentation: [Cosine Similarity Matrix](https://docs.flowjo.com/flowjo/experiment-based-platforms/plat-comp-overview/cosine-similarity-matrix/)

## License

MIT
