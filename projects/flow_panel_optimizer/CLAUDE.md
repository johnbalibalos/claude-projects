# Flow Panel Optimizer - Claude Code Instructions

## Project Overview

This is a Python CLI tool for calculating spectral similarity metrics for flow cytometry panel design. The project focuses on three key metrics:

1. **Cosine Similarity** - Standard mathematical similarity between emission spectra
2. **Cytek Complexity Index** - Proprietary-style metric for overall panel interference
3. **Spillover Spreading Matrix (SSM)** - Instrument-dependent spread estimation

## Project Structure

```
flow_panel_optimizer/
├── src/flow_panel_optimizer/
│   ├── __init__.py           # Package entry point
│   ├── cli.py                # Click CLI interface
│   ├── acquisition/          # Data acquisition modules
│   │   ├── fpbase_client.py  # FPbase GraphQL API client
│   │   ├── cytek_scraper.py  # Cytek PDF extraction
│   │   └── omip_loader.py    # OMIP panel definitions
│   ├── spectral/             # Metric calculations
│   │   ├── similarity.py     # Cosine similarity
│   │   ├── complexity.py     # Complexity index
│   │   └── spreading.py      # Spillover spreading
│   ├── models/               # Data models
│   │   ├── spectrum.py       # Spectrum dataclass
│   │   ├── fluorophore.py    # Fluorophore dataclass
│   │   └── panel.py          # Panel dataclass
│   └── validation/           # Testing infrastructure
│       ├── consensus.py      # 3-metric consensus checker
│       └── omip_validator.py # Compare against published OMIPs
└── tests/                    # Unit tests
```

## Key Commands

### Running Tests
```bash
cd flow_panel_optimizer
pip install -e ".[dev]"
pytest
```

### Using the CLI
```bash
# Calculate similarity matrix
flow-panel similarity PE FITC APC BV421

# Calculate complexity index
flow-panel complexity PE FITC APC BV421 PE-Cy5

# Calculate spreading matrix
flow-panel spreading PE FITC APC

# Run consensus check
flow-panel consensus PE FITC APC BV421

# Validate against OMIP panel
flow-panel validate-omip OMIP-069

# List available panels and fluorophores
flow-panel list-panels
flow-panel list-fluorophores
```

## Key Implementation Details

### Cosine Similarity (similarity.py)
- Uses scipy.spatial.distance.cosine
- Handles both numpy arrays and Spectrum objects
- Provides risk level classification (critical, high, moderate, low, minimal)

### Complexity Index (complexity.py)
- Approximation of Cytek's proprietary formula
- Measures cumulative interference from high-similarity pairs
- Scaled to roughly 0-100 range based on OMIP-069 reference (CI=54)

### Spillover Spreading (spreading.py)
- THEORETICAL estimates only (actual SSM requires instrument data)
- Based on Nguyen et al. (2013) formula principles
- Considers both spectral similarity and stain index

### Consensus Checker (consensus.py)
- Compares all three metrics for agreement
- Assigns risk levels from each metric independently
- Reports disagreements and overall consensus risk

## Data Sources

1. **Synthetic Spectra**: Built-in Gaussian approximations based on known emission peaks
2. **FPbase**: GraphQL API client for real spectral data (requires internet)
3. **Cytek PDFs**: PDF extraction for spread matrices (requires downloaded PDFs)
4. **OMIP Panels**: Manually curated definitions from published papers

## Testing Strategy

- Unit tests for all metric calculations
- Fixtures in conftest.py provide sample spectra and panels
- Validation tests compare against known OMIP values
- Use `pytest -v` for verbose output

## Common Tasks for Claude

### Adding a New Fluorophore
1. Add to `create_synthetic_test_spectra()` in `omip_validator.py`
2. Update `list-fluorophores` CLI command if needed

### Adding a New OMIP Panel
1. Add panel definition to `OMIP_PANELS` dict in `omip_loader.py`
2. Include published similarity pairs and complexity index if available

### Modifying Metric Calculations
- Always update corresponding tests in `tests/`
- Ensure backward compatibility with CLI output formats

## References

- FlowJo Cosine Similarity: https://docs.flowjo.com/flowjo/experiment-based-platforms/plat-comp-overview/cosine-similarity-matrix/
- Nguyen et al. (2013) SSM paper: Cytometry A. 2013;83(3):306-315
- OMIP-069 (40-color panel): Cytometry A. 2020;97(10):1044-1051
