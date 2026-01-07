#!/usr/bin/env python3
"""
Generate CSV with fluorophore combinations and calculated metrics for validation.

This allows double-checking our calculations against online tools like:
- Cytek Full Spectrum Viewer
- BD Spectrum Viewer
- BioLegend Spectra Analyzer
"""

import sys
from pathlib import Path
import csv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flow_panel_optimizer.mcp.server import analyze_panel
from flow_panel_optimizer.data.fluorophore_database import (
    FLUOROPHORE_DATABASE,
    get_fluorophore,
    calculate_spectral_overlap,
    get_known_overlap,
)


# Test panels to validate
TEST_PANELS = [
    {
        "name": "Basic T-cell panel (4-color)",
        "fluorophores": ["BV421", "FITC", "PE", "APC"],
        "description": "Common 4-color T-cell panel"
    },
    {
        "name": "Extended T-cell panel (8-color)",
        "fluorophores": ["BV421", "BV510", "FITC", "PE", "PE-Cy7", "APC", "APC-Cy7", "BV785"],
        "description": "Standard 8-color immunophenotyping"
    },
    {
        "name": "Similar green fluorophores",
        "fluorophores": ["FITC", "BB515", "Alexa Fluor 488"],
        "description": "Deliberately similar - should have high overlap"
    },
    {
        "name": "Similar far-red fluorophores",
        "fluorophores": ["APC", "Alexa Fluor 647", "APC-Cy7", "APC-Fire750"],
        "description": "Red laser fluorophores - check tandem overlaps"
    },
    {
        "name": "Violet laser panel",
        "fluorophores": ["BV421", "BV480", "BV510", "BV605", "BV650", "BV711", "BV785"],
        "description": "Full Brilliant Violet series"
    },
    {
        "name": "Well-separated panel",
        "fluorophores": ["BV421", "FITC", "PE", "APC", "BV785"],
        "description": "Spectrally distinct fluorophores"
    },
    {
        "name": "OMIP-069 subset (10 colors)",
        "fluorophores": ["BUV395", "BV421", "BV510", "BV605", "FITC", "PE", "PE-Cy5", "PerCP-Cy5.5", "APC", "APC-Cy7"],
        "description": "Subset from published 40-color panel"
    },
    {
        "name": "Problematic PE tandems",
        "fluorophores": ["PE", "PE-CF594", "PE-Cy5", "PE-Cy5.5", "PE-Cy7"],
        "description": "PE tandem series - check tandem spreading"
    },
]


def generate_panel_csv():
    """Generate CSV with panel analysis results."""

    output_file = Path(__file__).parent / "validation_panels.csv"

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "Panel Name",
            "Description",
            "Fluorophores",
            "N Fluorophores",
            "N Pairs",
            "Total Similarity Score",
            "Avg Similarity",
            "Max Similarity",
            "N Critical Pairs (>0.90)",
            "N High Risk Pairs (>0.70)",
            "Quality Rating",
        ])

        for panel in TEST_PANELS:
            result = analyze_panel(panel["fluorophores"])

            writer.writerow([
                panel["name"],
                panel["description"],
                ", ".join(panel["fluorophores"]),
                result.get("n_fluorophores", len(panel["fluorophores"])),
                result.get("n_pairs", 0),
                result.get("total_similarity_score", "N/A"),
                result.get("avg_similarity", "N/A"),
                result.get("max_similarity", "N/A"),
                result.get("n_critical_pairs", len(result.get("critical_pairs", []))),
                len(result.get("problematic_pairs", [])),
                result.get("quality_rating", "N/A"),
            ])

    print(f"Panel summary saved to: {output_file}")
    return output_file


def generate_pairwise_csv():
    """Generate CSV with all pairwise similarity values."""

    output_file = Path(__file__).parent / "validation_pairwise.csv"

    # Collect all unique fluorophores from test panels
    all_fluorophores = set()
    for panel in TEST_PANELS:
        all_fluorophores.update(panel["fluorophores"])

    all_fluorophores = sorted(all_fluorophores)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "Fluorophore 1",
            "Fluorophore 2",
            "Emission Max 1 (nm)",
            "Emission Max 2 (nm)",
            "Optimal Laser 1",
            "Optimal Laser 2",
            "Cosine Similarity",
            "Known Overlap Value",
            "Risk Level",
        ])

        for i, f1_name in enumerate(all_fluorophores):
            f1 = get_fluorophore(f1_name)
            if not f1:
                continue

            for f2_name in all_fluorophores[i+1:]:
                f2 = get_fluorophore(f2_name)
                if not f2:
                    continue

                # Get similarity
                known = get_known_overlap(f1_name, f2_name)
                calculated = calculate_spectral_overlap(f1, f2)

                similarity = known if known is not None else calculated

                # Determine risk level
                if similarity > 0.90:
                    risk = "CRITICAL"
                elif similarity > 0.70:
                    risk = "HIGH"
                elif similarity > 0.50:
                    risk = "MODERATE"
                else:
                    risk = "LOW"

                writer.writerow([
                    f1_name,
                    f2_name,
                    f1.em_max,
                    f2.em_max,
                    f1.optimal_laser,
                    f2.optimal_laser,
                    round(similarity, 4),
                    round(known, 4) if known else "calculated",
                    risk,
                ])

    print(f"Pairwise similarities saved to: {output_file}")
    return output_file


def generate_fluorophore_database_csv():
    """Generate CSV with all fluorophores in our database."""

    output_file = Path(__file__).parent / "validation_fluorophore_database.csv"

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "Name",
            "Excitation Max (nm)",
            "Emission Max (nm)",
            "Excitation Range",
            "Emission Range",
            "Optimal Laser (nm)",
            "Compatible Lasers",
            "Relative Brightness",
            "Category",
            "Primary Vendor",
            "Notes",
        ])

        for name in sorted(FLUOROPHORE_DATABASE.keys()):
            f = FLUOROPHORE_DATABASE[name]
            writer.writerow([
                f.name,
                f.ex_max,
                f.em_max,
                f"{f.ex_range[0]}-{f.ex_range[1]}",
                f"{f.em_range[0]}-{f.em_range[1]}",
                f.optimal_laser,
                ", ".join(map(str, f.compatible_lasers)),
                f.relative_brightness,
                f.category,
                f.vendor_primary,
                f.notes or "",
            ])

    print(f"Fluorophore database saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    print("Generating validation CSV files...\n")

    panel_file = generate_panel_csv()
    pairwise_file = generate_pairwise_csv()
    database_file = generate_fluorophore_database_csv()

    print("\n" + "=" * 60)
    print("GENERATED FILES:")
    print("=" * 60)
    print(f"1. {panel_file.name}")
    print("   - Panel-level analysis for 8 test panels")
    print(f"2. {pairwise_file.name}")
    print("   - All pairwise similarity values")
    print(f"3. {database_file.name}")
    print("   - Full fluorophore database with spectral properties")
    print("\nUse these to validate against:")
    print("- Cytek Full Spectrum Viewer")
    print("- BD Spectrum Viewer")
    print("- BioLegend Spectra Analyzer")
    print("- EasyPanel.ai")
