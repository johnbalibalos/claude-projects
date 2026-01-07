"""Command-line interface for Flow Panel Optimizer.

This CLI provides commands for calculating spectral similarity metrics
and validating panel designs.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Flow Panel Optimizer - Calculate spectral similarity metrics.

    This tool helps design flow cytometry panels by calculating:

    \b
    - Cosine similarity between fluorophore emission spectra
    - Cytek-style complexity index for overall panel interference
    - Theoretical spillover spreading matrix
    - Consensus risk assessment across all metrics

    Use the --help flag on any command for more details.
    """
    pass


@cli.command()
@click.argument("fluorophores", nargs=-1, required=True)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file for results",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table", "csv"]),
    default="table",
    help="Output format (default: table)",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.90,
    help="Similarity threshold for flagging pairs (default: 0.90)",
)
def similarity(fluorophores, output, output_format, threshold):
    """Calculate cosine similarity matrix for given fluorophores.

    Uses synthetic spectra based on known emission peaks. For accurate
    results with actual spectral data, use the Python API directly.

    Example:
        flow-panel similarity PE FITC APC BV421

        flow-panel similarity PE FITC APC --format json --output results.json
    """
    from flow_panel_optimizer.spectral.similarity import (
        build_similarity_matrix,
        find_high_similarity_pairs,
    )
    from flow_panel_optimizer.validation.omip_validator import (
        create_synthetic_test_spectra,
    )

    # Get synthetic spectra
    all_spectra = create_synthetic_test_spectra()

    # Filter to requested fluorophores
    spectra = {}
    missing = []
    for name in fluorophores:
        if name in all_spectra:
            spectra[name] = all_spectra[name]
        else:
            missing.append(name)

    if missing:
        click.echo(f"Warning: No spectra available for: {', '.join(missing)}", err=True)

    if not spectra:
        click.echo("Error: No valid fluorophores specified", err=True)
        sys.exit(1)

    # Calculate similarity matrix
    names, matrix = build_similarity_matrix(spectra)

    # Find high similarity pairs
    high_pairs = find_high_similarity_pairs(matrix, names, threshold)

    # Output results
    if output_format == "json":
        result = {
            "fluorophores": names,
            "similarity_matrix": matrix.tolist(),
            "high_similarity_pairs": [
                {"fluor_a": a, "fluor_b": b, "similarity": s}
                for a, b, s in high_pairs
            ],
            "threshold": threshold,
        }
        output_text = json.dumps(result, indent=2)
    elif output_format == "csv":
        lines = ["," + ",".join(names)]
        for i, name in enumerate(names):
            row = [name] + [f"{matrix[i,j]:.4f}" for j in range(len(names))]
            lines.append(",".join(row))
        output_text = "\n".join(lines)
    else:  # table
        output_text = _format_similarity_table(names, matrix, high_pairs, threshold)

    if output:
        Path(output).write_text(output_text)
        click.echo(f"Results written to {output}")
    else:
        click.echo(output_text)


@cli.command()
@click.argument("fluorophores", nargs=-1, required=True)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file for results",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.90,
    help="Similarity threshold for complexity (default: 0.90)",
)
def complexity(fluorophores, output, threshold):
    """Calculate complexity index for a panel.

    The complexity index measures overall interference among fluorophores.
    Lower values indicate better fluorophore combinations.

    Example:
        flow-panel complexity PE FITC APC BV421 PE-Cy5 APC-Cy7
    """
    from flow_panel_optimizer.spectral.similarity import build_similarity_matrix
    from flow_panel_optimizer.spectral.complexity import (
        complexity_index as calc_ci,
        identify_complexity_drivers,
        estimate_panel_quality,
    )
    from flow_panel_optimizer.validation.omip_validator import (
        create_synthetic_test_spectra,
    )

    # Get synthetic spectra
    all_spectra = create_synthetic_test_spectra()

    # Filter to requested fluorophores
    spectra = {}
    for name in fluorophores:
        if name in all_spectra:
            spectra[name] = all_spectra[name]

    if not spectra:
        click.echo("Error: No valid fluorophores specified", err=True)
        sys.exit(1)

    # Calculate
    names, sim_matrix = build_similarity_matrix(spectra)
    ci = calc_ci(sim_matrix, threshold)
    quality = estimate_panel_quality(ci, len(names))
    drivers = identify_complexity_drivers(sim_matrix, names, threshold)

    result = {
        "panel_size": len(names),
        "complexity_index": ci,
        "quality_rating": quality,
        "threshold": threshold,
        "top_complexity_drivers": drivers[:5],
    }

    if output:
        Path(output).write_text(json.dumps(result, indent=2))
        click.echo(f"Results written to {output}")
    else:
        click.echo(f"Panel size: {len(names)} fluorophores")
        click.echo(f"Complexity Index: {ci}")
        click.echo(f"Quality Rating: {quality}")
        if drivers:
            click.echo("\nTop complexity contributors:")
            for d in drivers[:5]:
                click.echo(
                    f"  {d['fluor_a']:15} - {d['fluor_b']:15} "
                    f"SI={d['similarity']:.3f}"
                )


@cli.command()
@click.argument("fluorophores", nargs=-1, required=True)
@click.option(
    "--stain-indices",
    type=click.Path(exists=True),
    help="JSON file with stain indices",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file for results",
)
def spreading(fluorophores, stain_indices, output):
    """Calculate theoretical spillover spreading matrix.

    Note: This provides THEORETICAL estimates. Actual spreading requires
    single-stained controls and instrument data.

    Example:
        flow-panel spreading PE FITC APC BV421
    """
    from flow_panel_optimizer.spectral.similarity import build_similarity_matrix
    from flow_panel_optimizer.spectral.spreading import (
        build_spreading_matrix,
        find_high_spreading_pairs,
        total_panel_spreading,
    )
    from flow_panel_optimizer.validation.omip_validator import (
        create_synthetic_test_spectra,
    )

    # Get synthetic spectra
    all_spectra = create_synthetic_test_spectra()

    # Filter to requested fluorophores
    spectra = {}
    for name in fluorophores:
        if name in all_spectra:
            spectra[name] = all_spectra[name]

    if not spectra:
        click.echo("Error: No valid fluorophores specified", err=True)
        sys.exit(1)

    # Load stain indices if provided
    si_values = None
    if stain_indices:
        with open(stain_indices) as f:
            si_data = json.load(f)
            si_values = np.array([si_data.get(name, 100.0) for name in spectra.keys()])

    # Calculate
    names, sim_matrix = build_similarity_matrix(spectra)
    ssm = build_spreading_matrix(sim_matrix, si_values)
    high_spread = find_high_spreading_pairs(ssm, names, threshold=5.0)
    total = total_panel_spreading(ssm)

    result = {
        "fluorophores": names,
        "spreading_matrix": ssm.tolist(),
        "total_spreading": round(total, 2),
        "high_spreading_pairs": [
            {"from": a, "to": b, "spread": s} for a, b, s in high_spread[:10]
        ],
    }

    if output:
        Path(output).write_text(json.dumps(result, indent=2))
        click.echo(f"Results written to {output}")
    else:
        click.echo(f"Panel size: {len(names)} fluorophores")
        click.echo(f"Total spreading: {total:.2f}")
        if high_spread:
            click.echo("\nHigh spreading pairs (top 10):")
            for a, b, s in high_spread[:10]:
                click.echo(f"  {a:15} -> {b:15}: {s:.2f}")


@cli.command()
@click.argument("fluorophores", nargs=-1, required=True)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file for results",
)
def consensus(fluorophores, output):
    """Run all three metrics and check for consensus.

    This command calculates cosine similarity, complexity contribution,
    and theoretical spreading for all pairs, then checks if the metrics
    agree on risk assessment.

    Example:
        flow-panel consensus PE FITC APC BV421 PE-Cy5 APC-Cy7
    """
    from flow_panel_optimizer.spectral.similarity import build_similarity_matrix
    from flow_panel_optimizer.spectral.spreading import build_spreading_matrix
    from flow_panel_optimizer.validation.consensus import (
        validate_panel_consensus,
        summarize_consensus,
    )
    from flow_panel_optimizer.validation.omip_validator import (
        create_synthetic_test_spectra,
    )

    # Get synthetic spectra
    all_spectra = create_synthetic_test_spectra()

    # Filter to requested fluorophores
    spectra = {}
    for name in fluorophores:
        if name in all_spectra:
            spectra[name] = all_spectra[name]

    if not spectra:
        click.echo("Error: No valid fluorophores specified", err=True)
        sys.exit(1)

    # Calculate matrices
    names, sim_matrix = build_similarity_matrix(spectra)
    ssm = build_spreading_matrix(sim_matrix)

    # Run consensus validation
    result = validate_panel_consensus(
        panel_name="Custom Panel",
        similarity_matrix=sim_matrix,
        spreading_matrix=ssm,
        fluorophore_names=names,
    )

    if output:
        # Prepare JSON-serializable output
        json_result = {
            "panel_name": result["panel_name"],
            "total_pairs": result["total_pairs"],
            "metrics_agree": result["metrics_agree"],
            "agreement_rate": result["agreement_rate"],
            "by_risk_level": result["by_risk_level"],
            "high_risk_pairs": result["high_risk_pairs"],
            "critical_pairs": result["critical_pairs"],
        }
        Path(output).write_text(json.dumps(json_result, indent=2))
        click.echo(f"Results written to {output}")
    else:
        summary = summarize_consensus(result)
        click.echo(summary)


@cli.command("validate-omip")
@click.argument("omip_id")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file for results",
)
def validate_omip(omip_id, output):
    """Validate calculations against a published OMIP panel.

    This compares our calculated metrics against published values
    from OMIP papers to verify accuracy.

    Available panels: OMIP-069, OMIP-042, OMIP-030, Test-High-Similarity

    Example:
        flow-panel validate-omip OMIP-069
    """
    from flow_panel_optimizer.validation.omip_validator import (
        OMIPValidator,
        create_synthetic_test_spectra,
    )
    from flow_panel_optimizer.acquisition.omip_loader import OMIP_PANELS

    if omip_id not in OMIP_PANELS:
        click.echo(f"Unknown panel: {omip_id}")
        click.echo(f"Available panels: {', '.join(OMIP_PANELS.keys())}")
        sys.exit(1)

    # Get synthetic spectra
    spectra = create_synthetic_test_spectra()

    # Run validation
    validator = OMIPValidator(tolerance=0.05)
    result = validator.validate_panel(omip_id, spectra)

    if output:
        json_result = {
            "panel_name": result.panel_name,
            "passed": result.passed,
            "similarity_tests": result.similarity_tests,
            "complexity_test": result.complexity_test,
            "notes": result.overall_notes,
        }
        Path(output).write_text(json.dumps(json_result, indent=2))
        click.echo(f"Results written to {output}")
    else:
        status = "PASSED" if result.passed else "FAILED"
        click.echo(f"Validation {status} for {result.panel_name}")
        click.echo()

        if result.similarity_tests:
            click.echo("Similarity pair tests:")
            for test in result.similarity_tests:
                if test.get("passed") is None:
                    click.echo(
                        f"  {test['fluor_a']} - {test['fluor_b']}: SKIPPED"
                    )
                else:
                    status_char = "✓" if test["passed"] else "✗"
                    click.echo(
                        f"  {status_char} {test['fluor_a']} - {test['fluor_b']}: "
                        f"expected={test['expected']:.3f}, "
                        f"calculated={test['calculated']:.3f}"
                    )

        if result.complexity_test:
            click.echo()
            click.echo("Complexity index test:")
            test = result.complexity_test
            status_char = "✓" if test["passed"] else "✗"
            click.echo(
                f"  {status_char} expected={test['expected']}, "
                f"calculated={test['calculated']:.1f}"
            )

        if result.overall_notes:
            click.echo()
            click.echo("Notes:")
            for note in result.overall_notes:
                click.echo(f"  - {note}")


@cli.command("list-panels")
def list_panels():
    """List available OMIP panels for validation."""
    from flow_panel_optimizer.acquisition.omip_loader import OMIP_PANELS

    click.echo("Available OMIP panels:")
    click.echo()
    for panel_id, info in OMIP_PANELS.items():
        click.echo(f"  {panel_id}")
        click.echo(f"    {info.get('description', 'No description')}")
        if info.get("reference"):
            click.echo(f"    Ref: {info['reference'][:60]}...")
        click.echo()


@cli.command("list-fluorophores")
def list_fluorophores():
    """List available fluorophores with synthetic spectra."""
    from flow_panel_optimizer.validation.omip_validator import (
        create_synthetic_test_spectra,
    )

    spectra = create_synthetic_test_spectra()
    click.echo(f"Available fluorophores ({len(spectra)}):")
    click.echo()

    # Group by laser
    by_laser = {
        "UV (355nm)": ["BUV395", "BUV496", "BUV563", "BUV615", "BUV661", "BUV737", "BUV805"],
        "Violet (405nm)": ["BV421", "BV480", "BV510", "BV570", "BV605", "BV650", "BV711", "BV750", "BV785", "Pacific Blue", "Super Bright 436"],
        "Blue (488nm)": ["FITC", "BB515", "Alexa Fluor 488", "PerCP", "PerCP-Cy5.5"],
        "Yellow-Green (561nm)": ["PE", "PE-CF594", "PE-Cy5", "PE-Cy5.5", "PE-Cy7"],
        "Red (633nm)": ["APC", "Alexa Fluor 647", "APC-R700", "APC-Fire750", "APC-Fire810"],
    }

    for laser, fluors in by_laser.items():
        click.echo(f"  {laser}:")
        available = [f for f in fluors if f in spectra]
        if available:
            click.echo(f"    {', '.join(available)}")
        click.echo()


def _format_similarity_table(
    names: list[str],
    matrix: np.ndarray,
    high_pairs: list[tuple],
    threshold: float,
) -> str:
    """Format similarity matrix as a table."""
    if RICH_AVAILABLE:
        console = Console(record=True)
        table = Table(title="Cosine Similarity Matrix")

        # Add header
        table.add_column("", style="bold")
        for name in names:
            table.add_column(name[:8], justify="center")

        # Add rows
        for i, name in enumerate(names):
            row = [name[:8]]
            for j in range(len(names)):
                val = matrix[i, j]
                if i == j:
                    row.append("[dim]1.0[/dim]")
                elif val >= 0.98:
                    row.append(f"[bold red]{val:.2f}[/bold red]")
                elif val >= threshold:
                    row.append(f"[yellow]{val:.2f}[/yellow]")
                else:
                    row.append(f"{val:.2f}")
            table.add_row(*row)

        console.print(table)
        output = console.export_text()
    else:
        # Plain text fallback
        lines = []
        header = "        " + "  ".join(f"{n[:6]:>6}" for n in names)
        lines.append(header)

        for i, name in enumerate(names):
            row = f"{name[:6]:>6}  "
            for j in range(len(names)):
                val = matrix[i, j]
                row += f"{val:6.3f}  "
            lines.append(row)

        output = "\n".join(lines)

    # Add high similarity pairs
    if high_pairs:
        output += f"\n\nHigh similarity pairs (threshold={threshold}):\n"
        for a, b, sim in high_pairs:
            output += f"  {a} - {b}: {sim:.4f}\n"

    return output


if __name__ == "__main__":
    cli()
