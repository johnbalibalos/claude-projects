"""Command-line interface for DrugDevBench."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from drugdevbench.data import load_annotations, FigureType, PromptCondition
from drugdevbench.evaluation import AblationConfig, run_ablation_study
from drugdevbench.models import estimate_benchmark_cost, SUPPORTED_MODELS

app = typer.Typer(
    name="drugdevbench",
    help="DrugDevBench: Benchmark for evaluating LLM interpretation of drug development figures",
)
console = Console()


@app.command()
def run(
    annotations_path: Path = typer.Argument(
        ..., help="Path to annotations JSONL file"
    ),
    models: list[str] = typer.Option(
        ["claude-haiku"],
        "--model",
        "-m",
        help="Models to evaluate (can specify multiple)",
    ),
    conditions: list[str] = typer.Option(
        None,
        "--condition",
        "-c",
        help="Conditions to test (default: all)",
    ),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output",
        "-o",
        help="Output directory for results",
    ),
    max_figures: Optional[int] = typer.Option(
        None,
        "--max-figures",
        help="Maximum number of figures to evaluate",
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        "-v/-q",
        help="Verbose output",
    ),
) -> None:
    """Run the benchmark ablation study."""
    # Load annotations
    annotations = load_annotations(annotations_path)
    if not annotations:
        console.print("[red]No annotations found![/red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(annotations)} annotations")

    # Parse conditions
    if conditions:
        parsed_conditions = [PromptCondition(c) for c in conditions]
    else:
        parsed_conditions = list(PromptCondition)

    # Create config
    config = AblationConfig(
        conditions=parsed_conditions,
        models=models,
        max_figures=max_figures,
        output_dir=output_dir,
        verbose=verbose,
    )

    # Run ablation
    results = run_ablation_study(annotations, config)

    console.print("\n[green]Ablation study complete![/green]")


@app.command()
def estimate_cost(
    n_figures: int = typer.Option(50, "--figures", "-f", help="Number of figures"),
    n_questions: int = typer.Option(4, "--questions", "-q", help="Questions per figure"),
    n_conditions: int = typer.Option(5, "--conditions", "-c", help="Number of conditions"),
    models: list[str] = typer.Option(
        ["claude-haiku", "gemini-flash"],
        "--model",
        "-m",
        help="Models to estimate",
    ),
) -> None:
    """Estimate the cost of running the benchmark."""
    estimate = estimate_benchmark_cost(
        n_figures=n_figures,
        n_questions_per_figure=n_questions,
        n_conditions=n_conditions,
        models=models,
    )

    console.print("\n[bold]Cost Estimate[/bold]")
    console.print(f"Total requests: {estimate['total_requests']}")

    table = Table(title="Estimated Costs by Model")
    table.add_column("Model", style="cyan")
    table.add_column("Cost (USD)", justify="right")

    for model, cost in estimate["costs_by_model"].items():
        table.add_row(model, f"${cost:.2f}")

    console.print(table)
    console.print(f"\n[bold]Total estimated cost: ${estimate['total_cost_usd']:.2f}[/bold]")


@app.command()
def list_models() -> None:
    """List supported models and their cost tiers."""
    table = Table(title="Supported Models")
    table.add_column("Short Name", style="cyan")
    table.add_column("LiteLLM Key")
    table.add_column("Cost Tier", justify="center")
    table.add_column("Vision", justify="center")

    for name, info in SUPPORTED_MODELS.items():
        table.add_row(
            name,
            info["litellm_key"],
            info["cost"].tier,
            "✓" if info["vision"] else "✗",
        )

    console.print(table)


@app.command()
def list_conditions() -> None:
    """List available ablation conditions."""
    table = Table(title="Ablation Conditions")
    table.add_column("Condition", style="cyan")
    table.add_column("Persona", justify="center")
    table.add_column("Base", justify="center")
    table.add_column("Skill", justify="center")
    table.add_column("Description")

    conditions_info = [
        ("vanilla", "✗", "✗", "✗", "Raw model capability"),
        ("base_only", "✗", "✓", "✗", "Generic scientific reasoning"),
        ("persona_only", "✓", "✗", "✗", "Domain expertise value"),
        ("base_plus_skill", "✗", "✓", "✓", "Skill-based improvement"),
        ("full_stack", "✓", "✓", "✓", "Complete system"),
        ("wrong_skill", "✗", "✓", "✗*", "Skill specificity (mismatched)"),
    ]

    for name, persona, base, skill, desc in conditions_info:
        table.add_row(name, persona, base, skill, desc)

    console.print(table)


@app.command()
def list_figure_types() -> None:
    """List supported figure types and their personas."""
    from drugdevbench.prompts.dispatcher import FIGURE_TYPE_TO_PERSONA

    table = Table(title="Figure Types")
    table.add_column("Category", style="cyan")
    table.add_column("Figure Type")
    table.add_column("Persona")

    categories = {
        "Protein Analysis": [FigureType.WESTERN_BLOT, FigureType.COOMASSIE_GEL, FigureType.DOT_BLOT],
        "Binding Assays": [FigureType.ELISA, FigureType.DOSE_RESPONSE, FigureType.IC50_EC50],
        "Pharmacokinetics": [FigureType.PK_CURVE, FigureType.AUC_PLOT, FigureType.COMPARTMENT_MODEL],
        "Flow Cytometry": [FigureType.FLOW_BIAXIAL, FigureType.FLOW_HISTOGRAM, FigureType.GATING_STRATEGY],
        "Genomics": [FigureType.HEATMAP, FigureType.VOLCANO_PLOT, FigureType.PATHWAY_ENRICHMENT],
        "Cell Assays": [FigureType.VIABILITY_CURVE, FigureType.PROLIFERATION, FigureType.CYTOTOXICITY],
    }

    for category, figure_types in categories.items():
        for i, ft in enumerate(figure_types):
            cat_display = category if i == 0 else ""
            persona = FIGURE_TYPE_TO_PERSONA[ft].value.replace("_", " ").title()
            table.add_row(cat_display, ft.value, persona)

    console.print(table)


@app.command()
def download(
    source: str = typer.Option(
        "all",
        "--source",
        "-s",
        help="Data source: sourcedata, openpmc, or all",
    ),
    max_figures: int = typer.Option(
        100,
        "--max",
        "-n",
        help="Maximum figures to download per source",
    ),
    output_dir: Path = typer.Option(
        Path("data/figures"),
        "--output",
        "-o",
        help="Output directory for figures",
    ),
    figure_types: list[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by figure type(s)",
    ),
    openpmc_path: Optional[str] = typer.Option(
        None,
        "--openpmc-path",
        help="Hugging Face path for Open-PMC dataset",
    ),
) -> None:
    """Download figures from SourceData (EMBO) and/or Open-PMC-18M."""
    from drugdevbench.data.sources.sourcedata import SourceDataSource
    from drugdevbench.data.sources.openpmc import OpenPMCSource
    from drugdevbench.data import save_annotations

    # Parse figure types
    ft_filter = None
    if figure_types:
        ft_filter = [FigureType(t) for t in figure_types]

    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    if source in ("sourcedata", "all"):
        console.print("\n[bold cyan]Downloading from SourceData (EMBO)...[/bold cyan]")
        try:
            sd_source = SourceDataSource(output_dir=output_dir)
            figures = sd_source.download_figures(
                figure_types=ft_filter,
                max_figures=max_figures,
            )
            annotations = sd_source.create_annotations_from_figures(figures)
            ann_path = output_dir.parent / "annotations" / "sourcedata_annotations.jsonl"
            save_annotations(annotations, ann_path)
            console.print(f"[green]SourceData: {len(figures)} figures, {len(annotations)} annotations[/green]")
            total += len(figures)
        except Exception as e:
            console.print(f"[red]SourceData error: {e}[/red]")

    if source in ("openpmc", "all"):
        console.print("\n[bold cyan]Downloading from Open-PMC-18M...[/bold cyan]")
        try:
            pmc_source = OpenPMCSource(output_dir=output_dir, dataset_path=openpmc_path)
            figures = pmc_source.download_figures(
                figure_types=ft_filter,
                max_figures=max_figures,
            )
            annotations = pmc_source.create_annotations_from_figures(figures)
            ann_path = output_dir.parent / "annotations" / "openpmc_annotations.jsonl"
            save_annotations(annotations, ann_path)
            console.print(f"[green]Open-PMC: {len(figures)} figures, {len(annotations)} annotations[/green]")
            total += len(figures)
        except Exception as e:
            console.print(f"[red]Open-PMC error: {e}[/red]")

    console.print(f"\n[bold]Total figures downloaded: {total}[/bold]")


@app.command()
def list_sources() -> None:
    """List available data sources for figures."""
    table = Table(title="Data Sources")
    table.add_column("Source", style="cyan")
    table.add_column("Description")
    table.add_column("Dataset")

    sources = [
        ("sourcedata", "EMBO SourceData - semantic annotations", "EMBO/SourceData"),
        ("openpmc", "Open-PMC-18M - 18M image-text pairs", "Various HF paths"),
        ("biorxiv", "bioRxiv preprints", "API: api.biorxiv.org"),
        ("pubmed", "PubMed Central open access", "NCBI E-utilities"),
    ]

    for name, desc, dataset in sources:
        table.add_row(name, desc, dataset)

    console.print(table)


if __name__ == "__main__":
    app()
