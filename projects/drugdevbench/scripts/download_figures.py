#!/usr/bin/env python3
"""Download figures from SourceData and Open-PMC-18M datasets.

Usage:
    python scripts/download_figures.py --source sourcedata --max 100
    python scripts/download_figures.py --source openpmc --max 100
    python scripts/download_figures.py --source all --max 100
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drugdevbench.data.schemas import FigureType
from drugdevbench.data.sources.sourcedata import SourceDataSource
from drugdevbench.data.sources.openpmc import OpenPMCSource
from drugdevbench.data import save_annotations


def download_from_sourcedata(
    output_dir: Path,
    max_figures: int = 100,
    figure_types: list[FigureType] | None = None,
) -> int:
    """Download figures from SourceData (EMBO).

    Args:
        output_dir: Directory to save figures
        max_figures: Maximum number of figures to download
        figure_types: List of figure types to include

    Returns:
        Number of figures downloaded
    """
    print("\n" + "=" * 60)
    print("Downloading from SourceData (EMBO/SourceData)")
    print("=" * 60)

    source = SourceDataSource(output_dir=output_dir)

    try:
        figures = source.download_figures(
            figure_types=figure_types,
            max_figures=max_figures,
        )

        # Create annotations
        annotations = source.create_annotations_from_figures(figures)

        # Save annotations
        annotations_path = output_dir.parent / "annotations" / "sourcedata_annotations.jsonl"
        save_annotations(annotations, annotations_path)
        print(f"\nSaved {len(annotations)} annotations to {annotations_path}")

        return len(figures)

    except Exception as e:
        print(f"Error downloading from SourceData: {e}")
        return 0


def download_from_openpmc(
    output_dir: Path,
    max_figures: int = 100,
    figure_types: list[FigureType] | None = None,
    dataset_path: str | None = None,
) -> int:
    """Download figures from Open-PMC-18M.

    Args:
        output_dir: Directory to save figures
        max_figures: Maximum number of figures to download
        figure_types: List of figure types to include
        dataset_path: Specific Hugging Face dataset path

    Returns:
        Number of figures downloaded
    """
    print("\n" + "=" * 60)
    print("Downloading from Open-PMC-18M")
    print("=" * 60)

    source = OpenPMCSource(
        output_dir=output_dir,
        dataset_path=dataset_path,
    )

    try:
        figures = source.download_figures(
            figure_types=figure_types,
            max_figures=max_figures,
        )

        # Create annotations
        annotations = source.create_annotations_from_figures(figures)

        # Save annotations
        annotations_path = output_dir.parent / "annotations" / "openpmc_annotations.jsonl"
        save_annotations(annotations, annotations_path)
        print(f"\nSaved {len(annotations)} annotations to {annotations_path}")

        return len(figures)

    except Exception as e:
        print(f"Error downloading from Open-PMC: {e}")
        print("Note: Open-PMC-18M may require specific dataset path.")
        print("Try: --openpmc-path 'dataset/path' to specify manually")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download figures from scientific figure datasets"
    )
    parser.add_argument(
        "--source",
        choices=["sourcedata", "openpmc", "all"],
        default="all",
        help="Which source to download from (default: all)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=100,
        help="Maximum figures to download per source (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/figures"),
        help="Output directory for figures (default: data/figures)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=[ft.value for ft in FigureType],
        help="Filter by figure types (default: all types)",
    )
    parser.add_argument(
        "--openpmc-path",
        type=str,
        default=None,
        help="Specific Hugging Face path for Open-PMC dataset",
    )

    args = parser.parse_args()

    # Parse figure types
    figure_types = None
    if args.types:
        figure_types = [FigureType(t) for t in args.types]

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    # Download from selected sources
    if args.source in ("sourcedata", "all"):
        count = download_from_sourcedata(
            output_dir=args.output,
            max_figures=args.max,
            figure_types=figure_types,
        )
        total_downloaded += count
        print(f"\nSourceData: Downloaded {count} figures")

    if args.source in ("openpmc", "all"):
        count = download_from_openpmc(
            output_dir=args.output,
            max_figures=args.max,
            figure_types=figure_types,
            dataset_path=args.openpmc_path,
        )
        total_downloaded += count
        print(f"\nOpen-PMC: Downloaded {count} figures")

    print("\n" + "=" * 60)
    print(f"Total figures downloaded: {total_downloaded}")
    print("=" * 60)

    # Summary by type
    print("\nFigures by type:")
    for subdir in args.output.iterdir():
        if subdir.is_dir():
            count = len(list(subdir.glob("*.png")))
            if count > 0:
                print(f"  {subdir.name}: {count}")


if __name__ == "__main__":
    main()
