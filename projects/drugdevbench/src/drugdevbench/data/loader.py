"""Data loading utilities for DrugDevBench."""

import json
from pathlib import Path

from drugdevbench.data.schemas import Annotation, Figure, FigureType


def get_data_dir() -> Path:
    """Get the data directory path."""
    # Look for data directory relative to package
    package_dir = Path(__file__).parent.parent.parent.parent
    data_dir = package_dir / "data"
    if data_dir.exists():
        return data_dir
    # Fallback to current directory
    return Path("data")


def get_figure_path(figure_id: str, figure_type: FigureType) -> Path:
    """Get the path to a figure image file.

    Args:
        figure_id: Unique identifier for the figure
        figure_type: Type of figure (used to determine subdirectory)

    Returns:
        Path to the figure image file
    """
    data_dir = get_data_dir()
    # Map figure types to directory names
    type_to_dir = {
        FigureType.WESTERN_BLOT: "western_blots",
        FigureType.COOMASSIE_GEL: "western_blots",  # Same directory
        FigureType.DOT_BLOT: "western_blots",
        FigureType.ELISA: "elisa",
        FigureType.DOSE_RESPONSE: "dose_response",
        FigureType.IC50_EC50: "dose_response",
        FigureType.PK_CURVE: "pk_curves",
        FigureType.AUC_PLOT: "pk_curves",
        FigureType.COMPARTMENT_MODEL: "pk_curves",
        FigureType.FLOW_BIAXIAL: "flow_cytometry",
        FigureType.FLOW_HISTOGRAM: "flow_cytometry",
        FigureType.GATING_STRATEGY: "flow_cytometry",
        FigureType.HEATMAP: "heatmaps",
        FigureType.VOLCANO_PLOT: "heatmaps",
        FigureType.PATHWAY_ENRICHMENT: "heatmaps",
        FigureType.VIABILITY_CURVE: "dose_response",
        FigureType.PROLIFERATION: "dose_response",
        FigureType.CYTOTOXICITY: "dose_response",
    }
    subdir = type_to_dir.get(figure_type, "misc")
    return data_dir / "figures" / subdir / f"{figure_id}.png"


def load_annotations(annotations_path: Path | str | None = None) -> list[Annotation]:
    """Load annotations from a JSONL file.

    Args:
        annotations_path: Path to annotations JSONL file. If None, uses default.

    Returns:
        List of Annotation objects
    """
    if annotations_path is None:
        annotations_path = get_data_dir() / "annotations" / "annotations.jsonl"
    else:
        annotations_path = Path(annotations_path)

    if not annotations_path.exists():
        return []

    annotations = []
    with open(annotations_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                annotations.append(Annotation(**data))

    return annotations


def load_figures(figure_type: FigureType | None = None) -> list[Figure]:
    """Load all figures, optionally filtered by type.

    Args:
        figure_type: If provided, only return figures of this type

    Returns:
        List of Figure objects
    """
    annotations = load_annotations()
    figures = [a.figure for a in annotations]

    if figure_type is not None:
        figures = [f for f in figures if f.figure_type == figure_type]

    return figures


def save_annotations(annotations: list[Annotation], output_path: Path | str | None = None) -> None:
    """Save annotations to a JSONL file.

    Args:
        annotations: List of Annotation objects to save
        output_path: Path to output JSONL file. If None, uses default.
    """
    if output_path is None:
        output_path = get_data_dir() / "annotations" / "annotations.jsonl"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for annotation in annotations:
            f.write(annotation.model_dump_json() + "\n")


def load_paper_metadata(metadata_path: Path | str | None = None) -> dict:
    """Load paper metadata from JSON file.

    Args:
        metadata_path: Path to metadata JSON file. If None, uses default.

    Returns:
        Dictionary of paper metadata keyed by DOI
    """
    if metadata_path is None:
        metadata_path = get_data_dir() / "papers" / "metadata.json"
    else:
        metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        return {}

    with open(metadata_path) as f:
        return json.load(f)


def save_paper_metadata(metadata: dict, output_path: Path | str | None = None) -> None:
    """Save paper metadata to JSON file.

    Args:
        metadata: Dictionary of paper metadata
        output_path: Path to output JSON file. If None, uses default.
    """
    if output_path is None:
        output_path = get_data_dir() / "papers" / "metadata.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
