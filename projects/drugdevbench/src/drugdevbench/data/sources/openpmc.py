"""Open-PMC-18M data source for PubMed Central figures.

Open-PMC-18M is a large-scale dataset with 18M image-text pairs from
PubMed Central, with specialized support for compound figure separation.

Dataset may be available on Hugging Face or via direct download.
"""

import json
import re
from pathlib import Path
from typing import Any, Generator

from PIL import Image

from drugdevbench.data.schemas import (
    Annotation,
    Figure,
    FigureType,
    Question,
    QuestionType,
)


# Keywords to identify figure types from captions
FIGURE_TYPE_KEYWORDS = {
    FigureType.WESTERN_BLOT: [
        "western blot", "immunoblot", "western analysis", "wb analysis",
        "anti-", "antibody", "kda", "loading control", "β-actin", "gapdh",
    ],
    FigureType.DOSE_RESPONSE: [
        "dose-response", "dose response", "ic50", "ec50", "inhibition curve",
        "concentration-response", "sigmoidal", "hill slope",
    ],
    FigureType.PK_CURVE: [
        "pharmacokinetic", "pk profile", "concentration-time", "plasma concentration",
        "half-life", "t1/2", "cmax", "tmax", "auc", "clearance",
    ],
    FigureType.FLOW_BIAXIAL: [
        "flow cytometry", "facs", "biaxial", "dot plot", "scatter plot",
        "cd4", "cd8", "cd3", "cd45", "gating",
    ],
    FigureType.FLOW_HISTOGRAM: [
        "histogram", "fluorescence intensity", "mfi", "overlay",
    ],
    FigureType.HEATMAP: [
        "heatmap", "heat map", "expression profile", "hierarchical clustering",
        "gene expression", "rna-seq", "microarray",
    ],
    FigureType.VOLCANO_PLOT: [
        "volcano plot", "volcano", "differential expression", "-log10",
        "fold change", "log2fc",
    ],
    FigureType.ELISA: [
        "elisa", "enzyme-linked", "immunoassay", "standard curve",
        "absorbance", "od450",
    ],
    FigureType.VIABILITY_CURVE: [
        "viability", "cell viability", "mtt", "mts", "cck-8",
        "alamar blue", "live/dead",
    ],
    FigureType.CYTOTOXICITY: [
        "cytotoxicity", "cytotoxic", "cell death", "apoptosis",
        "annexin", "propidium iodide",
    ],
    FigureType.COOMASSIE_GEL: [
        "coomassie", "sds-page", "gel electrophoresis", "protein gel",
        "silver stain",
    ],
}


def infer_figure_type_from_caption(caption: str) -> FigureType | None:
    """Infer figure type from caption text using keyword matching.

    Args:
        caption: Figure caption text

    Returns:
        Inferred FigureType or None if cannot determine
    """
    caption_lower = caption.lower()

    # Score each figure type by keyword matches
    scores = {}
    for fig_type, keywords in FIGURE_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in caption_lower)
        if score > 0:
            scores[fig_type] = score

    if not scores:
        return None

    # Return the type with highest score
    return max(scores, key=scores.get)


class OpenPMCSource:
    """Fetch figures from Open-PMC-18M dataset.

    This source handles the Open-PMC-18M dataset which contains 18M
    image-text pairs from PubMed Central with compound figure separation.
    """

    # Common Hugging Face dataset locations to try
    DATASET_CANDIDATES = [
        "OpenPMC/Open-PMC-18M",
        "UCSC-VLAA/Open-PMC-18M",
        "pmc/open-pmc-18m",
    ]

    def __init__(
        self,
        cache_dir: Path | None = None,
        output_dir: Path | None = None,
        dataset_path: str | None = None,
    ):
        """Initialize Open-PMC source.

        Args:
            cache_dir: Directory for caching
            output_dir: Directory to save downloaded figures
            dataset_path: Specific dataset path on Hugging Face (optional)
        """
        self.cache_dir = cache_dir or Path("data/cache/openpmc")
        self.output_dir = output_dir or Path("data/figures")
        self.dataset_path = dataset_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dataset = None

    def _load_dataset(self, split: str = "train") -> Any:
        """Load the Open-PMC dataset from Hugging Face.

        Args:
            split: Dataset split to load

        Returns:
            Loaded dataset
        """
        if self._dataset is not None:
            return self._dataset

        from datasets import load_dataset

        # Try specified path first, then candidates
        paths_to_try = (
            [self.dataset_path] if self.dataset_path
            else self.DATASET_CANDIDATES
        )

        for dataset_path in paths_to_try:
            try:
                print(f"Trying to load dataset from: {dataset_path}")
                self._dataset = load_dataset(
                    dataset_path,
                    split=split,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                    streaming=True,  # Use streaming for large dataset
                )
                print(f"Successfully loaded from {dataset_path}")
                return self._dataset
            except Exception as e:
                print(f"Failed to load from {dataset_path}: {e}")
                continue

        raise RuntimeError(
            f"Could not load Open-PMC dataset. Tried: {paths_to_try}. "
            "Please specify dataset_path manually or check Hugging Face."
        )

    def get_figures_by_type(
        self,
        figure_types: list[FigureType] | None = None,
        max_figures: int = 100,
        split: str = "train",
    ) -> Generator[dict[str, Any], None, None]:
        """Get figures filtered by type.

        Args:
            figure_types: List of figure types to include (None = all)
            max_figures: Maximum number of figures to return
            split: Dataset split to use

        Yields:
            Dictionary with figure data
        """
        dataset = self._load_dataset(split)
        count = 0

        for example in dataset:
            if count >= max_figures:
                break

            # Get caption - try multiple field names
            caption = (
                example.get("caption") or
                example.get("text") or
                example.get("subcaption") or
                ""
            )

            # Infer figure type
            fig_type = infer_figure_type_from_caption(caption)

            # Skip if we can't determine type or it's not in requested types
            if fig_type is None:
                continue
            if figure_types and fig_type not in figure_types:
                continue

            # Get image
            image = example.get("image") or example.get("img")
            if image is None:
                continue

            yield {
                "id": example.get("id") or example.get("figure_id") or f"pmc_{count}",
                "image": image,
                "caption": caption,
                "figure_type": fig_type,
                "subfigure_id": example.get("subfigure_id") or example.get("panel_id"),
                "pmcid": example.get("pmcid") or example.get("pmc_id"),
                "doi": example.get("doi"),
                "raw": example,
            }
            count += 1

    def download_figures(
        self,
        figure_types: list[FigureType] | None = None,
        max_figures: int = 100,
        split: str = "train",
    ) -> list[Figure]:
        """Download figures and save to output directory.

        Args:
            figure_types: List of figure types to include (None = all)
            max_figures: Maximum number of figures to download
            split: Dataset split to use

        Returns:
            List of Figure objects
        """
        figures = []

        for fig_data in self.get_figures_by_type(
            figure_types=figure_types,
            max_figures=max_figures,
            split=split,
        ):
            try:
                # Get output path based on figure type
                fig_type = fig_data["figure_type"]
                type_dir = self._get_type_directory(fig_type)

                # Create unique filename
                fig_id = fig_data["id"]
                subfig = fig_data.get("subfigure_id")
                if subfig:
                    filename = f"pmc_{fig_id}_{subfig}.png"
                else:
                    filename = f"pmc_{fig_id}.png"
                output_path = type_dir / filename

                # Save image
                image = fig_data["image"]
                if image is not None:
                    if isinstance(image, Image.Image):
                        image.save(output_path)
                    else:
                        Image.open(image).save(output_path)

                # Create Figure object
                figure_id = f"openpmc_{fig_id}"
                if subfig:
                    figure_id = f"{figure_id}_{subfig}"

                figure = Figure(
                    figure_id=figure_id,
                    figure_type=fig_type,
                    image_path=str(output_path),
                    legend_text=fig_data["caption"],
                    paper_doi=fig_data.get("doi"),
                    source="openpmc",
                    metadata={
                        "pmcid": fig_data.get("pmcid"),
                        "subfigure_id": subfig,
                    },
                )
                figures.append(figure)
                print(f"Downloaded: {figure.figure_id} ({fig_type.value})")

            except Exception as e:
                print(f"Error downloading figure {fig_data['id']}: {e}")
                continue

        return figures

    def _get_type_directory(self, figure_type: FigureType) -> Path:
        """Get the output directory for a figure type."""
        type_to_dir = {
            FigureType.WESTERN_BLOT: "western_blots",
            FigureType.COOMASSIE_GEL: "western_blots",
            FigureType.DOT_BLOT: "western_blots",
            FigureType.ELISA: "elisa",
            FigureType.DOSE_RESPONSE: "dose_response",
            FigureType.IC50_EC50: "dose_response",
            FigureType.PK_CURVE: "pk_curves",
            FigureType.AUC_PLOT: "pk_curves",
            FigureType.FLOW_BIAXIAL: "flow_cytometry",
            FigureType.FLOW_HISTOGRAM: "flow_cytometry",
            FigureType.GATING_STRATEGY: "flow_cytometry",
            FigureType.HEATMAP: "heatmaps",
            FigureType.VOLCANO_PLOT: "heatmaps",
            FigureType.VIABILITY_CURVE: "dose_response",
            FigureType.CYTOTOXICITY: "dose_response",
            FigureType.PROLIFERATION: "dose_response",
        }
        subdir = type_to_dir.get(figure_type, "misc")
        output_dir = self.output_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def create_annotations_from_figures(
        self,
        figures: list[Figure],
    ) -> list[Annotation]:
        """Create annotations with auto-generated questions for figures.

        Args:
            figures: List of Figure objects

        Returns:
            List of Annotation objects with questions
        """
        from drugdevbench.questions import generate_questions

        annotations = []

        for figure in figures:
            questions = generate_questions(
                figure_id=figure.figure_id,
                figure_type=figure.figure_type,
                legend_text=figure.legend_text,
            )

            annotation = Annotation(
                figure=figure,
                questions=questions,
                annotator="openpmc_auto",
            )
            annotations.append(annotation)

        return annotations


def download_openpmc_figures(
    output_dir: Path | str = "data/figures",
    max_figures: int = 100,
    figure_types: list[FigureType] | None = None,
    dataset_path: str | None = None,
) -> list[Figure]:
    """Convenience function to download figures from Open-PMC-18M.

    Args:
        output_dir: Directory to save figures
        max_figures: Maximum number of figures to download
        figure_types: List of figure types to include (None = all)
        dataset_path: Specific Hugging Face dataset path

    Returns:
        List of downloaded Figure objects
    """
    source = OpenPMCSource(
        output_dir=Path(output_dir),
        dataset_path=dataset_path,
    )
    return source.download_figures(
        figure_types=figure_types,
        max_figures=max_figures,
    )
