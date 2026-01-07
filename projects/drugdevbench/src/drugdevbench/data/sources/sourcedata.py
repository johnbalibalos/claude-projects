"""SourceData (EMBO) data source for semantically annotated figures.

SourceData provides deep semantic annotations linking figure elements to
biological entities using standardized ontologies (Uniprot, NCBI Taxon).

Dataset: https://huggingface.co/datasets/EMBO/SourceData
"""

import json
import re
from pathlib import Path
from typing import Any, Generator

from datasets import load_dataset
from PIL import Image

from drugdevbench.data.schemas import (
    Annotation,
    Figure,
    FigureType,
    Question,
    QuestionType,
)


# Mapping from SourceData panel types to DrugDevBench figure types
PANEL_TYPE_MAPPING = {
    "western blot": FigureType.WESTERN_BLOT,
    "western": FigureType.WESTERN_BLOT,
    "immunoblot": FigureType.WESTERN_BLOT,
    "blot": FigureType.WESTERN_BLOT,
    "gel": FigureType.COOMASSIE_GEL,
    "sds-page": FigureType.COOMASSIE_GEL,
    "flow cytometry": FigureType.FLOW_BIAXIAL,
    "facs": FigureType.FLOW_BIAXIAL,
    "flow": FigureType.FLOW_BIAXIAL,
    "histogram": FigureType.FLOW_HISTOGRAM,
    "dose-response": FigureType.DOSE_RESPONSE,
    "dose response": FigureType.DOSE_RESPONSE,
    "ic50": FigureType.IC50_EC50,
    "ec50": FigureType.IC50_EC50,
    "pharmacokinetic": FigureType.PK_CURVE,
    "pk": FigureType.PK_CURVE,
    "concentration-time": FigureType.PK_CURVE,
    "elisa": FigureType.ELISA,
    "heatmap": FigureType.HEATMAP,
    "heat map": FigureType.HEATMAP,
    "volcano": FigureType.VOLCANO_PLOT,
    "volcano plot": FigureType.VOLCANO_PLOT,
    "viability": FigureType.VIABILITY_CURVE,
    "cytotoxicity": FigureType.CYTOTOXICITY,
    "proliferation": FigureType.PROLIFERATION,
}


def infer_figure_type(caption: str, panel_type: str | None = None) -> FigureType | None:
    """Infer figure type from caption text and panel type annotation.

    Args:
        caption: Figure caption/legend text
        panel_type: Optional panel type from SourceData annotations

    Returns:
        Inferred FigureType or None if cannot determine
    """
    text = f"{caption} {panel_type or ''}".lower()

    for keyword, fig_type in PANEL_TYPE_MAPPING.items():
        if keyword in text:
            return fig_type

    return None


class SourceDataSource:
    """Fetch figures and annotations from SourceData (EMBO) via Hugging Face."""

    DATASET_NAME = "EMBO/SourceData"

    def __init__(
        self,
        cache_dir: Path | None = None,
        output_dir: Path | None = None,
    ):
        """Initialize SourceData source.

        Args:
            cache_dir: Directory for Hugging Face cache
            output_dir: Directory to save downloaded figures
        """
        self.cache_dir = cache_dir or Path("data/cache/sourcedata")
        self.output_dir = output_dir or Path("data/figures")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dataset = None

    def _load_dataset(self, split: str = "train") -> Any:
        """Load the SourceData dataset from Hugging Face.

        Args:
            split: Dataset split to load

        Returns:
            Loaded dataset
        """
        if self._dataset is None:
            print(f"Loading SourceData dataset from Hugging Face...")
            try:
                self._dataset = load_dataset(
                    self.DATASET_NAME,
                    split=split,
                    cache_dir=str(self.cache_dir),
                )
                print(f"Loaded {len(self._dataset)} examples")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                print("Falling back to streaming mode...")
                self._dataset = load_dataset(
                    self.DATASET_NAME,
                    split=split,
                    cache_dir=str(self.cache_dir),
                    streaming=True,
                )
        return self._dataset

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
            Dictionary with figure data and annotations
        """
        dataset = self._load_dataset(split)
        count = 0

        for example in dataset:
            if count >= max_figures:
                break

            # Extract caption/legend
            caption = example.get("caption", "") or example.get("legend", "") or ""

            # Try to infer figure type
            panel_type = example.get("panel_type")
            fig_type = infer_figure_type(caption, panel_type)

            # Skip if we can't determine type or it's not in requested types
            if fig_type is None:
                continue
            if figure_types and fig_type not in figure_types:
                continue

            yield {
                "id": example.get("id") or f"sd_{count}",
                "image": example.get("image"),
                "caption": caption,
                "figure_type": fig_type,
                "panel_type": panel_type,
                "entities": example.get("entities", []),
                "source_doi": example.get("doi"),
                "source_pmid": example.get("pmid"),
                "journal": example.get("journal"),
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
            figure_types: List of figure types to include (None = all types)
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
                output_path = type_dir / f"sd_{fig_data['id']}.png"

                # Save image
                image = fig_data["image"]
                if image is not None:
                    if isinstance(image, Image.Image):
                        image.save(output_path)
                    else:
                        # Handle other image formats if needed
                        Image.open(image).save(output_path)

                # Create Figure object
                figure = Figure(
                    figure_id=f"sourcedata_{fig_data['id']}",
                    figure_type=fig_type,
                    image_path=str(output_path),
                    legend_text=fig_data["caption"],
                    paper_doi=fig_data.get("source_doi"),
                    source="sourcedata",
                    metadata={
                        "panel_type": fig_data.get("panel_type"),
                        "entities": fig_data.get("entities", []),
                        "journal": fig_data.get("journal"),
                        "pmid": fig_data.get("source_pmid"),
                    },
                )
                figures.append(figure)
                print(f"Downloaded: {figure.figure_id} ({fig_type.value})")

            except Exception as e:
                print(f"Error downloading figure {fig_data['id']}: {e}")
                continue

        return figures

    def _get_type_directory(self, figure_type: FigureType) -> Path:
        """Get the output directory for a figure type.

        Args:
            figure_type: Type of figure

        Returns:
            Path to the output directory
        """
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
                annotator="sourcedata_auto",
            )
            annotations.append(annotation)

        return annotations


def download_sourcedata_figures(
    output_dir: Path | str = "data/figures",
    max_figures: int = 100,
    figure_types: list[FigureType] | None = None,
) -> list[Figure]:
    """Convenience function to download figures from SourceData.

    Args:
        output_dir: Directory to save figures
        max_figures: Maximum number of figures to download
        figure_types: List of figure types to include (None = all)

    Returns:
        List of downloaded Figure objects
    """
    source = SourceDataSource(output_dir=Path(output_dir))
    return source.download_figures(
        figure_types=figure_types,
        max_figures=max_figures,
    )
