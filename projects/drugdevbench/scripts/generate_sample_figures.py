#!/usr/bin/env python3
"""Generate sample placeholder figures for testing the DrugDevBench pipeline.

This script creates synthetic placeholder figures when external data sources
are unavailable (e.g., network restrictions).

Usage:
    python scripts/generate_sample_figures.py --count 100
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image, ImageDraw, ImageFont

from drugdevbench.data.schemas import (
    Annotation,
    Figure,
    FigureType,
    Question,
    QuestionType,
)
from drugdevbench.data import save_annotations
from drugdevbench.questions import generate_questions


# Sample captions for each figure type
SAMPLE_CAPTIONS = {
    FigureType.WESTERN_BLOT: [
        "Western blot analysis of p53 expression. β-actin served as loading control. MW markers shown on left (kDa).",
        "Immunoblot showing ERK1/2 phosphorylation after treatment. GAPDH loading control. n=3 independent experiments.",
        "Western blot of cleaved caspase-3 (17 kDa) indicating apoptosis. Tubulin loading control.",
    ],
    FigureType.DOSE_RESPONSE: [
        "Dose-response curve for Compound A. IC50 = 2.3 μM. Data represent mean ± SEM, n=3.",
        "Inhibition of cell proliferation by Drug X. EC50 = 15 nM. Hill slope = 1.2.",
        "Concentration-response relationship. IC50 = 450 nM (95% CI: 380-530 nM).",
    ],
    FigureType.PK_CURVE: [
        "Plasma concentration-time profile after 10 mg/kg IV dose. t1/2 = 4.2 hours, Cmax = 1250 ng/mL.",
        "Pharmacokinetic profile in mice (n=6). Tmax = 0.5 hr, AUC = 5420 ng·h/mL.",
        "Single-dose PK study. Terminal half-life = 8.5 hours. Bioavailability = 45%.",
    ],
    FigureType.FLOW_BIAXIAL: [
        "Flow cytometry analysis of CD4+ and CD8+ T cells. Numbers indicate percentage of parent gate.",
        "FACS analysis showing CD45+CD3+ populations. Gated on live singlets.",
        "Biaxial plot of surface markers. Q1: 12.3%, Q2: 45.6%, Q3: 8.9%, Q4: 33.2%.",
    ],
    FigureType.FLOW_HISTOGRAM: [
        "Histogram overlay showing GFP expression. Blue: control, Red: treated. MFI fold-change = 3.2.",
        "Flow cytometry histogram of CD69 expression after stimulation.",
        "Fluorescence intensity distribution. Percent positive: 78.4%.",
    ],
    FigureType.HEATMAP: [
        "Heatmap of differentially expressed genes (FDR < 0.05, |log2FC| > 1). Hierarchical clustering shown.",
        "Gene expression heatmap. Color scale: log2 normalized counts. n=3 per group.",
        "RNA-seq expression profile. Z-score normalized. 1,234 genes shown.",
    ],
    FigureType.ELISA: [
        "ELISA standard curve. 4-parameter logistic fit, R² = 0.998. LOD = 15 pg/mL.",
        "Cytokine levels measured by ELISA. Detection range: 31.25-2000 pg/mL.",
        "Sandwich ELISA results. Intra-assay CV < 10%.",
    ],
    FigureType.VIABILITY_CURVE: [
        "Cell viability assay (MTT). IC50 = 5.8 μM after 72h treatment.",
        "Dose-dependent cytotoxicity. Cell viability normalized to DMSO control.",
        "CellTiter-Glo viability assay. EC50 = 125 nM. n=4 replicates.",
    ],
    FigureType.VOLCANO_PLOT: [
        "Volcano plot of differential expression. Red: upregulated (log2FC > 1, padj < 0.05).",
        "Differential expression analysis. 456 genes upregulated, 312 downregulated.",
        "RNA-seq volcano plot. Horizontal line: FDR = 0.05. Vertical lines: |log2FC| = 1.",
    ],
}

# Colors for different figure types
FIGURE_COLORS = {
    FigureType.WESTERN_BLOT: (50, 50, 80),
    FigureType.DOSE_RESPONSE: (80, 50, 50),
    FigureType.PK_CURVE: (50, 80, 50),
    FigureType.FLOW_BIAXIAL: (80, 80, 50),
    FigureType.FLOW_HISTOGRAM: (80, 50, 80),
    FigureType.HEATMAP: (50, 80, 80),
    FigureType.ELISA: (70, 60, 50),
    FigureType.VIABILITY_CURVE: (60, 70, 50),
    FigureType.VOLCANO_PLOT: (60, 50, 70),
}


def create_placeholder_image(
    figure_type: FigureType,
    figure_id: str,
    caption: str,
    width: int = 800,
    height: int = 600,
) -> Image.Image:
    """Create a placeholder image for a figure type.

    Args:
        figure_type: Type of figure
        figure_id: Unique identifier
        caption: Figure caption
        width: Image width
        height: Image height

    Returns:
        PIL Image object
    """
    # Get color for this figure type
    bg_color = FIGURE_COLORS.get(figure_type, (60, 60, 60))

    # Create image
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw border
    border_color = tuple(min(c + 40, 255) for c in bg_color)
    draw.rectangle([0, 0, width - 1, height - 1], outline=border_color, width=3)

    # Draw figure type label
    label = f"[{figure_type.value.upper()}]"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    # Center the label
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    x = (width - text_width) // 2
    draw.text((x, 30), label, fill=(255, 255, 255), font=font)

    # Draw figure ID
    draw.text((20, height - 40), f"ID: {figure_id}", fill=(200, 200, 200), font=small_font)

    # Draw placeholder content based on figure type
    _draw_placeholder_content(draw, figure_type, width, height)

    # Draw caption (wrapped)
    caption_y = height - 80
    max_chars = width // 8
    wrapped = [caption[i : i + max_chars] for i in range(0, len(caption), max_chars)][:2]
    for i, line in enumerate(wrapped):
        draw.text((20, caption_y + i * 18), line, fill=(180, 180, 180), font=small_font)

    return img


def _draw_placeholder_content(
    draw: ImageDraw.Draw,
    figure_type: FigureType,
    width: int,
    height: int,
) -> None:
    """Draw placeholder content specific to each figure type."""
    center_x, center_y = width // 2, height // 2

    if figure_type == FigureType.WESTERN_BLOT:
        # Draw gel lanes
        lane_width = 40
        for i in range(6):
            x = 150 + i * 90
            # Draw lane background
            draw.rectangle([x, 100, x + lane_width, 350], fill=(30, 30, 40))
            # Draw bands
            for j in range(random.randint(1, 4)):
                band_y = 120 + j * 60 + random.randint(-10, 10)
                intensity = random.randint(100, 255)
                draw.rectangle(
                    [x + 5, band_y, x + lane_width - 5, band_y + 15],
                    fill=(intensity, intensity, intensity),
                )

    elif figure_type in (FigureType.DOSE_RESPONSE, FigureType.VIABILITY_CURVE):
        # Draw sigmoidal curve
        points = []
        for i in range(20):
            x = 100 + i * 30
            # Sigmoid function
            t = (i - 10) / 3
            y = center_y + 100 - int(200 / (1 + 2.718 ** (-t)))
            points.append((x, y))
        draw.line(points, fill=(100, 200, 100), width=3)
        # Draw points
        for p in points[::2]:
            draw.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(150, 250, 150))

    elif figure_type == FigureType.PK_CURVE:
        # Draw PK curve (exponential decay)
        points = []
        for i in range(25):
            x = 80 + i * 25
            y = center_y + 100 - int(180 * (0.9 ** i))
            points.append((x, y))
        draw.line(points, fill=(100, 100, 200), width=3)
        for p in points[::3]:
            draw.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(150, 150, 250))

    elif figure_type == FigureType.FLOW_BIAXIAL:
        # Draw quadrant plot
        draw.line([(center_x, 100), (center_x, 400)], fill=(150, 150, 150), width=2)
        draw.line([(150, center_y), (650, center_y)], fill=(150, 150, 150), width=2)
        # Draw scattered points in each quadrant
        for _ in range(200):
            qx = random.choice([-1, 1])
            qy = random.choice([-1, 1])
            x = center_x + qx * random.randint(20, 200)
            y = center_y + qy * random.randint(20, 120)
            color = (100 + qx * 50, 100 + qy * 50, 150)
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)

    elif figure_type == FigureType.FLOW_HISTOGRAM:
        # Draw histogram
        for i in range(30):
            x = 100 + i * 20
            h = int(100 * (2.718 ** (-((i - 15) ** 2) / 50)))
            draw.rectangle([x, center_y + 50 - h, x + 15, center_y + 50], fill=(100, 150, 200))

    elif figure_type == FigureType.HEATMAP:
        # Draw heatmap grid
        cell_size = 25
        for i in range(12):
            for j in range(15):
                val = random.randint(0, 255)
                color = (val, 50, 255 - val)  # Blue to red
                x = 150 + j * cell_size
                y = 100 + i * cell_size
                draw.rectangle([x, y, x + cell_size - 2, y + cell_size - 2], fill=color)

    elif figure_type == FigureType.VOLCANO_PLOT:
        # Draw volcano plot
        for _ in range(300):
            x = center_x + random.randint(-250, 250)
            # Higher y near center
            spread = abs(x - center_x)
            y = center_y + random.randint(-50, 150) - spread // 3
            # Color by significance
            if abs(x - center_x) > 150 and y < center_y - 50:
                color = (250, 100, 100) if x > center_x else (100, 100, 250)
            else:
                color = (150, 150, 150)
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)


def generate_sample_figures(
    output_dir: Path,
    count_per_type: int = 100,
) -> list[Figure]:
    """Generate sample placeholder figures for all types.

    Args:
        output_dir: Directory to save figures
        count_per_type: Number of figures per type

    Returns:
        List of Figure objects
    """
    figures = []

    figure_types = [
        FigureType.WESTERN_BLOT,
        FigureType.DOSE_RESPONSE,
        FigureType.PK_CURVE,
        FigureType.FLOW_BIAXIAL,
        FigureType.FLOW_HISTOGRAM,
        FigureType.HEATMAP,
        FigureType.ELISA,
        FigureType.VIABILITY_CURVE,
        FigureType.VOLCANO_PLOT,
    ]

    type_to_dir = {
        FigureType.WESTERN_BLOT: "western_blots",
        FigureType.DOSE_RESPONSE: "dose_response",
        FigureType.PK_CURVE: "pk_curves",
        FigureType.FLOW_BIAXIAL: "flow_cytometry",
        FigureType.FLOW_HISTOGRAM: "flow_cytometry",
        FigureType.HEATMAP: "heatmaps",
        FigureType.ELISA: "elisa",
        FigureType.VIABILITY_CURVE: "dose_response",
        FigureType.VOLCANO_PLOT: "heatmaps",
    }

    for fig_type in figure_types:
        print(f"\nGenerating {count_per_type} {fig_type.value} figures...")
        subdir = output_dir / type_to_dir[fig_type]
        subdir.mkdir(parents=True, exist_ok=True)

        captions = SAMPLE_CAPTIONS.get(fig_type, ["Sample figure caption."])

        for i in range(count_per_type):
            figure_id = f"sample_{fig_type.value}_{i:04d}"
            caption = random.choice(captions)

            # Create and save image
            img = create_placeholder_image(fig_type, figure_id, caption)
            img_path = subdir / f"{figure_id}.png"
            img.save(img_path)

            # Create Figure object
            figure = Figure(
                figure_id=figure_id,
                figure_type=fig_type,
                image_path=str(img_path),
                legend_text=caption,
                source="sample",
                metadata={"generated": datetime.now().isoformat()},
            )
            figures.append(figure)

            if (i + 1) % 25 == 0:
                print(f"  Generated {i + 1}/{count_per_type}")

    return figures


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample placeholder figures for DrugDevBench"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of figures per type (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/figures"),
        help="Output directory (default: data/figures)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        help="Specific figure types to generate (default: all)",
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Sample Figures for DrugDevBench")
    print("=" * 60)

    figures = generate_sample_figures(
        output_dir=args.output,
        count_per_type=args.count,
    )

    print(f"\nGenerated {len(figures)} total figures")

    # Generate questions and annotations
    print("\nGenerating annotations...")
    annotations = []
    for fig in figures:
        questions = generate_questions(
            figure_id=fig.figure_id,
            figure_type=fig.figure_type,
            legend_text=fig.legend_text,
        )
        annotations.append(
            Annotation(
                figure=fig,
                questions=questions,
                annotator="sample_generator",
            )
        )

    # Save annotations
    ann_path = args.output.parent / "annotations" / "sample_annotations.jsonl"
    save_annotations(annotations, ann_path)
    print(f"Saved {len(annotations)} annotations to {ann_path}")

    # Summary by type
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    from collections import Counter

    type_counts = Counter(f.figure_type.value for f in figures)
    for fig_type, count in sorted(type_counts.items()):
        print(f"  {fig_type}: {count}")


if __name__ == "__main__":
    main()
