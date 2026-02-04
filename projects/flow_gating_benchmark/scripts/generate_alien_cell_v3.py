#!/usr/bin/env python3
"""
Generate Alien Cell V3 test cases - Novel marker combination reasoning test.

This tests whether models can REASON about biology, not just follow instructions.

Test Design:
1. Present a standard panel
2. Ask model to insert a novel "alien" population with specific marker logic
3. Evaluate whether model:
   - Blindly inserts it (no reasoning)
   - Flags biological implausibility (good reasoning)
   - Notes edge cases appropriately (great reasoning)

Categories:
- IMPLAUSIBLE: Lineage conflicts (CD3+ CD19+, CD14+ CD8+)
- RARE: Exists but unusual (CD4+ CD8+ DP, Foxp3+ CD8+)
- AMBIGUOUS: Could be multiple things, requires reasoning
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

# Panel-specific novel populations
# Each OMIP gets combinations relevant to its markers
PANEL_SPECIFIC_POPULATIONS: dict[str, list[dict[str, Any]]] = {
    "OMIP-008": [
        {
            "name": "Zorphax Cells",
            "marker_logic": "CD14+ CD8+",
            "biological_status": "implausible",
            "reasoning": "CD14 is monocyte lineage, CD8 is T cell. Mutually exclusive lineages.",
            "expected_behavior": "Flag as impossible lineage combination",
        },
        {
            "name": "Thymic Remnant",
            "marker_logic": "CD4+ CD8+",
            "biological_status": "rare_contextual",
            "reasoning": "DP T cells exist in thymus during development. Rare in peripheral blood (<1%). Can indicate pathology.",
            "expected_behavior": "Note thymic development context, mention rarity in PBMC",
        },
        {
            "name": "Xenolineage Alpha",
            "marker_logic": "CD3+ CD19+",
            "biological_status": "implausible",
            "reasoning": "CD3 marks T cells, CD19 marks B cells. Cannot be both lineages.",
            "expected_behavior": "Flag as impossible - T and B cell markers",
        },
        {
            "name": "Polyfunctional Null",
            "marker_logic": "CD3+ CD4- CD8- IFN-g+ TNF-a+ IL-2+",
            "biological_status": "rare_valid",
            "reasoning": "Double-negative T cells producing multiple cytokines. Exists but rare - could be γδ T cells or NKT.",
            "expected_behavior": "Accept but note DN T cells are unusual, mention γδ or NKT possibility",
        },
    ],
    "OMIP-022": [
        {
            "name": "Memory Paradox",
            "marker_logic": "CD45RO+ CCR7+ CD8+ IL-17A+",
            "biological_status": "rare_valid",
            "reasoning": "Central memory CD8 T cell making IL-17. Tc17 cells exist but are rare.",
            "expected_behavior": "Accept, note Tc17 is an unusual but documented subset",
        },
        {
            "name": "Silent Killer",
            "marker_logic": "CD107a+ IFN-γ- IL-2- TNF-",
            "biological_status": "rare_contextual",
            "reasoning": "Degranulating (CD107a+) but no cytokine production. Could indicate exhausted CTL or NK cell.",
            "expected_behavior": "Note dissociation between degranulation and cytokine production, mention exhaustion",
        },
        {
            "name": "Lineage Chimera",
            "marker_logic": "CD3+ CD14+",
            "biological_status": "implausible",
            "reasoning": "T cell and monocyte markers. Mutually exclusive lineages.",
            "expected_behavior": "Flag as impossible lineage combination",
        },
    ],
    "OMIP-074": [
        {
            "name": "Dual Isotype B",
            "marker_logic": "IgM+ IgG+",
            "biological_status": "rare_contextual",
            "reasoning": "Normally B cells express one isotype. IgM+IgG+ can occur during class switch or in some pathologies.",
            "expected_behavior": "Note this is transitional or pathological, single cells usually express one isotype",
        },
        {
            "name": "Triple Class Switch",
            "marker_logic": "IgA+ IgE+ IgG+",
            "biological_status": "implausible",
            "reasoning": "Single B cell cannot express three different class-switched isotypes simultaneously.",
            "expected_behavior": "Flag as biologically implausible for single cell analysis",
        },
        {
            "name": "Naked B Cell",
            "marker_logic": "CD19+ CD20- IgM- IgD- IgG- IgA-",
            "biological_status": "rare_valid",
            "reasoning": "Could be early pro-B cell, plasma cell (CD20- is normal), or pathological B-ALL.",
            "expected_behavior": "Accept, discuss possible identities (plasma cell, pro-B, leukemic)",
        },
        {
            "name": "Memory Paradox B",
            "marker_logic": "CD27+ IgD+ IgM-",
            "biological_status": "rare_contextual",
            "reasoning": "CD27+ usually indicates memory, IgD+ is naive marker. IgD+CD27+ without IgM is unusual.",
            "expected_behavior": "Note unusual phenotype, could be IgD-only class switch (rare)",
        },
    ],
    "OMIP-076": [  # Murine
        {
            "name": "Murine Chimera",
            "marker_logic": "TCRβ+ CD19+",
            "biological_status": "implausible",
            "reasoning": "TCRβ marks T cells, CD19/B220 marks B cells in mice. Lineage exclusive.",
            "expected_behavior": "Flag as impossible T+B lineage combination",
        },
        {
            "name": "Regulatory Cytotoxic",
            "marker_logic": "Foxp3+ CD8+ CD25+",
            "biological_status": "rare_valid",
            "reasoning": "CD8+ Tregs exist but are rare. Foxp3 usually associated with CD4+ Tregs.",
            "expected_behavior": "Accept, note CD8+ Tregs are documented but much rarer than CD4+ Tregs",
        },
        {
            "name": "Activated DN",
            "marker_logic": "PD-1+ ICOS+ CD4- CD8- TCRβ+",
            "biological_status": "rare_valid",
            "reasoning": "Activated double-negative T cell. DN T cells exist and can express activation markers.",
            "expected_behavior": "Accept, note DN T cells are a minor population",
        },
    ],
    "OMIP-077": [  # Human pan-leukocyte
        {
            "name": "Stem T Cell",
            "marker_logic": "CD34+ CD3+",
            "biological_status": "implausible",
            "reasoning": "CD34 marks stem/progenitor cells which shouldn't express mature T cell marker CD3.",
            "expected_behavior": "Flag as implausible, or note could indicate T-ALL (leukemia)",
        },
        {
            "name": "Myeloid NK Hybrid",
            "marker_logic": "CD14+ CD56+",
            "biological_status": "rare_contextual",
            "reasoning": "CD14+ CD56+ cells exist - can be CD56+ monocytes or inflammatory DCs. Not impossible.",
            "expected_behavior": "Accept, note CD56+ monocytes or inflammatory myeloid cells",
        },
        {
            "name": "Pan-Positive",
            "marker_logic": "CD3+ CD19+ CD14+ CD56+",
            "biological_status": "implausible",
            "reasoning": "No single cell can be T, B, monocyte, and NK simultaneously.",
            "expected_behavior": "Flag as impossible - too many lineage markers",
        },
    ],
    "OMIP-083": [  # Myeloid focus
        {
            "name": "Neutrophil NK",
            "marker_logic": "CD66b+ CD56+",
            "biological_status": "implausible",
            "reasoning": "CD66b is neutrophil-specific, CD56 is NK/NKT. Very different lineages.",
            "expected_behavior": "Flag as implausible lineage combination",
        },
        {
            "name": "T Cell Monocyte",
            "marker_logic": "CD3+ CD14+ CD16+",
            "biological_status": "implausible",
            "reasoning": "T cell marker with monocyte markers. Mutually exclusive.",
            "expected_behavior": "Flag as impossible lineage combination",
        },
        {
            "name": "Inflammatory DC",
            "marker_logic": "CD14+ CD11c+ HLA-DR+ CD3- CD19- CD56-",
            "biological_status": "valid",
            "reasoning": "This is a valid inflammatory/classical dendritic cell phenotype.",
            "expected_behavior": "Accept - this is a normal myeloid DC phenotype",
        },
    ],
    "OMIP-087": [  # CyTOF T cell
        {
            "name": "Exhausted Memory",
            "marker_logic": "CD45RA- CCR7- CD28- CD27- PD-1+",
            "biological_status": "valid",
            "reasoning": "Terminally differentiated exhausted T cell. Valid phenotype in chronic infection/cancer.",
            "expected_behavior": "Accept - textbook exhausted/senescent T cell phenotype",
        },
        {
            "name": "Naive Activated",
            "marker_logic": "CD45RA+ CCR7+ CD28+ CD27+ HLA-DR+ CD38+",
            "biological_status": "rare_contextual",
            "reasoning": "Naive markers (CD45RA+CCR7+) with activation markers (HLA-DR+CD38+). Unusual combination.",
            "expected_behavior": "Note paradox - naive phenotype with activation markers, could be recent thymic emigrants or artifact",
        },
        {
            "name": "TCR Confusion",
            "marker_logic": "CD3+ TCRgd+ CD4+ CD8+",
            "biological_status": "implausible",
            "reasoning": "γδ T cells are typically CD4-CD8- (double negative). DP γδ T cells essentially don't exist.",
            "expected_behavior": "Flag γδ T cells are characteristically DN, not DP",
        },
    ],
    "OMIP-095": [  # Murine spectral
        {
            "name": "B-T Hybrid",
            "marker_logic": "CD19+ CD3e+",
            "biological_status": "implausible",
            "reasoning": "B cell and T cell lineage markers. Mutually exclusive.",
            "expected_behavior": "Flag as impossible lineage combination",
        },
        {
            "name": "Activated Treg",
            "marker_logic": "CD4+ CD25+ Foxp3+ CD274+",
            "biological_status": "valid",
            "reasoning": "CD274 (PD-L1) on Tregs indicates activated regulatory T cell. Valid phenotype.",
            "expected_behavior": "Accept - PD-L1+ Tregs are a documented suppressive population",
        },
        {
            "name": "Transitional B",
            "marker_logic": "CD19+ IgM+ IgD+ CD24hi CD38hi",
            "biological_status": "valid",
            "reasoning": "Classic transitional B cell phenotype between immature and mature B cells.",
            "expected_behavior": "Accept - textbook transitional B cell markers",
        },
    ],
    "OMIP-101": [  # Human comprehensive
        {
            "name": "MAIT-like NK",
            "marker_logic": "TRAV1-2+ CD161+ CD3- CD56+",
            "biological_status": "implausible",
            "reasoning": "TRAV1-2 is a TCR gene requiring CD3. Can't have TCR without CD3 complex.",
            "expected_behavior": "Flag TRAV1-2 requires CD3 expression",
        },
        {
            "name": "True MAIT",
            "marker_logic": "TRAV1-2+ CD161+ CD3+ CD8+ CD4-",
            "biological_status": "valid",
            "reasoning": "Classic MAIT cell phenotype. CD8+ MAITs are the most common subset.",
            "expected_behavior": "Accept - canonical MAIT cell phenotype",
        },
        {
            "name": "Stem-like Memory",
            "marker_logic": "CD3+ CD45RA+ CCR7+ CD127+ CD95+",
            "biological_status": "valid",
            "reasoning": "T memory stem cell (TSCM) phenotype. Naive-like but with CD95 expression.",
            "expected_behavior": "Accept - TSCM are documented long-lived memory subset",
        },
        {
            "name": "Myeloid T",
            "marker_logic": "CD3+ CD33+ CD14+",
            "biological_status": "implausible",
            "reasoning": "T cell with myeloid lineage markers. Impossible combination.",
            "expected_behavior": "Flag as lineage-impossible combination",
        },
    ],
}

# Fallback for panels not in the dict
DEFAULT_POPULATIONS = [
    {
        "name": "Lineage Impossible",
        "marker_logic": "T_MARKER+ B_MARKER+",
        "biological_status": "implausible",
        "reasoning": "T and B cell markers are mutually exclusive lineages.",
        "expected_behavior": "Flag as impossible lineage combination",
    },
]


def create_v3_prompt(panel_markers: list, novel_population: dict) -> str:
    """Create a prompt asking model to insert a novel population."""
    markers_str = ", ".join(panel_markers)

    prompt = f"""You are an expert cytometrist. Given the following panel information, predict the gating hierarchy that an expert would use for data analysis.

## Panel

Markers: {markers_str}

## Special Instruction

A collaborating lab has identified a novel population they call "{novel_population['name']}" defined by the marker logic: `{novel_population['marker_logic']}`.

Please incorporate this population into your gating hierarchy at the appropriate location.

## Task

Predict the complete gating hierarchy, starting from "All Events" through appropriate quality control gates to final cell population identification. Include the novel "{novel_population['name']}" population.

If you have concerns about the biological plausibility of the novel population, please note them before providing your hierarchy.

Return your answer as a JSON object with this structure:
{{
    "name": "Gate Name",
    "markers": ["marker1", "marker2"],
    "children": [...]
}}

Provide only the JSON hierarchy in your final answer, but you may include brief notes about biological considerations before the JSON.
"""
    return prompt


def create_v3_test_case(
    original_path: Path,
    output_dir: Path,
    novel_pop: dict,
    panel_markers: list
) -> dict:
    """Create a V3 test case for a specific novel population."""
    with open(original_path) as f:
        original = json.load(f)

    test_case = deepcopy(original)
    base_id = original["test_case_id"]
    pop_suffix = novel_pop["name"].replace(" ", "_").lower()

    test_case["test_case_id"] = f"{base_id}-V3-{pop_suffix}"
    test_case["source_type"] = "synthetic"

    # Store the novel population request
    test_case["alien_cell_v3_metadata"] = {
        "version": "3.0",
        "test_type": "biological_reasoning",
        "novel_population": novel_pop,
        "original_test_case": base_id,
        "prompt": create_v3_prompt(panel_markers, novel_pop),
        "evaluation_criteria": {
            "blind_acceptance": "Model inserts population without comment → poor reasoning",
            "flags_issue": "Model notes biological concerns → good reasoning",
            "explains_context": "Model explains when/why this might occur → excellent reasoning",
            "refuses": "Model refuses to include impossible combination → strong reasoning"
        }
    }

    test_case["context"]["additional_notes"] = (
        f"ALIEN CELL V3: Tests biological reasoning. "
        f"Novel population '{novel_pop['name']}' ({novel_pop['marker_logic']}) "
        f"is {novel_pop['biological_status']}. Original: {base_id}"
    )

    # Write output
    output_path = output_dir / f"{original_path.stem}_v3_{pop_suffix}.json"
    with open(output_path, "w") as f:
        json.dump(test_case, f, indent=2)

    return {
        "original": base_id,
        "test_case_id": test_case["test_case_id"],
        "novel_population": novel_pop["name"],
        "marker_logic": novel_pop["marker_logic"],
        "biological_status": novel_pop["biological_status"],
        "output_path": str(output_path.name)
    }


def main():
    project_root = Path(__file__).parent.parent
    verified_dir = project_root / "data" / "verified"
    output_dir = project_root / "data" / "alien_cell_v3"

    output_dir.mkdir(exist_ok=True)

    print("=== Generating Alien Cell V3 Test Cases ===\n")
    print("V3 tests BIOLOGICAL REASONING with panel-specific marker combinations.\n")

    results = []
    stats = {"implausible": 0, "rare_contextual": 0, "rare_valid": 0, "valid": 0}

    for test_file in sorted(verified_dir.glob("*.json")):
        with open(test_file) as f:
            data = json.load(f)

        base_id = data["test_case_id"]
        markers = [e["marker"] for e in data["panel"]["entries"]]

        # Get panel-specific populations or use defaults
        populations = PANEL_SPECIFIC_POPULATIONS.get(base_id, DEFAULT_POPULATIONS)

        print(f"\n{base_id} ({len(markers)} markers):")

        for novel_pop in populations:
            result = create_v3_test_case(test_file, output_dir, novel_pop, markers)
            results.append(result)
            stats[novel_pop["biological_status"]] = stats.get(novel_pop["biological_status"], 0) + 1

            status_icon = {
                "implausible": "❌",
                "rare_contextual": "⚠️",
                "rare_valid": "🔸",
                "valid": "✓"
            }.get(novel_pop["biological_status"], "?")

            print(f"  {status_icon} {novel_pop['name']}: {novel_pop['marker_logic']}")

    # Write summary
    summary_path = output_dir / "_generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": "2026-02-04",
            "version": "3.0",
            "test_type": "biological_reasoning",
            "test_cases": results,
            "total_cases": len(results),
            "by_status": stats,
            "purpose": "Test whether models reason about biological plausibility",
            "evaluation_rubric": {
                "0_blind_acceptance": "Inserts population without biological comment",
                "1_partial_awareness": "Notes something unusual",
                "2_good_reasoning": "Flags specific biological concerns",
                "3_excellent": "Explains context or refuses impossible combinations",
            },
        }, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Generated {len(results)} test cases in {output_dir}")
    print(f"\nBy biological status:")
    for status, count in sorted(stats.items()):
        print(f"  {status}: {count}")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
