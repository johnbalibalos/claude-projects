"""
Gate name normalization for flow cytometry hierarchies.

Provides text normalization and semantic synonym matching for
comparing gate names between predicted and ground truth hierarchies.
"""

from __future__ import annotations

import re

# Cell type synonyms for semantic matching
# Maps variations to a canonical form
CELL_TYPE_SYNONYMS: dict[str, str] = {
    # T cell variations
    "t cells": "t_cells",
    "t-cells": "t_cells",
    "t lymphocytes": "t_cells",
    "t lymphs": "t_cells",
    "cd3+ t cells": "t_cells",
    "cd3+ t": "t_cells",
    "cd3+": "t_cells",
    "ab t cells": "t_cells",
    "αβ t cells": "t_cells",
    # B cell variations
    "b cells": "b_cells",
    "b-cells": "b_cells",
    "b lymphocytes": "b_cells",
    "b lymphs": "b_cells",
    "b lineage": "b_cells",
    "cd19+ b cells": "b_cells",
    "cd19+ b": "b_cells",
    "cd19+": "b_cells",
    "cd20+ b cells": "b_cells",
    "cd20+ b": "b_cells",
    "cd20+": "b_cells",
    # NK cell variations
    "nk cells": "nk_cells",
    "nk": "nk_cells",
    "natural killer cells": "nk_cells",
    "natural killer": "nk_cells",
    "cd56+ nk cells": "nk_cells",
    "cd56+ nk": "nk_cells",
    "cd56+cd3-": "nk_cells",
    "cd3- cd56+": "nk_cells",
    "cd3-cd56+": "nk_cells",
    # NK subset variations
    "cd56bright": "cd56bright_nk",
    "cd56bright nk": "cd56bright_nk",
    "cd56bright nk cells": "cd56bright_nk",
    "cd56dim": "cd56dim_nk",
    "cd56dim nk": "cd56dim_nk",
    "cd56dim nk cells": "cd56dim_nk",
    # NKT variations
    "nkt cells": "nkt_cells",
    "nkt": "nkt_cells",
    "nk-t cells": "nkt_cells",
    "nkt-like cells": "nkt_cells",
    "inkt cells": "inkt_cells",
    "inkt": "inkt_cells",
    # Monocyte variations
    "monocytes": "monocytes",
    "monos": "monocytes",
    "cd14+ monocytes": "monocytes",
    "cd14+ monos": "monocytes",
    "cd14+": "monocytes",
    "cd64+ monocytes": "monocytes",
    # Classical monocyte variations
    "classical monocytes": "classical_monocytes",
    "classical monos": "classical_monocytes",
    "classical": "classical_monocytes",
    # Non-classical monocyte variations
    "non-classical monocytes": "nonclassical_monocytes",
    "nonclassical monocytes": "nonclassical_monocytes",
    "non-classical": "nonclassical_monocytes",
    "nonclassical": "nonclassical_monocytes",
    "slan+ nonclassical monocytes": "nonclassical_monocytes",
    "slan+ non-classical": "nonclassical_monocytes",
    "slan- nonclassical monocytes": "nonclassical_monocytes",
    # Intermediate monocyte variations
    "intermediate monocytes": "intermediate_monocytes",
    "intermediate monos": "intermediate_monocytes",
    "intermediate": "intermediate_monocytes",
    # Lymphocyte variations
    "lymphocytes": "lymphocytes",
    "lymphs": "lymphocytes",
    # Singlet variations
    "singlets": "singlets",
    "singlets (fsc)": "singlets",
    "singlets (ssc)": "singlets",
    "singlets (fsc-a vs fsc-h)": "singlets",
    "singlets (ssc-a vs ssc-h)": "singlets",
    "fsc singlets": "singlets",
    "ssc singlets": "singlets",
    "singlet": "singlets",
    "non-aggregates": "singlets",
    # Live/Dead variations
    "live cells": "live",
    "live": "live",
    "live/dead": "live",
    "viable cells": "live",
    "viable": "live",
    # Leukocyte variations
    "leukocytes": "leukocytes",
    "cd45+ leukocytes": "leukocytes",
    "cd45+": "leukocytes",
    "cd45+ cells": "leukocytes",
    "cd45+ tils": "leukocytes",
    "white blood cells": "leukocytes",
    "wbc": "leukocytes",
    # CD4+ T cell variations
    "cd4+ t cells": "cd4_t_cells",
    "cd4+ t": "cd4_t_cells",
    "cd4 t cells": "cd4_t_cells",
    "cd4 t": "cd4_t_cells",
    "helper t cells": "cd4_t_cells",
    "t helper": "cd4_t_cells",
    "th cells": "cd4_t_cells",
    # CD8+ T cell variations
    "cd8+ t cells": "cd8_t_cells",
    "cd8+ t": "cd8_t_cells",
    "cd8 t cells": "cd8_t_cells",
    "cd8 t": "cd8_t_cells",
    "cytotoxic t cells": "cd8_t_cells",
    "cytotoxic t": "cd8_t_cells",
    "ctl": "cd8_t_cells",
    # Naive CD4 T cell variations
    "naive cd4": "naive_cd4",
    "cd4 naive": "naive_cd4",
    "cd4+ naive": "naive_cd4",
    "naive cd4 t cells": "naive_cd4",
    "naive cd4+": "naive_cd4",
    # Memory CD4 T cell variations
    "memory cd4": "memory_cd4",
    "cd4 memory": "memory_cd4",
    "cd4+ memory": "memory_cd4",
    "memory cd4 t cells": "memory_cd4",
    "cd4+ memory subsets": "memory_cd4",
    # CD4 central memory variations
    "cd4 cm": "cd4_cm",
    "cd4+ cm": "cd4_cm",
    "cd4 central memory": "cd4_cm",
    # CD4 effector memory variations
    "cd4 em": "cd4_em",
    "cd4+ em": "cd4_em",
    "cd4 effector memory": "cd4_em",
    # CD4 TEMRA variations
    "cd4 temra": "cd4_temra",
    "cd4+ temra": "cd4_temra",
    # Naive CD8 T cell variations
    "naive cd8": "naive_cd8",
    "cd8 naive": "naive_cd8",
    "cd8+ naive": "naive_cd8",
    "naive cd8 t cells": "naive_cd8",
    "naive cd8+": "naive_cd8",
    # Memory CD8 T cell variations
    "memory cd8": "memory_cd8",
    "cd8 memory": "memory_cd8",
    "cd8+ memory": "memory_cd8",
    "memory cd8 t cells": "memory_cd8",
    "cd8+ memory subsets": "memory_cd8",
    # CD8 central memory variations
    "cd8 cm": "cd8_cm",
    "cd8+ cm": "cd8_cm",
    "cd8 central memory": "cd8_cm",
    # CD8 effector memory variations
    "cd8 em": "cd8_em",
    "cd8+ em": "cd8_em",
    "cd8 effector memory": "cd8_em",
    # CD8 TEMRA variations
    "cd8 temra": "cd8_temra",
    "cd8+ temra": "cd8_temra",
    # General central/effector memory
    "central memory t cells": "central_memory_t",
    "cm t cells": "central_memory_t",
    "effector memory t cells": "effector_memory_t",
    "em t cells": "effector_memory_t",
    # Dendritic cell variations (general)
    "dendritic cells": "dendritic_cells",
    "dc": "dendritic_cells",
    "dcs": "dendritic_cells",
    # Myeloid dendritic cell variations
    "myeloid dendritic cells": "mdc",
    "mdcs": "mdc",
    "mdc": "mdc",
    "cdc": "mdc",
    "cdcs": "mdc",
    "conventional dendritic cells": "mdc",
    # cDC1 variations
    "cdc1": "cdc1",
    "cdc1s": "cdc1",
    "cd141+ mdcs": "cdc1",
    "cd141+ mdc": "cdc1",
    "cd141+ mdc1": "cdc1",
    # cDC2 variations
    "cdc2": "cdc2",
    "cdc2s": "cdc2",
    "cd1c+ mdcs": "cdc2",
    "cd1c+ mdc": "cdc2",
    "cd1c+ mdc2": "cdc2",
    # Plasmacytoid dendritic cell variations
    "plasmacytoid dendritic cells": "pdc",
    "pdc": "pdc",
    "pdcs": "pdc",
    # Gamma-delta T cell variations
    "gd t cells": "gamma_delta_t",
    "gd t": "gamma_delta_t",
    "gamma delta t cells": "gamma_delta_t",
    "γδ t cells": "gamma_delta_t",
    "gammadelta t": "gamma_delta_t",
    "vδ2+ γδ t cells": "gamma_delta_t",
    "vδ2- γδ t cells": "gamma_delta_t",
    # Regulatory T cell variations
    "tregs": "regulatory_t",
    "treg": "regulatory_t",
    "regulatory t cells": "regulatory_t",
    "regulatory t": "regulatory_t",
    "cd4+cd25+foxp3+": "regulatory_t",
    "tconv": "conventional_t",
    # Naive/memory Treg variations
    "naive tregs": "naive_tregs",
    "memory tregs": "memory_tregs",
    # Tfh variations
    "tfh": "tfh",
    "t follicular helper cells": "tfh",
    "t follicular helper": "tfh",
    "follicular helper t cells": "tfh",
    # Th subset variations
    "th1": "th1",
    "th1 cells": "th1",
    "th2": "th2",
    "th2 cells": "th2",
    "th17": "th17",
    "th17 cells": "th17",
    "th22": "th22",
    "th22 cells": "th22",
    # Naive B cell variations
    "naive b cells": "naive_b",
    "naive b": "naive_b",
    "cd27- naive b cells": "naive_b",
    # Memory B cell variations
    "memory b cells": "memory_b",
    "memory b": "memory_b",
    "switched memory": "switched_memory_b",
    "switched memory b cells": "switched_memory_b",
    "unswitched memory": "unswitched_memory_b",
    "unswitched memory b cells": "unswitched_memory_b",
    # Plasma/Plasmablast variations
    "plasma cells": "plasma_cells",
    "plasmablasts": "plasmablasts",
    "b220+ plasmablasts": "plasmablasts",
    "asc subsets": "plasmablasts",
    # Marginal zone B variations
    "marginal zone": "marginal_zone_b",
    "marginal zone b cells": "marginal_zone_b",
    # Transitional B variations
    "transitional": "transitional_b",
    "transitional b cells": "transitional_b",
    # Neutrophil variations
    "neutrophils": "neutrophils",
    "neuts": "neutrophils",
    # Basophil variations
    "basophils": "basophils",
    "basos": "basophils",
    # Granulocyte variations
    "granulocytes": "granulocytes",
    "grans": "granulocytes",
    "non-granulocytes": "non_granulocytes",
    # Myeloid variations
    "myeloid": "myeloid",
    "myeloid cells": "myeloid",
    # Time gate variations
    "time gate": "time",
    "time": "time",
    # All events variations
    "all events": "all_events",
    "all": "all_events",
    "root": "all_events",
    "ungated": "all_events",
}


def normalize_gate_name(name: str) -> str:
    """
    Normalize gate name for comparison.

    Handles common variations in gate naming conventions:
    - Parenthetical qualifiers: "Singlets (FSC)" -> "singlets"
    - Positive/negative notation: "CD4 positive" -> "cd4+"
    - Common abbreviations: "lymphocytes" -> "lymphs"

    Args:
        name: Raw gate name

    Returns:
        Normalized gate name for comparison
    """
    normalized = name.lower().strip()

    # Remove parenthetical qualifiers like "(FSC)", "(SSC-A vs SSC-H)", etc.
    normalized = re.sub(r'\s*\([^)]*\)\s*', ' ', normalized)

    # Common replacements
    replacements = [
        (" positive", "+"),
        ("positive", "+"),
        (" negative", "-"),
        ("negative", "-"),
        (" cells", ""),
        (" cell", ""),
        ("lymphocytes", "lymphs"),
        ("monocytes", "monos"),
        ("neutrophils", "neuts"),
    ]

    for old, new in replacements:
        normalized = normalized.replace(old, new)

    # Normalize whitespace
    normalized = " ".join(normalized.split())

    return normalized


def normalize_gate_semantic(name: str) -> str:
    """
    Normalize gate name with semantic synonym matching.

    More aggressive normalization that maps cell type variations
    to canonical forms. Use for parent matching where "CD3+ T cells"
    should match "T cells".

    Args:
        name: Raw gate name

    Returns:
        Canonical form if synonym found, otherwise basic normalized form
    """
    normalized = normalize_gate_name(name)

    # Check for exact match in synonyms
    if normalized in CELL_TYPE_SYNONYMS:
        return CELL_TYPE_SYNONYMS[normalized]

    # Check if normalized name contains a known synonym pattern
    for synonym, canonical in CELL_TYPE_SYNONYMS.items():
        if len(synonym) <= 2:
            # Short synonyms need word boundaries
            pattern = r'(^|[^a-z])' + re.escape(synonym) + r'($|[^a-z])'
            if re.search(pattern, normalized):
                return canonical
        elif synonym in normalized:
            return canonical

    return normalized


def are_gates_equivalent(name1: str, name2: str, semantic: bool = True) -> bool:
    """
    Check if two gate names are equivalent.

    Args:
        name1: First gate name
        name2: Second gate name
        semantic: If True, use semantic matching (more permissive)

    Returns:
        True if gates are considered equivalent
    """
    if semantic:
        return normalize_gate_semantic(name1) == normalize_gate_semantic(name2)
    return normalize_gate_name(name1) == normalize_gate_name(name2)
