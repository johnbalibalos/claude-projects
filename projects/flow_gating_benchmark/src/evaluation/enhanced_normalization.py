"""
Enhanced gate name normalization with biological context awareness.

Improvements over normalization.py:
1. Expanded cell type synonyms (~200 vs ~100)
2. Marker alias integration (CCR7=CD197, etc.)
3. Polarity-aware marker extraction
4. Hierarchical matching support
"""

from __future__ import annotations

import re
from typing import Set

# =============================================================================
# MARKER ALIASES (CD numbers and common aliases)
# =============================================================================

MARKER_ALIAS_GROUPS = [
    # B cell markers
    ["B220", "CD45R"],
    # T cell activation/checkpoint
    ["ICOS", "CD278"],
    ["PD-1", "PD1", "CD279", "PDCD1"],
    ["PD-L1", "PDL1", "CD274"],
    ["PD-L2", "PDL2", "CD273"],
    ["CTLA-4", "CTLA4", "CD152"],
    ["LAG-3", "LAG3", "CD223"],
    ["TIM-3", "TIM3", "CD366", "HAVCR2"],
    # Chemokine receptors
    ["CXCR5", "CD185"],
    ["CCR7", "CD197"],
    ["CXCR3", "CD183"],
    ["CCR5", "CD195"],
    ["CXCR4", "CD184"],
    ["CCR4", "CD194"],
    ["CCR6", "CD196"],
    ["CX3CR1", "CD369"],
    # Integrins/adhesion
    ["CD103", "ITGAE", "INTEGRIN AE"],
    ["CD49D", "ITGA4", "VLA-4"],
    ["CD29", "ITGB1"],
    ["L-SELECTIN", "CD62L"],
    # Plasma cell markers
    ["CD138", "SYND-1", "SYNDECAN-1", "SDC1"],
    # MHC markers
    ["MHC CLASS II", "HLA-DR", "I-A/I-E", "MHC-II", "IA-IE", "MHCII"],
    # IL receptors
    ["IL-7R", "IL7R", "CD127"],
    ["IL-2R", "IL2R", "CD25"],
    ["IL-15R", "IL15R", "CD122"],
    # Other
    ["KLRG1", "KLRG-1"],
    ["NKG2A", "CD159A"],
    ["NKG2D", "CD314"],
    ["NKP46", "NCR1", "CD335"],
    ["NKP44", "NCR2", "CD336"],
    ["NKP30", "NCR3", "CD337"],
]

# Build bidirectional alias lookup
MARKER_TO_CANONICAL: dict[str, str] = {}
for group in MARKER_ALIAS_GROUPS:
    canonical = group[0].upper()
    for alias in group:
        MARKER_TO_CANONICAL[alias.upper()] = canonical
        # Also handle with/without hyphens
        MARKER_TO_CANONICAL[alias.upper().replace("-", "")] = canonical
        MARKER_TO_CANONICAL[alias.upper().replace("-", " ")] = canonical


# =============================================================================
# EXPANDED CELL TYPE SYNONYMS
# =============================================================================

CELL_TYPE_SYNONYMS: dict[str, str] = {
    # ==========================================================================
    # QC / CLEANUP GATES
    # ==========================================================================
    "singlets": "singlets",
    "single cells": "singlets",
    "singlet": "singlets",
    "fsc singlets": "singlets",
    "ssc singlets": "singlets",
    "fsc-a vs fsc-h": "singlets",
    "fsc-h vs fsc-a": "singlets",
    "ssc-a vs ssc-h": "singlets",
    "non-aggregates": "singlets",
    "non aggregates": "singlets",
    "nonaggregates": "singlets",
    "doublet exclusion": "singlets",
    "aggregate 1": "singlets",
    "aggregate 2": "singlets",

    # Scatter gate
    "scatter gate": "scatter_gate",
    "fsc/ssc": "scatter_gate",
    "fsc vs ssc": "scatter_gate",
    "fsc-a vs ssc-a": "scatter_gate",
    "debris exclusion": "scatter_gate",
    "cells": "scatter_gate",

    # Live/Dead
    "live cells": "live",
    "live": "live",
    "viable cells": "live",
    "viable": "live",
    "live/dead": "live",
    "viability": "live",

    # Time gate
    "time gate": "time",
    "time": "time",

    # Nucleated cells
    "dna+": "nucleated",
    "nucleated cells": "nucleated",
    "nucleated": "nucleated",

    # All events
    "all events": "all_events",
    "all": "all_events",
    "root": "all_events",
    "ungated": "all_events",
    "total events": "all_events",

    # ==========================================================================
    # LYMPHOCYTES / LEUKOCYTES
    # ==========================================================================
    "lymphocytes": "lymphocytes",
    "lymphs": "lymphocytes",
    "lymphocyte": "lymphocytes",

    "leukocytes": "leukocytes",
    "cd45+ leukocytes": "leukocytes",
    "cd45+": "leukocytes",
    "cd45+ cells": "leukocytes",
    "white blood cells": "leukocytes",
    "wbc": "leukocytes",

    "non-granulocytes": "non_granulocytes",
    "nongranulocytes": "non_granulocytes",

    # ==========================================================================
    # T CELLS
    # ==========================================================================
    "t cells": "t_cells",
    "t-cells": "t_cells",
    "t lymphocytes": "t_cells",
    "cd3+ t cells": "t_cells",
    "cd3+ t": "t_cells",
    "cd3+": "cd3_t_cells",

    # CD4+ T cells
    "cd4+ t cells": "cd4_t_cells",
    "cd4 t cells": "cd4_t_cells",
    "cd4+ t": "cd4_t_cells",
    "helper t cells": "cd4_t_cells",
    "th cells": "cd4_t_cells",
    "t helper cells": "cd4_t_cells",

    # Conventional CD4+ T cells
    "conventional cd4+ t cells": "tconv",
    "conventional cd4 t cells": "tconv",
    "cd4+ tconv": "tconv",
    "tconv": "tconv",
    "conventional t cells": "tconv",

    # CD8+ T cells
    "cd8+ t cells": "cd8_t_cells",
    "cd8 t cells": "cd8_t_cells",
    "cd8+ t": "cd8_t_cells",
    "cytotoxic t cells": "cd8_t_cells",
    "cytotoxic t": "cd8_t_cells",
    "ctl": "cd8_t_cells",
    "ctls": "cd8_t_cells",

    # Memory subsets
    "naive": "naive",
    "naive t cells": "naive_t",
    "naive t": "naive_t",
    "tn": "naive_t",

    "central memory": "cm",
    "cm": "cm",
    "tcm": "cm",
    "central memory t cells": "cm",

    "effector memory": "em",
    "em": "em",
    "tem": "em",
    "effector memory t cells": "em",

    "temra": "temra",
    "emra": "temra",
    "effector memory ra": "temra",
    "cd45ra+ effector memory": "temra",

    # Tregs
    "tregs": "tregs",
    "treg": "tregs",
    "regulatory t cells": "tregs",
    "regulatory t": "tregs",
    "cd4+cd25+foxp3+": "tregs",
    "foxp3+ tregs": "tregs",

    # Tfh
    "tfh": "tfh",
    "tfh cells": "tfh",
    "t follicular helper cells": "tfh",
    "t follicular helper": "tfh",
    "follicular helper t cells": "tfh",
    "cxcr5+ cd4+ t cells": "tfh",

    # Gamma-delta T cells
    "gamma delta t cells": "gd_t_cells",
    "gd t cells": "gd_t_cells",
    "γδ t cells": "gd_t_cells",
    "gammadelta t cells": "gd_t_cells",
    "gd t": "gd_t_cells",

    # Double negative T cells
    "double negative t cells": "dn_t_cells",
    "dn t cells": "dn_t_cells",
    "cd4- cd8- t cells": "dn_t_cells",
    "cd4-cd8- t cells": "dn_t_cells",

    # Th subsets
    "th1": "th1",
    "th1 cells": "th1",
    "th2": "th2",
    "th2 cells": "th2",
    "th17": "th17",
    "th17 cells": "th17",
    "th22": "th22",
    "th22 cells": "th22",

    # ==========================================================================
    # B CELLS
    # ==========================================================================
    "b cells": "b_cells",
    "b-cells": "b_cells",
    "b lymphocytes": "b_cells",
    "cd19+ b cells": "b_cells",
    "cd19+ b": "b_cells",
    "cd19+": "cd19_b_cells",
    "cd20+ b cells": "b_cells",
    "cd20+": "cd20_b_cells",

    # Naive B cells
    "naive b cells": "naive_b",
    "naive b": "naive_b",
    "cd27- b cells": "naive_b",

    # Memory B cells
    "memory b cells": "memory_b",
    "memory b": "memory_b",
    "cd27+ b cells": "memory_b",
    "switched memory b cells": "switched_memory_b",
    "class-switched memory b cells": "switched_memory_b",
    "class switched memory b cells": "switched_memory_b",
    "unswitched memory b cells": "unswitched_memory_b",

    # Plasma cells
    "plasma cells": "plasma_cells",
    "plasmablasts": "plasma_cells",
    "asc": "plasma_cells",
    "antibody secreting cells": "plasma_cells",
    "cd138+ b cells": "plasma_cells",

    # Transitional B cells
    "transitional b cells": "transitional_b",
    "transitional b": "transitional_b",

    # B-2 cells
    "b-2 cells": "b2_cells",
    "b2 cells": "b2_cells",
    "conventional b cells": "b2_cells",

    # Marginal zone
    "marginal zone b cells": "mz_b_cells",
    "mz b cells": "mz_b_cells",

    # ==========================================================================
    # NK CELLS
    # ==========================================================================
    "nk cells": "nk_cells",
    "nk": "nk_cells",
    "natural killer cells": "nk_cells",
    "natural killer": "nk_cells",
    "cd56+ nk cells": "nk_cells",
    "cd56+ nk": "nk_cells",
    "cd56+cd3-": "nk_cells",
    "cd3-cd56+": "nk_cells",
    "cd3- cd56+": "nk_cells",

    # NK subsets
    "cd56bright nk cells": "cd56bright_nk",
    "cd56bright nk": "cd56bright_nk",
    "cd56bright": "cd56bright_nk",
    "cd56hi nk cells": "cd56bright_nk",
    "cd56hi": "cd56bright_nk",

    "cd56dim nk cells": "cd56dim_nk",
    "cd56dim nk": "cd56dim_nk",
    "cd56dim": "cd56dim_nk",
    "cd56lo nk cells": "cd56dim_nk",
    "cd56lo": "cd56dim_nk",

    # NKT cells
    "nkt cells": "nkt_cells",
    "nkt": "nkt_cells",
    "nk-t cells": "nkt_cells",
    "cd3+cd56+": "nkt_cells",

    # ==========================================================================
    # MONOCYTES
    # ==========================================================================
    "monocytes": "monocytes",
    "monos": "monocytes",
    "monocyte": "monocytes",
    "cd14+ monocytes": "monocytes",
    "cd14+ monos": "monocytes",
    "cd14+": "cd14_monocytes",

    # Classical monocytes
    "classical monocytes": "classical_monocytes",
    "classical monos": "classical_monocytes",
    "cd14++cd16-": "classical_monocytes",
    "cd14++cd16- monocytes": "classical_monocytes",
    "cd14+cd16-": "classical_monocytes",

    # Intermediate monocytes
    "intermediate monocytes": "intermediate_monocytes",
    "intermediate monos": "intermediate_monocytes",
    "cd14++cd16+": "intermediate_monocytes",
    "cd14+cd16+": "intermediate_monocytes",

    # Non-classical monocytes
    "non-classical monocytes": "nonclassical_monocytes",
    "nonclassical monocytes": "nonclassical_monocytes",
    "non classical monocytes": "nonclassical_monocytes",
    "cd14+cd16++": "nonclassical_monocytes",
    "cd14dimcd16+": "nonclassical_monocytes",
    "cd16+ monocytes": "nonclassical_monocytes",

    # ==========================================================================
    # GRANULOCYTES
    # ==========================================================================
    "granulocytes": "granulocytes",
    "grans": "granulocytes",
    "pmn": "granulocytes",
    "polymorphonuclear cells": "granulocytes",

    "neutrophils": "neutrophils",
    "neuts": "neutrophils",
    "pmns": "neutrophils",
    "cd66b+ neutrophils": "neutrophils",

    "eosinophils": "eosinophils",
    "eos": "eosinophils",
    "siglec-8+ eosinophils": "eosinophils",

    "basophils": "basophils",
    "basos": "basophils",
    "cd123+ basophils": "basophils",

    # ==========================================================================
    # DENDRITIC CELLS
    # ==========================================================================
    "dendritic cells": "dcs",
    "dcs": "dcs",
    "dc": "dcs",

    # Myeloid / Conventional DCs
    "myeloid dcs": "mdcs",
    "mdcs": "mdcs",
    "mdc": "mdcs",
    "conventional dcs": "cdcs",
    "cdcs": "cdcs",
    "cdc": "cdcs",

    # cDC1
    "cdc1": "cdc1",
    "cdc1s": "cdc1",
    "cd141+ dc": "cdc1",
    "cd141+ dcs": "cdc1",
    "clec9a+ dc": "cdc1",

    # cDC2
    "cdc2": "cdc2",
    "cdc2s": "cdc2",
    "cd1c+ dc": "cdc2",
    "cd1c+ dcs": "cdc2",

    # Plasmacytoid DCs
    "plasmacytoid dcs": "pdcs",
    "plasmacytoid dendritic cells": "pdcs",
    "pdcs": "pdcs",
    "pdc": "pdcs",
    "cd123+ dcs": "pdcs",
    "cd303+ dcs": "pdcs",

    # ==========================================================================
    # OTHER
    # ==========================================================================
    "other": "other",
    "other cells": "other",

    # ILCs
    "ilcs": "ilcs",
    "innate lymphoid cells": "ilcs",
    "ilc1": "ilc1",
    "ilc2": "ilc2",
    "ilc3": "ilc3",

    # Beads
    "non-beads": "non_beads",
    "nonbeads": "non_beads",
    "beads excluded": "non_beads",
}


def normalize_marker_in_name(name: str) -> str:
    """Replace marker aliases with canonical forms in a gate name."""
    result = name.upper()

    # Sort by length descending to match longer patterns first
    for alias, canonical in sorted(MARKER_TO_CANONICAL.items(), key=lambda x: -len(x[0])):
        # Use word boundary matching
        pattern = r'\b' + re.escape(alias) + r'\b'
        result = re.sub(pattern, canonical, result, flags=re.IGNORECASE)

    return result.lower()


def normalize_gate_name_enhanced(name: str) -> str:
    """
    Comprehensive gate name normalization with marker alias support.

    Improvements over basic normalization:
    - Handles marker aliases (CCR7 = CD197)
    - More extensive synonym mapping
    - Normalizes +/- notation
    """
    if not name:
        return ""

    n = name.lower().strip()

    # Normalize markers first (handles aliases)
    n = normalize_marker_in_name(n)

    # Normalize +/- notation
    n = re.sub(r'\s*\+\s*', '+', n)
    n = re.sub(r'\s*-\s*', '-', n)
    n = re.sub(r'\s+positive\b', '+', n)
    n = re.sub(r'\s+negative\b', '-', n)
    n = re.sub(r'\bpositive\b', '+', n)
    n = re.sub(r'\bnegative\b', '-', n)

    # Remove parenthetical qualifiers like (FSC), (SSC-A vs SSC-H)
    n = re.sub(r'\s*\([^)]*\)\s*', ' ', n)

    # Normalize whitespace
    n = ' '.join(n.split())

    # Check cell type synonyms
    if n in CELL_TYPE_SYNONYMS:
        return CELL_TYPE_SYNONYMS[n]

    # Check partial matches
    for synonym, canonical in CELL_TYPE_SYNONYMS.items():
        if synonym in n:
            return canonical

    return n


def extract_markers_with_polarity(gate_name: str) -> dict[str, str]:
    """
    Extract marker names from a gate name with their polarity.

    Returns dict mapping marker -> polarity ('+', '-', or '?')
    """
    markers = {}
    name = gate_name.upper()

    # CD markers: CD followed by digits, optionally with letter suffix, then +/-
    cd_pattern = r'\b(CD\d+[A-Z]?)([+-])?\b'
    for match in re.finditer(cd_pattern, name):
        marker = match.group(1).lower()
        polarity = match.group(2) or "?"
        markers[marker] = polarity

    # Chemokine receptors
    chemokine_pattern = r'\b(CCR\d+|CXCR\d+|CX3CR\d+)([+-])?\b'
    for match in re.finditer(chemokine_pattern, name, re.IGNORECASE):
        marker = match.group(1).lower()
        polarity = match.group(2) or "?"
        markers[marker] = polarity

    return markers


def get_positive_markers(gate_name: str) -> Set[str]:
    """Get markers that are explicitly positive in a gate name."""
    markers = extract_markers_with_polarity(gate_name)
    return {m for m, p in markers.items() if p == "+"}


def are_gates_equivalent_enhanced(name1: str, name2: str) -> bool:
    """
    Check if two gate names are equivalent using enhanced normalization.
    """
    return normalize_gate_name_enhanced(name1) == normalize_gate_name_enhanced(name2)
