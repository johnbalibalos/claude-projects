#!/usr/bin/env python3
"""
Improved structure error analysis with biological context awareness.

Key improvements over analyze_structure_errors.py:
1. Marker alias integration (CCR7 = CD197, etc.)
2. Sample type context (CD45 optional for PBMCs)
3. Hard vs soft constraint separation
4. Semantic population matching
5. Valid gating order alternatives
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =============================================================================
# MARKER ALIASES (from marker_aliases.py)
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
# CELL TYPE SYNONYMS
# =============================================================================

CELL_TYPE_SYNONYMS: dict[str, str] = {
    # ==========================================================================
    # QC / CLEANUP GATES
    # ==========================================================================
    # Singlets / Doublet exclusion
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

    # Scatter gate / Debris exclusion
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

    # All events / Root
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
    # General T cells
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

    # CCR7/CD45RA subsets
    "ccr7/cd45ra subsets": "memory_subsets",
    "cd45ra/ccr7 subsets": "memory_subsets",
    "memory subsets": "memory_subsets",

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
    "unswitched memory b cells": "unswitched_memory_b",

    # Plasma cells / Plasmablasts
    "plasma cells": "plasma_cells",
    "plasmablasts": "plasma_cells",
    "asc": "plasma_cells",
    "antibody secreting cells": "plasma_cells",
    "cd138+ b cells": "plasma_cells",

    # Transitional B cells
    "transitional b cells": "transitional_b",
    "transitional b": "transitional_b",

    # B-2 cells (conventional B cells)
    "b-2 cells": "b2_cells",
    "b2 cells": "b2_cells",
    "conventional b cells": "b2_cells",

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

    # Classical monocytes (handle case variations)
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
    # OTHER POPULATIONS
    # ==========================================================================
    "other": "other",
    "other cells": "other",

    # ILCs
    "ilcs": "ilcs",
    "innate lymphoid cells": "ilcs",
    "ilc1": "ilc1",
    "ilc2": "ilc2",
    "ilc3": "ilc3",

    # ==========================================================================
    # ADDITIONAL QC GATES
    # ==========================================================================
    # Aggregate gates (part of singlet gating)
    "aggregate 1": "singlets",
    "aggregate 2": "singlets",
    "aggregates": "singlets",

    # Bead exclusion
    "non-beads": "non_beads",
    "nonbeads": "non_beads",
    "beads excluded": "non_beads",

    # ==========================================================================
    # ADDITIONAL B CELL SUBSETS
    # ==========================================================================
    "class-switched memory b cells": "switched_memory_b",
    "class switched memory b cells": "switched_memory_b",
    "switched memory b": "switched_memory_b",
    "igd- memory b cells": "switched_memory_b",
    "igd+ memory b cells": "unswitched_memory_b",

    "ige+ b cells": "ige_b_cells",
    "ige+ b": "ige_b_cells",

    "igg+ b cells": "igg_b_cells",
    "iga+ b cells": "iga_b_cells",
    "igm+ b cells": "igm_b_cells",

    "marginal zone b cells": "mz_b_cells",
    "mz b cells": "mz_b_cells",

    # ==========================================================================
    # ADDITIONAL T CELL SUBSETS
    # ==========================================================================
    "cd4+ cd8- t cells": "cd4_t_cells",
    "cd4+cd8- t cells": "cd4_t_cells",
    "cd8+ cd4- t cells": "cd8_t_cells",
    "cd8+cd4- t cells": "cd8_t_cells",

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
    # CYTOKINE GATES (functional analysis)
    # ==========================================================================
    "ifn-γ+": "ifng_positive",
    "ifng+": "ifng_positive",
    "ifn-gamma+": "ifng_positive",
    "tnf+": "tnf_positive",
    "tnf-α+": "tnf_positive",
    "il-2+": "il2_positive",
    "il2+": "il2_positive",
    "il-17+": "il17_positive",
    "il-17a+": "il17_positive",
    "il17+": "il17_positive",
}


# =============================================================================
# HARD BIOLOGICAL CONSTRAINTS
# =============================================================================

# Mutually exclusive lineage markers (should NEVER co-occur in single cell)
LINEAGE_EXCLUSIVITY_RULES = [
    ({"cd3", "cd3+"}, {"cd19", "cd19+", "cd20", "cd20+"}),  # T vs B
    ({"cd3", "cd3+"}, {"cd14", "cd14+"}),  # T vs Monocyte
    ({"cd19", "cd19+", "cd20", "cd20+"}, {"cd14", "cd14+"}),  # B vs Monocyte
]


# =============================================================================
# SOFT CONSTRAINTS (context-dependent)
# =============================================================================

# Sample types where CD45 gating is optional
CD45_OPTIONAL_SAMPLES = [
    "pbmc", "pbmcs", "human pbmc",
    "peripheral blood mononuclear",
    "whole blood", "blood",
    "cryopreserved pbmc",
]

# Valid gating order alternatives (these are NOT errors)
VALID_GATING_ORDERS = [
    # Both singlets→live and live→singlets are acceptable
    ("singlets", "live"),
    ("live", "singlets"),
    # Both orders for lymphocyte gating
    ("live", "lymphocytes"),
    ("lymphocytes", "live"),  # Less common but valid
]


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_marker_in_name(name: str) -> str:
    """Replace marker aliases with canonical forms in a gate name."""
    result = name.upper()

    # Sort by length descending to match longer patterns first
    for alias, canonical in sorted(MARKER_TO_CANONICAL.items(), key=lambda x: -len(x[0])):
        # Use word boundary matching
        pattern = r'\b' + re.escape(alias) + r'\b'
        result = re.sub(pattern, canonical, result, flags=re.IGNORECASE)

    return result.lower()


def normalize_gate_name(name: str) -> str:
    """Comprehensive gate name normalization."""
    if not name:
        return ""

    n = name.lower().strip()

    # Normalize markers first
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


def extract_markers_from_gate(gate_name: str) -> dict[str, str]:
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


def get_positive_markers(gate: dict) -> set[str]:
    """Get markers that are explicitly positive in a gate."""
    markers = gate.get("markers", {})
    if isinstance(markers, set):
        return markers  # Old format compatibility
    return {m for m, p in markers.items() if p == "+"}


# =============================================================================
# HIERARCHY UTILITIES
# =============================================================================

def extract_all_gates(hierarchy: dict) -> list[dict]:
    """Extract all gates with their context."""
    gates = []

    def traverse(node: dict, parent: str | None = None, depth: int = 0, path: list[str] = None):
        if path is None:
            path = []

        if "name" in node:
            current_path = path + [node["name"]]
            gates.append({
                "name": node["name"],
                "normalized": normalize_gate_name(node["name"]),
                "parent": parent,
                "depth": depth,
                "path": current_path,
                "markers": extract_markers_from_gate(node["name"]),
            })
            for child in node.get("children", []):
                traverse(child, node["name"], depth + 1, current_path)

    if "root" in hierarchy:
        traverse(hierarchy["root"])
    elif "name" in hierarchy:
        traverse(hierarchy)

    return gates


def get_terminal_populations(hierarchy: dict) -> list[dict]:
    """Get leaf nodes (populations with no children)."""
    all_gates = extract_all_gates(hierarchy)

    # Build parent set
    parents = set()
    for gate in all_gates:
        if gate["parent"]:
            parents.add(normalize_gate_name(gate["parent"]))

    # Return gates that are not parents
    return [g for g in all_gates if g["normalized"] not in parents]


def is_valid_ancestor(ancestor_norm: str, descendant_norm: str) -> bool:
    """
    Check if ancestor could be a valid parent for descendant based on biological lineage.

    Valid simplifications (skipping intermediate gates):
    - Lymphocytes → CD4+ T cells (skipping CD3+ T cells)
    - T cells → Tregs (skipping CD4+ T cells)
    - B cells → Memory B cells (skipping intermediate markers)
    """
    # Define valid direct parent relationships (can skip intermediate)
    VALID_LINEAGE_SHORTCUTS = {
        # T cell lineage
        "lymphocytes": {"t_cells", "cd3_t_cells", "cd4_t_cells", "cd8_t_cells", "tregs",
                        "tfh", "naive_t", "cm", "em", "temra", "gd_t_cells", "dn_t_cells",
                        "th1", "th2", "th17", "tconv"},
        "t_cells": {"cd4_t_cells", "cd8_t_cells", "tregs", "tfh", "naive_t", "cm", "em",
                    "temra", "gd_t_cells", "dn_t_cells", "th1", "th2", "th17", "tconv"},
        "cd3_t_cells": {"cd4_t_cells", "cd8_t_cells", "tregs", "tfh", "naive_t", "cm",
                        "em", "temra", "gd_t_cells", "dn_t_cells"},
        "cd4_t_cells": {"tregs", "tfh", "naive_t", "cm", "em", "temra", "th1", "th2",
                        "th17", "tconv"},
        "cd8_t_cells": {"naive_t", "cm", "em", "temra"},

        # B cell lineage
        "lymphocytes": {"b_cells", "cd19_b_cells", "cd20_b_cells", "naive_b", "memory_b",
                        "switched_memory_b", "unswitched_memory_b", "plasma_cells",
                        "transitional_b", "b2_cells"},
        "b_cells": {"naive_b", "memory_b", "switched_memory_b", "unswitched_memory_b",
                    "plasma_cells", "transitional_b", "b2_cells"},
        "cd19_b_cells": {"naive_b", "memory_b", "switched_memory_b", "plasma_cells"},
        "memory_b": {"switched_memory_b", "unswitched_memory_b"},

        # NK cell lineage
        "lymphocytes": {"nk_cells", "cd56bright_nk", "cd56dim_nk", "nkt_cells"},
        "nk_cells": {"cd56bright_nk", "cd56dim_nk"},

        # Monocyte lineage
        "monocytes": {"classical_monocytes", "intermediate_monocytes", "nonclassical_monocytes"},
        "cd14_monocytes": {"classical_monocytes", "intermediate_monocytes", "nonclassical_monocytes"},

        # DC lineage
        "dcs": {"mdcs", "cdcs", "cdc1", "cdc2", "pdcs"},
        "cdcs": {"cdc1", "cdc2"},
        "mdcs": {"cdc1", "cdc2"},

        # Granulocyte lineage
        "granulocytes": {"neutrophils", "eosinophils", "basophils"},

        # General leukocyte → anything
        "leukocytes": {"lymphocytes", "monocytes", "granulocytes", "dcs", "nk_cells",
                       "t_cells", "b_cells", "neutrophils", "eosinophils", "basophils"},

        # QC gates can parent anything
        "singlets": {"live", "lymphocytes", "leukocytes", "scatter_gate"},
        "live": {"lymphocytes", "leukocytes", "singlets", "scatter_gate"},
        "scatter_gate": {"live", "singlets", "lymphocytes", "leukocytes"},
    }

    # Check if this is a valid shortcut
    if ancestor_norm in VALID_LINEAGE_SHORTCUTS:
        return descendant_norm in VALID_LINEAGE_SHORTCUTS[ancestor_norm]

    return False


def get_ancestor_chain(gate: dict) -> list[str]:
    """Get the normalized names of all ancestors from path."""
    if not gate.get("path"):
        return []
    return [normalize_gate_name(p) for p in gate["path"][:-1]]  # Exclude self


# =============================================================================
# GATE ORDER VALIDATION
# =============================================================================

# Standard gating order rules
# Format: (earlier_gate, later_gate) - earlier should come before later in hierarchy
REQUIRED_GATE_ORDER = [
    # QC gates should come first
    ("singlets", "live"),  # Can be either order, but both before lymphocytes
    ("live", "singlets"),  # Acceptable alternative
    ("singlets", "lymphocytes"),
    ("live", "lymphocytes"),
    ("scatter_gate", "lymphocytes"),

    # Leukocyte gating before lineage
    ("leukocytes", "t_cells"),
    ("leukocytes", "b_cells"),
    ("leukocytes", "nk_cells"),
    ("leukocytes", "monocytes"),

    # Lymphocyte gating before specific lineages
    ("lymphocytes", "t_cells"),
    ("lymphocytes", "b_cells"),
    ("lymphocytes", "nk_cells"),

    # T cell hierarchy
    ("t_cells", "cd4_t_cells"),
    ("t_cells", "cd8_t_cells"),
    ("cd3_t_cells", "cd4_t_cells"),
    ("cd3_t_cells", "cd8_t_cells"),
    ("cd4_t_cells", "tregs"),
    ("cd4_t_cells", "tfh"),
    ("cd4_t_cells", "th1"),
    ("cd4_t_cells", "th2"),
    ("cd4_t_cells", "th17"),

    # Memory subsets after lineage
    ("cd4_t_cells", "naive_t"),
    ("cd4_t_cells", "cm"),
    ("cd4_t_cells", "em"),
    ("cd4_t_cells", "temra"),
    ("cd8_t_cells", "naive_t"),
    ("cd8_t_cells", "cm"),
    ("cd8_t_cells", "em"),
    ("cd8_t_cells", "temra"),

    # B cell hierarchy
    ("b_cells", "naive_b"),
    ("b_cells", "memory_b"),
    ("b_cells", "plasma_cells"),

    # Monocyte subsets after monocyte gate
    ("monocytes", "classical_monocytes"),
    ("monocytes", "intermediate_monocytes"),
    ("monocytes", "nonclassical_monocytes"),

    # NK subsets after NK gate
    ("nk_cells", "cd56bright_nk"),
    ("nk_cells", "cd56dim_nk"),
]

# Gates that should NOT come before certain gates (order violations)
# Format: (gate_that_should_not_be_earlier, gate_that_should_not_be_later)
INVALID_GATE_ORDERS = [
    # Lineage should not come before QC
    ("t_cells", "singlets"),
    ("b_cells", "singlets"),
    ("nk_cells", "singlets"),
    ("t_cells", "live"),
    ("b_cells", "live"),
    ("nk_cells", "live"),

    # Specific subsets should not come before their parent lineage
    ("cd4_t_cells", "t_cells"),
    ("cd8_t_cells", "t_cells"),
    ("tregs", "cd4_t_cells"),
    ("tfh", "cd4_t_cells"),

    # Memory should not come before lineage
    ("cm", "t_cells"),
    ("em", "t_cells"),
    ("naive_t", "t_cells"),
    ("temra", "t_cells"),
]


def validate_gate_order(pred_gates: list[dict]) -> list[str]:
    """
    Validate that gates are in a biologically sensible order.

    INVALID_GATE_ORDERS contains (X, Y) pairs meaning "X should NOT come before Y".
    A violation occurs when X is at lower depth (closer to root) than Y.

    Returns list of order violations.
    """
    violations = []

    # Build order map: gate_name -> depth (lower depth = closer to root)
    gate_depths = {}
    for gate in pred_gates:
        norm = gate["normalized"]
        depth = gate["depth"]
        # Keep the minimum depth if gate appears multiple times
        if norm not in gate_depths or depth < gate_depths[norm]:
            gate_depths[norm] = depth

    # Check for invalid orders
    for should_not_be_earlier, should_not_be_later in INVALID_GATE_ORDERS:
        if should_not_be_earlier in gate_depths and should_not_be_later in gate_depths:
            # Violation occurs when:
            # - should_not_be_earlier has lower depth (comes first) than should_not_be_later
            # Lower depth = closer to root = comes earlier in hierarchy
            if gate_depths[should_not_be_earlier] < gate_depths[should_not_be_later]:
                violations.append(
                    f"ORDER: '{should_not_be_earlier}' (depth {gate_depths[should_not_be_earlier]}) "
                    f"incorrectly appears before '{should_not_be_later}' (depth {gate_depths[should_not_be_later]})"
                )

    # Check ancestry relationships - more precise check
    for gate in pred_gates:
        norm = gate["normalized"]
        ancestors = get_ancestor_chain(gate)

        # Check if any ancestor should actually be a descendant
        for ancestor_norm in ancestors:
            for should_not_be_earlier, should_not_be_later in INVALID_GATE_ORDERS:
                # If gate is should_not_be_earlier and ancestor is should_not_be_later,
                # that means should_not_be_later is parent of should_not_be_earlier,
                # which means should_not_be_earlier comes AFTER should_not_be_later (correct, not violation)
                #
                # The violation is the opposite: if should_not_be_earlier is an ANCESTOR
                # of should_not_be_later
                if ancestor_norm == should_not_be_earlier and norm == should_not_be_later:
                    violations.append(
                        f"ORDER: '{should_not_be_earlier}' is ancestor of '{should_not_be_later}' (invalid)"
                    )

    return violations


# =============================================================================
# BIOLOGICAL VALIDATION
# =============================================================================

def check_lineage_exclusivity(gate: dict) -> list[str]:
    """Check if a gate violates lineage exclusivity rules (positive markers only)."""
    violations = []
    positive_markers = get_positive_markers(gate)
    name = gate["name"]

    for group_a, group_b in LINEAGE_EXCLUSIVITY_RULES:
        # Only check POSITIVE markers - CD3- CD19+ is fine
        has_a = bool(positive_markers & group_a)
        has_b = bool(positive_markers & group_b)

        if has_a and has_b:
            violations.append(
                f"HARD: '{name}' has mutually exclusive POSITIVE markers "
                f"({positive_markers & group_a} + {positive_markers & group_b})"
            )

    return violations


def check_cd4_cd8_double_positive(gate: dict, context: dict) -> list[str]:
    """Check for CD4+CD8+ which is rare in periphery."""
    warnings = []
    markers = gate.get("markers", {})

    # Check if BOTH are explicitly positive
    has_cd4_positive = markers.get("cd4") == "+"
    has_cd8_positive = markers.get("cd8") == "+"

    if has_cd4_positive and has_cd8_positive:
        sample = context.get("sample_type", "").lower()
        if "thymus" not in sample and "thymocyte" not in sample:
            warnings.append(
                f"WARNING: '{gate['name']}' is CD4+CD8+ (rare in periphery, "
                f"<3% normal, check if intentional)"
            )

    return warnings


# =============================================================================
# IMPROVED MATCHING
# =============================================================================

@dataclass
class MatchResult:
    """Result of comparing predicted vs ground truth."""
    # Exact/semantic matches
    matched_gates: list[tuple[str, str]] = field(default_factory=list)  # (pred, gt)

    # Hierarchically matched (terminal population correct, intermediate skipped)
    hierarchically_matched: list[tuple[str, str, str]] = field(default_factory=list)  # (pred, gt, reason)

    # Skipped intermediate gates (valid simplification, not an error)
    skipped_intermediates: list[tuple[str, str]] = field(default_factory=list)  # (gate, reason)

    # True missing (not in prediction, no equivalent found)
    truly_missing: list[str] = field(default_factory=list)

    # Missing but acceptable (context-dependent)
    acceptable_missing: list[tuple[str, str]] = field(default_factory=list)  # (gate, reason)

    # Extra predictions (not necessarily errors)
    extra_predictions: list[str] = field(default_factory=list)

    # Hard biological violations
    hard_violations: list[str] = field(default_factory=list)

    # Soft structural deviations
    soft_deviations: list[str] = field(default_factory=list)

    # Statistics
    total_gt_gates: int = 0
    total_pred_gates: int = 0


def compare_hierarchies(
    predicted: dict,
    ground_truth: dict,
    context: dict,
) -> MatchResult:
    """
    Compare predicted hierarchy against ground truth with biological awareness.
    """
    result = MatchResult()

    pred_gates = extract_all_gates(predicted)
    gt_gates = extract_all_gates(ground_truth)

    result.total_pred_gates = len(pred_gates)
    result.total_gt_gates = len(gt_gates)

    # Build normalized lookup
    pred_normalized = {g["normalized"]: g for g in pred_gates}
    pred_names_lower = {g["name"].lower(): g for g in pred_gates}

    gt_normalized = {g["normalized"]: g for g in gt_gates}

    sample_type = context.get("sample_type", "").lower()
    is_pbmc = any(s in sample_type for s in CD45_OPTIONAL_SAMPLES)

    matched_gt = set()
    matched_pred = set()

    # Pass 1: Exact normalized matches
    for gt_norm, gt_gate in gt_normalized.items():
        if gt_norm in pred_normalized:
            result.matched_gates.append((pred_normalized[gt_norm]["name"], gt_gate["name"]))
            matched_gt.add(gt_gate["name"])
            matched_pred.add(pred_normalized[gt_norm]["name"])

    # Pass 2: Fuzzy matching for unmatched
    for gt_gate in gt_gates:
        if gt_gate["name"] in matched_gt:
            continue

        gt_norm = gt_gate["normalized"]
        gt_name_lower = gt_gate["name"].lower()

        # Try direct lowercase match
        if gt_name_lower in pred_names_lower:
            pred_gate = pred_names_lower[gt_name_lower]
            if pred_gate["name"] not in matched_pred:
                result.matched_gates.append((pred_gate["name"], gt_gate["name"]))
                matched_gt.add(gt_gate["name"])
                matched_pred.add(pred_gate["name"])
                continue

        # Try partial match (for gates like "CD4+ T cells" vs "CD4 T cells")
        for pred_gate in pred_gates:
            if pred_gate["name"] in matched_pred:
                continue

            pred_norm = pred_gate["normalized"]

            # Check if one contains the other
            if gt_norm in pred_norm or pred_norm in gt_norm:
                result.matched_gates.append((pred_gate["name"], gt_gate["name"]))
                matched_gt.add(gt_gate["name"])
                matched_pred.add(pred_gate["name"])
                break

    # Pass 3: Hierarchical matching for terminal populations
    # If a terminal population in GT is found in prediction with valid (shorter) ancestry,
    # credit it as hierarchically matched
    gt_terminals = get_terminal_populations(ground_truth)
    pred_terminals = get_terminal_populations(predicted)

    # Build lookup for prediction terminals
    pred_terminal_lookup = {g["normalized"]: g for g in pred_terminals}

    for gt_terminal in gt_terminals:
        if gt_terminal["name"] in matched_gt:
            continue

        gt_norm = gt_terminal["normalized"]

        # Check if this terminal exists in prediction (possibly with different ancestry)
        if gt_norm in pred_terminal_lookup:
            pred_terminal = pred_terminal_lookup[gt_norm]
            if pred_terminal["name"] not in matched_pred:
                # Terminal population matches - check if ancestry is valid
                gt_ancestors = get_ancestor_chain(gt_terminal)
                pred_ancestors = get_ancestor_chain(pred_terminal)

                # Prediction may have shorter ancestry (skipped intermediates)
                # Check if prediction's parent is a valid ancestor of GT terminal
                if pred_ancestors:
                    pred_parent_norm = pred_ancestors[-1] if pred_ancestors else None

                    # Valid if pred parent appears in GT ancestors OR is a valid shortcut
                    is_valid = False
                    reason = ""

                    if pred_parent_norm in gt_ancestors:
                        is_valid = True
                        reason = f"Valid: pred parent '{pred_parent_norm}' is GT ancestor"
                    elif any(is_valid_ancestor(pred_parent_norm, gt_norm) for _ in [1]):
                        is_valid = True
                        skipped = [a for a in gt_ancestors if a not in pred_ancestors]
                        reason = f"Valid shortcut: skipped {skipped}"

                    if is_valid:
                        result.hierarchically_matched.append(
                            (pred_terminal["name"], gt_terminal["name"], reason)
                        )
                        matched_gt.add(gt_terminal["name"])
                        matched_pred.add(pred_terminal["name"])
                        continue

        # Also check if terminal exists with fuzzy match in pred terminals
        for pred_terminal in pred_terminals:
            if pred_terminal["name"] in matched_pred:
                continue

            pred_norm = pred_terminal["normalized"]

            # Fuzzy terminal match
            if gt_norm in pred_norm or pred_norm in gt_norm:
                result.hierarchically_matched.append(
                    (pred_terminal["name"], gt_terminal["name"],
                     "Fuzzy terminal match")
                )
                matched_gt.add(gt_terminal["name"])
                matched_pred.add(pred_terminal["name"])
                break

    # Pass 4: Identify skipped intermediate gates
    # For each matched gate, check if its GT ancestors are missing
    # If both the gate and a higher ancestor are matched but intermediates are missing,
    # those intermediates are "skipped" (valid simplification) not "truly missing"
    skipped_gates = set()  # Track unique skipped gates to avoid duplicates

    for pred_name, gt_name in result.matched_gates:
        # Find this GT gate
        gt_gate = None
        for g in gt_gates:
            if g["name"] == gt_name:
                gt_gate = g
                break

        if not gt_gate or not gt_gate.get("path"):
            continue

        # Get GT ancestor names (from path)
        gt_path = gt_gate["path"]

        # Find corresponding prediction gate
        pred_gate = None
        for g in pred_gates:
            if g["name"] == pred_name:
                pred_gate = g
                break

        if not pred_gate or not pred_gate.get("path"):
            continue

        pred_path = pred_gate["path"]
        pred_path_normalized = [normalize_gate_name(p) for p in pred_path]

        # Check each GT ancestor - if it's not matched but a higher ancestor is,
        # and the terminal is matched, then this intermediate was skipped
        for i, gt_ancestor in enumerate(gt_path[:-1]):  # Exclude the gate itself
            gt_ancestor_norm = normalize_gate_name(gt_ancestor)

            # Skip if already processed
            if gt_ancestor in skipped_gates or gt_ancestor in matched_gt:
                continue

            # Check if any earlier ancestor IS matched (in prediction)
            earlier_matched = False
            for j in range(i):
                earlier_norm = normalize_gate_name(gt_path[j])
                if earlier_norm in pred_path_normalized or gt_path[j] in matched_gt:
                    earlier_matched = True
                    break

            # Also consider if root/QC gates are implicitly matched
            if gt_ancestor_norm in ["singlets", "live", "scatter_gate", "all_events"]:
                earlier_matched = True

            # If an earlier ancestor is matched and the terminal is matched,
            # this intermediate was validly skipped
            if earlier_matched:
                skipped_gates.add(gt_ancestor)

    # Add unique skipped gates to result
    for gate in skipped_gates:
        result.skipped_intermediates.append((gate, "Skipped intermediate in hierarchy"))

    # Mark skipped gates as matched (they're accounted for)
    matched_gt.update(skipped_gates)

    # Categorize unmatched GT gates
    for gt_gate in gt_gates:
        if gt_gate["name"] in matched_gt:
            continue

        gt_name = gt_gate["name"]
        gt_norm = gt_gate["normalized"]

        # Check if this is an acceptable missing gate

        # CD45 gates in PBMC samples
        if is_pbmc and ("cd45" in gt_norm or "leukocyte" in gt_norm):
            result.acceptable_missing.append((gt_name, "CD45 optional for PBMC samples"))
            continue

        # Time gate (optional)
        if gt_norm in ["time", "time gate"]:
            result.acceptable_missing.append((gt_name, "Time gate optional if acquisition stable"))
            continue

        # Otherwise truly missing
        result.truly_missing.append(gt_name)

    # Extra predictions
    for pred_gate in pred_gates:
        if pred_gate["name"] not in matched_pred:
            result.extra_predictions.append(pred_gate["name"])

    # Check for hard biological violations in predictions
    for pred_gate in pred_gates:
        violations = check_lineage_exclusivity(pred_gate)
        result.hard_violations.extend(violations)

        warnings = check_cd4_cd8_double_positive(pred_gate, context)
        result.soft_deviations.extend(warnings)

    # Check gate order
    order_violations = validate_gate_order(pred_gates)
    result.soft_deviations.extend(order_violations)

    return result


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def parse_hierarchy_from_response(raw_response: str) -> dict | None:
    """Extract JSON hierarchy from LLM response."""
    if not raw_response:
        return None

    # Try to find JSON in the response
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw_response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    first_brace = raw_response.find('{')
    if first_brace != -1:
        depth = 0
        for i, char in enumerate(raw_response[first_brace:], first_brace):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw_response[first_brace:i+1])
                    except json.JSONDecodeError:
                        pass
                    break

    return None


def load_test_cases(data_dir: Path) -> dict[str, dict]:
    """Load ground truth test cases."""
    test_cases = {}
    for json_file in data_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            test_case_id = data.get("test_case_id", json_file.stem)
            test_cases[test_case_id] = data
    return test_cases


def main():
    """Run improved analysis."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "full_benchmark_20260114"
    data_dir = project_root / "data" / "verified"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    predictions_file = results_dir / "predictions.json"
    if not predictions_file.exists():
        print(f"Predictions file not found: {predictions_file}")
        return

    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file) as f:
        predictions = json.load(f)

    print(f"Loading test cases from {data_dir}...")
    test_cases = load_test_cases(data_dir)
    print(f"Loaded {len(test_cases)} test cases")

    # Aggregate results
    all_results: list[MatchResult] = []
    results_by_model: dict[str, list[MatchResult]] = defaultdict(list)
    results_by_test_case: dict[str, list[MatchResult]] = defaultdict(list)

    hard_violations_all: list[str] = []
    soft_deviations_all: list[str] = []

    print(f"\nAnalyzing {len(predictions)} predictions with biological context...")

    parsed_count = 0
    for i, pred in enumerate(predictions):
        test_case_id = pred.get("test_case_id")
        model = pred.get("model", "unknown")

        if test_case_id not in test_cases:
            continue

        tc = test_cases[test_case_id]
        gt_hierarchy = tc.get("gating_hierarchy", tc.get("hierarchy", {}))

        # Build context
        context = tc.get("context", {})
        if "sample_type" not in context:
            context["sample_type"] = context.get("sample", "unknown")

        # Parse predicted hierarchy
        raw_response = pred.get("raw_response", "")
        parsed = parse_hierarchy_from_response(raw_response)
        if not parsed:
            continue

        parsed_count += 1

        # Compare
        match_result = compare_hierarchies(parsed, gt_hierarchy, context)
        all_results.append(match_result)
        results_by_model[model].append(match_result)
        results_by_test_case[test_case_id].append(match_result)

        hard_violations_all.extend(match_result.hard_violations)
        soft_deviations_all.extend(match_result.soft_deviations)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(predictions)}")

    # ==========================================================================
    # PRINT RESULTS
    # ==========================================================================

    print("\n" + "=" * 80)
    print("IMPROVED ANALYSIS RESULTS (with biological context)")
    print("=" * 80)

    print(f"\nPredictions analyzed: {parsed_count}")

    # Aggregate statistics
    total_matched = sum(len(r.matched_gates) for r in all_results)
    total_hierarchically_matched = sum(len(r.hierarchically_matched) for r in all_results)
    total_skipped = sum(len(r.skipped_intermediates) for r in all_results)
    total_all_matched = total_matched + total_hierarchically_matched + total_skipped
    total_truly_missing = sum(len(r.truly_missing) for r in all_results)
    total_acceptable_missing = sum(len(r.acceptable_missing) for r in all_results)
    total_extra = sum(len(r.extra_predictions) for r in all_results)
    total_gt = sum(r.total_gt_gates for r in all_results)
    total_pred = sum(r.total_pred_gates for r in all_results)

    print(f"\n--- Gate Matching Summary ---")
    print(f"Total GT gates across all predictions: {total_gt}")
    print(f"Total predicted gates: {total_pred}")
    print(f"Exact/semantic matches: {total_matched} ({100*total_matched/total_gt:.1f}% of GT)")
    print(f"Hierarchically matched: {total_hierarchically_matched} ({100*total_hierarchically_matched/total_gt:.1f}% of GT)")
    print(f"Skipped intermediates: {total_skipped} ({100*total_skipped/total_gt:.1f}% of GT)")
    print(f"TOTAL ACCOUNTED FOR: {total_all_matched} ({100*total_all_matched/total_gt:.1f}% of GT)")
    print(f"Truly missing gates: {total_truly_missing} ({100*total_truly_missing/total_gt:.1f}% of GT)")
    print(f"Acceptable missing (context-dependent): {total_acceptable_missing}")
    print(f"Extra predictions: {total_extra}")

    # Compare to old analysis
    old_missing = 47928  # From previous analysis
    improvement = old_missing - total_truly_missing
    print(f"\n--- Comparison to Previous Analysis ---")
    print(f"Previous 'MISSING_GATE' count: {old_missing}")
    print(f"New 'truly missing' count: {total_truly_missing}")
    print(f"Reduction: {improvement} ({100*improvement/old_missing:.1f}% were false positives)")

    # Hard violations
    print(f"\n--- Hard Biological Violations ---")
    print(f"Total: {len(hard_violations_all)}")
    if hard_violations_all:
        violation_counts = Counter(hard_violations_all)
        print("Most common:")
        for v, count in violation_counts.most_common(10):
            print(f"  {count:4d}x {v}")
    else:
        print("  None found! Models respect lineage exclusivity rules.")

    # Soft deviations
    print(f"\n--- Soft Deviations (warnings, not errors) ---")
    print(f"Total: {len(soft_deviations_all)}")
    if soft_deviations_all:
        for d in soft_deviations_all[:5]:
            print(f"  • {d}")

    # By model
    print(f"\n--- Results by Model ---")
    print(f"{'Model':<20} {'Exact':>8} {'Hier':>6} {'Skip':>6} {'Total':>8} {'Missing':>10} {'Rate':>8}")
    print("-" * 80)
    for model in sorted(results_by_model.keys()):
        results = results_by_model[model]
        matched = sum(len(r.matched_gates) for r in results)
        hier_matched = sum(len(r.hierarchically_matched) for r in results)
        skipped = sum(len(r.skipped_intermediates) for r in results)
        total_match = matched + hier_matched + skipped
        missing = sum(len(r.truly_missing) for r in results)
        gt_total = sum(r.total_gt_gates for r in results)
        rate = 100 * total_match / gt_total if gt_total > 0 else 0
        print(f"{model:<20} {matched:>8} {hier_matched:>6} {skipped:>6} {total_match:>8} {missing:>10} {rate:>7.1f}%")

    # By test case
    print(f"\n--- Results by Test Case ---")
    print(f"{'Test Case':<20} {'Sample Type':<30} {'Matched':>10} {'Truly Missing':>15}")
    print("-" * 80)
    for tc_id in sorted(results_by_test_case.keys()):
        results = results_by_test_case[tc_id]
        tc = test_cases.get(tc_id, {})
        sample = tc.get("context", {}).get("sample_type", tc.get("context", {}).get("sample", "unknown"))[:28]
        matched = sum(len(r.matched_gates) for r in results)
        missing = sum(len(r.truly_missing) for r in results)
        print(f"{tc_id:<20} {sample:<30} {matched:>10} {missing:>15}")

    # Sample hierarchically matched gates
    print(f"\n--- Sample of Hierarchically Matched Gates ---")
    hier_sample = []
    for r in all_results[:100]:
        hier_sample.extend(r.hierarchically_matched[:3])

    if hier_sample:
        print("Examples of valid simplifications:")
        for pred, gt, reason in hier_sample[:10]:
            print(f"  • Pred: '{pred}' ↔ GT: '{gt}'")
            print(f"    Reason: {reason}")
    else:
        print("  None found in sample")

    # Sample skipped intermediate gates
    print(f"\n--- Sample of Skipped Intermediate Gates ---")
    skipped_sample = []
    for r in all_results[:100]:
        skipped_sample.extend(r.skipped_intermediates[:5])

    if skipped_sample:
        # Count unique gates
        skipped_counts = Counter(gate for gate, reason in skipped_sample)
        print("Most commonly skipped intermediates (valid simplifications):")
        for gate, count in skipped_counts.most_common(15):
            print(f"  {count:3d}x {gate}")
    else:
        print("  None found in sample")

    # Sample truly missing gates
    print(f"\n--- Sample of Truly Missing Gates ---")
    truly_missing_sample = []
    for r in all_results[:100]:
        truly_missing_sample.extend(r.truly_missing[:3])

    missing_counts = Counter(truly_missing_sample)
    print("Most commonly missing (sample):")
    for gate, count in missing_counts.most_common(15):
        print(f"  {count:3d}x {gate}")

    # Sample acceptable missing
    print(f"\n--- Sample of Acceptable Missing (not errors) ---")
    acceptable_sample = []
    for r in all_results[:100]:
        acceptable_sample.extend(r.acceptable_missing[:3])

    for gate, reason in acceptable_sample[:10]:
        print(f"  • {gate}: {reason}")

    # Save results
    output_file = results_dir / "improved_analysis_results.json"
    output_data = {
        "summary": {
            "predictions_analyzed": parsed_count,
            "total_gt_gates": total_gt,
            "total_pred_gates": total_pred,
            "exact_matched_gates": total_matched,
            "hierarchically_matched_gates": total_hierarchically_matched,
            "skipped_intermediate_gates": total_skipped,
            "total_accounted_gates": total_all_matched,
            "truly_missing": total_truly_missing,
            "acceptable_missing": total_acceptable_missing,
            "extra_predictions": total_extra,
            "hard_violations": len(hard_violations_all),
            "soft_deviations": len(soft_deviations_all),
            "exact_match_rate": total_matched / total_gt if total_gt > 0 else 0,
            "total_accounted_rate": total_all_matched / total_gt if total_gt > 0 else 0,
        },
        "comparison_to_previous": {
            "old_missing_count": old_missing,
            "new_truly_missing": total_truly_missing,
            "false_positive_rate": improvement / old_missing if old_missing > 0 else 0,
        },
        "hard_violations": list(set(hard_violations_all)),
        "by_model": {
            model: {
                "exact_matched": sum(len(r.matched_gates) for r in results),
                "hierarchically_matched": sum(len(r.hierarchically_matched) for r in results),
                "skipped_intermediates": sum(len(r.skipped_intermediates) for r in results),
                "total_accounted": sum(len(r.matched_gates) + len(r.hierarchically_matched) + len(r.skipped_intermediates) for r in results),
                "truly_missing": sum(len(r.truly_missing) for r in results),
                "acceptable_missing": sum(len(r.acceptable_missing) for r in results),
            }
            for model, results in results_by_model.items()
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
