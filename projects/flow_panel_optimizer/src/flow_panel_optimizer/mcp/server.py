"""
MCP Server for Flow Cytometry Panel Design.

This module provides spectral analysis tools that can be called by Claude
to make informed fluorophore selections for flow cytometry panels.

Tools:
- analyze_panel: Get complexity index and problematic pairs for a panel
- check_compatibility: Check if a candidate fluorophore fits with existing panel
- suggest_fluorophores: Get ranked suggestions for a marker given current panel
- get_fluorophore_info: Get detailed spectral info for a fluorophore
- find_alternatives: Find alternatives to a problematic fluorophore
"""

import sys
from pathlib import Path
from typing import Optional
import json

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flow_panel_optimizer.data.fluorophore_database import (
    FLUOROPHORE_DATABASE,
    FluorophoreData,
    get_fluorophore,
    get_fluorophores_by_laser,
    list_all_fluorophores,
    calculate_spectral_overlap,
    get_known_overlap,
)


def analyze_panel(fluorophores: list[str]) -> dict:
    """
    Analyze a panel of fluorophores for spectral conflicts.

    Uses EasyPanel-style metrics:
    - Total Similarity Score: Sum of ALL pairwise similarities (lower is better)
    - Critical pairs: Pairs with similarity > 0.90 that should be avoided

    Args:
        fluorophores: List of fluorophore names in the panel

    Returns:
        dict with:
        - total_similarity_score: Sum of all pairwise similarities (lower is better)
        - complexity_index: Alias for total_similarity_score (for compatibility)
        - avg_similarity: Average pairwise similarity
        - max_similarity: Highest pairwise similarity
        - critical_pairs: List of pairs with similarity > 0.90
        - problematic_pairs: List of pairs with similarity > 0.70
        - unknown_fluorophores: Fluorophores not in database
    """
    if not fluorophores:
        return {
            "total_similarity_score": 0.0,
            "complexity_index": 0.0,
            "avg_similarity": 0.0,
            "max_similarity": 0.0,
            "problematic_pairs": [],
            "critical_pairs": [],
            "unknown_fluorophores": [],
            "summary": "Empty panel"
        }

    # Resolve fluorophores
    resolved = []
    unknown = []
    for name in fluorophores:
        f = get_fluorophore(name)
        if f:
            resolved.append((name, f))
        else:
            unknown.append(name)

    if len(resolved) < 2:
        return {
            "total_similarity_score": 0.0,
            "complexity_index": 0.0,
            "avg_similarity": 0.0,
            "max_similarity": 0.0,
            "problematic_pairs": [],
            "critical_pairs": [],
            "unknown_fluorophores": unknown,
            "summary": f"Need at least 2 known fluorophores. Unknown: {unknown}"
        }

    # Calculate pairwise similarities
    similarities = []
    problematic = []
    critical = []
    max_sim = 0.0
    total_sim = 0.0

    for i, (name1, f1) in enumerate(resolved):
        for j, (name2, f2) in enumerate(resolved):
            if i >= j:
                continue

            # Use known overlap if available, otherwise calculate
            sim = get_known_overlap(name1, name2)
            if sim is None:
                sim = calculate_spectral_overlap(f1, f2)

            similarities.append({
                "pair": (name1, name2),
                "similarity": sim,
                "laser1": f1.optimal_laser,
                "laser2": f2.optimal_laser,
            })

            # Sum ALL pairs for total similarity score (EasyPanel method)
            total_sim += sim
            max_sim = max(max_sim, sim)

            if sim > 0.90:
                critical.append({
                    "pair": f"{name1} / {name2}",
                    "similarity": round(sim, 3),
                    "risk": "CRITICAL - nearly identical spectra, will cause unmixing errors",
                    "recommendation": "Replace one of these fluorophores"
                })
            elif sim > 0.70:
                problematic.append({
                    "pair": f"{name1} / {name2}",
                    "similarity": round(sim, 3),
                    "risk": "HIGH - significant spectral overlap",
                    "recommendation": "Consider alternatives if markers are co-expressed"
                })

    # Calculate metrics
    num_pairs = len(similarities)
    avg_sim = total_sim / num_pairs if num_pairs > 0 else 0.0

    # Total similarity score is the EasyPanel optimization target
    total_similarity_score = round(total_sim, 4)

    # Determine quality rating based on average similarity
    if avg_sim < 0.25 and len(critical) == 0:
        quality = "EXCELLENT"
    elif avg_sim < 0.35 and len(critical) == 0:
        quality = "GOOD"
    elif avg_sim < 0.50 and len(critical) <= 1:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"

    return {
        "total_similarity_score": total_similarity_score,
        "complexity_index": total_similarity_score,  # Alias for compatibility
        "avg_similarity": round(avg_sim, 4),
        "max_similarity": round(max_sim, 4),
        "n_pairs": num_pairs,
        "n_critical_pairs": len(critical),
        "problematic_pairs": problematic,
        "critical_pairs": critical,
        "unknown_fluorophores": unknown,
        "n_fluorophores": len(resolved),
        "quality_rating": quality,
        "summary": f"Panel quality: {quality}. Total similarity={total_similarity_score:.2f}, "
                   f"Avg={avg_sim:.3f}, Max={max_sim:.3f}, "
                   f"{len(critical)} critical pairs (>0.90), {len(problematic)} high-risk pairs (>0.70)"
    }


def check_compatibility(candidate: str, existing_panel: list[str]) -> dict:
    """
    Check if a candidate fluorophore is compatible with existing selections.

    Args:
        candidate: Name of fluorophore to check
        existing_panel: List of fluorophores already in panel

    Returns:
        dict with:
        - compatible: bool
        - max_similarity: Highest similarity to existing fluorophores
        - recommendation: "safe" / "caution" / "avoid"
        - conflicts: List of problematic existing fluorophores
        - fluorophore_info: Details about the candidate
    """
    cand = get_fluorophore(candidate)
    if not cand:
        return {
            "compatible": False,
            "max_similarity": None,
            "recommendation": "unknown",
            "conflicts": [],
            "error": f"Unknown fluorophore: {candidate}. Check spelling or use list_fluorophores()."
        }

    if not existing_panel:
        return {
            "compatible": True,
            "max_similarity": 0.0,
            "recommendation": "safe",
            "conflicts": [],
            "fluorophore_info": {
                "name": cand.name,
                "emission_max": cand.em_max,
                "optimal_laser": cand.optimal_laser,
                "brightness": cand.relative_brightness,
            }
        }

    # Check against each existing fluorophore
    max_sim = 0.0
    conflicts = []

    for existing_name in existing_panel:
        existing = get_fluorophore(existing_name)
        if not existing:
            continue

        sim = get_known_overlap(candidate, existing_name)
        if sim is None:
            sim = calculate_spectral_overlap(cand, existing)

        max_sim = max(max_sim, sim)

        if sim > 0.70:
            conflicts.append({
                "fluorophore": existing_name,
                "similarity": round(sim, 3),
                "same_laser": cand.optimal_laser == existing.optimal_laser,
            })

    # Determine recommendation
    if max_sim > 0.90:
        recommendation = "AVOID"
        compatible = False
    elif max_sim > 0.75:
        recommendation = "CAUTION"
        compatible = True
    elif max_sim > 0.60:
        recommendation = "ACCEPTABLE"
        compatible = True
    else:
        recommendation = "SAFE"
        compatible = True

    return {
        "compatible": compatible,
        "max_similarity": round(max_sim, 4),
        "recommendation": recommendation,
        "conflicts": conflicts,
        "fluorophore_info": {
            "name": cand.name,
            "emission_max": cand.em_max,
            "optimal_laser": cand.optimal_laser,
            "brightness": cand.relative_brightness,
            "category": cand.category,
        }
    }


def suggest_fluorophores(
    existing_panel: list[str],
    expression_level: str = "medium",
    preferred_laser: Optional[int] = None,
    exclude: Optional[list[str]] = None,
    top_n: int = 5
) -> dict:
    """
    Suggest best fluorophores given current panel and marker expression.

    Args:
        existing_panel: List of fluorophores already selected
        expression_level: "high", "medium", or "low"
        preferred_laser: Optional laser to prefer (355, 405, 488, 561, 640)
        exclude: Fluorophores to exclude from suggestions
        top_n: Number of suggestions to return

    Returns:
        dict with ranked fluorophore suggestions
    """
    exclude = exclude or []

    # Define brightness requirements based on expression
    brightness_ranges = {
        "high": (20, 150),  # Can use dimmer dyes
        "medium": (50, 150),  # Need moderately bright
        "low": (70, 150),  # Need brightest dyes
    }

    min_bright, max_bright = brightness_ranges.get(expression_level, (50, 150))

    # Score all available fluorophores
    candidates = []

    for name, fluor in FLUOROPHORE_DATABASE.items():
        if name in exclude or name in existing_panel:
            continue

        # Check brightness
        if not (min_bright <= fluor.relative_brightness <= max_bright):
            continue

        # Check laser preference
        laser_match = 1.0
        if preferred_laser and preferred_laser not in fluor.compatible_lasers:
            laser_match = 0.5  # Penalize but don't exclude

        # Calculate max similarity to existing panel
        max_sim = 0.0
        for existing_name in existing_panel:
            existing = get_fluorophore(existing_name)
            if existing:
                sim = get_known_overlap(name, existing_name)
                if sim is None:
                    sim = calculate_spectral_overlap(fluor, existing)
                max_sim = max(max_sim, sim)

        # Score: lower similarity is better, brightness helps
        # Score = (1 - max_similarity) * brightness_factor * laser_match
        brightness_factor = fluor.relative_brightness / 100.0
        score = (1 - max_sim) * brightness_factor * laser_match

        candidates.append({
            "fluorophore": name,
            "score": round(score, 3),
            "max_similarity_to_panel": round(max_sim, 3),
            "brightness": fluor.relative_brightness,
            "optimal_laser": fluor.optimal_laser,
            "emission_max": fluor.em_max,
            "category": fluor.category,
            "vendor": fluor.vendor_primary,
        })

    # Sort by score (descending)
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Take top N
    top_suggestions = candidates[:top_n]

    # Add recommendation text
    for i, s in enumerate(top_suggestions):
        if s["max_similarity_to_panel"] < 0.5:
            s["compatibility"] = "EXCELLENT"
        elif s["max_similarity_to_panel"] < 0.7:
            s["compatibility"] = "GOOD"
        else:
            s["compatibility"] = "CAUTION"

    return {
        "expression_level": expression_level,
        "existing_panel_size": len(existing_panel),
        "suggestions": top_suggestions,
        "total_candidates_evaluated": len(candidates),
    }


def get_fluorophore_info(name: str) -> dict:
    """
    Get detailed information about a specific fluorophore.

    Args:
        name: Fluorophore name

    Returns:
        dict with spectral properties and usage recommendations
    """
    fluor = get_fluorophore(name)
    if not fluor:
        # Try to suggest similar names
        all_names = list_all_fluorophores()
        similar = [n for n in all_names if name.lower() in n.lower()]

        return {
            "error": f"Unknown fluorophore: {name}",
            "suggestions": similar[:5] if similar else all_names[:10],
        }

    return {
        "name": fluor.name,
        "excitation_max": fluor.ex_max,
        "emission_max": fluor.em_max,
        "excitation_range": fluor.ex_range,
        "emission_range": fluor.em_range,
        "optimal_laser": fluor.optimal_laser,
        "compatible_lasers": fluor.compatible_lasers,
        "relative_brightness": fluor.relative_brightness,
        "brightness_category": (
            "very bright" if fluor.relative_brightness >= 80 else
            "bright" if fluor.relative_brightness >= 50 else
            "moderate" if fluor.relative_brightness >= 30 else
            "dim"
        ),
        "category": fluor.category,
        "vendor": fluor.vendor_primary,
        "notes": fluor.notes,
        "recommended_for": (
            "low expression markers" if fluor.relative_brightness >= 70 else
            "medium-high expression markers" if fluor.relative_brightness >= 40 else
            "high expression markers only"
        ),
    }


def find_alternatives(
    fluorophore: str,
    existing_panel: list[str],
    top_n: int = 5
) -> dict:
    """
    Find alternative fluorophores similar in properties but better separated.

    Args:
        fluorophore: The problematic fluorophore to replace
        existing_panel: Current panel (excluding the fluorophore to replace)
        top_n: Number of alternatives to return

    Returns:
        dict with alternative suggestions
    """
    original = get_fluorophore(fluorophore)
    if not original:
        return {"error": f"Unknown fluorophore: {fluorophore}"}

    # Look for fluorophores with similar emission but different enough
    candidates = []

    for name, fluor in FLUOROPHORE_DATABASE.items():
        if name == fluorophore or name in existing_panel:
            continue

        # Similar emission range (within 100nm)
        if abs(fluor.em_max - original.em_max) > 100:
            continue

        # Similar brightness (within 30%)
        brightness_diff = abs(fluor.relative_brightness - original.relative_brightness)
        if brightness_diff > 30:
            continue

        # Calculate similarity to existing panel
        max_sim = 0.0
        for existing_name in existing_panel:
            existing = get_fluorophore(existing_name)
            if existing:
                sim = get_known_overlap(name, existing_name)
                if sim is None:
                    sim = calculate_spectral_overlap(fluor, existing)
                max_sim = max(max_sim, sim)

        # Check improvement over original
        original_max_sim = 0.0
        for existing_name in existing_panel:
            existing = get_fluorophore(existing_name)
            if existing:
                sim = get_known_overlap(fluorophore, existing_name)
                if sim is None:
                    sim = calculate_spectral_overlap(original, existing)
                original_max_sim = max(original_max_sim, sim)

        improvement = original_max_sim - max_sim

        if improvement > 0:  # Only suggest if it's actually better
            candidates.append({
                "fluorophore": name,
                "improvement": round(improvement, 3),
                "max_similarity": round(max_sim, 3),
                "original_similarity": round(original_max_sim, 3),
                "emission_max": fluor.em_max,
                "brightness": fluor.relative_brightness,
                "optimal_laser": fluor.optimal_laser,
                "vendor": fluor.vendor_primary,
            })

    # Sort by improvement
    candidates.sort(key=lambda x: x["improvement"], reverse=True)

    return {
        "original": fluorophore,
        "original_emission": original.em_max,
        "original_brightness": original.relative_brightness,
        "alternatives": candidates[:top_n],
        "message": f"Found {len(candidates)} alternatives with better spectral separation"
    }


def list_fluorophores_by_laser(laser_nm: int) -> dict:
    """List all fluorophores compatible with a specific laser."""
    fluors = get_fluorophores_by_laser(laser_nm)

    return {
        "laser": laser_nm,
        "count": len(fluors),
        "fluorophores": [
            {
                "name": f.name,
                "emission_max": f.em_max,
                "brightness": f.relative_brightness,
                "category": f.category,
            }
            for f in sorted(fluors, key=lambda x: x.em_max)
        ]
    }


# =========================================================================
# ANTIBODY AVAILABILITY TOOLS
# =========================================================================

def check_antibody_availability(marker: str, fluorophore: str) -> dict:
    """
    Check if a marker-fluorophore antibody conjugate is commercially available.

    Args:
        marker: Target antigen (e.g., "CD4")
        fluorophore: Fluorochrome (e.g., "BV421")

    Returns:
        dict with availability, vendor, clone, and catalog info
    """
    from flow_panel_optimizer.data.antibody_availability import check_availability
    return check_availability(marker, fluorophore)


def validate_panel_reagents(assignments: list[dict]) -> dict:
    """
    Validate that all panel assignments are commercially available.

    Args:
        assignments: List of {"marker": X, "fluorophore": Y} dicts

    Returns:
        dict with availability report and ordering info
    """
    from flow_panel_optimizer.data.antibody_availability import validate_panel_availability
    return validate_panel_availability(assignments)


def suggest_available_reagents(
    marker: str,
    existing_panel: list[str],
    expression_level: str = "medium"
) -> dict:
    """
    Suggest fluorophores for a marker that are BOTH spectrally compatible
    AND commercially available as antibody conjugates.

    Args:
        marker: Target antigen
        existing_panel: Currently selected fluorophores
        expression_level: "high", "medium", or "low"

    Returns:
        dict with ranked suggestions including vendor/catalog info
    """
    from flow_panel_optimizer.data.antibody_availability import (
        suggest_available_fluorophores,
        get_available_fluorophores,
    )

    # First check what's available for this marker
    available = get_available_fluorophores(marker)

    if not available:
        return {
            "marker": marker,
            "error": f"No antibody conjugates found for {marker} in database",
            "suggestions": [],
        }

    # Get suggestions with availability info
    suggestions = suggest_available_fluorophores(marker, existing_panel, expression_level)

    return {
        "marker": marker,
        "expression_level": expression_level,
        "existing_panel_size": len(existing_panel),
        "total_available": len(available),
        "suggestions": suggestions[:8],  # Top 8
    }


# =========================================================================
# CO-EXPRESSION PROFILE HANDLING
# =========================================================================

# Common co-expression patterns in immunology
# Markers that are typically mutually exclusive can safely use high-overlap fluorophores
COEXPRESSION_PROFILES = {
    # T cell lineage - mutually exclusive
    "CD4": {"exclusive_with": ["CD8", "CD19", "CD14", "CD56"], "coexpressed_with": ["CD3", "CD27", "CD28"]},
    "CD8": {"exclusive_with": ["CD4", "CD19", "CD14"], "coexpressed_with": ["CD3", "CD27", "CD28", "CD57"]},

    # B cell vs T cell - mutually exclusive
    "CD19": {"exclusive_with": ["CD3", "CD4", "CD8", "CD14", "CD56"], "coexpressed_with": ["CD20", "CD27", "IgD"]},
    "CD20": {"exclusive_with": ["CD3", "CD14"], "coexpressed_with": ["CD19", "CD27", "IgD"]},

    # Monocyte markers - exclusive from lymphocytes
    "CD14": {"exclusive_with": ["CD3", "CD19", "CD56"], "coexpressed_with": ["CD16", "HLA-DR", "CD11c"]},

    # NK cells
    "CD56": {"exclusive_with": ["CD19", "CD14"], "coexpressed_with": ["CD16", "NKG2D"]},

    # T cell differentiation - mutually exclusive states
    "CD45RA": {"exclusive_with": ["CD45RO"], "coexpressed_with": ["CCR7"]},
    "CD45RO": {"exclusive_with": ["CD45RA"], "coexpressed_with": []},

    # Th subsets - often mutually exclusive
    "CXCR3": {"exclusive_with": ["CCR4", "CRTH2"], "coexpressed_with": []},  # Th1
    "CCR4": {"exclusive_with": ["CXCR3"], "coexpressed_with": ["CCR6"]},  # Th2/Th17
    "CXCR5": {"exclusive_with": [], "coexpressed_with": ["PD-1", "ICOS"]},  # Tfh
}


def check_coexpression(marker1: str, marker2: str) -> dict:
    """
    Check if two markers are co-expressed or mutually exclusive.

    This is critical for panel design:
    - Mutually exclusive markers can safely use high-overlap fluorophores
    - Co-expressed markers MUST use well-separated fluorophores

    Args:
        marker1: First marker name
        marker2: Second marker name

    Returns:
        dict with:
        - relationship: "exclusive", "coexpressed", or "unknown"
        - can_use_similar_fluorophores: bool
        - recommendation: str
    """
    m1_profile = COEXPRESSION_PROFILES.get(marker1, {})
    m2_profile = COEXPRESSION_PROFILES.get(marker2, {})

    # Check if mutually exclusive
    if marker2 in m1_profile.get("exclusive_with", []) or \
       marker1 in m2_profile.get("exclusive_with", []):
        return {
            "marker1": marker1,
            "marker2": marker2,
            "relationship": "exclusive",
            "can_use_similar_fluorophores": True,
            "recommendation": f"{marker1} and {marker2} are mutually exclusive - "
                            f"safe to use fluorophores with high spectral overlap"
        }

    # Check if co-expressed
    if marker2 in m1_profile.get("coexpressed_with", []) or \
       marker1 in m2_profile.get("coexpressed_with", []):
        return {
            "marker1": marker1,
            "marker2": marker2,
            "relationship": "coexpressed",
            "can_use_similar_fluorophores": False,
            "recommendation": f"{marker1} and {marker2} are co-expressed - "
                            f"MUST use fluorophores with low spectral overlap (<0.70)"
        }

    # Unknown relationship
    return {
        "marker1": marker1,
        "marker2": marker2,
        "relationship": "unknown",
        "can_use_similar_fluorophores": False,  # Err on side of caution
        "recommendation": f"Co-expression pattern unknown for {marker1}/{marker2} - "
                        f"recommend using low-overlap fluorophores to be safe"
    }


def optimize_panel_with_coexpression(
    marker_fluorophore_pairs: list[dict],
) -> dict:
    """
    Analyze a panel considering co-expression profiles.

    Identifies pairs where high spectral overlap is acceptable (mutually exclusive)
    vs problematic (co-expressed).

    Args:
        marker_fluorophore_pairs: List of {"marker": str, "fluorophore": str}

    Returns:
        Analysis with co-expression-aware recommendations
    """
    if len(marker_fluorophore_pairs) < 2:
        return {"error": "Need at least 2 marker-fluorophore pairs"}

    # Extract fluorophores for spectral analysis
    fluorophores = [p["fluorophore"] for p in marker_fluorophore_pairs]
    markers = [p["marker"] for p in marker_fluorophore_pairs]

    # Get spectral analysis
    spectral_analysis = analyze_panel(fluorophores)

    # Analyze each problematic pair considering co-expression
    acceptable_overlaps = []
    problematic_overlaps = []

    for i, p1 in enumerate(marker_fluorophore_pairs):
        for j, p2 in enumerate(marker_fluorophore_pairs):
            if i >= j:
                continue

            # Get spectral similarity
            f1 = get_fluorophore(p1["fluorophore"])
            f2 = get_fluorophore(p2["fluorophore"])

            if not f1 or not f2:
                continue

            sim = get_known_overlap(p1["fluorophore"], p2["fluorophore"])
            if sim is None:
                sim = calculate_spectral_overlap(f1, f2)

            # Check co-expression
            coexp = check_coexpression(p1["marker"], p2["marker"])

            pair_info = {
                "markers": f"{p1['marker']} / {p2['marker']}",
                "fluorophores": f"{p1['fluorophore']} / {p2['fluorophore']}",
                "similarity": round(sim, 3),
                "coexpression": coexp["relationship"],
            }

            if sim > 0.70:
                if coexp["can_use_similar_fluorophores"]:
                    pair_info["status"] = "ACCEPTABLE - markers are mutually exclusive"
                    acceptable_overlaps.append(pair_info)
                else:
                    pair_info["status"] = "PROBLEMATIC - markers may be co-expressed"
                    pair_info["recommendation"] = "Consider swapping fluorophores"
                    problematic_overlaps.append(pair_info)

    return {
        "total_similarity_score": spectral_analysis["total_similarity_score"],
        "quality_rating": spectral_analysis["quality_rating"],
        "n_high_overlap_pairs": len(acceptable_overlaps) + len(problematic_overlaps),
        "acceptable_overlaps": acceptable_overlaps,
        "problematic_overlaps": problematic_overlaps,
        "summary": f"Found {len(problematic_overlaps)} problematic high-overlap pairs "
                  f"and {len(acceptable_overlaps)} acceptable high-overlap pairs "
                  f"(mutually exclusive markers)"
    }


# Tool definitions for Claude
MCP_TOOLS = [
    {
        "name": "analyze_panel",
        "description": "Analyze a panel of fluorophores for spectral conflicts. Returns complexity index, problematic pairs, and quality rating.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fluorophores": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of fluorophore names in the panel"
                }
            },
            "required": ["fluorophores"]
        }
    },
    {
        "name": "check_compatibility",
        "description": "Check if a candidate fluorophore is compatible with existing panel selections. Returns similarity score and recommendation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate": {
                    "type": "string",
                    "description": "Name of fluorophore to check"
                },
                "existing_panel": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of fluorophores already in panel"
                }
            },
            "required": ["candidate", "existing_panel"]
        }
    },
    {
        "name": "suggest_fluorophores",
        "description": "Get ranked fluorophore suggestions for a marker given current panel and expression level.",
        "input_schema": {
            "type": "object",
            "properties": {
                "existing_panel": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of fluorophores already selected"
                },
                "expression_level": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Expression level of the marker"
                },
                "preferred_laser": {
                    "type": "integer",
                    "description": "Optional preferred laser wavelength (355, 405, 488, 561, 640)"
                }
            },
            "required": ["existing_panel", "expression_level"]
        }
    },
    {
        "name": "get_fluorophore_info",
        "description": "Get detailed spectral information about a specific fluorophore.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Fluorophore name"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "find_alternatives",
        "description": "Find alternative fluorophores with better spectral separation than a problematic choice.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fluorophore": {
                    "type": "string",
                    "description": "The fluorophore to find alternatives for"
                },
                "existing_panel": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Current panel (excluding the fluorophore to replace)"
                }
            },
            "required": ["fluorophore", "existing_panel"]
        }
    },
    {
        "name": "check_coexpression",
        "description": "Check if two markers are co-expressed or mutually exclusive. Critical for determining if high-overlap fluorophores are safe.",
        "input_schema": {
            "type": "object",
            "properties": {
                "marker1": {
                    "type": "string",
                    "description": "First marker name (e.g., CD4)"
                },
                "marker2": {
                    "type": "string",
                    "description": "Second marker name (e.g., CD8)"
                }
            },
            "required": ["marker1", "marker2"]
        }
    },
    {
        "name": "optimize_panel_with_coexpression",
        "description": "Analyze a full panel considering both spectral overlap AND co-expression profiles. Returns which high-overlap pairs are acceptable vs problematic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "marker_fluorophore_pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "marker": {"type": "string"},
                            "fluorophore": {"type": "string"}
                        },
                        "required": ["marker", "fluorophore"]
                    },
                    "description": "List of marker-fluorophore assignments"
                }
            },
            "required": ["marker_fluorophore_pairs"]
        }
    }
]


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute an MCP tool and return the result."""
    tool_functions = {
        "analyze_panel": lambda args: analyze_panel(args["fluorophores"]),
        "check_compatibility": lambda args: check_compatibility(
            args["candidate"],
            args["existing_panel"]
        ),
        "suggest_fluorophores": lambda args: suggest_fluorophores(
            args["existing_panel"],
            args.get("expression_level", "medium"),
            args.get("preferred_laser"),
        ),
        "get_fluorophore_info": lambda args: get_fluorophore_info(args["name"]),
        "find_alternatives": lambda args: find_alternatives(
            args["fluorophore"],
            args["existing_panel"]
        ),
        "list_fluorophores_by_laser": lambda args: list_fluorophores_by_laser(args["laser_nm"]),
        "check_coexpression": lambda args: check_coexpression(
            args["marker1"],
            args["marker2"]
        ),
        "optimize_panel_with_coexpression": lambda args: optimize_panel_with_coexpression(
            args["marker_fluorophore_pairs"]
        ),
    }

    if tool_name not in tool_functions:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        result = tool_functions[tool_name](arguments)
        return result
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test the tools
    print("Testing MCP tools...\n")

    # Test analyze_panel
    panel = ["BV421", "FITC", "PE", "APC", "PE-Cy7", "APC-Cy7"]
    result = analyze_panel(panel)
    print(f"analyze_panel({panel}):")
    print(json.dumps(result, indent=2))
    print()

    # Test check_compatibility
    result = check_compatibility("BV510", ["BV421", "FITC", "PE"])
    print(f"check_compatibility('BV510', ['BV421', 'FITC', 'PE']):")
    print(json.dumps(result, indent=2))
    print()

    # Test suggest_fluorophores
    result = suggest_fluorophores(["BV421", "FITC", "PE"], "low")
    print(f"suggest_fluorophores(['BV421', 'FITC', 'PE'], 'low'):")
    print(json.dumps(result, indent=2))
