"""
Exact OMIP panel specifications from published literature.

These are used as ground truth for testing if:
1. Claude can reproduce expert panel designs
2. Claude + MCP can improve on published panels (lower complexity)

Each panel contains:
- markers: List of markers ONLY (what Claude receives as input)
- published_assignments: The actual fluorophore assignments from the paper
- metadata: Publication info, instrument, cell types identified
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OMIPPanel:
    """Complete OMIP panel specification."""
    omip_number: int
    name: str
    description: str
    doi: str
    url: str
    year: int
    authors: str
    instrument: str
    cell_types: list[str]
    markers: list[dict]  # name, expression, notes
    published_assignments: list[dict]  # marker, fluorophore, clone, vendor
    notes: Optional[str] = None


# =============================================================================
# OMIP-030: Human T Cell Subsets (10-color)
# Reference: Wingender & Kronenberg, Cytometry A 2015
# =============================================================================
OMIP_030 = OMIPPanel(
    omip_number=30,
    name="OMIP-030: Human T Cell Subsets",
    description="Characterization of human T cell subsets via surface markers",
    doi="10.1002/cyto.a.22788",
    url="https://doi.org/10.1002/cyto.a.22788",
    year=2015,
    authors="Wingender G, Kronenberg M",
    instrument="BD LSRFortessa (4-laser: 405, 488, 561, 640nm)",
    cell_types=[
        "CD4+ T cells",
        "CD8+ T cells",
        "Th1, Th2, Th17, Th22, Tfh subsets",
        "Regulatory T cells (Tregs)",
        "NKT-like cells",
    ],
    markers=[
        {"name": "CD3", "expression": "high", "notes": "Pan T-cell marker"},
        {"name": "CD4", "expression": "high", "notes": "Helper T cells"},
        {"name": "CD8", "expression": "high", "notes": "Cytotoxic T cells"},
        {"name": "CD45RA", "expression": "high", "notes": "Naive marker"},
        {"name": "CD127", "expression": "medium", "notes": "IL-7R, Treg exclusion"},
        {"name": "CD25", "expression": "medium", "notes": "IL-2R, Treg inclusion"},
        {"name": "CCR7", "expression": "medium", "notes": "Memory/naive distinction"},
        {"name": "CD161", "expression": "medium", "notes": "NKT-like identification"},
        {"name": "CXCR3", "expression": "low", "notes": "Th1 marker"},
        {"name": "Viability", "expression": "high", "notes": "Dead cell exclusion"},
    ],
    published_assignments=[
        {"marker": "CD3", "fluorophore": "Pacific Blue", "clone": "UCHT1", "vendor": "BD"},
        {"marker": "CD4", "fluorophore": "PerCP-Cy5.5", "clone": "SK3", "vendor": "BD"},
        {"marker": "CD8", "fluorophore": "APC-Fire750", "clone": "SK1", "vendor": "BioLegend"},
        {"marker": "CD45RA", "fluorophore": "FITC", "clone": "HI100", "vendor": "BD"},
        {"marker": "CD127", "fluorophore": "PE", "clone": "HIL-7R-M21", "vendor": "BD"},
        {"marker": "CD25", "fluorophore": "PE-Cy7", "clone": "M-A251", "vendor": "BD"},
        {"marker": "CCR7", "fluorophore": "PE-Cy5", "clone": "G043H7", "vendor": "BioLegend"},
        {"marker": "CD161", "fluorophore": "BV711", "clone": "HP-3G10", "vendor": "BioLegend"},
        {"marker": "CXCR3", "fluorophore": "APC", "clone": "G025H7", "vendor": "BioLegend"},
        {"marker": "Viability", "fluorophore": "LIVE/DEAD Blue", "clone": "N/A", "vendor": "Thermo"},
    ],
)

# =============================================================================
# OMIP-044: Human Dendritic Cells (28-color)
# Reference: Mair & Prlic, Cytometry A 2018
# =============================================================================
OMIP_044 = OMIPPanel(
    omip_number=44,
    name="OMIP-044: Human Dendritic Cells",
    description="28-color immunophenotyping of the human dendritic cell compartment",
    doi="10.1002/cyto.a.23331",
    url="https://doi.org/10.1002/cyto.a.23331",
    year=2018,
    authors="Mair F, Prlic M",
    instrument="BD FACSymphony (5-laser: 355, 405, 488, 561, 640nm)",
    cell_types=[
        "cDC1 (CD141+)",
        "cDC2 (CD1c+)",
        "Plasmacytoid DCs",
        "Classical monocytes",
        "Non-classical monocytes",
    ],
    markers=[
        {"name": "Viability", "expression": "high", "notes": "Dead cell exclusion"},
        {"name": "CD3", "expression": "high", "notes": "T cell lineage exclusion"},
        {"name": "CD14", "expression": "high", "notes": "Monocyte marker"},
        {"name": "CD19", "expression": "high", "notes": "B cell lineage exclusion"},
        {"name": "CD56", "expression": "high", "notes": "NK lineage exclusion"},
        {"name": "HLA-DR", "expression": "high", "notes": "APC identification"},
        {"name": "CD11c", "expression": "high", "notes": "DC marker"},
        {"name": "CD123", "expression": "medium", "notes": "pDC marker"},
        {"name": "CD141", "expression": "low", "notes": "cDC1 marker"},
        {"name": "CD1c", "expression": "medium", "notes": "cDC2 marker"},
        {"name": "CD16", "expression": "medium", "notes": "Non-classical monocytes"},
        {"name": "CD40", "expression": "medium", "notes": "Costimulation"},
        {"name": "CD80", "expression": "low", "notes": "Costimulation"},
        {"name": "CD86", "expression": "medium", "notes": "Costimulation"},
        {"name": "CCR7", "expression": "low", "notes": "Migration"},
        {"name": "CD11b", "expression": "high", "notes": "Integrin"},
        {"name": "CD45RA", "expression": "high", "notes": "T cell memory"},
        {"name": "CD4", "expression": "high", "notes": "T helper"},
        {"name": "CD8", "expression": "high", "notes": "Cytotoxic T"},
    ],
    published_assignments=[
        {"marker": "Viability", "fluorophore": "LIVE/DEAD Blue", "clone": "N/A", "vendor": "Thermo"},
        {"marker": "CD3", "fluorophore": "BUV395", "clone": "UCHT1", "vendor": "BD"},
        {"marker": "CD14", "fluorophore": "BUV737", "clone": "M5E2", "vendor": "BD"},
        {"marker": "CD19", "fluorophore": "BUV395", "clone": "SJ25C1", "vendor": "BD"},
        {"marker": "CD56", "fluorophore": "BUV395", "clone": "NCAM16.2", "vendor": "BD"},
        {"marker": "HLA-DR", "fluorophore": "BV785", "clone": "L243", "vendor": "BioLegend"},
        {"marker": "CD11c", "fluorophore": "PE-Cy7", "clone": "Bu15", "vendor": "BioLegend"},
        {"marker": "CD123", "fluorophore": "BV650", "clone": "6H6", "vendor": "BioLegend"},
        {"marker": "CD141", "fluorophore": "APC", "clone": "M80", "vendor": "BioLegend"},
        {"marker": "CD1c", "fluorophore": "PE-Dazzle 594", "clone": "L161", "vendor": "BioLegend"},
        {"marker": "CD16", "fluorophore": "BV605", "clone": "3G8", "vendor": "BioLegend"},
        {"marker": "CD40", "fluorophore": "BV421", "clone": "5C3", "vendor": "BioLegend"},
        {"marker": "CD80", "fluorophore": "FITC", "clone": "2D10", "vendor": "BioLegend"},
        {"marker": "CD86", "fluorophore": "BV510", "clone": "IT2.2", "vendor": "BioLegend"},
        {"marker": "CCR7", "fluorophore": "PE", "clone": "G043H7", "vendor": "BioLegend"},
        {"marker": "CD11b", "fluorophore": "PerCP-Cy5.5", "clone": "ICRF44", "vendor": "BioLegend"},
        {"marker": "CD45RA", "fluorophore": "BV711", "clone": "HI100", "vendor": "BioLegend"},
        {"marker": "CD4", "fluorophore": "APC-Cy7", "clone": "RPA-T4", "vendor": "BioLegend"},
        {"marker": "CD8", "fluorophore": "Alexa Fluor 700", "clone": "SK1", "vendor": "BioLegend"},
    ],
)

# =============================================================================
# OMIP-069: 40-color Full Spectrum Panel
# Reference: Park et al., Cytometry A 2020
# =============================================================================
OMIP_069 = OMIPPanel(
    omip_number=69,
    name="OMIP-069: 40-Color Full Spectrum",
    description="40-color full spectrum flow cytometry panel for deep immunophenotyping",
    doi="10.1002/cyto.a.24213",
    url="https://doi.org/10.1002/cyto.a.24213",
    year=2020,
    authors="Park LM, Lannigan J, Jaimes MC",
    instrument="Cytek Aurora (5-laser spectral: 355, 405, 488, 561, 640nm)",
    cell_types=[
        "CD4+ T cells (naive, memory, effector)",
        "CD8+ T cells (naive, memory, effector)",
        "Regulatory T cells",
        "γδ T cells",
        "NKT-like cells",
        "B cells (naive, memory, plasmablasts)",
        "NK cells (early, mature, terminal)",
        "Monocytes (classical, intermediate, non-classical)",
        "Dendritic cells",
        "Basophils",
        "Innate lymphoid cells",
    ],
    markers=[
        # T cell core
        {"name": "CD3", "expression": "high", "notes": "Pan T-cell"},
        {"name": "CD4", "expression": "high", "notes": "Helper T"},
        {"name": "CD8", "expression": "high", "notes": "Cytotoxic T"},
        {"name": "TCR-gd", "expression": "medium", "notes": "Gamma-delta T"},
        # T cell memory/activation
        {"name": "CD45RA", "expression": "high", "notes": "Naive marker"},
        {"name": "CD45RO", "expression": "high", "notes": "Memory marker"},
        {"name": "CCR7", "expression": "medium", "notes": "Central memory"},
        {"name": "CD27", "expression": "medium", "notes": "Memory/naive"},
        {"name": "CD28", "expression": "medium", "notes": "Costimulation"},
        {"name": "CD57", "expression": "medium", "notes": "Terminal differentiation"},
        {"name": "CD95", "expression": "medium", "notes": "Fas"},
        # Tregs
        {"name": "CD25", "expression": "medium", "notes": "IL-2R alpha"},
        {"name": "CD127", "expression": "medium", "notes": "IL-7R"},
        {"name": "CD39", "expression": "low", "notes": "Treg functional marker"},
        # Th subsets / chemokine receptors
        {"name": "CXCR3", "expression": "low", "notes": "Th1"},
        {"name": "CCR4", "expression": "low", "notes": "Th2/Treg"},
        {"name": "CCR6", "expression": "low", "notes": "Th17"},
        {"name": "CXCR5", "expression": "low", "notes": "Tfh"},
        # B cells
        {"name": "CD19", "expression": "high", "notes": "Pan B-cell"},
        {"name": "CD20", "expression": "high", "notes": "B cell"},
        {"name": "IgD", "expression": "medium", "notes": "Naive B"},
        {"name": "CD38", "expression": "medium", "notes": "Plasmablasts"},
        # NK cells
        {"name": "CD56", "expression": "high", "notes": "NK/NKT"},
        {"name": "CD16", "expression": "medium", "notes": "NK cytotoxicity"},
        {"name": "NKG2D", "expression": "low", "notes": "NK activation"},
        # Myeloid
        {"name": "CD14", "expression": "high", "notes": "Classical monocytes"},
        {"name": "HLA-DR", "expression": "high", "notes": "APCs"},
        {"name": "CD11c", "expression": "high", "notes": "DCs/monocytes"},
        {"name": "CD123", "expression": "medium", "notes": "pDCs/basophils"},
        # General
        {"name": "CD45", "expression": "high", "notes": "Pan-leukocyte"},
        {"name": "Viability", "expression": "high", "notes": "Dead cell exclusion"},
    ],
    published_assignments=[
        # These are from the published OMIP-069 paper
        {"marker": "CD45", "fluorophore": "BUV395", "clone": "HI30", "vendor": "BD"},
        {"marker": "Viability", "fluorophore": "LIVE/DEAD Blue", "clone": "N/A", "vendor": "Thermo"},
        {"marker": "CD3", "fluorophore": "BUV496", "clone": "UCHT1", "vendor": "BD"},
        {"marker": "CD4", "fluorophore": "BUV563", "clone": "SK3", "vendor": "BD"},
        {"marker": "CD8", "fluorophore": "BUV805", "clone": "SK1", "vendor": "BD"},
        {"marker": "TCR-gd", "fluorophore": "BV421", "clone": "B1", "vendor": "BioLegend"},
        {"marker": "CD45RA", "fluorophore": "BV570", "clone": "HI100", "vendor": "BioLegend"},
        {"marker": "CD45RO", "fluorophore": "BV650", "clone": "UCHL1", "vendor": "BD"},
        {"marker": "CCR7", "fluorophore": "BV785", "clone": "G043H7", "vendor": "BioLegend"},
        {"marker": "CD27", "fluorophore": "BV510", "clone": "O323", "vendor": "BioLegend"},
        {"marker": "CD28", "fluorophore": "BB515", "clone": "CD28.2", "vendor": "BD"},
        {"marker": "CD57", "fluorophore": "FITC", "clone": "HNK-1", "vendor": "BD"},
        {"marker": "CD95", "fluorophore": "PE-Cy5", "clone": "DX2", "vendor": "BioLegend"},
        {"marker": "CD25", "fluorophore": "PE-Cy7", "clone": "M-A251", "vendor": "BD"},
        {"marker": "CD127", "fluorophore": "PE-Dazzle 594", "clone": "A019D5", "vendor": "BioLegend"},
        {"marker": "CD39", "fluorophore": "PE-Cy5.5", "clone": "eBioA1", "vendor": "Thermo"},
        {"marker": "CXCR3", "fluorophore": "BV711", "clone": "G025H7", "vendor": "BioLegend"},
        {"marker": "CCR4", "fluorophore": "PE", "clone": "1G1", "vendor": "BD"},
        {"marker": "CCR6", "fluorophore": "BV605", "clone": "G034E3", "vendor": "BioLegend"},
        {"marker": "CXCR5", "fluorophore": "BV750", "clone": "J252D4", "vendor": "BioLegend"},
        {"marker": "CD19", "fluorophore": "BUV661", "clone": "SJ25C1", "vendor": "BD"},
        {"marker": "CD20", "fluorophore": "APC-R700", "clone": "2H7", "vendor": "BD"},
        {"marker": "IgD", "fluorophore": "PerCP-Cy5.5", "clone": "IA6-2", "vendor": "BioLegend"},
        {"marker": "CD38", "fluorophore": "APC-Fire750", "clone": "HB-7", "vendor": "BioLegend"},
        {"marker": "CD56", "fluorophore": "BUV737", "clone": "NCAM16.2", "vendor": "BD"},
        {"marker": "CD16", "fluorophore": "APC-Cy7", "clone": "3G8", "vendor": "BioLegend"},
        {"marker": "NKG2D", "fluorophore": "APC", "clone": "1D11", "vendor": "BioLegend"},
        {"marker": "CD14", "fluorophore": "BUV615", "clone": "M5E2", "vendor": "BD"},
        {"marker": "HLA-DR", "fluorophore": "Alexa Fluor 700", "clone": "L243", "vendor": "BioLegend"},
        {"marker": "CD11c", "fluorophore": "PE-CF594", "clone": "B-ly6", "vendor": "BD"},
        {"marker": "CD123", "fluorophore": "Super Bright 436", "clone": "6H6", "vendor": "Thermo"},
    ],
    notes="First 40-color fluorescent panel published. Requires spectral unmixing.",
)

# =============================================================================
# OMIP-047: B Cell Phenotyping (16-color)
# Reference: Liechti et al., Cytometry A 2018
# =============================================================================
OMIP_047 = OMIPPanel(
    omip_number=47,
    name="OMIP-047: B Cell Phenotyping",
    description="High-dimensional phenotypic characterization of B cells",
    doi="10.1002/cyto.a.23488",
    url="https://doi.org/10.1002/cyto.a.23488",
    year=2018,
    authors="Liechti T, Gunthard HF, Trkola A",
    instrument="BD LSRFortessa (4-laser: 405, 488, 561, 640nm)",
    cell_types=[
        "Naive B cells",
        "Memory B cells (switched/unswitched)",
        "Marginal zone-like B cells",
        "Plasmablasts",
        "Transitional B cells",
    ],
    markers=[
        {"name": "Viability", "expression": "high", "notes": "Dead cell exclusion"},
        {"name": "CD3", "expression": "high", "notes": "T cell exclusion"},
        {"name": "CD14", "expression": "high", "notes": "Monocyte exclusion"},
        {"name": "CD19", "expression": "high", "notes": "Pan B-cell"},
        {"name": "CD20", "expression": "high", "notes": "B cell"},
        {"name": "CD27", "expression": "medium", "notes": "Memory B"},
        {"name": "IgD", "expression": "medium", "notes": "Naive/MZ B"},
        {"name": "CD38", "expression": "medium", "notes": "Plasmablasts"},
        {"name": "CD21", "expression": "medium", "notes": "Complement receptor"},
        {"name": "CD10", "expression": "low", "notes": "Transitional B"},
        {"name": "IgG", "expression": "medium", "notes": "Class-switched"},
        {"name": "IgA", "expression": "medium", "notes": "Class-switched"},
        {"name": "CXCR3", "expression": "low", "notes": "Chemokine receptor"},
        {"name": "CCR7", "expression": "low", "notes": "Chemokine receptor"},
        {"name": "IL-21R", "expression": "low", "notes": "Cytokine receptor"},
        {"name": "Ki67", "expression": "low", "notes": "Proliferation", "intracellular": True},
    ],
    published_assignments=[
        {"marker": "Viability", "fluorophore": "LIVE/DEAD Aqua", "clone": "N/A", "vendor": "Thermo"},
        {"marker": "CD3", "fluorophore": "BV421", "clone": "UCHT1", "vendor": "BioLegend"},
        {"marker": "CD14", "fluorophore": "BV421", "clone": "M5E2", "vendor": "BioLegend"},
        {"marker": "CD19", "fluorophore": "BV785", "clone": "HIB19", "vendor": "BioLegend"},
        {"marker": "CD20", "fluorophore": "APC-Cy7", "clone": "2H7", "vendor": "BioLegend"},
        {"marker": "CD27", "fluorophore": "BV605", "clone": "O323", "vendor": "BioLegend"},
        {"marker": "IgD", "fluorophore": "PE-Cy7", "clone": "IA6-2", "vendor": "BioLegend"},
        {"marker": "CD38", "fluorophore": "BB515", "clone": "HIT2", "vendor": "BD"},
        {"marker": "CD21", "fluorophore": "BV711", "clone": "Bu32", "vendor": "BioLegend"},
        {"marker": "CD10", "fluorophore": "PE", "clone": "HI10a", "vendor": "BioLegend"},
        {"marker": "IgG", "fluorophore": "PerCP-Cy5.5", "clone": "G18-145", "vendor": "BD"},
        {"marker": "IgA", "fluorophore": "APC", "clone": "IS11-8E10", "vendor": "Miltenyi"},
        {"marker": "CXCR3", "fluorophore": "FITC", "clone": "G025H7", "vendor": "BioLegend"},
        {"marker": "CCR7", "fluorophore": "PE-Dazzle 594", "clone": "G043H7", "vendor": "BioLegend"},
        {"marker": "IL-21R", "fluorophore": "BV510", "clone": "2G1-K12", "vendor": "BioLegend"},
        {"marker": "Ki67", "fluorophore": "Alexa Fluor 700", "clone": "Ki-67", "vendor": "BioLegend"},
    ],
)

# =============================================================================
# OMIP-063: Broad Human Immunophenotyping (28-color)
# Reference: Payne K et al., Cytometry A 2020
# =============================================================================
OMIP_063 = OMIPPanel(
    omip_number=63,
    name="OMIP-063: Broad Immunophenotyping",
    description="28-color flow cytometry panel for broad human immunophenotyping",
    doi="10.1002/cyto.a.24018",
    url="https://doi.org/10.1002/cyto.a.24018",
    year=2020,
    authors="Payne K, et al.",
    instrument="BD FACSymphony A5 (5-laser: 355, 405, 488, 561, 640nm)",
    cell_types=[
        "T cells (CD4, CD8, naive, memory, effector)",
        "B cells",
        "NK cells",
        "Monocytes",
        "Dendritic cells",
        "Granulocytes",
    ],
    markers=[
        {"name": "Viability", "expression": "high", "notes": "Dead cell exclusion"},
        {"name": "CD45", "expression": "high", "notes": "Pan-leukocyte"},
        {"name": "CD3", "expression": "high", "notes": "T cells"},
        {"name": "CD4", "expression": "high", "notes": "Helper T"},
        {"name": "CD8", "expression": "high", "notes": "Cytotoxic T"},
        {"name": "CD19", "expression": "high", "notes": "B cells"},
        {"name": "CD56", "expression": "high", "notes": "NK cells"},
        {"name": "CD14", "expression": "high", "notes": "Monocytes"},
        {"name": "CD16", "expression": "medium", "notes": "NK/monocyte subsets"},
        {"name": "HLA-DR", "expression": "high", "notes": "APCs"},
        {"name": "CD45RA", "expression": "high", "notes": "Naive T"},
        {"name": "CCR7", "expression": "medium", "notes": "Central memory"},
        {"name": "CD27", "expression": "medium", "notes": "Memory"},
        {"name": "CD28", "expression": "medium", "notes": "Costimulation"},
        {"name": "CD57", "expression": "medium", "notes": "Senescence"},
        {"name": "CD127", "expression": "medium", "notes": "IL-7R"},
        {"name": "CD25", "expression": "medium", "notes": "IL-2R/Tregs"},
        {"name": "CD38", "expression": "medium", "notes": "Activation"},
        {"name": "CD11c", "expression": "high", "notes": "DCs"},
        {"name": "CD123", "expression": "medium", "notes": "pDCs/basophils"},
    ],
    published_assignments=[
        {"marker": "Viability", "fluorophore": "LIVE/DEAD Blue", "clone": "N/A", "vendor": "Thermo"},
        {"marker": "CD45", "fluorophore": "BUV395", "clone": "HI30", "vendor": "BD"},
        {"marker": "CD3", "fluorophore": "BUV496", "clone": "UCHT1", "vendor": "BD"},
        {"marker": "CD4", "fluorophore": "BV750", "clone": "SK3", "vendor": "BD"},
        {"marker": "CD8", "fluorophore": "BUV805", "clone": "SK1", "vendor": "BD"},
        {"marker": "CD19", "fluorophore": "BV480", "clone": "HIB19", "vendor": "BD"},
        {"marker": "CD56", "fluorophore": "BV605", "clone": "NCAM16.2", "vendor": "BD"},
        {"marker": "CD14", "fluorophore": "BV650", "clone": "M5E2", "vendor": "BD"},
        {"marker": "CD16", "fluorophore": "APC-Cy7", "clone": "3G8", "vendor": "BioLegend"},
        {"marker": "HLA-DR", "fluorophore": "BV785", "clone": "L243", "vendor": "BioLegend"},
        {"marker": "CD45RA", "fluorophore": "BV510", "clone": "HI100", "vendor": "BD"},
        {"marker": "CCR7", "fluorophore": "PE", "clone": "G043H7", "vendor": "BioLegend"},
        {"marker": "CD27", "fluorophore": "BV421", "clone": "O323", "vendor": "BioLegend"},
        {"marker": "CD28", "fluorophore": "BB515", "clone": "CD28.2", "vendor": "BD"},
        {"marker": "CD57", "fluorophore": "FITC", "clone": "HNK-1", "vendor": "BD"},
        {"marker": "CD127", "fluorophore": "PE-Cy7", "clone": "A019D5", "vendor": "BioLegend"},
        {"marker": "CD25", "fluorophore": "PE-Dazzle 594", "clone": "M-A251", "vendor": "BD"},
        {"marker": "CD38", "fluorophore": "PerCP-Cy5.5", "clone": "HIT2", "vendor": "BioLegend"},
        {"marker": "CD11c", "fluorophore": "APC", "clone": "Bu15", "vendor": "BioLegend"},
        {"marker": "CD123", "fluorophore": "BV711", "clone": "6H6", "vendor": "BioLegend"},
    ],
)


# =============================================================================
# OMIP-008: NK Cell Panel (10-color)
# Reference: Villanueva et al., Cytometry A 2014
# =============================================================================
OMIP_008 = OMIPPanel(
    omip_number=8,
    name="OMIP-008: NK Cell Panel",
    description="10-color panel for comprehensive NK cell phenotyping",
    doi="10.1002/cyto.a.22502",
    url="https://doi.org/10.1002/cyto.a.22502",
    year=2014,
    authors="Villanueva J, et al.",
    instrument="BD FACSCanto II (3-laser: 405, 488, 640nm)",
    cell_types=[
        "CD56bright NK cells",
        "CD56dim NK cells",
        "Mature NK cells",
        "Immature NK cells",
        "Cytotoxic NK subsets",
    ],
    markers=[
        {"name": "Viability", "expression": "high", "notes": "Dead cell exclusion"},
        {"name": "CD3", "expression": "high", "notes": "T cell exclusion"},
        {"name": "CD56", "expression": "high", "notes": "NK cell identification"},
        {"name": "CD16", "expression": "medium", "notes": "NK maturation/cytotoxicity"},
        {"name": "NKG2A", "expression": "medium", "notes": "Inhibitory receptor"},
        {"name": "NKG2D", "expression": "medium", "notes": "Activating receptor"},
        {"name": "CD57", "expression": "medium", "notes": "Terminal differentiation"},
        {"name": "CD94", "expression": "medium", "notes": "Heterodimeric receptor"},
        {"name": "CD158a", "expression": "low", "notes": "KIR2DL1"},
        {"name": "CD158b", "expression": "low", "notes": "KIR2DL2/3"},
    ],
    published_assignments=[
        {"marker": "Viability", "fluorophore": "LIVE/DEAD Aqua", "clone": "N/A", "vendor": "Thermo"},
        {"marker": "CD3", "fluorophore": "Pacific Blue", "clone": "UCHT1", "vendor": "BD"},
        {"marker": "CD56", "fluorophore": "PE-Cy7", "clone": "B159", "vendor": "BD"},
        {"marker": "CD16", "fluorophore": "FITC", "clone": "3G8", "vendor": "BD"},
        {"marker": "NKG2A", "fluorophore": "PE", "clone": "Z199", "vendor": "Beckman"},
        {"marker": "NKG2D", "fluorophore": "APC", "clone": "1D11", "vendor": "BioLegend"},
        {"marker": "CD57", "fluorophore": "BV421", "clone": "HNK-1", "vendor": "BD"},
        {"marker": "CD94", "fluorophore": "PerCP-Cy5.5", "clone": "HP-3D9", "vendor": "BD"},
        {"marker": "CD158a", "fluorophore": "APC-Cy7", "clone": "HP-3E4", "vendor": "BD"},
        {"marker": "CD158b", "fluorophore": "PE-Cy5", "clone": "CH-L", "vendor": "BD"},
    ],
)

# =============================================================================
# OMIP-020: Regulatory T Cell Panel (12-color)
# Reference: Santegoets et al., Cytometry A 2015
# =============================================================================
OMIP_020 = OMIPPanel(
    omip_number=20,
    name="OMIP-020: Regulatory T Cells",
    description="12-color panel for identification and phenotyping of regulatory T cells",
    doi="10.1002/cyto.a.22586",
    url="https://doi.org/10.1002/cyto.a.22586",
    year=2015,
    authors="Santegoets SJ, et al.",
    instrument="BD LSRFortessa (4-laser: 405, 488, 561, 640nm)",
    cell_types=[
        "Natural Tregs (nTreg)",
        "Induced Tregs (iTreg)",
        "Memory Tregs",
        "Naive Tregs",
        "Activated Tregs",
    ],
    markers=[
        {"name": "Viability", "expression": "high", "notes": "Dead cell exclusion"},
        {"name": "CD3", "expression": "high", "notes": "T cell identification"},
        {"name": "CD4", "expression": "high", "notes": "Helper T cells"},
        {"name": "CD25", "expression": "medium", "notes": "IL-2R, Treg marker"},
        {"name": "CD127", "expression": "medium", "notes": "IL-7R, low on Tregs"},
        {"name": "FOXP3", "expression": "medium", "notes": "Treg transcription factor", "intracellular": True},
        {"name": "CD45RA", "expression": "high", "notes": "Naive Tregs"},
        {"name": "CTLA-4", "expression": "low", "notes": "Suppressive function", "intracellular": True},
        {"name": "ICOS", "expression": "low", "notes": "Activated Tregs"},
        {"name": "CD39", "expression": "low", "notes": "Effector Treg marker"},
        {"name": "Helios", "expression": "medium", "notes": "nTreg vs iTreg", "intracellular": True},
        {"name": "Ki67", "expression": "low", "notes": "Proliferating Tregs", "intracellular": True},
    ],
    published_assignments=[
        {"marker": "Viability", "fluorophore": "LIVE/DEAD Blue", "clone": "N/A", "vendor": "Thermo"},
        {"marker": "CD3", "fluorophore": "BV510", "clone": "UCHT1", "vendor": "BioLegend"},
        {"marker": "CD4", "fluorophore": "PerCP-Cy5.5", "clone": "OKT4", "vendor": "BioLegend"},
        {"marker": "CD25", "fluorophore": "PE-Cy7", "clone": "M-A251", "vendor": "BD"},
        {"marker": "CD127", "fluorophore": "BV421", "clone": "A019D5", "vendor": "BioLegend"},
        {"marker": "FOXP3", "fluorophore": "PE", "clone": "259D/C7", "vendor": "BD"},
        {"marker": "CD45RA", "fluorophore": "FITC", "clone": "HI100", "vendor": "BD"},
        {"marker": "CTLA-4", "fluorophore": "APC", "clone": "BNI3", "vendor": "BD"},
        {"marker": "ICOS", "fluorophore": "Alexa Fluor 700", "clone": "C398.4A", "vendor": "BioLegend"},
        {"marker": "CD39", "fluorophore": "BV605", "clone": "A1", "vendor": "BioLegend"},
        {"marker": "Helios", "fluorophore": "Pacific Blue", "clone": "22F6", "vendor": "BioLegend"},
        {"marker": "Ki67", "fluorophore": "APC-Cy7", "clone": "B56", "vendor": "BD"},
    ],
)

# =============================================================================
# OMIP-041: Myeloid Cell Panel (28-color)
# Reference: Dutertre et al., Cytometry A 2017
# =============================================================================
OMIP_041 = OMIPPanel(
    omip_number=41,
    name="OMIP-041: Myeloid Cells",
    description="28-color panel for comprehensive myeloid cell phenotyping",
    doi="10.1002/cyto.a.23293",
    url="https://doi.org/10.1002/cyto.a.23293",
    year=2017,
    authors="Dutertre CA, et al.",
    instrument="BD FACSymphony (5-laser: 355, 405, 488, 561, 640nm)",
    cell_types=[
        "Classical monocytes (CD14+CD16-)",
        "Intermediate monocytes (CD14+CD16+)",
        "Non-classical monocytes (CD14dimCD16+)",
        "cDC1 (CD141+)",
        "cDC2 (CD1c+)",
        "Plasmacytoid DCs",
        "Granulocytes",
    ],
    markers=[
        {"name": "Viability", "expression": "high", "notes": "Dead cell exclusion"},
        {"name": "CD45", "expression": "high", "notes": "Pan-leukocyte"},
        {"name": "HLA-DR", "expression": "high", "notes": "APC marker"},
        {"name": "CD14", "expression": "high", "notes": "Classical monocyte"},
        {"name": "CD16", "expression": "medium", "notes": "Non-classical monocyte"},
        {"name": "CD11c", "expression": "high", "notes": "DC marker"},
        {"name": "CD11b", "expression": "high", "notes": "Integrin"},
        {"name": "CD123", "expression": "medium", "notes": "pDC marker"},
        {"name": "CD141", "expression": "low", "notes": "cDC1 marker"},
        {"name": "CD1c", "expression": "medium", "notes": "cDC2 marker"},
        {"name": "CD163", "expression": "medium", "notes": "M2 macrophage"},
        {"name": "CD206", "expression": "medium", "notes": "Mannose receptor"},
        {"name": "CD64", "expression": "medium", "notes": "FcγRI"},
        {"name": "CD32", "expression": "medium", "notes": "FcγRII"},
        {"name": "Siglec-1", "expression": "low", "notes": "CD169"},
        {"name": "CX3CR1", "expression": "low", "notes": "Fractalkine receptor"},
    ],
    published_assignments=[
        {"marker": "Viability", "fluorophore": "LIVE/DEAD Blue", "clone": "N/A", "vendor": "Thermo"},
        {"marker": "CD45", "fluorophore": "BUV395", "clone": "HI30", "vendor": "BD"},
        {"marker": "HLA-DR", "fluorophore": "BV785", "clone": "L243", "vendor": "BioLegend"},
        {"marker": "CD14", "fluorophore": "BUV737", "clone": "M5E2", "vendor": "BD"},
        {"marker": "CD16", "fluorophore": "BV605", "clone": "3G8", "vendor": "BioLegend"},
        {"marker": "CD11c", "fluorophore": "PE-Cy7", "clone": "Bu15", "vendor": "BioLegend"},
        {"marker": "CD11b", "fluorophore": "PerCP-Cy5.5", "clone": "ICRF44", "vendor": "BioLegend"},
        {"marker": "CD123", "fluorophore": "BV650", "clone": "6H6", "vendor": "BioLegend"},
        {"marker": "CD141", "fluorophore": "APC", "clone": "M80", "vendor": "BioLegend"},
        {"marker": "CD1c", "fluorophore": "PE-Dazzle 594", "clone": "L161", "vendor": "BioLegend"},
        {"marker": "CD163", "fluorophore": "BV421", "clone": "GHI/61", "vendor": "BioLegend"},
        {"marker": "CD206", "fluorophore": "BB515", "clone": "15-2", "vendor": "BD"},
        {"marker": "CD64", "fluorophore": "PE", "clone": "10.1", "vendor": "BioLegend"},
        {"marker": "CD32", "fluorophore": "FITC", "clone": "FUN-2", "vendor": "BioLegend"},
        {"marker": "Siglec-1", "fluorophore": "APC-Cy7", "clone": "7-239", "vendor": "BioLegend"},
        {"marker": "CX3CR1", "fluorophore": "BV711", "clone": "2A9-1", "vendor": "BioLegend"},
    ],
)

# =============================================================================
# OMIP-052: Innate Lymphoid Cells (ILC) Panel (18-color)
# Reference: Simoni et al., Cytometry A 2018
# =============================================================================
OMIP_052 = OMIPPanel(
    omip_number=52,
    name="OMIP-052: Innate Lymphoid Cells",
    description="18-color panel for comprehensive ILC phenotyping",
    doi="10.1002/cyto.a.23496",
    url="https://doi.org/10.1002/cyto.a.23496",
    year=2018,
    authors="Simoni Y, et al.",
    instrument="BD LSRFortessa X-20 (5-laser: 355, 405, 488, 561, 640nm)",
    cell_types=[
        "ILC1",
        "ILC2",
        "ILC3 (NCR+ and NCR-)",
        "NK cells",
        "LTi-like cells",
    ],
    markers=[
        {"name": "Viability", "expression": "high", "notes": "Dead cell exclusion"},
        {"name": "CD45", "expression": "high", "notes": "Pan-leukocyte"},
        {"name": "CD3", "expression": "high", "notes": "T cell exclusion"},
        {"name": "CD19", "expression": "high", "notes": "B cell exclusion"},
        {"name": "CD14", "expression": "high", "notes": "Monocyte exclusion"},
        {"name": "CD127", "expression": "medium", "notes": "ILC marker"},
        {"name": "CD56", "expression": "medium", "notes": "NK/ILC"},
        {"name": "CD117", "expression": "medium", "notes": "c-Kit, ILC2/ILC3"},
        {"name": "CRTH2", "expression": "medium", "notes": "ILC2 marker"},
        {"name": "CD161", "expression": "medium", "notes": "ILC/NK"},
        {"name": "NKp44", "expression": "low", "notes": "NCR+ ILC3"},
        {"name": "NKp46", "expression": "low", "notes": "NK receptor"},
        {"name": "CD294", "expression": "low", "notes": "CRTH2/ILC2"},
        {"name": "CD336", "expression": "low", "notes": "NKp44"},
        {"name": "T-bet", "expression": "medium", "notes": "ILC1 TF", "intracellular": True},
        {"name": "GATA3", "expression": "medium", "notes": "ILC2 TF", "intracellular": True},
        {"name": "RORgt", "expression": "medium", "notes": "ILC3 TF", "intracellular": True},
        {"name": "Eomes", "expression": "medium", "notes": "NK TF", "intracellular": True},
    ],
    published_assignments=[
        {"marker": "Viability", "fluorophore": "LIVE/DEAD Aqua", "clone": "N/A", "vendor": "Thermo"},
        {"marker": "CD45", "fluorophore": "BUV395", "clone": "HI30", "vendor": "BD"},
        {"marker": "CD3", "fluorophore": "BV510", "clone": "UCHT1", "vendor": "BioLegend"},
        {"marker": "CD19", "fluorophore": "BV510", "clone": "HIB19", "vendor": "BioLegend"},
        {"marker": "CD14", "fluorophore": "BV510", "clone": "M5E2", "vendor": "BioLegend"},
        {"marker": "CD127", "fluorophore": "PE-Cy7", "clone": "A019D5", "vendor": "BioLegend"},
        {"marker": "CD56", "fluorophore": "BV605", "clone": "HCD56", "vendor": "BioLegend"},
        {"marker": "CD117", "fluorophore": "BV421", "clone": "104D2", "vendor": "BioLegend"},
        {"marker": "CRTH2", "fluorophore": "PE", "clone": "BM16", "vendor": "BioLegend"},
        {"marker": "CD161", "fluorophore": "BV711", "clone": "HP-3G10", "vendor": "BioLegend"},
        {"marker": "NKp44", "fluorophore": "PerCP-Cy5.5", "clone": "P44-8", "vendor": "BioLegend"},
        {"marker": "NKp46", "fluorophore": "FITC", "clone": "9E2", "vendor": "BioLegend"},
        {"marker": "CD294", "fluorophore": "APC", "clone": "BM16", "vendor": "BioLegend"},
        {"marker": "CD336", "fluorophore": "APC-Cy7", "clone": "P44-8", "vendor": "BD"},
        {"marker": "T-bet", "fluorophore": "PE-Dazzle 594", "clone": "4B10", "vendor": "BioLegend"},
        {"marker": "GATA3", "fluorophore": "BV785", "clone": "16E10A23", "vendor": "BioLegend"},
        {"marker": "RORgt", "fluorophore": "Alexa Fluor 647", "clone": "Q21-559", "vendor": "BD"},
        {"marker": "Eomes", "fluorophore": "Alexa Fluor 700", "clone": "WD1928", "vendor": "Thermo"},
    ],
)

# =============================================================================
# OMIP-077: 30-color T Cell Panel
# Reference: Ruhland et al., Cytometry A 2021
# =============================================================================
OMIP_077 = OMIPPanel(
    omip_number=77,
    name="OMIP-077: 30-Color T Cell Panel",
    description="30-color spectral flow cytometry panel for comprehensive T cell phenotyping",
    doi="10.1002/cyto.a.24333",
    url="https://doi.org/10.1002/cyto.a.24333",
    year=2021,
    authors="Ruhland MK, et al.",
    instrument="Cytek Aurora (5-laser spectral: 355, 405, 488, 561, 640nm)",
    cell_types=[
        "CD4+ T cells (naive, Tcm, Tem, Temra)",
        "CD8+ T cells (naive, Tcm, Tem, Temra)",
        "Regulatory T cells",
        "Th1, Th2, Th17, Tfh subsets",
        "Exhausted/senescent T cells",
    ],
    markers=[
        {"name": "Viability", "expression": "high", "notes": "Dead cell exclusion"},
        {"name": "CD45", "expression": "high", "notes": "Pan-leukocyte"},
        {"name": "CD3", "expression": "high", "notes": "Pan T-cell"},
        {"name": "CD4", "expression": "high", "notes": "Helper T cells"},
        {"name": "CD8", "expression": "high", "notes": "Cytotoxic T cells"},
        {"name": "CD45RA", "expression": "high", "notes": "Naive marker"},
        {"name": "CD45RO", "expression": "high", "notes": "Memory marker"},
        {"name": "CCR7", "expression": "medium", "notes": "Central memory"},
        {"name": "CD27", "expression": "medium", "notes": "Costimulation"},
        {"name": "CD28", "expression": "medium", "notes": "Costimulation"},
        {"name": "CD57", "expression": "medium", "notes": "Senescence"},
        {"name": "CD25", "expression": "medium", "notes": "IL-2R/Tregs"},
        {"name": "CD127", "expression": "medium", "notes": "IL-7R"},
        {"name": "CXCR3", "expression": "low", "notes": "Th1"},
        {"name": "CCR4", "expression": "low", "notes": "Th2/Treg"},
        {"name": "CCR6", "expression": "low", "notes": "Th17"},
        {"name": "CXCR5", "expression": "low", "notes": "Tfh"},
        {"name": "PD-1", "expression": "medium", "notes": "Exhaustion"},
        {"name": "TIM-3", "expression": "low", "notes": "Exhaustion"},
        {"name": "LAG-3", "expression": "low", "notes": "Exhaustion"},
    ],
    published_assignments=[
        {"marker": "Viability", "fluorophore": "LIVE/DEAD Blue", "clone": "N/A", "vendor": "Thermo"},
        {"marker": "CD45", "fluorophore": "BUV395", "clone": "HI30", "vendor": "BD"},
        {"marker": "CD3", "fluorophore": "BUV496", "clone": "UCHT1", "vendor": "BD"},
        {"marker": "CD4", "fluorophore": "BUV563", "clone": "SK3", "vendor": "BD"},
        {"marker": "CD8", "fluorophore": "BUV805", "clone": "SK1", "vendor": "BD"},
        {"marker": "CD45RA", "fluorophore": "BV570", "clone": "HI100", "vendor": "BioLegend"},
        {"marker": "CD45RO", "fluorophore": "BV650", "clone": "UCHL1", "vendor": "BD"},
        {"marker": "CCR7", "fluorophore": "BV785", "clone": "G043H7", "vendor": "BioLegend"},
        {"marker": "CD27", "fluorophore": "BV510", "clone": "O323", "vendor": "BioLegend"},
        {"marker": "CD28", "fluorophore": "BB515", "clone": "CD28.2", "vendor": "BD"},
        {"marker": "CD57", "fluorophore": "FITC", "clone": "HNK-1", "vendor": "BD"},
        {"marker": "CD25", "fluorophore": "PE-Cy7", "clone": "M-A251", "vendor": "BD"},
        {"marker": "CD127", "fluorophore": "PE-Dazzle 594", "clone": "A019D5", "vendor": "BioLegend"},
        {"marker": "CXCR3", "fluorophore": "BV711", "clone": "G025H7", "vendor": "BioLegend"},
        {"marker": "CCR4", "fluorophore": "PE", "clone": "1G1", "vendor": "BD"},
        {"marker": "CCR6", "fluorophore": "BV605", "clone": "G034E3", "vendor": "BioLegend"},
        {"marker": "CXCR5", "fluorophore": "BV421", "clone": "J252D4", "vendor": "BioLegend"},
        {"marker": "PD-1", "fluorophore": "APC", "clone": "EH12.2H7", "vendor": "BioLegend"},
        {"marker": "TIM-3", "fluorophore": "APC-Cy7", "clone": "F38-2E2", "vendor": "BioLegend"},
        {"marker": "LAG-3", "fluorophore": "Alexa Fluor 700", "clone": "11C3C65", "vendor": "BioLegend"},
    ],
)

# Collection of all panels
ALL_OMIP_PANELS = {
    8: OMIP_008,
    20: OMIP_020,
    30: OMIP_030,
    41: OMIP_041,
    44: OMIP_044,
    47: OMIP_047,
    52: OMIP_052,
    63: OMIP_063,
    69: OMIP_069,
    77: OMIP_077,
}


def get_omip_panel(number: int) -> OMIPPanel:
    """Get an OMIP panel by number."""
    if number not in ALL_OMIP_PANELS:
        raise ValueError(f"OMIP-{number:03d} not found. Available: {list(ALL_OMIP_PANELS.keys())}")
    return ALL_OMIP_PANELS[number]


def get_markers_only(panel: OMIPPanel) -> list[dict]:
    """Extract just the markers (no fluorophores) for testing."""
    return panel.markers


def get_published_assignments(panel: OMIPPanel) -> dict[str, str]:
    """Get marker -> fluorophore mapping from published panel."""
    return {a["marker"]: a["fluorophore"] for a in panel.published_assignments}
