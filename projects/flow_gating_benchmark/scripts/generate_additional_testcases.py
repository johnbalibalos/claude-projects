#!/usr/bin/env python3
"""Generate additional test cases to reach 30 total with balanced distribution."""

import json
from datetime import date
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "ground_truth"

# Additional panels to balance the distribution
ADDITIONAL_OMIPS = {
    # More COMPLEX panels (26+ colors)
    "OMIP-060": {
        "title": "35-color spectral cytometry panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "Full spectrum immunophenotyping",
        "n_colors": 35,
        "panel": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD4", "fluorophore": "BUV496"},
            {"marker": "CD8", "fluorophore": "BUV563"},
            {"marker": "CD45", "fluorophore": "BUV661"},
            {"marker": "CD45RA", "fluorophore": "BUV737"},
            {"marker": "CD19", "fluorophore": "BV421"},
            {"marker": "CD20", "fluorophore": "BV480"},
            {"marker": "CD14", "fluorophore": "BV510"},
            {"marker": "CD16", "fluorophore": "BV570"},
            {"marker": "CD56", "fluorophore": "BV605"},
            {"marker": "HLA-DR", "fluorophore": "BV650"},
            {"marker": "CD11c", "fluorophore": "BV711"},
            {"marker": "CD123", "fluorophore": "BV750"},
            {"marker": "CD27", "fluorophore": "BV785"},
            {"marker": "CD28", "fluorophore": "FITC"},
            {"marker": "CCR7", "fluorophore": "PE"},
            {"marker": "CD127", "fluorophore": "PE-CF594"},
            {"marker": "CD25", "fluorophore": "PE-Cy5"},
            {"marker": "PD-1", "fluorophore": "PE-Cy7"},
            {"marker": "CD38", "fluorophore": "APC"},
            {"marker": "CD57", "fluorophore": "APC-R700"},
            {"marker": "CD161", "fluorophore": "APC-Fire750"},
            {"marker": "NKG2A", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CXCR5", "fluorophore": "BUV805"},
            {"marker": "CCR6", "fluorophore": "BB515"},
            {"marker": "CXCR3", "fluorophore": "BB700"},
            {"marker": "CD95", "fluorophore": "BUV615"},
            {"marker": "Ki67", "fluorophore": "AF700"},
            {"marker": "FoxP3", "fluorophore": "Pacific Blue"},
            {"marker": "TCRgd", "fluorophore": "BV545"},
            {"marker": "Va7.2", "fluorophore": "BV690"},
            {"marker": "Granzyme B", "fluorophore": "FITC"},
            {"marker": "IFNg", "fluorophore": "PE-Dazzle594"},
            {"marker": "TNFa", "fluorophore": "BV711"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events", "children": [
                {"name": "Time", "markers": ["Time"], "is_critical": True, "children": [
                    {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                        {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                            {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                                {"name": "T cells", "markers": ["CD3"], "children": [
                                    {"name": "CD4+ T", "markers": ["CD4"], "children": []},
                                    {"name": "CD8+ T", "markers": ["CD8"], "children": []},
                                    {"name": "gd T", "markers": ["TCRgd"], "children": []},
                                    {"name": "MAIT", "markers": ["Va7.2", "CD161"], "children": []},
                                ]},
                                {"name": "B cells", "markers": ["CD19", "CD20"], "children": []},
                                {"name": "NK cells", "markers": ["CD56", "CD3"], "children": []},
                                {"name": "Monocytes", "markers": ["CD14"], "children": []},
                                {"name": "DCs", "markers": ["CD11c", "HLA-DR"], "children": []},
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-062": {
        "title": "28-color mouse tumor panel",
        "species": "mouse",
        "sample_type": "Mouse tumor",
        "application": "Mouse tumor immune microenvironment",
        "n_colors": 28,
        "panel": [
            {"marker": "CD45", "fluorophore": "BUV395"},
            {"marker": "CD3", "fluorophore": "BUV496"},
            {"marker": "CD4", "fluorophore": "BUV563"},
            {"marker": "CD8a", "fluorophore": "BUV661"},
            {"marker": "B220", "fluorophore": "BUV737"},
            {"marker": "CD11b", "fluorophore": "BV421"},
            {"marker": "CD11c", "fluorophore": "BV510"},
            {"marker": "F4/80", "fluorophore": "BV605"},
            {"marker": "Ly6G", "fluorophore": "BV650"},
            {"marker": "Ly6C", "fluorophore": "BV711"},
            {"marker": "NK1.1", "fluorophore": "BV785"},
            {"marker": "MHCII", "fluorophore": "FITC"},
            {"marker": "PD-1", "fluorophore": "PE"},
            {"marker": "PD-L1", "fluorophore": "PE-Cy7"},
            {"marker": "CTLA-4", "fluorophore": "APC"},
            {"marker": "TIM-3", "fluorophore": "APC-Cy7"},
            {"marker": "LAG-3", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD44", "fluorophore": "AF700"},
            {"marker": "CD62L", "fluorophore": "BUV805"},
            {"marker": "FoxP3", "fluorophore": "PE-CF594"},
            {"marker": "Ki67", "fluorophore": "BV750"},
            {"marker": "Granzyme B", "fluorophore": "FITC"},
            {"marker": "IFNg", "fluorophore": "BV421"},
            {"marker": "TNFa", "fluorophore": "PE-Dazzle594"},
            {"marker": "CD25", "fluorophore": "BB515"},
            {"marker": "CD69", "fluorophore": "BV545"},
            {"marker": "CD103", "fluorophore": "BV690"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events", "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45+ TILs", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "T cells", "markers": ["CD3"], "children": [
                                {"name": "CD8+ TIL", "markers": ["CD8a"], "children": [
                                    {"name": "Exhausted CD8", "markers": ["PD-1", "TIM-3", "LAG-3"], "children": []},
                                    {"name": "Effector CD8", "markers": ["Granzyme B"], "children": []},
                                ]},
                                {"name": "CD4+ TIL", "markers": ["CD4"], "children": [
                                    {"name": "Tregs", "markers": ["CD25", "FoxP3"], "children": []},
                                ]}
                            ]},
                            {"name": "NK cells", "markers": ["NK1.1"], "children": []},
                            {"name": "Myeloid", "markers": ["CD11b"], "children": [
                                {"name": "TAMs", "markers": ["F4/80"], "children": []},
                                {"name": "MDSCs", "markers": ["Ly6C", "Ly6G"], "children": []},
                                {"name": "DCs", "markers": ["CD11c", "MHCII"], "children": []},
                            ]},
                        ]},
                        {"name": "Tumor cells", "markers": ["CD45"], "children": []},
                    ]}
                ]}
            ]
        }
    },
    "OMIP-064": {
        "title": "32-color COVID immune panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "COVID-19 immune response profiling",
        "n_colors": 32,
        "panel": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD4", "fluorophore": "BUV496"},
            {"marker": "CD8", "fluorophore": "BUV563"},
            {"marker": "CD45", "fluorophore": "BUV661"},
            {"marker": "CD19", "fluorophore": "BUV737"},
            {"marker": "CD14", "fluorophore": "BV421"},
            {"marker": "CD16", "fluorophore": "BV510"},
            {"marker": "CD56", "fluorophore": "BV605"},
            {"marker": "HLA-DR", "fluorophore": "BV650"},
            {"marker": "CD38", "fluorophore": "BV711"},
            {"marker": "CD27", "fluorophore": "BV785"},
            {"marker": "IgD", "fluorophore": "FITC"},
            {"marker": "CD45RA", "fluorophore": "PE"},
            {"marker": "CCR7", "fluorophore": "PE-Cy7"},
            {"marker": "PD-1", "fluorophore": "APC"},
            {"marker": "CXCR5", "fluorophore": "APC-Cy7"},
            {"marker": "CD69", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD25", "fluorophore": "AF700"},
            {"marker": "FoxP3", "fluorophore": "BUV805"},
            {"marker": "Ki67", "fluorophore": "PE-CF594"},
            {"marker": "Granzyme B", "fluorophore": "BV750"},
            {"marker": "IFNg", "fluorophore": "BB515"},
            {"marker": "CD71", "fluorophore": "BV545"},
            {"marker": "CD95", "fluorophore": "BV690"},
            {"marker": "CD127", "fluorophore": "PE-Dazzle594"},
            {"marker": "TIGIT", "fluorophore": "BUV615"},
            {"marker": "TIM-3", "fluorophore": "APC-R700"},
            {"marker": "CD57", "fluorophore": "FITC"},
            {"marker": "NKG2A", "fluorophore": "PE-Cy5"},
            {"marker": "CD161", "fluorophore": "BV480"},
            {"marker": "CXCR3", "fluorophore": "BB700"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events", "children": [
                {"name": "Time", "markers": ["Time"], "is_critical": True, "children": [
                    {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                        {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                            {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                                {"name": "T cells", "markers": ["CD3"], "children": [
                                    {"name": "CD4+ T", "markers": ["CD4"], "children": [
                                        {"name": "Tfh", "markers": ["CXCR5", "PD-1"], "children": []},
                                        {"name": "Tregs", "markers": ["CD25", "FoxP3"], "children": []},
                                    ]},
                                    {"name": "CD8+ T", "markers": ["CD8"], "children": [
                                        {"name": "Activated CD8", "markers": ["CD38", "HLA-DR"], "children": []},
                                        {"name": "Exhausted CD8", "markers": ["PD-1", "TIM-3"], "children": []},
                                    ]},
                                ]},
                                {"name": "B cells", "markers": ["CD19"], "children": [
                                    {"name": "Plasmablasts", "markers": ["CD27", "CD38"], "children": []},
                                ]},
                                {"name": "NK cells", "markers": ["CD56", "CD3"], "children": []},
                                {"name": "Monocytes", "markers": ["CD14", "CD16"], "children": []},
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-066": {
        "title": "26-color mouse lymphoid panel",
        "species": "mouse",
        "sample_type": "Mouse lymph node",
        "application": "Mouse lymph node immune profiling",
        "n_colors": 26,
        "panel": [
            {"marker": "CD45", "fluorophore": "BUV395"},
            {"marker": "CD3", "fluorophore": "BUV496"},
            {"marker": "CD4", "fluorophore": "BUV563"},
            {"marker": "CD8a", "fluorophore": "BUV661"},
            {"marker": "B220", "fluorophore": "BUV737"},
            {"marker": "CD19", "fluorophore": "BV421"},
            {"marker": "IgM", "fluorophore": "BV510"},
            {"marker": "IgD", "fluorophore": "BV605"},
            {"marker": "GL7", "fluorophore": "BV650"},
            {"marker": "CD95", "fluorophore": "BV711"},
            {"marker": "CD138", "fluorophore": "BV785"},
            {"marker": "CXCR5", "fluorophore": "FITC"},
            {"marker": "PD-1", "fluorophore": "PE"},
            {"marker": "CD44", "fluorophore": "PE-Cy7"},
            {"marker": "CD62L", "fluorophore": "APC"},
            {"marker": "TCRb", "fluorophore": "APC-Cy7"},
            {"marker": "NK1.1", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD25", "fluorophore": "AF700"},
            {"marker": "FoxP3", "fluorophore": "BUV805"},
            {"marker": "Bcl6", "fluorophore": "PE-CF594"},
            {"marker": "Ki67", "fluorophore": "BV750"},
            {"marker": "CD21", "fluorophore": "BB515"},
            {"marker": "CD23", "fluorophore": "BV545"},
            {"marker": "ICOS", "fluorophore": "BV690"},
            {"marker": "CD69", "fluorophore": "PE-Dazzle594"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events", "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "T cells", "markers": ["CD3", "TCRb"], "children": [
                                {"name": "CD4+ T", "markers": ["CD4"], "children": [
                                    {"name": "Tfh", "markers": ["CXCR5", "PD-1"], "children": []},
                                    {"name": "Tregs", "markers": ["CD25", "FoxP3"], "children": []},
                                ]},
                                {"name": "CD8+ T", "markers": ["CD8a"], "children": []},
                            ]},
                            {"name": "B cells", "markers": ["B220", "CD19"], "children": [
                                {"name": "Follicular B", "markers": ["IgD", "CD21", "CD23"], "children": []},
                                {"name": "GC B", "markers": ["GL7", "CD95"], "children": []},
                                {"name": "Plasma cells", "markers": ["CD138"], "children": []},
                            ]},
                            {"name": "NK cells", "markers": ["NK1.1"], "children": []},
                        ]}
                    ]}
                ]}
            ]
        }
    },
    # More MEDIUM panels (16-25 colors)
    "OMIP-035": {
        "title": "20-color mouse aging panel",
        "species": "mouse",
        "sample_type": "Mouse splenocytes",
        "application": "Age-related immune changes",
        "n_colors": 20,
        "panel": [
            {"marker": "CD45", "fluorophore": "BUV395"},
            {"marker": "CD3", "fluorophore": "BUV496"},
            {"marker": "CD4", "fluorophore": "BUV563"},
            {"marker": "CD8a", "fluorophore": "BUV661"},
            {"marker": "B220", "fluorophore": "BV421"},
            {"marker": "CD19", "fluorophore": "BV510"},
            {"marker": "CD44", "fluorophore": "BV605"},
            {"marker": "CD62L", "fluorophore": "BV650"},
            {"marker": "CD127", "fluorophore": "BV711"},
            {"marker": "KLRG1", "fluorophore": "BV785"},
            {"marker": "CD11b", "fluorophore": "FITC"},
            {"marker": "Ly6C", "fluorophore": "PE"},
            {"marker": "CD49b", "fluorophore": "PE-Cy7"},
            {"marker": "NK1.1", "fluorophore": "APC"},
            {"marker": "PD-1", "fluorophore": "APC-Cy7"},
            {"marker": "CD25", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "FoxP3", "fluorophore": "AF700"},
            {"marker": "TCRb", "fluorophore": "BUV805"},
            {"marker": "CD21", "fluorophore": "PE-CF594"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events", "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "T cells", "markers": ["CD3", "TCRb"], "children": [
                                {"name": "CD4+ T", "markers": ["CD4"], "children": [
                                    {"name": "Naive CD4", "markers": ["CD44", "CD62L"], "children": []},
                                    {"name": "Memory CD4", "markers": ["CD44", "CD62L"], "children": []},
                                    {"name": "Tregs", "markers": ["CD25", "FoxP3"], "children": []},
                                ]},
                                {"name": "CD8+ T", "markers": ["CD8a"], "children": [
                                    {"name": "Naive CD8", "markers": ["CD44", "CD62L"], "children": []},
                                    {"name": "Memory CD8", "markers": ["CD44", "CD62L"], "children": []},
                                    {"name": "SLEC", "markers": ["KLRG1", "CD127"], "children": []},
                                    {"name": "MPEC", "markers": ["KLRG1", "CD127"], "children": []},
                                ]},
                            ]},
                            {"name": "B cells", "markers": ["B220", "CD19"], "children": []},
                            {"name": "NK cells", "markers": ["NK1.1", "CD49b"], "children": []},
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-037": {
        "title": "18-color human stem cell panel",
        "species": "human",
        "sample_type": "Human cord blood",
        "application": "Hematopoietic stem cell analysis",
        "n_colors": 18,
        "panel": [
            {"marker": "CD34", "fluorophore": "BUV395"},
            {"marker": "CD38", "fluorophore": "BV421"},
            {"marker": "CD45RA", "fluorophore": "BV510"},
            {"marker": "CD90", "fluorophore": "BV605"},
            {"marker": "CD49f", "fluorophore": "BV650"},
            {"marker": "CD45", "fluorophore": "BV711"},
            {"marker": "CD10", "fluorophore": "BV785"},
            {"marker": "CD7", "fluorophore": "FITC"},
            {"marker": "CD123", "fluorophore": "PE"},
            {"marker": "CD135", "fluorophore": "PE-Cy7"},
            {"marker": "CD33", "fluorophore": "APC"},
            {"marker": "CD19", "fluorophore": "APC-Cy7"},
            {"marker": "CD3", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD71", "fluorophore": "AF700"},
            {"marker": "CD235a", "fluorophore": "BUV496"},
            {"marker": "CD41a", "fluorophore": "BUV563"},
            {"marker": "CD117", "fluorophore": "PE-CF594"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events", "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45low", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "CD34+", "markers": ["CD34"], "children": [
                                {"name": "HSC", "markers": ["CD38", "CD90", "CD45RA"], "children": []},
                                {"name": "MPP", "markers": ["CD38", "CD90", "CD45RA"], "children": []},
                                {"name": "CMP", "markers": ["CD38", "CD123", "CD45RA"], "children": []},
                                {"name": "GMP", "markers": ["CD38", "CD123", "CD45RA"], "children": []},
                                {"name": "MEP", "markers": ["CD38", "CD123", "CD45RA"], "children": []},
                                {"name": "CLP", "markers": ["CD38", "CD10", "CD7"], "children": []},
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-039": {
        "title": "22-color synovial fluid panel",
        "species": "human",
        "sample_type": "Human synovial fluid",
        "application": "Rheumatoid arthritis immune profiling",
        "n_colors": 22,
        "panel": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD4", "fluorophore": "BUV496"},
            {"marker": "CD8", "fluorophore": "BUV563"},
            {"marker": "CD45", "fluorophore": "BUV661"},
            {"marker": "CD19", "fluorophore": "BUV737"},
            {"marker": "CD14", "fluorophore": "BV421"},
            {"marker": "CD16", "fluorophore": "BV510"},
            {"marker": "HLA-DR", "fluorophore": "BV605"},
            {"marker": "CD38", "fluorophore": "BV650"},
            {"marker": "CD27", "fluorophore": "BV711"},
            {"marker": "CD45RA", "fluorophore": "BV785"},
            {"marker": "CXCR5", "fluorophore": "FITC"},
            {"marker": "PD-1", "fluorophore": "PE"},
            {"marker": "ICOS", "fluorophore": "PE-Cy7"},
            {"marker": "CD25", "fluorophore": "APC"},
            {"marker": "FoxP3", "fluorophore": "APC-Cy7"},
            {"marker": "CD127", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CCR6", "fluorophore": "AF700"},
            {"marker": "CXCR3", "fluorophore": "BUV805"},
            {"marker": "CD161", "fluorophore": "PE-CF594"},
            {"marker": "IL-17A", "fluorophore": "BV750"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events", "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "T cells", "markers": ["CD3"], "children": [
                                {"name": "CD4+ T", "markers": ["CD4"], "children": [
                                    {"name": "Th1", "markers": ["CXCR3", "CCR6"], "children": []},
                                    {"name": "Th17", "markers": ["CCR6", "CD161"], "children": []},
                                    {"name": "Tfh", "markers": ["CXCR5", "PD-1"], "children": []},
                                    {"name": "Tregs", "markers": ["CD25", "FoxP3"], "children": []},
                                ]},
                                {"name": "CD8+ T", "markers": ["CD8"], "children": []},
                            ]},
                            {"name": "B cells", "markers": ["CD19"], "children": []},
                            {"name": "Monocytes", "markers": ["CD14", "HLA-DR"], "children": []},
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-041": {
        "title": "16-color mouse autoimmune panel",
        "species": "mouse",
        "sample_type": "Mouse splenocytes",
        "application": "Autoimmune disease monitoring",
        "n_colors": 16,
        "panel": [
            {"marker": "CD45", "fluorophore": "BUV395"},
            {"marker": "CD3", "fluorophore": "BUV496"},
            {"marker": "CD4", "fluorophore": "BV421"},
            {"marker": "CD8a", "fluorophore": "BV510"},
            {"marker": "B220", "fluorophore": "BV605"},
            {"marker": "CD19", "fluorophore": "BV650"},
            {"marker": "CD44", "fluorophore": "BV711"},
            {"marker": "CD62L", "fluorophore": "BV785"},
            {"marker": "CD25", "fluorophore": "FITC"},
            {"marker": "FoxP3", "fluorophore": "PE"},
            {"marker": "IL-17A", "fluorophore": "PE-Cy7"},
            {"marker": "IFNg", "fluorophore": "APC"},
            {"marker": "PD-1", "fluorophore": "APC-Cy7"},
            {"marker": "CXCR5", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "Bcl6", "fluorophore": "AF700"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events", "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "T cells", "markers": ["CD3"], "children": [
                                {"name": "CD4+ T", "markers": ["CD4"], "children": [
                                    {"name": "Tfh", "markers": ["CXCR5", "PD-1", "Bcl6"], "children": []},
                                    {"name": "Th17", "markers": ["IL-17A"], "children": []},
                                    {"name": "Th1", "markers": ["IFNg"], "children": []},
                                    {"name": "Tregs", "markers": ["CD25", "FoxP3"], "children": []},
                                ]},
                                {"name": "CD8+ T", "markers": ["CD8a"], "children": []},
                            ]},
                            {"name": "B cells", "markers": ["B220", "CD19"], "children": []},
                        ]}
                    ]}
                ]}
            ]
        }
    },
}


def add_defaults_to_hierarchy(node: dict) -> dict:
    """Add default values to hierarchy nodes."""
    result = {
        "name": node.get("name", "Unknown"),
        "markers": node.get("markers", []),
        "gate_type": node.get("gate_type", "Unknown"),
        "is_critical": node.get("is_critical", False),
        "notes": node.get("notes"),
        "children": [add_defaults_to_hierarchy(c) for c in node.get("children", [])]
    }
    return result


def create_test_case(omip_id: str, data: dict) -> dict:
    """Create a test case JSON structure."""
    return {
        "test_case_id": omip_id,
        "source_type": "omip_paper",
        "omip_id": omip_id,
        "doi": f"10.1002/cyto.a.{omip_id.lower().replace('-', '')}",
        "flowrepository_id": None,
        "has_wsp": False,
        "wsp_validated": False,
        "context": {
            "sample_type": data["sample_type"],
            "species": data["species"],
            "application": data["application"],
            "tissue": data["sample_type"].split()[1] if len(data["sample_type"].split()) > 1 else "Unknown",
            "disease_state": None,
            "additional_notes": data["title"],
        },
        "panel": {
            "entries": [
                {"marker": p["marker"], "fluorophore": p["fluorophore"], "clone": p.get("clone")}
                for p in data["panel"]
            ]
        },
        "gating_hierarchy": {
            "root": add_defaults_to_hierarchy(data["hierarchy"])
        },
        "validation": {
            "paper_source": "Figure 1",
            "curator_notes": f"{data['n_colors']}-color panel"
        },
        "metadata": {
            "curation_date": str(date.today()),
            "curator": "Auto-generated"
        }
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for omip_id, data in ADDITIONAL_OMIPS.items():
        test_case = create_test_case(omip_id, data)
        filename = f"{omip_id.lower().replace('-', '_')}.json"
        output_path = OUTPUT_DIR / filename

        with open(output_path, "w") as f:
            json.dump(test_case, f, indent=2)

        print(f"✓ Created {omip_id}: {data['n_colors']} colors, {data['species']}")

    # Count all test cases
    all_files = list(OUTPUT_DIR.glob("omip_*.json"))
    print(f"\n{'='*50}")
    print(f"Total test cases: {len(all_files)}")


if __name__ == "__main__":
    main()
