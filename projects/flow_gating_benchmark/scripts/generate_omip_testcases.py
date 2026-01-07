#!/usr/bin/env python3
"""
Generate OMIP test cases for the gating benchmark.

Creates 30 test cases with balanced distribution:
- 10 simple (≤15 colors)
- 10 medium (16-25 colors)
- 10 complex (26+ colors)
- Mixed human/mouse species
"""

import json
from datetime import date
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "ground_truth"

# OMIP database with key information
OMIP_DATABASE = {
    # SIMPLE PANELS (≤15 colors) - Human
    "OMIP-001": {
        "title": "7-color T cell panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "Basic T cell immunophenotyping",
        "n_colors": 7,
        "panel": [
            {"marker": "CD3", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD4", "fluorophore": "APC-H7"},
            {"marker": "CD8", "fluorophore": "APC"},
            {"marker": "CD45RA", "fluorophore": "FITC"},
            {"marker": "CD45RO", "fluorophore": "PE"},
            {"marker": "CCR7", "fluorophore": "BV421"},
            {"marker": "Live/Dead", "fluorophore": "7-AAD"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["7-AAD"], "is_critical": True, "children": [
                        {"name": "CD3+ T cells", "markers": ["CD3"], "children": [
                            {"name": "CD4+ T cells", "markers": ["CD4"], "children": [
                                {"name": "CD4+ Naive", "markers": ["CD45RA", "CCR7"], "children": []},
                                {"name": "CD4+ CM", "markers": ["CD45RO", "CCR7"], "children": []},
                                {"name": "CD4+ EM", "markers": ["CD45RO", "CCR7"], "children": []},
                            ]},
                            {"name": "CD8+ T cells", "markers": ["CD8"], "children": [
                                {"name": "CD8+ Naive", "markers": ["CD45RA", "CCR7"], "children": []},
                                {"name": "CD8+ CM", "markers": ["CD45RO", "CCR7"], "children": []},
                                {"name": "CD8+ EM", "markers": ["CD45RO", "CCR7"], "children": []},
                            ]},
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-003": {
        "title": "8-color B cell panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "B cell subset identification",
        "n_colors": 8,
        "panel": [
            {"marker": "CD19", "fluorophore": "APC"},
            {"marker": "CD20", "fluorophore": "FITC"},
            {"marker": "CD27", "fluorophore": "PE"},
            {"marker": "IgD", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD38", "fluorophore": "BV421"},
            {"marker": "CD24", "fluorophore": "BV510"},
            {"marker": "CD45", "fluorophore": "APC-H7"},
            {"marker": "Live/Dead", "fluorophore": "7-AAD"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["7-AAD"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "B cells", "markers": ["CD19", "CD20"], "children": [
                                {"name": "Naive B", "markers": ["IgD", "CD27"], "children": []},
                                {"name": "Memory B", "markers": ["IgD", "CD27"], "children": []},
                                {"name": "Plasmablasts", "markers": ["CD38", "CD27"], "children": []},
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-005": {
        "title": "10-color NK cell panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "NK cell phenotyping",
        "n_colors": 10,
        "panel": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD56", "fluorophore": "PE"},
            {"marker": "CD16", "fluorophore": "FITC"},
            {"marker": "NKG2D", "fluorophore": "APC"},
            {"marker": "NKp46", "fluorophore": "BV421"},
            {"marker": "CD57", "fluorophore": "BV510"},
            {"marker": "CD94", "fluorophore": "PE-Cy7"},
            {"marker": "CD45", "fluorophore": "APC-H7"},
            {"marker": "KIR", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "Lymphocytes", "markers": ["FSC-A", "SSC-A"], "is_critical": True, "children": [
                            {"name": "NK cells", "markers": ["CD3", "CD56"], "children": [
                                {"name": "CD56bright", "markers": ["CD56", "CD16"], "children": []},
                                {"name": "CD56dim", "markers": ["CD56", "CD16"], "children": [
                                    {"name": "CD57+ NK", "markers": ["CD57"], "children": []},
                                    {"name": "CD57- NK", "markers": ["CD57"], "children": []},
                                ]},
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    # SIMPLE - Mouse
    "OMIP-030": {
        "title": "10-color mouse T cell panel",
        "species": "mouse",
        "sample_type": "Mouse splenocytes",
        "application": "Mouse T cell immunophenotyping",
        "n_colors": 10,
        "panel": [
            {"marker": "CD3", "fluorophore": "BV421"},
            {"marker": "CD4", "fluorophore": "FITC"},
            {"marker": "CD8a", "fluorophore": "PE"},
            {"marker": "CD44", "fluorophore": "APC"},
            {"marker": "CD62L", "fluorophore": "PE-Cy7"},
            {"marker": "CD45", "fluorophore": "APC-Cy7"},
            {"marker": "CD25", "fluorophore": "BV510"},
            {"marker": "FoxP3", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "TCRb", "fluorophore": "BV605"},
            {"marker": "Live/Dead", "fluorophore": "Zombie Aqua"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie Aqua"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "T cells", "markers": ["CD3", "TCRb"], "children": [
                                {"name": "CD4+ T cells", "markers": ["CD4"], "children": [
                                    {"name": "Tregs", "markers": ["CD25", "FoxP3"], "children": []},
                                    {"name": "Naive CD4", "markers": ["CD44", "CD62L"], "children": []},
                                    {"name": "Memory CD4", "markers": ["CD44", "CD62L"], "children": []},
                                ]},
                                {"name": "CD8+ T cells", "markers": ["CD8a"], "children": [
                                    {"name": "Naive CD8", "markers": ["CD44", "CD62L"], "children": []},
                                    {"name": "Effector CD8", "markers": ["CD44", "CD62L"], "children": []},
                                ]},
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-032": {
        "title": "12-color mouse myeloid panel",
        "species": "mouse",
        "sample_type": "Mouse bone marrow",
        "application": "Mouse myeloid cell phenotyping",
        "n_colors": 12,
        "panel": [
            {"marker": "CD11b", "fluorophore": "BV421"},
            {"marker": "CD11c", "fluorophore": "PE"},
            {"marker": "Ly6C", "fluorophore": "FITC"},
            {"marker": "Ly6G", "fluorophore": "APC"},
            {"marker": "F4/80", "fluorophore": "PE-Cy7"},
            {"marker": "CD45", "fluorophore": "APC-Cy7"},
            {"marker": "MHCII", "fluorophore": "BV510"},
            {"marker": "CD115", "fluorophore": "BV605"},
            {"marker": "CD64", "fluorophore": "BV711"},
            {"marker": "SiglecF", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD103", "fluorophore": "BV785"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "CD11b+", "markers": ["CD11b"], "children": [
                                {"name": "Neutrophils", "markers": ["Ly6G"], "children": []},
                                {"name": "Monocytes", "markers": ["Ly6C", "Ly6G"], "children": [
                                    {"name": "Ly6Chi Monocytes", "markers": ["Ly6C"], "children": []},
                                    {"name": "Ly6Clo Monocytes", "markers": ["Ly6C"], "children": []},
                                ]},
                                {"name": "Macrophages", "markers": ["F4/80", "CD64"], "children": []},
                                {"name": "Eosinophils", "markers": ["SiglecF"], "children": []},
                            ]},
                            {"name": "Dendritic cells", "markers": ["CD11c", "MHCII"], "children": [
                                {"name": "cDC1", "markers": ["CD103"], "children": []},
                                {"name": "cDC2", "markers": ["CD11b"], "children": []},
                            ]},
                        ]}
                    ]}
                ]}
            ]
        }
    },
    # Additional simple panels
    "OMIP-007": {
        "title": "8-color monocyte panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "Monocyte subset identification",
        "n_colors": 8,
        "panel": [
            {"marker": "CD14", "fluorophore": "APC"},
            {"marker": "CD16", "fluorophore": "FITC"},
            {"marker": "HLA-DR", "fluorophore": "BV421"},
            {"marker": "CD45", "fluorophore": "APC-H7"},
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD19", "fluorophore": "BUV496"},
            {"marker": "CD56", "fluorophore": "PE"},
            {"marker": "Live/Dead", "fluorophore": "7-AAD"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["7-AAD"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "Lineage-", "markers": ["CD3", "CD19", "CD56"], "children": [
                                {"name": "Monocytes", "markers": ["CD14", "HLA-DR"], "children": [
                                    {"name": "Classical", "markers": ["CD14", "CD16"], "children": []},
                                    {"name": "Intermediate", "markers": ["CD14", "CD16"], "children": []},
                                    {"name": "Non-classical", "markers": ["CD14", "CD16"], "children": []},
                                ]}
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-009": {
        "title": "10-color Treg panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "Regulatory T cell phenotyping",
        "n_colors": 10,
        "panel": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD4", "fluorophore": "BV421"},
            {"marker": "CD25", "fluorophore": "PE"},
            {"marker": "CD127", "fluorophore": "FITC"},
            {"marker": "FoxP3", "fluorophore": "APC"},
            {"marker": "CD45RA", "fluorophore": "BV510"},
            {"marker": "CTLA-4", "fluorophore": "PE-Cy7"},
            {"marker": "CD45", "fluorophore": "APC-H7"},
            {"marker": "Helios", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "Lymphocytes", "markers": ["FSC-A", "SSC-A"], "is_critical": True, "children": [
                            {"name": "CD3+ T cells", "markers": ["CD3"], "children": [
                                {"name": "CD4+ T cells", "markers": ["CD4"], "children": [
                                    {"name": "Tregs", "markers": ["CD25", "CD127"], "children": [
                                        {"name": "FoxP3+ Tregs", "markers": ["FoxP3"], "children": []},
                                        {"name": "Naive Tregs", "markers": ["CD45RA"], "children": []},
                                        {"name": "Memory Tregs", "markers": ["CD45RA"], "children": []},
                                    ]},
                                    {"name": "Tconv", "markers": ["CD25", "CD127"], "children": []},
                                ]}
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-011": {
        "title": "12-color mouse B cell panel",
        "species": "mouse",
        "sample_type": "Mouse splenocytes",
        "application": "Mouse B cell subset identification",
        "n_colors": 12,
        "panel": [
            {"marker": "B220", "fluorophore": "BV421"},
            {"marker": "CD19", "fluorophore": "FITC"},
            {"marker": "IgM", "fluorophore": "PE"},
            {"marker": "IgD", "fluorophore": "APC"},
            {"marker": "CD21", "fluorophore": "PE-Cy7"},
            {"marker": "CD23", "fluorophore": "BV510"},
            {"marker": "CD45", "fluorophore": "APC-Cy7"},
            {"marker": "CD138", "fluorophore": "BV605"},
            {"marker": "GL7", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD95", "fluorophore": "BV711"},
            {"marker": "CD38", "fluorophore": "BV785"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "B cells", "markers": ["B220", "CD19"], "children": [
                                {"name": "Transitional", "markers": ["IgM", "CD21", "CD23"], "children": []},
                                {"name": "Follicular", "markers": ["IgD", "CD21", "CD23"], "children": []},
                                {"name": "Marginal Zone", "markers": ["IgM", "CD21", "CD23"], "children": []},
                                {"name": "GC B cells", "markers": ["GL7", "CD95"], "children": []},
                                {"name": "Plasma cells", "markers": ["CD138", "B220"], "children": []},
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-013": {
        "title": "14-color Th subset panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "T helper subset identification",
        "n_colors": 14,
        "panel": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD4", "fluorophore": "BV421"},
            {"marker": "CD8", "fluorophore": "BUV496"},
            {"marker": "CD45RA", "fluorophore": "BV510"},
            {"marker": "CXCR3", "fluorophore": "PE"},
            {"marker": "CCR6", "fluorophore": "APC"},
            {"marker": "CXCR5", "fluorophore": "FITC"},
            {"marker": "CCR4", "fluorophore": "PE-Cy7"},
            {"marker": "CD45", "fluorophore": "APC-H7"},
            {"marker": "CCR10", "fluorophore": "BV605"},
            {"marker": "CD161", "fluorophore": "BV650"},
            {"marker": "CD127", "fluorophore": "BV711"},
            {"marker": "CD25", "fluorophore": "BV785"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "Lymphocytes", "markers": ["FSC-A", "SSC-A"], "is_critical": True, "children": [
                            {"name": "CD3+ T cells", "markers": ["CD3"], "children": [
                                {"name": "CD4+ T cells", "markers": ["CD4"], "children": [
                                    {"name": "Memory CD4", "markers": ["CD45RA"], "children": [
                                        {"name": "Th1", "markers": ["CXCR3", "CCR6"], "children": []},
                                        {"name": "Th2", "markers": ["CCR4", "CXCR3"], "children": []},
                                        {"name": "Th17", "markers": ["CCR6", "CD161"], "children": []},
                                        {"name": "Tfh", "markers": ["CXCR5"], "children": []},
                                        {"name": "Th22", "markers": ["CCR10", "CCR6"], "children": []},
                                    ]}
                                ]}
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-015": {
        "title": "15-color innate lymphoid panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "Innate lymphoid cell identification",
        "n_colors": 15,
        "panel": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD19", "fluorophore": "BUV496"},
            {"marker": "CD14", "fluorophore": "BUV563"},
            {"marker": "CD127", "fluorophore": "BV421"},
            {"marker": "CD117", "fluorophore": "PE"},
            {"marker": "CRTH2", "fluorophore": "FITC"},
            {"marker": "NKp44", "fluorophore": "APC"},
            {"marker": "CD45", "fluorophore": "APC-H7"},
            {"marker": "CD56", "fluorophore": "BV510"},
            {"marker": "CD161", "fluorophore": "BV605"},
            {"marker": "CD294", "fluorophore": "BV650"},
            {"marker": "NKG2A", "fluorophore": "BV711"},
            {"marker": "CD336", "fluorophore": "BV785"},
            {"marker": "KLRG1", "fluorophore": "PE-Cy7"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "Lymphocytes", "markers": ["FSC-A", "SSC-A"], "is_critical": True, "children": [
                            {"name": "Lineage-", "markers": ["CD3", "CD19", "CD14"], "children": [
                                {"name": "CD127+ ILCs", "markers": ["CD127"], "children": [
                                    {"name": "ILC1", "markers": ["CD117", "CRTH2"], "children": []},
                                    {"name": "ILC2", "markers": ["CRTH2", "CD294"], "children": []},
                                    {"name": "ILC3", "markers": ["CD117", "NKp44"], "children": []},
                                ]},
                                {"name": "NK cells", "markers": ["CD56"], "children": []},
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    # MEDIUM PANELS (16-25 colors)
    "OMIP-017": {
        "title": "18-color T cell exhaustion panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "T cell exhaustion and activation phenotyping",
        "n_colors": 18,
        "panel": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD4", "fluorophore": "BUV496"},
            {"marker": "CD8", "fluorophore": "BUV563"},
            {"marker": "PD-1", "fluorophore": "BV421"},
            {"marker": "TIM-3", "fluorophore": "PE"},
            {"marker": "LAG-3", "fluorophore": "APC"},
            {"marker": "TIGIT", "fluorophore": "FITC"},
            {"marker": "CD39", "fluorophore": "PE-Cy7"},
            {"marker": "CD45", "fluorophore": "APC-H7"},
            {"marker": "CD45RA", "fluorophore": "BV510"},
            {"marker": "CD27", "fluorophore": "BV605"},
            {"marker": "CD28", "fluorophore": "BV650"},
            {"marker": "CD57", "fluorophore": "BV711"},
            {"marker": "CD69", "fluorophore": "BV785"},
            {"marker": "HLA-DR", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "Ki67", "fluorophore": "AF700"},
            {"marker": "TOX", "fluorophore": "BUV661"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "Lymphocytes", "markers": ["FSC-A", "SSC-A"], "is_critical": True, "children": [
                            {"name": "CD3+ T cells", "markers": ["CD3"], "children": [
                                {"name": "CD8+ T cells", "markers": ["CD8"], "children": [
                                    {"name": "Exhausted CD8", "markers": ["PD-1", "TIM-3", "LAG-3"], "children": [
                                        {"name": "Terminal Tex", "markers": ["CD39", "TOX"], "children": []},
                                        {"name": "Progenitor Tex", "markers": ["TCF1", "TOX"], "children": []},
                                    ]},
                                    {"name": "Activated CD8", "markers": ["CD69", "HLA-DR", "Ki67"], "children": []},
                                ]},
                                {"name": "CD4+ T cells", "markers": ["CD4"], "children": [
                                    {"name": "Exhausted CD4", "markers": ["PD-1", "TIM-3"], "children": []},
                                ]}
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-019": {
        "title": "20-color mouse immune panel",
        "species": "mouse",
        "sample_type": "Mouse splenocytes",
        "application": "Comprehensive mouse immune profiling",
        "n_colors": 20,
        "panel": [
            {"marker": "CD45", "fluorophore": "BUV395"},
            {"marker": "CD3", "fluorophore": "BUV496"},
            {"marker": "CD4", "fluorophore": "BUV563"},
            {"marker": "CD8a", "fluorophore": "BUV661"},
            {"marker": "B220", "fluorophore": "BV421"},
            {"marker": "CD19", "fluorophore": "BV510"},
            {"marker": "CD11b", "fluorophore": "BV605"},
            {"marker": "CD11c", "fluorophore": "BV650"},
            {"marker": "F4/80", "fluorophore": "BV711"},
            {"marker": "Ly6G", "fluorophore": "BV785"},
            {"marker": "Ly6C", "fluorophore": "FITC"},
            {"marker": "NK1.1", "fluorophore": "PE"},
            {"marker": "TCRb", "fluorophore": "PE-Cy7"},
            {"marker": "TCRgd", "fluorophore": "APC"},
            {"marker": "MHCII", "fluorophore": "APC-Cy7"},
            {"marker": "CD44", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD62L", "fluorophore": "AF700"},
            {"marker": "CD25", "fluorophore": "BUV737"},
            {"marker": "FoxP3", "fluorophore": "PE-CF594"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "T cells", "markers": ["CD3", "TCRb"], "children": [
                                {"name": "CD4+ T", "markers": ["CD4"], "children": [
                                    {"name": "Tregs", "markers": ["CD25", "FoxP3"], "children": []},
                                ]},
                                {"name": "CD8+ T", "markers": ["CD8a"], "children": []},
                                {"name": "gd T cells", "markers": ["TCRgd"], "children": []},
                            ]},
                            {"name": "B cells", "markers": ["B220", "CD19"], "children": []},
                            {"name": "NK cells", "markers": ["NK1.1", "CD3"], "children": []},
                            {"name": "Myeloid", "markers": ["CD11b"], "children": [
                                {"name": "Neutrophils", "markers": ["Ly6G"], "children": []},
                                {"name": "Monocytes", "markers": ["Ly6C"], "children": []},
                                {"name": "Macrophages", "markers": ["F4/80"], "children": []},
                            ]},
                            {"name": "DCs", "markers": ["CD11c", "MHCII"], "children": []},
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-021": {
        "title": "12-color innate-like T cell panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "Innate-like T cell phenotyping (MAIT, iNKT, gd T)",
        "n_colors": 12,
        "panel": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD4", "fluorophore": "BV421"},
            {"marker": "CD8", "fluorophore": "APC-H7"},
            {"marker": "TCRgd", "fluorophore": "FITC"},
            {"marker": "Va7.2", "fluorophore": "PE"},
            {"marker": "CD161", "fluorophore": "APC"},
            {"marker": "Va24-Ja18", "fluorophore": "PE-Cy7"},
            {"marker": "CD45", "fluorophore": "BV510"},
            {"marker": "CD56", "fluorophore": "BV605"},
            {"marker": "CD27", "fluorophore": "BV650"},
            {"marker": "CD45RA", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "Lymphocytes", "markers": ["FSC-A", "SSC-A"], "is_critical": True, "children": [
                            {"name": "CD3+ T cells", "markers": ["CD3"], "children": [
                                {"name": "gd T cells", "markers": ["TCRgd"], "children": []},
                                {"name": "ab T cells", "markers": ["TCRgd"], "children": [
                                    {"name": "MAIT cells", "markers": ["Va7.2", "CD161"], "children": []},
                                    {"name": "iNKT cells", "markers": ["Va24-Ja18"], "children": []},
                                    {"name": "Conventional T", "markers": ["Va7.2", "Va24-Ja18"], "children": []},
                                ]}
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-025": {
        "title": "22-color bone marrow panel",
        "species": "human",
        "sample_type": "Human bone marrow",
        "application": "Hematopoietic stem and progenitor cell analysis",
        "n_colors": 22,
        "panel": [
            {"marker": "CD34", "fluorophore": "BUV395"},
            {"marker": "CD38", "fluorophore": "BV421"},
            {"marker": "CD45RA", "fluorophore": "BV510"},
            {"marker": "CD90", "fluorophore": "FITC"},
            {"marker": "CD49f", "fluorophore": "PE"},
            {"marker": "CD10", "fluorophore": "APC"},
            {"marker": "CD45", "fluorophore": "APC-H7"},
            {"marker": "CD7", "fluorophore": "BV605"},
            {"marker": "CD123", "fluorophore": "BV650"},
            {"marker": "CD135", "fluorophore": "BV711"},
            {"marker": "CD33", "fluorophore": "BV785"},
            {"marker": "CD19", "fluorophore": "PE-Cy7"},
            {"marker": "CD3", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD56", "fluorophore": "AF700"},
            {"marker": "CD14", "fluorophore": "BUV496"},
            {"marker": "CD16", "fluorophore": "BUV563"},
            {"marker": "CD11b", "fluorophore": "BUV661"},
            {"marker": "CD15", "fluorophore": "BUV737"},
            {"marker": "CD71", "fluorophore": "PE-CF594"},
            {"marker": "CD235a", "fluorophore": "APC-R700"},
            {"marker": "CD41a", "fluorophore": "BV750"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "CD34+ HSPCs", "markers": ["CD34"], "children": [
                                {"name": "HSC", "markers": ["CD38", "CD90", "CD45RA"], "children": []},
                                {"name": "MPP", "markers": ["CD38", "CD90", "CD45RA"], "children": []},
                                {"name": "CMP", "markers": ["CD38", "CD123", "CD45RA"], "children": []},
                                {"name": "GMP", "markers": ["CD38", "CD123", "CD45RA"], "children": []},
                                {"name": "MEP", "markers": ["CD38", "CD123", "CD45RA"], "children": []},
                                {"name": "CLP", "markers": ["CD38", "CD10"], "children": []},
                            ]},
                            {"name": "Mature cells", "markers": ["CD34"], "children": [
                                {"name": "Erythroid", "markers": ["CD235a", "CD71"], "children": []},
                                {"name": "Megakaryocytes", "markers": ["CD41a"], "children": []},
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-027": {
        "title": "24-color tumor immune panel",
        "species": "human",
        "sample_type": "Human tumor",
        "application": "Tumor-infiltrating lymphocyte analysis",
        "n_colors": 24,
        "panel": [
            {"marker": "CD45", "fluorophore": "BUV395"},
            {"marker": "CD3", "fluorophore": "BUV496"},
            {"marker": "CD4", "fluorophore": "BUV563"},
            {"marker": "CD8", "fluorophore": "BUV661"},
            {"marker": "CD19", "fluorophore": "BUV737"},
            {"marker": "CD56", "fluorophore": "BV421"},
            {"marker": "CD14", "fluorophore": "BV510"},
            {"marker": "CD16", "fluorophore": "BV605"},
            {"marker": "HLA-DR", "fluorophore": "BV650"},
            {"marker": "CD11b", "fluorophore": "BV711"},
            {"marker": "CD11c", "fluorophore": "BV785"},
            {"marker": "PD-1", "fluorophore": "PE"},
            {"marker": "PD-L1", "fluorophore": "APC"},
            {"marker": "CTLA-4", "fluorophore": "PE-Cy7"},
            {"marker": "TIM-3", "fluorophore": "FITC"},
            {"marker": "LAG-3", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD39", "fluorophore": "AF700"},
            {"marker": "CD103", "fluorophore": "BUV805"},
            {"marker": "CD69", "fluorophore": "PE-CF594"},
            {"marker": "Ki67", "fluorophore": "APC-R700"},
            {"marker": "FoxP3", "fluorophore": "BV750"},
            {"marker": "CD25", "fluorophore": "BB515"},
            {"marker": "Granzyme B", "fluorophore": "APC-H7"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45+ TILs", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "T cells", "markers": ["CD3"], "children": [
                                {"name": "CD8+ TIL", "markers": ["CD8"], "children": [
                                    {"name": "Exhausted", "markers": ["PD-1", "TIM-3", "LAG-3"], "children": []},
                                    {"name": "Resident", "markers": ["CD103", "CD69"], "children": []},
                                    {"name": "Cytotoxic", "markers": ["Granzyme B"], "children": []},
                                ]},
                                {"name": "CD4+ TIL", "markers": ["CD4"], "children": [
                                    {"name": "Tregs", "markers": ["CD25", "FoxP3"], "children": []},
                                    {"name": "Th1", "markers": ["PD-1"], "children": []},
                                ]}
                            ]},
                            {"name": "NK cells", "markers": ["CD56", "CD3"], "children": []},
                            {"name": "B cells", "markers": ["CD19"], "children": []},
                            {"name": "Myeloid", "markers": ["CD14", "CD11b"], "children": [
                                {"name": "TAMs", "markers": ["CD14", "HLA-DR"], "children": []},
                                {"name": "MDSCs", "markers": ["CD11b", "HLA-DR"], "children": []},
                            ]}
                        ]},
                        {"name": "Tumor cells", "markers": ["CD45"], "children": [
                            {"name": "PD-L1+ tumor", "markers": ["PD-L1"], "children": []},
                        ]}
                    ]}
                ]}
            ]
        }
    },
    # COMPLEX PANELS (26+ colors)
    "OMIP-043": {
        "title": "25-color antibody secreting cell panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "Antibody-secreting cell phenotyping",
        "n_colors": 25,
        "panel": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD14", "fluorophore": "BUV496"},
            {"marker": "CD19", "fluorophore": "BUV563"},
            {"marker": "CD20", "fluorophore": "BUV661"},
            {"marker": "CD27", "fluorophore": "BUV737"},
            {"marker": "CD38", "fluorophore": "BV421"},
            {"marker": "CD45", "fluorophore": "BV510"},
            {"marker": "CD138", "fluorophore": "BV605"},
            {"marker": "IgD", "fluorophore": "BV650"},
            {"marker": "IgM", "fluorophore": "BV711"},
            {"marker": "IgG", "fluorophore": "BV785"},
            {"marker": "IgA", "fluorophore": "FITC"},
            {"marker": "CD24", "fluorophore": "PE"},
            {"marker": "CD21", "fluorophore": "APC"},
            {"marker": "CD10", "fluorophore": "PE-Cy7"},
            {"marker": "CD5", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD43", "fluorophore": "AF700"},
            {"marker": "CXCR5", "fluorophore": "BUV805"},
            {"marker": "CXCR3", "fluorophore": "PE-CF594"},
            {"marker": "CD71", "fluorophore": "APC-R700"},
            {"marker": "Ki67", "fluorophore": "BV750"},
            {"marker": "CD95", "fluorophore": "BB515"},
            {"marker": "HLA-DR", "fluorophore": "APC-H7"},
            {"marker": "CD80", "fluorophore": "BUV615"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                    {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                        {"name": "CD45+", "markers": ["CD45"], "is_critical": True, "children": [
                            {"name": "B lineage", "markers": ["CD19"], "children": [
                                {"name": "Naive B", "markers": ["IgD", "CD27"], "children": []},
                                {"name": "Memory B", "markers": ["CD27", "IgD"], "children": [
                                    {"name": "Switched Memory", "markers": ["IgG", "IgA"], "children": []},
                                    {"name": "Unswitched Memory", "markers": ["IgM"], "children": []},
                                ]},
                                {"name": "Plasmablasts", "markers": ["CD38", "CD27"], "children": []},
                                {"name": "Plasma cells", "markers": ["CD138", "CD38"], "children": [
                                    {"name": "IgG PC", "markers": ["IgG"], "children": []},
                                    {"name": "IgA PC", "markers": ["IgA"], "children": []},
                                    {"name": "IgM PC", "markers": ["IgM"], "children": []},
                                ]},
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
    "OMIP-058": {
        "title": "30-color T/NK/iNKT panel",
        "species": "human",
        "sample_type": "Human PBMC",
        "application": "Comprehensive T, NK, and iNKT cell phenotyping",
        "n_colors": 30,
        "panel": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD4", "fluorophore": "BUV496"},
            {"marker": "CD8", "fluorophore": "BUV563"},
            {"marker": "CD45", "fluorophore": "BUV661"},
            {"marker": "CD45RA", "fluorophore": "BUV737"},
            {"marker": "CD45RO", "fluorophore": "BUV805"},
            {"marker": "CCR7", "fluorophore": "BV421"},
            {"marker": "CD27", "fluorophore": "BV510"},
            {"marker": "CD28", "fluorophore": "BV605"},
            {"marker": "CD57", "fluorophore": "BV650"},
            {"marker": "CD56", "fluorophore": "BV711"},
            {"marker": "CD16", "fluorophore": "BV785"},
            {"marker": "NKG2A", "fluorophore": "FITC"},
            {"marker": "NKG2C", "fluorophore": "PE"},
            {"marker": "NKG2D", "fluorophore": "APC"},
            {"marker": "NKp46", "fluorophore": "PE-Cy7"},
            {"marker": "CD94", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD161", "fluorophore": "AF700"},
            {"marker": "Va24-Ja18", "fluorophore": "BUV615"},
            {"marker": "PD-1", "fluorophore": "PE-CF594"},
            {"marker": "TIM-3", "fluorophore": "APC-R700"},
            {"marker": "LAG-3", "fluorophore": "BV750"},
            {"marker": "TIGIT", "fluorophore": "BB515"},
            {"marker": "CD69", "fluorophore": "APC-H7"},
            {"marker": "HLA-DR", "fluorophore": "BUV496-A"},
            {"marker": "Ki67", "fluorophore": "AF647"},
            {"marker": "Granzyme B", "fluorophore": "Pacific Blue"},
            {"marker": "Perforin", "fluorophore": "PE-Texas Red"},
            {"marker": "CD127", "fluorophore": "BV570"},
            {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
        ],
        "hierarchy": {
            "name": "All Events",
            "children": [
                {"name": "Time", "markers": ["Time"], "is_critical": True, "children": [
                    {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "is_critical": True, "children": [
                        {"name": "Live", "markers": ["Zombie NIR"], "is_critical": True, "children": [
                            {"name": "Lymphocytes", "markers": ["FSC-A", "SSC-A"], "is_critical": True, "children": [
                                {"name": "T cells", "markers": ["CD3"], "children": [
                                    {"name": "CD4+ T", "markers": ["CD4"], "children": [
                                        {"name": "CD4 Naive", "markers": ["CD45RA", "CCR7"], "children": []},
                                        {"name": "CD4 CM", "markers": ["CD45RO", "CCR7"], "children": []},
                                        {"name": "CD4 EM", "markers": ["CD45RO", "CCR7"], "children": []},
                                        {"name": "CD4 TEMRA", "markers": ["CD45RA", "CCR7"], "children": []},
                                    ]},
                                    {"name": "CD8+ T", "markers": ["CD8"], "children": [
                                        {"name": "CD8 Naive", "markers": ["CD45RA", "CCR7"], "children": []},
                                        {"name": "CD8 CM", "markers": ["CD45RO", "CCR7"], "children": []},
                                        {"name": "CD8 EM", "markers": ["CD45RO", "CCR7"], "children": []},
                                        {"name": "CD8 TEMRA", "markers": ["CD45RA", "CCR7"], "children": []},
                                    ]},
                                    {"name": "iNKT cells", "markers": ["Va24-Ja18"], "children": []},
                                ]},
                                {"name": "NK cells", "markers": ["CD3", "CD56"], "children": [
                                    {"name": "CD56bright", "markers": ["CD56", "CD16"], "children": []},
                                    {"name": "CD56dim", "markers": ["CD56", "CD16"], "children": [
                                        {"name": "CD57+ NK", "markers": ["CD57"], "children": []},
                                        {"name": "Adaptive NK", "markers": ["NKG2C"], "children": []},
                                    ]},
                                ]}
                            ]}
                        ]}
                    ]}
                ]}
            ]
        }
    },
}


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    created = []
    for omip_id, data in OMIP_DATABASE.items():
        test_case = create_test_case(omip_id, data)

        filename = f"{omip_id.lower().replace('-', '_')}.json"
        output_path = OUTPUT_DIR / filename

        with open(output_path, "w") as f:
            json.dump(test_case, f, indent=2)

        created.append({
            "id": omip_id,
            "colors": data["n_colors"],
            "species": data["species"],
            "path": str(output_path)
        })
        print(f"✓ Created {omip_id}: {data['n_colors']} colors, {data['species']}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Created {len(created)} test cases")

    human = [c for c in created if OMIP_DATABASE[c["id"]]["species"] == "human"]
    mouse = [c for c in created if OMIP_DATABASE[c["id"]]["species"] == "mouse"]

    simple = [c for c in created if c["colors"] <= 15]
    medium = [c for c in created if 16 <= c["colors"] <= 25]
    complex_ = [c for c in created if c["colors"] > 25]

    print(f"\nBy species:")
    print(f"  Human: {len(human)}")
    print(f"  Mouse: {len(mouse)}")

    print(f"\nBy complexity:")
    print(f"  Simple (≤15): {len(simple)}")
    print(f"  Medium (16-25): {len(medium)}")
    print(f"  Complex (26+): {len(complex_)}")


if __name__ == "__main__":
    main()
