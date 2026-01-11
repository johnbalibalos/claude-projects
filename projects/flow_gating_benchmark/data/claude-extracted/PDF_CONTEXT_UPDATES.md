# PDF Context Updates Tracking

Tracks which `claude-extracted/*.json` files have been updated with context information from PDFs in `~/claude-projects/docs/papers/OMIP/`.

## PDF to OMIP Mapping

| PDF File | OMIP ID | PMC ID | Status |
|----------|---------|--------|--------|
| nihms363148.pdf | OMIP-008 | PMC3418807 | DONE |
| omip022.pdf | OMIP-022 | PMC4231567 | DONE |
| nihms694613.pdf | OMIP-025 | PMC4454451 | DONE |
| nihms866639.pdf | OMIP-035 | PMC5494841 | DONE |
| nihms-1017368.pdf | OMIP-053 | PMC6497402 | DONE |
| nihms-1684062.pdf | OMIP-064 | PMC8485754 | DONE |
| CYTO-99-880.pdf | OMIP-074 | PMC8453800 | DONE |
| CYTO-99-888.pdf | OMIP-076 | PMC9546025 | DONE |
| CYTO-101-15.pdf | OMIP-077 | PMC9292053 | DONE |
| OMIP_083__A_21...pdf | OMIP-083 | PMC9310743 | DONE |
| nihms-1926997.pdf | OMIP-095 | PMC10843696 | DONE |
| nihms-1964512.pdf | OMIP-101 | PMC10958279 | DONE |

## Context Fields Updated

When updating a JSON file, the following context fields are populated from PDF Table 1:

- `sample_type`: Cell/tissue type (e.g., "Human PBMC", "Mouse spleen")
- `species`: Species name (e.g., "human", "mouse", "macaque")
- `application`: Purpose from Table 1
- `cross_references`: Array of related OMIPs mentioned
- `notes`: Additional details about the panel
- `pdf_verified`: true
- `pdf_source`: Path to the PDF file

## Completed Context Updates

### OMIP-008 (nihms363148.pdf)
- Sample: PBMC, T cell clones/lines, tumor-resident T cell lines
- Species: human
- Application: T cell cytokine production following in vitro stimulation
- Cross-refs: OMIP-001

### OMIP-022 (omip022.pdf)
- Sample: Cryopreserved PBMC (adult and infant)
- Species: human
- Application: T-cell phenotyping, memory categorization, cytokine production, and function following in vitro stimulation
- Cross-refs: OMIP-001, OMIP-008, OMIP-009, OMIP-014

### OMIP-025 (nihms694613.pdf)
- Sample: Cryopreserved PBMC
- Species: human
- Application: Characterization of antigen-specific T cells, TFH-like cells and NK cells
- Cross-refs: OMIP-014

### OMIP-035 (nihms866639.pdf)
- Sample: 721.221 stimulated PBMC
- Species: Macaca mulatta (rhesus macaque)
- Application: Define functional capability of various NK cell subsets
- Cross-refs: OMIP-007, OMIP-027, OMIP-028

### OMIP-053 (nihms-1017368.pdf)
- Sample: PBMC, long-term cultured Treg
- Species: human
- Application: Human CD4+ Treg identification, enumeration, sorting, and classification
- Cross-refs: OMIP-004, OMIP-006, OMIP-015

### OMIP-064 (nihms-1684062.pdf)
- Sample: PBMC
- Species: human
- Application: Extensive phenotyping of NK cells, ILCs, MAIT cells, and γδ T cells
- Cross-refs: OMIP-029, OMIP-039, OMIP-044, OMIP-055, OMIP-056, OMIP-058

### OMIP-074 (CYTO-99-880.pdf)
- Sample: Fresh or cryopreserved PBMC
- Species: human
- Application: Phenotyping IgA and IgG subclasses on B cells
- Cross-refs: OMIP-003, OMIP-033, OMIP-043, OMIP-047, OMIP-051

### OMIP-076 (CYTO-99-888.pdf)
- Sample: Freshly isolated splenocytes
- Species: mouse
- Application: Comprehensive immunophenotyping of T-cell, B-cell, and ASC subsets
- Cross-refs: OMIP-031, OMIP-032, OMIP-054, OMIP-061

### OMIP-077 (CYTO-101-15.pdf)
- Sample: Leukocytes from human WB (anticoagulated, erythrocyte-depleted)
- Species: human
- Application: Comprehensive phenotyping of leukocyte subsets in WB using flow cytometry with just 14 colors
- Cross-refs: OMIP-023, OMIP-024, OMIP-042, OMIP-051, OMIP-063

### OMIP-083 (OMIP_083__A_21...pdf)
- Sample: PBMCs
- Species: human
- Application: Delineation of peripheral monocyte subsets and deep phenotyping of monocyte function
- Cross-refs: OMIP-023, OMIP-024, OMIP-034, OMIP-038, OMIP-042, OMIP-069

### OMIP-095 (nihms-1926997.pdf)
- Sample: Spleen, Blood, Bone Marrow, Thymus, Inguinal Lymph Nodes, Peyer's Patches
- Species: mouse (C57BL/6)
- Application: Deep immunophenotyping of all major leukocyte populations in murine lymphoid tissues
- Cross-refs: OMIP-031, OMIP-032, OMIP-054, OMIP-057, OMIP-059, OMIP-061, OMIP-076, OMIP-079, OMIP-093

### OMIP-101 (nihms-1964512.pdf)
- Sample: Fresh or fixed, cryopreserved whole blood
- Species: human
- Application: Broad immunophenotyping of leukocytes including myeloid and lymphoid cell lineages
- Cross-refs: OMIP-062, OMIP-063, OMIP-069, OMIP-077, OMIP-078

## JSON Files Without PDFs

These files have extractions but no corresponding PDF in the OMIP folder:

| JSON File | Has Panel | Has Hierarchy | Notes |
|-----------|-----------|---------------|-------|
| omip_087.json | xml, llm | llm | No PDF available |

## Update Log

- 2026-01-11: Created tracking document
- 2026-01-11: Identified 7 PDFs, mapped to OMIPs
- 2026-01-11: Found OMIP-074 and OMIP-077 already have context populated
- 2026-01-11: Updated context for OMIP-008, OMIP-025, OMIP-035, OMIP-095, OMIP-101 from PDFs
- 2026-01-11: Added OMIP-022 from omip022.pdf (in papers/ folder)
- 2026-01-11: Discovered OMIP-053, OMIP-064, OMIP-076, OMIP-083 already had context from PDFs - updated tracking document
