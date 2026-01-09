"""
Reasoning benchmark test cases.

These tests evaluate genuine biological reasoning, not pattern matching.
Each test has clear fail/pass criteria based on biological correctness.
"""

from .schemas import (
    ReasoningTestCase,
    ReasoningTestType,
    ExpectedBehavior,
    EvaluationCriteria,
)


# =============================================================================
# LINEAGE NEGATIVE (DUMP CHANNEL) TESTS
# =============================================================================
# These test whether the model understands that cell types are defined
# by exclusion, not just positive markers.

LINEAGE_NEGATIVE_TESTS = [
    ReasoningTestCase(
        test_id="LN001_nk_cells_basic",
        test_type=ReasoningTestType.LINEAGE_NEGATIVE,
        difficulty="medium",
        prompt=(
            "I have a sample stained for CD45, CD3, CD19, CD56, CD14, and a "
            "Viability Dye. Design a gating hierarchy to identify NK cells."
        ),
        markers=["CD45", "CD3", "CD19", "CD56", "CD14", "Viability"],
        target_population="NK cells",
        tissue_context="PBMC",
        expected=ExpectedBehavior(
            fail_patterns=[
                "CD45+ → CD56+",
                "Gate on CD56 positive",
            ],
            fail_description=(
                "Pattern matching response gates directly on CD56+ without "
                "excluding T cells (CD3+), B cells (CD19+), or monocytes (CD14+). "
                "This produces a 'dirty' NK gate contaminated with other lineages."
            ),
            pass_requirements=[
                "Exclude CD3+",
                "Exclude CD19+",
                "Exclude CD14+",
                "Lineage negative",
                "Dump channel",
            ],
            pass_description=(
                "Reasoning response uses a dump channel or sequential exclusion: "
                "CD45+ → Lin- (CD3-CD19-CD14-) → CD56+. Understands NK cells are "
                "defined by what they are NOT."
            ),
            required_concepts=[
                "lineage negative",
                "exclusion",
                "dump channel",
                "CD3 negative",
            ],
            pattern_matching_indicators=[
                "simply gate on CD56",
                "CD56 bright",
            ],
        ),
        criteria=EvaluationCriteria(
            required_gates_in_order=[
                "viability",
                "singlets",
                "CD45+",
                "lineage negative",
                "CD56+",
            ],
            gate_prerequisites={
                "CD56+": ["lineage negative", "CD3-", "CD19-", "CD14-"],
            },
            exclusion_gates=["CD3", "CD19", "CD14"],
            required_reasoning_concepts=[
                "NK cells are lineage negative",
                "must exclude T cells",
                "must exclude B cells",
                "dump channel",
            ],
            failure_indicators=[
                "gate directly on CD56",
                "CD45+ then CD56+",
            ],
        ),
        rationale=(
            "NK cells express CD56 but so do some T cells (NKT cells). "
            "True NK cells are CD3-CD19-CD14-CD56+. A reasoning model must "
            "understand that lineage exclusion is required for purity."
        ),
    ),
    ReasoningTestCase(
        test_id="LN002_t_helper_tumor",
        test_type=ReasoningTestType.LINEAGE_NEGATIVE,
        difficulty="hard",
        prompt=(
            "I have a tumor digest sample. My markers are CD45, CD3, CD4, EpCAM, "
            "and a Viability Dye. Design a gating tree to sort pure T helper cells."
        ),
        markers=["CD45", "CD3", "CD4", "EpCAM", "Viability"],
        target_population="T helper cells",
        tissue_context="tumor digest",
        expected=ExpectedBehavior(
            fail_patterns=[
                "CD45+ → CD3+ → CD4+",
                "Gate CD3 then CD4",
            ],
            fail_description=(
                "Pattern matching ignores tumor cells (EpCAM+) and dead cells. "
                "A tumor digest has high debris and tumor cell contamination."
            ),
            pass_requirements=[
                "Viability gate",
                "Singlet discrimination",
                "EpCAM exclusion",
                "Time gate",
            ],
            pass_description=(
                "Reasoning response: Time → Singlets → Live → EpCAM- → CD45+ → CD3+ → CD4+. "
                "Excludes tumor cells before immune gating."
            ),
            required_concepts=[
                "exclude tumor cells",
                "EpCAM negative",
                "viability",
                "doublet exclusion",
            ],
        ),
        criteria=EvaluationCriteria(
            required_gates_in_order=[
                "time",
                "singlets",
                "live",
                "EpCAM-",
                "CD45+",
                "CD3+",
                "CD4+",
            ],
            gate_prerequisites={
                "CD45+": ["EpCAM-", "live"],
                "CD3+": ["CD45+"],
            },
            exclusion_gates=["EpCAM", "dead cells"],
            required_reasoning_concepts=[
                "tumor digest requires tumor exclusion",
                "EpCAM marks epithelial/tumor cells",
                "dead cell exclusion critical for digests",
            ],
            failure_indicators=[
                "skip EpCAM",
                "no viability gate",
                "same as blood",
            ],
        ),
        rationale=(
            "Tumor digests contain epithelial tumor cells (EpCAM+) that must be "
            "excluded before immune cell gating. Dead cells are also abundant. "
            "This tests adaptation to tissue context."
        ),
    ),
    ReasoningTestCase(
        test_id="LN003_monocyte_subsets",
        test_type=ReasoningTestType.LINEAGE_NEGATIVE,
        difficulty="hard",
        prompt=(
            "I need to identify classical, intermediate, and non-classical monocytes. "
            "My panel has CD45, CD14, CD16, CD3, CD19, CD56, HLA-DR, and Viability. "
            "Design the gating strategy."
        ),
        markers=["CD45", "CD14", "CD16", "CD3", "CD19", "CD56", "HLA-DR", "Viability"],
        target_population="monocyte subsets",
        tissue_context="PBMC",
        expected=ExpectedBehavior(
            fail_patterns=[
                "CD14+ → subdivide by CD16",
                "Gate HLA-DR+ then CD14/CD16",
            ],
            fail_description=(
                "Pattern matching misses that monocytes must first be isolated from "
                "other HLA-DR+ cells (B cells, DCs) and that CD14lowCD16+ non-classical "
                "monocytes would be missed by a CD14+ gate."
            ),
            pass_requirements=[
                "Exclude lymphocytes first",
                "HLA-DR+ gate includes all monocytes",
                "CD14/CD16 quadrant or biaxial plot",
            ],
            pass_description=(
                "Correct: Live → Lin- (CD3-CD19-CD56-) → HLA-DR+ → CD14 vs CD16 quadrant. "
                "Classical=CD14++CD16-, Intermediate=CD14++CD16+, Non-classical=CD14+CD16++"
            ),
            required_concepts=[
                "lineage exclusion before monocytes",
                "CD14 vs CD16 biaxial",
                "three monocyte populations",
                "HLA-DR positive myeloid",
            ],
        ),
        criteria=EvaluationCriteria(
            required_gates_in_order=[
                "live",
                "singlets",
                "lineage negative",
                "HLA-DR+",
                "CD14 vs CD16",
            ],
            exclusion_gates=["CD3", "CD19", "CD56"],
            required_reasoning_concepts=[
                "must exclude lymphocytes",
                "CD14/CD16 defines three subsets",
                "non-classical are CD14 low",
            ],
        ),
        rationale=(
            "Monocyte subset identification requires understanding that all three "
            "subsets are HLA-DR+ but differ in CD14/CD16 expression. Non-classical "
            "monocytes are CD14low and would be lost with a simple CD14+ gate."
        ),
    ),
]


# =============================================================================
# BIOLOGICAL IMPOSSIBILITY TESTS
# =============================================================================
# These test whether the model questions biological contradictions.

BIOLOGICAL_IMPOSSIBILITY_TESTS = [
    ReasoningTestCase(
        test_id="BI001_triple_positive_doublet",
        test_type=ReasoningTestType.BIOLOGICAL_IMPOSSIBILITY,
        difficulty="medium",
        prompt=(
            "I found a population that is CD3+, CD14+, and CD19+. "
            "What type of immune cell is this, and how should I gate for it?"
        ),
        markers=["CD3", "CD14", "CD19"],
        target_population="unknown",
        expected=ExpectedBehavior(
            fail_patterns=[
                "rare hybrid cell",
                "T-B cell interaction",
                "activation state",
                "novel subset",
            ],
            fail_description=(
                "Pattern matching hallucinating a rare cell type instead of "
                "recognizing the biological impossibility."
            ),
            pass_requirements=[
                "doublet",
                "aggregates",
                "FSC-A vs FSC-H",
                "artifact",
                "question the data",
            ],
            pass_description=(
                "Reasoning response flags this as likely doublets (two cells stuck "
                "together) or nonspecific binding. Suggests checking singlet gate "
                "with FSC-A vs FSC-H or SSC-A vs SSC-H."
            ),
            required_concepts=[
                "doublet",
                "doublet discrimination",
                "FSC-A vs FSC-H",
                "biologically impossible",
                "artifact",
            ],
            pattern_matching_indicators=[
                "novel cell type",
                "rare subset",
                "activated",
                "transitional",
            ],
        ),
        criteria=EvaluationCriteria(
            required_reasoning_concepts=[
                "this is not a real cell type",
                "doublet discrimination needed",
                "check singlet gate",
                "nonspecific binding possible",
            ],
            failure_indicators=[
                "this could be",
                "rare population",
                "novel subset",
                "gate for this population",
            ],
        ),
        rationale=(
            "A cell cannot simultaneously be a T cell (CD3+), monocyte (CD14+), "
            "and B cell (CD19+). This pattern indicates doublets or technical "
            "artifacts. A reasoning model should question the premise."
        ),
    ),
    ReasoningTestCase(
        test_id="BI002_cd4_cd8_double_positive_blood",
        test_type=ReasoningTestType.BIOLOGICAL_IMPOSSIBILITY,
        difficulty="easy",
        prompt=(
            "In my PBMC sample, I'm seeing a large CD4+CD8+ double positive "
            "population (about 15% of T cells). How should I analyze this population?"
        ),
        markers=["CD3", "CD4", "CD8"],
        target_population="CD4+CD8+ T cells",
        tissue_context="PBMC",
        expected=ExpectedBehavior(
            fail_patterns=[
                "double positive T cells",
                "DP thymocytes",
                "gate on this population",
            ],
            fail_description=(
                "While CD4+CD8+ DP cells exist in thymus, they are <1% in peripheral "
                "blood. 15% is a red flag for technical issues."
            ),
            pass_requirements=[
                "technical artifact",
                "compensation issue",
                "spillover",
                "check controls",
            ],
            pass_description=(
                "15% DP in blood is abnormal - suggests compensation/spillover issues "
                "between CD4 and CD8 channels. Check single-stained controls."
            ),
            required_concepts=[
                "compensation",
                "spillover",
                "abnormally high percentage",
                "check controls",
            ],
        ),
        criteria=EvaluationCriteria(
            required_reasoning_concepts=[
                "DP rare in blood",
                "compensation artifact",
                "check single stains",
                "spectral overlap",
            ],
            failure_indicators=[
                "analyze the DP population",
                "thymic emigrants",
                "normal finding",
            ],
        ),
        rationale=(
            "CD4+CD8+ double positive cells are found in thymus during T cell "
            "development, but are <1% in peripheral blood. A 15% population "
            "strongly suggests compensation/spillover artifacts."
        ),
    ),
    ReasoningTestCase(
        test_id="BI003_foxp3_without_permeabilization",
        test_type=ReasoningTestType.BIOLOGICAL_IMPOSSIBILITY,
        difficulty="medium",
        prompt=(
            "I stained my PBMCs with surface markers (CD4, CD25) and FoxP3 without "
            "any fixation or permeabilization. I'm seeing clear FoxP3+ populations. "
            "Are these real Tregs?"
        ),
        markers=["CD4", "CD25", "FoxP3"],
        target_population="Tregs",
        expected=ExpectedBehavior(
            fail_patterns=[
                "yes these are Tregs",
                "gate on FoxP3+",
                "CD4+CD25+FoxP3+",
            ],
            fail_description=(
                "FoxP3 is an intracellular transcription factor. Without "
                "permeabilization, antibodies cannot access it."
            ),
            pass_requirements=[
                "intracellular",
                "permeabilization required",
                "cannot detect without fixation",
                "artifact",
                "surface binding",
            ],
            pass_description=(
                "FoxP3 is intracellular and requires fix/perm for detection. "
                "The signal is likely nonspecific surface binding or autofluorescence."
            ),
            required_concepts=[
                "intracellular staining required",
                "fixation/permeabilization",
                "FoxP3 is nuclear",
                "nonspecific binding",
            ],
        ),
        criteria=EvaluationCriteria(
            required_reasoning_concepts=[
                "FoxP3 is intracellular",
                "needs permeabilization",
                "signal is artifact",
                "cannot detect surface FoxP3",
            ],
            failure_indicators=[
                "real Tregs",
                "gate the population",
                "analyze FoxP3+",
            ],
        ),
        rationale=(
            "FoxP3 is a nuclear transcription factor. Antibodies cannot access "
            "it without fixation and permeabilization. Any 'FoxP3+' signal without "
            "these steps is an artifact."
        ),
    ),
]


# =============================================================================
# CONTEXT SWITCH TESTS
# =============================================================================
# These test adaptation to tissue-specific constraints.

CONTEXT_SWITCH_TESTS = [
    ReasoningTestCase(
        test_id="CS001_lung_autofluorescence",
        test_type=ReasoningTestType.CONTEXT_SWITCH,
        difficulty="hard",
        prompt=(
            "Create a gating strategy for alveolar macrophages (CD45+CD11b+CD11c+CD64+) "
            "in a high-autofluorescence lung tissue sample. "
            "My panel uses FITC, PE, APC, and BV421."
        ),
        markers=["CD45", "CD11b", "CD11c", "CD64", "Viability"],
        target_population="Alveolar macrophages",
        tissue_context="lung tissue",
        constraints={
            "high_autofluorescence": True,
            "fluorophores": ["FITC", "PE", "APC", "BV421"],
        },
        expected=ExpectedBehavior(
            fail_patterns=[
                "same as blood",
                "CD45+ → CD11b+ → CD11c+ → CD64+",
            ],
            fail_description=(
                "Pattern matching gives standard blood hierarchy, ignoring "
                "lung autofluorescence that bleeds into FITC/GFP channel."
            ),
            pass_requirements=[
                "autofluorescence gate",
                "avoid FITC channel",
                "empty channel",
                "unstained control",
            ],
            pass_description=(
                "Reasoning response addresses autofluorescence: either uses an "
                "'empty channel' gate to exclude autofluorescent cells, avoids "
                "FITC for critical markers, or mentions unstained control for "
                "autofluorescence baseline."
            ),
            required_concepts=[
                "lung autofluorescence",
                "FITC channel affected",
                "empty channel gate",
                "unstained control",
            ],
        ),
        criteria=EvaluationCriteria(
            required_reasoning_concepts=[
                "autofluorescence is a problem in lung",
                "FITC/GFP channel most affected",
                "need empty channel or autofluorescence gate",
                "check unstained control",
            ],
            failure_indicators=[
                "standard gating",
                "same as PBMC",
                "no mention of autofluorescence",
            ],
            bonus_concepts=[
                "use red-shifted fluorophores",
                "APC/far-red less affected",
                "spectral unmixing",
            ],
        ),
        rationale=(
            "Lung tissue has high autofluorescence, especially in the FITC/GFP "
            "channel, due to NADH, flavins, and lipofuscin. A reasoning model "
            "must adapt the strategy to handle this."
        ),
    ),
    ReasoningTestCase(
        test_id="CS002_bone_marrow_precursors",
        test_type=ReasoningTestType.CONTEXT_SWITCH,
        difficulty="hard",
        prompt=(
            "I need to identify mature B cells in a bone marrow sample. "
            "My markers are CD45, CD19, CD20, CD10, and CD34. "
            "How should I gate differently than I would for blood?"
        ),
        markers=["CD45", "CD19", "CD20", "CD10", "CD34"],
        target_population="Mature B cells",
        tissue_context="bone marrow",
        expected=ExpectedBehavior(
            fail_patterns=[
                "CD19+ → CD20+",
                "same as blood",
            ],
            fail_description=(
                "Blood gating ignores that bone marrow contains B cell precursors "
                "at various maturation stages."
            ),
            pass_requirements=[
                "exclude precursors",
                "CD10 negative for mature",
                "CD34 negative",
                "maturation stages",
            ],
            pass_description=(
                "Mature B cells are CD19+CD20+CD10-CD34-. Must exclude pro-B "
                "(CD34+CD19+), pre-B (CD10+CD20-), and immature B cells (CD10+CD20+)."
            ),
            required_concepts=[
                "B cell maturation stages",
                "CD10 marks immature",
                "CD34 marks progenitors",
                "bone marrow has precursors",
            ],
        ),
        criteria=EvaluationCriteria(
            exclusion_gates=["CD10+", "CD34+"],
            required_reasoning_concepts=[
                "bone marrow contains precursors",
                "CD10 marks immature B cells",
                "mature B cells are CD10-",
                "different from blood",
            ],
            failure_indicators=[
                "same as blood",
                "just gate CD19+CD20+",
                "no mention of precursors",
            ],
        ),
        rationale=(
            "Bone marrow contains B cells at all maturation stages. Mature B cells "
            "must be distinguished from precursors using CD10 and CD34 negativity."
        ),
    ),
    ReasoningTestCase(
        test_id="CS003_csf_low_cellularity",
        test_type=ReasoningTestType.CONTEXT_SWITCH,
        difficulty="medium",
        prompt=(
            "I'm analyzing CSF (cerebrospinal fluid) which has very low cellularity "
            "(~5 cells/μL). My panel has CD45, CD3, CD4, CD8, CD19, CD14. "
            "What special considerations are needed?"
        ),
        markers=["CD45", "CD3", "CD4", "CD8", "CD19", "CD14"],
        target_population="Lymphocytes",
        tissue_context="CSF",
        constraints={"low_cellularity": True},
        expected=ExpectedBehavior(
            fail_patterns=[
                "standard PBMC gating",
                "no special considerations",
            ],
            fail_description=(
                "Low cellularity requires different approach - standard statistics "
                "and gates may not apply."
            ),
            pass_requirements=[
                "low event count",
                "counting beads",
                "absolute counts",
                "wider gates",
                "background significance",
            ],
            pass_description=(
                "Low cellularity requires: counting beads for absolute numbers, "
                "awareness that small populations may be noise, potentially wider "
                "gates to capture rare events, and careful interpretation."
            ),
            required_concepts=[
                "low event count issues",
                "counting beads",
                "absolute quantification",
                "statistical uncertainty",
            ],
        ),
        criteria=EvaluationCriteria(
            required_reasoning_concepts=[
                "very few cells to analyze",
                "need absolute counts",
                "statistical limitations",
                "counting beads recommended",
            ],
            bonus_concepts=[
                "longer acquisition time",
                "sample concentration",
                "background subtraction",
            ],
        ),
        rationale=(
            "CSF has extremely low cellularity. Standard gating statistics "
            "don't apply, and absolute quantification requires counting beads."
        ),
    ),
]


# =============================================================================
# FMO LOGIC TESTS
# =============================================================================
# These test understanding of gating boundary determination.

FMO_LOGIC_TESTS = [
    ReasoningTestCase(
        test_id="FMO001_cd25_boundary",
        test_type=ReasoningTestType.FMO_LOGIC,
        difficulty="medium",
        prompt=(
            "I have a smear of CD25 expression on my CD4+ T cells - there's no clear "
            "positive and negative population. How do I determine the cutoff for "
            "CD25 positive vs negative?"
        ),
        markers=["CD4", "CD25"],
        target_population="CD25+ T cells",
        expected=ExpectedBehavior(
            fail_patterns=[
                "draw the gate at the valley",
                "use isotype control",
                "separate positive and negative",
            ],
            fail_description=(
                "Pattern matching suggests subjective gating or outdated isotype "
                "controls. This fails for continuous expression markers."
            ),
            pass_requirements=[
                "FMO",
                "Fluorescence Minus One",
                "background spread",
                "spillover",
            ],
            pass_description=(
                "Need an FMO (Fluorescence Minus One) control - stain for everything "
                "except CD25. This reveals background fluorescence spread from other "
                "channels, allowing proper boundary setting."
            ),
            required_concepts=[
                "FMO control",
                "Fluorescence Minus One",
                "background spread",
                "spillover from other channels",
            ],
            pattern_matching_indicators=[
                "isotype control",
                "draw at the dip",
                "obvious separation",
            ],
        ),
        criteria=EvaluationCriteria(
            required_reasoning_concepts=[
                "FMO is the correct control",
                "reveals spillover background",
                "isotype is outdated",
                "continuous expression requires FMO",
            ],
            failure_indicators=[
                "use isotype",
                "subjective gating",
                "draw between populations",
            ],
        ),
        rationale=(
            "For markers with continuous expression (no clear positive/negative), "
            "FMO controls are essential. They show the spread of background "
            "fluorescence into the channel of interest."
        ),
    ),
    ReasoningTestCase(
        test_id="FMO002_why_not_isotype",
        test_type=ReasoningTestType.FMO_LOGIC,
        difficulty="easy",
        prompt=(
            "My PI insists I use isotype controls to set my gates. "
            "Why might FMO controls be better for multicolor flow cytometry?"
        ),
        markers=[],
        target_population="N/A",
        expected=ExpectedBehavior(
            fail_patterns=[
                "isotype controls are fine",
                "both work equally well",
            ],
            fail_description=(
                "Pattern matching doesn't understand the specific advantages "
                "of FMO in multicolor panels."
            ),
            pass_requirements=[
                "spillover",
                "spreading error",
                "isotype doesn't account for",
                "multicolor specific",
            ],
            pass_description=(
                "FMO controls account for spillover spreading from all other "
                "fluorophores in the panel. Isotype controls only show nonspecific "
                "binding of ONE antibody, missing the multicolor compensation effects."
            ),
            required_concepts=[
                "spillover spreading error",
                "isotype misses compensation effects",
                "FMO is panel-specific",
                "multicolor interactions",
            ],
        ),
        criteria=EvaluationCriteria(
            required_reasoning_concepts=[
                "FMO accounts for spillover",
                "isotype only shows one antibody",
                "spreading error in multicolor",
                "FMO is gold standard",
            ],
        ),
        rationale=(
            "Isotype controls show nonspecific binding of one antibody but miss "
            "the spreading error from compensation in multicolor panels. FMO is "
            "the current gold standard for gate setting."
        ),
    ),
]


# =============================================================================
# PANEL SUBSET DESIGN TESTS
# =============================================================================
# These test optimization under instrument constraints.

PANEL_SUBSET_TESTS = [
    ReasoningTestCase(
        test_id="PS001_reduce_to_5_color",
        test_type=ReasoningTestType.PANEL_SUBSET,
        difficulty="hard",
        prompt=(
            "I have a 20-color T cell panel with CD3, CD4, CD8, CD45RA, CCR7, "
            "CD27, CD28, CD127, CD25, FoxP3, PD-1, TIGIT, CD69, HLA-DR, CD38, "
            "Ki-67, CD95, CD57, KLRG1, and Viability. "
            "I need to reduce this to a 5-color panel for routine clinical monitoring "
            "of T cell health. Which markers should I keep and why?"
        ),
        markers=[
            "CD3", "CD4", "CD8", "CD45RA", "CCR7", "CD27", "CD28",
            "CD127", "CD25", "FoxP3", "PD-1", "TIGIT", "CD69", "HLA-DR",
            "CD38", "Ki-67", "CD95", "CD57", "KLRG1", "Viability",
        ],
        target_population="T cells for clinical monitoring",
        constraints={
            "target_colors": 5,
            "application": "routine clinical monitoring",
        },
        expected=ExpectedBehavior(
            fail_patterns=[
                "keep the first 5",
                "random selection",
            ],
            fail_description=(
                "Pattern matching doesn't optimize for clinical utility or "
                "justify marker selection."
            ),
            pass_requirements=[
                "viability essential",
                "CD3/CD4/CD8 for lineage",
                "justification for each",
                "what you lose",
            ],
            pass_description=(
                "Reasoning response: Viability, CD3, CD4, CD8, + one clinical marker "
                "(CD45RA for memory or CD38 for activation). Explains trade-offs and "
                "what populations can't be identified with reduced panel."
            ),
            required_concepts=[
                "viability is non-negotiable",
                "lineage markers first",
                "clinical utility",
                "trade-off explanation",
            ],
        ),
        criteria=EvaluationCriteria(
            required_reasoning_concepts=[
                "keep essential lineage markers",
                "viability required",
                "explain what is lost",
                "clinical utility drives selection",
            ],
            bonus_concepts=[
                "combine information",
                "memory vs activation choice",
                "surface vs intracellular priority",
            ],
        ),
        rationale=(
            "Panel reduction requires understanding marker hierarchy: some are "
            "essential (viability, lineage), some are high-value (memory), and "
            "some are specialized (exhaustion markers)."
        ),
    ),
    ReasoningTestCase(
        test_id="PS002_12_color_b_cell",
        test_type=ReasoningTestType.PANEL_SUBSET,
        difficulty="medium",
        prompt=(
            "Design a 12-color panel for comprehensive B cell subset analysis "
            "in human PBMC. Must include viability and be able to identify: "
            "naive, memory, class-switched, plasmablasts, and transitional B cells."
        ),
        markers=[],  # Open-ended design
        target_population="B cell subsets",
        constraints={
            "target_colors": 12,
            "required_populations": [
                "naive B cells",
                "memory B cells",
                "class-switched B cells",
                "plasmablasts",
                "transitional B cells",
            ],
        },
        expected=ExpectedBehavior(
            pass_requirements=[
                "CD19 or CD20",
                "IgD for naive/memory",
                "CD27 for memory",
                "CD38 for plasmablasts",
                "CD24/CD38 for transitional",
            ],
            pass_description=(
                "Complete panel needs: Viability, CD45, CD19/CD20, IgD, IgM, CD27, "
                "CD38, CD24, plus markers for class switch (IgG, IgA) and lineage "
                "exclusion (CD3). Explains marker logic."
            ),
            required_concepts=[
                "lineage exclusion needed",
                "IgD/CD27 for memory staging",
                "CD38 high for plasmablasts",
                "CD24/CD38 for transitional",
            ],
        ),
        criteria=EvaluationCriteria(
            required_reasoning_concepts=[
                "marker logic for each population",
                "how to distinguish subsets",
                "no redundant markers",
                "lineage exclusion included",
            ],
        ),
        rationale=(
            "B cell subset identification requires understanding marker biology: "
            "IgD/CD27 define memory stages, CD38 intensity distinguishes "
            "plasmablasts, and CD24/CD38 pattern identifies transitional cells."
        ),
    ),
]


def get_all_reasoning_tests() -> list[ReasoningTestCase]:
    """Get all reasoning benchmark test cases."""
    return (
        LINEAGE_NEGATIVE_TESTS
        + BIOLOGICAL_IMPOSSIBILITY_TESTS
        + CONTEXT_SWITCH_TESTS
        + FMO_LOGIC_TESTS
        + PANEL_SUBSET_TESTS
    )


def get_tests_by_type(test_type: ReasoningTestType) -> list[ReasoningTestCase]:
    """Get test cases of a specific type."""
    type_map = {
        ReasoningTestType.LINEAGE_NEGATIVE: LINEAGE_NEGATIVE_TESTS,
        ReasoningTestType.BIOLOGICAL_IMPOSSIBILITY: BIOLOGICAL_IMPOSSIBILITY_TESTS,
        ReasoningTestType.CONTEXT_SWITCH: CONTEXT_SWITCH_TESTS,
        ReasoningTestType.FMO_LOGIC: FMO_LOGIC_TESTS,
        ReasoningTestType.PANEL_SUBSET: PANEL_SUBSET_TESTS,
    }
    return type_map.get(test_type, [])


def get_tests_by_difficulty(difficulty: str) -> list[ReasoningTestCase]:
    """Get test cases of a specific difficulty."""
    return [t for t in get_all_reasoning_tests() if t.difficulty == difficulty]
