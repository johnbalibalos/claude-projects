"""
Modular Hypothesis Pipeline

A flexible framework for running hypothesis testing experiments with LLMs.

Supports independent configuration of:
- Reasoning strategies (CoT, WoT, direct, few-shot, ReAct)
- RAG providers (vector, hybrid, oracle, negative)
- Context levels (minimal, standard, rich, oracle)
- Tool/MCP configurations

Example:
    from hypothesis_pipeline import (
        HypothesisPipeline,
        PipelineConfig,
        ReasoningType,
        ContextLevel,
        RAGMode,
        TrialInput,
    )

    # Define your evaluator
    class MyEvaluator(Evaluator):
        def extract(self, response: str) -> Any:
            return json.loads(response)

        def score(self, extracted: Any, ground_truth: Any) -> dict[str, float]:
            return {"accuracy": 1.0 if extracted == ground_truth else 0.0}

    # Create trial inputs
    trials = [
        TrialInput(id="1", raw_input=data, prompt="...", ground_truth=expected),
    ]

    # Configure pipeline
    config = PipelineConfig(
        name="my_experiment",
        models=["claude-sonnet-4-20250514"],
        reasoning_types=[ReasoningType.DIRECT, ReasoningType.COT],
        context_levels=[ContextLevel.MINIMAL, ContextLevel.STANDARD],
        rag_modes=[RAGMode.NONE, RAGMode.VECTOR],
    )

    # Run pipeline
    pipeline = HypothesisPipeline(config, MyEvaluator(), trials)
    results = pipeline.run()
    print(pipeline.generate_report(results))
"""

# Models
from .models import (
    ReasoningType,
    ContextLevel,
    RAGMode,
    DataSource,
    ToolConfig,
    HypothesisCondition,
    TrialInput,
    TrialResult,
    ExperimentResults,
)

# Base classes
from .base import (
    PromptStrategy,
    RAGProvider,
    ContextBuilder,
    ToolRegistry,
    Evaluator,
    ModelClient,
)

# Strategies
from .strategies import (
    DirectStrategy,
    ChainOfThoughtStrategy,
    WebOfThoughtStrategy,
    FewShotStrategy,
    SelfConsistencyStrategy,
    ReActStrategy,
    get_strategy,
    DIRECT,
    COT,
    WOT,
)

# RAG Providers
from .rag import (
    NoRAGProvider,
    OracleRAGProvider,
    NegativeRAGProvider,
    VectorRAGProvider,
    HybridRAGProvider,
    CallbackRAGProvider,
    RAGRegistry,
    register_rag_provider,
    get_rag_provider,
    get_provider_for_mode,
)

# Context Builders
from .context import (
    MinimalContextBuilder,
    StandardContextBuilder,
    RichContextBuilder,
    OracleContextBuilder,
    ComposableContextBuilder,
    CallbackContextBuilder,
    get_context_builder,
)

# Pipeline (use config.PipelineConfig for full features)
from .pipeline import (
    HypothesisPipeline,
    create_simple_pipeline,
)

# Config
from .config import (
    PipelineConfig,
    ConfigLoader,
    parse_cli_overrides,
    create_minimal_config,
    create_ablation_config,
    create_full_config,
)

# Tracker
from .tracker import (
    ExperimentTracker,
    ExperimentMetadata,
    ExperimentConclusion,
    ExperimentRecord,
)

# Cost estimation
from .cost import (
    CostEstimate,
    ModelPricing,
    estimate_experiment_cost,
    confirm_experiment_cost,
    get_model_pricing_table,
)

# Statistical rigor
from .statistics import (
    BootstrapResult,
    bootstrap_ci,
    bootstrap_compare,
    PermutationTestResult,
    paired_permutation_test,
    independent_permutation_test,
    EffectSizeResult,
    cohens_d,
    hedges_g,
    glass_delta,
    cliff_delta,
    MultipleComparisonResult,
    bonferroni_correction,
    holm_correction,
    benjamini_hochberg_correction,
    apply_correction,
    PowerAnalysisResult,
    power_analysis_two_sample,
    minimum_detectable_effect,
    ComprehensiveComparison,
    compare_conditions,
    DescriptiveStats,
    describe,
)

# Contamination detection
from .contamination import (
    MemorizationResult,
    ContaminationReport,
    MemorizationDetector,
    TestCaseVariant,
    DynamicTestGenerator,
    VerbatimMatch,
    detect_verbatim_reproduction,
    TestSetFingerprint,
    create_test_set_fingerprint,
    check_fingerprint_in_output,
)

# Calibration metrics
from .calibration import (
    ParsedConfidence,
    parse_confidence_from_response,
    get_confidence_prompt,
    CONFIDENCE_ELICITATION_PROMPTS,
    CalibrationBin,
    CalibrationResult,
    expected_calibration_error,
    static_calibration_error,
    TemperatureScalingResult,
    find_optimal_temperature,
    BrierScoreResult,
    brier_score,
    ReliabilityDiagramData,
    get_reliability_diagram_data,
    plot_reliability_diagram_ascii,
    AggregateCalibrationAnalysis,
    compare_calibration,
)

# Robustness evaluation
from .robustness import (
    introduce_typos,
    replace_with_synonyms,
    reorder_sentences,
    random_case_change,
    add_whitespace_noise,
    add_punctuation_noise,
    PERTURBATIONS,
    get_perturbation,
    apply_perturbation,
    PerturbationResult,
    RobustnessResult,
    RobustnessEvaluator,
    ConsistencyResult,
    ConsistencyTester,
    AdversarialResult,
    test_adversarial_robustness,
    ComprehensiveRobustnessAnalysis,
    comprehensive_robustness_analysis,
)

# LLM-as-judge
from .llm_judge import (
    RubricLevel,
    RubricCriterion,
    EvaluationRubric,
    JudgmentScore,
    JudgmentResult,
    LLMJudge,
    PairwiseComparisonResult,
    PairwiseJudge,
    InterJudgeAgreement,
    compute_inter_judge_agreement,
    EnsembleJudgmentResult,
    EnsembleJudge,
    JudgeCalibrationResult,
    calibrate_judge,
    create_default_judge,
    quick_evaluate,
)

# Experiment tracking
from .experiment_tracking import (
    MLflowTracker,
    WandBTracker,
    LocalTracker,
    RunMetadata,
    create_tracker,
    RunComparison,
    compare_experiments,
    generate_comparison_report,
    TrackedExperiment,
    track_pipeline_run,
)

# Selective prediction
from .selective_prediction import (
    SelectivePredictionResult,
    selective_prediction_metrics,
    RiskCoverageCurve,
    compute_risk_coverage_curve,
    plot_risk_coverage_ascii,
    ThresholdOptimizationResult,
    optimize_threshold,
    AbstentionStrategy,
    ConfidenceThresholdStrategy,
    AdaptiveThresholdStrategy,
    UncertaintyBasedStrategy,
    EnsembleDisagreementStrategy,
    SelectiveEvaluationResult,
    evaluate_selective_prediction,
    CostSensitiveResult,
    cost_sensitive_selective_prediction,
)

# Test set diversity
from .diversity import (
    DistributionAnalysis,
    analyze_distribution,
    DiversityReport,
    analyze_test_set_diversity,
    CoverageCell,
    CrossFeatureCoverage,
    analyze_cross_feature_coverage,
    StratificationResult,
    analyze_stratification,
    ContentDiversityResult,
    analyze_content_diversity,
    DifficultyAnalysis,
    analyze_difficulty_distribution,
    ComprehensiveDiversityAnalysis,
    comprehensive_diversity_analysis,
)

# Cost-performance Pareto analysis
from .pareto import (
    ModelResult,
    ParetoPoint,
    ParetoAnalysis,
    compute_pareto_frontier,
    MultiObjectiveParetoResult,
    multi_objective_pareto,
    CostEffectivenessMetrics,
    compute_cost_effectiveness,
    ROIAnalysis,
    compute_upgrade_roi,
    ScalingAnalysis,
    analyze_cost_scaling,
    CostPerformanceReport,
    generate_cost_performance_report,
    print_report as print_pareto_report,
)

# Prompt sensitivity
from .prompt_sensitivity import (
    PROMPT_VARIATION_TEMPLATES,
    get_prompt_variations,
    VariationResult,
    SensitivityResult,
    PromptSensitivityAnalyzer,
    InstructionFollowingResult,
    analyze_instruction_following,
    FormatSensitivityResult,
    analyze_format_sensitivity,
    OptimizedPrompt,
    optimize_prompt_simple,
    ComprehensivePromptAnalysis,
    comprehensive_prompt_analysis,
)

__all__ = [
    # Models
    "ReasoningType",
    "ContextLevel",
    "RAGMode",
    "DataSource",
    "ToolConfig",
    "HypothesisCondition",
    "TrialInput",
    "TrialResult",
    "ExperimentResults",
    # Base classes
    "PromptStrategy",
    "RAGProvider",
    "ContextBuilder",
    "ToolRegistry",
    "Evaluator",
    "ModelClient",
    # Strategies
    "DirectStrategy",
    "ChainOfThoughtStrategy",
    "WebOfThoughtStrategy",
    "FewShotStrategy",
    "SelfConsistencyStrategy",
    "ReActStrategy",
    "get_strategy",
    "DIRECT",
    "COT",
    "WOT",
    # RAG
    "NoRAGProvider",
    "OracleRAGProvider",
    "NegativeRAGProvider",
    "VectorRAGProvider",
    "HybridRAGProvider",
    "CallbackRAGProvider",
    "RAGRegistry",
    "register_rag_provider",
    "get_rag_provider",
    "get_provider_for_mode",
    # Context
    "MinimalContextBuilder",
    "StandardContextBuilder",
    "RichContextBuilder",
    "OracleContextBuilder",
    "ComposableContextBuilder",
    "CallbackContextBuilder",
    "get_context_builder",
    # Pipeline
    "HypothesisPipeline",
    "create_simple_pipeline",
    # Config
    "PipelineConfig",
    "ConfigLoader",
    "parse_cli_overrides",
    "create_minimal_config",
    "create_ablation_config",
    "create_full_config",
    # Tracker
    "ExperimentTracker",
    "ExperimentMetadata",
    "ExperimentConclusion",
    "ExperimentRecord",
    # Cost estimation
    "CostEstimate",
    "ModelPricing",
    "estimate_experiment_cost",
    "confirm_experiment_cost",
    "get_model_pricing_table",
    # Statistical rigor
    "BootstrapResult",
    "bootstrap_ci",
    "bootstrap_compare",
    "PermutationTestResult",
    "paired_permutation_test",
    "independent_permutation_test",
    "EffectSizeResult",
    "cohens_d",
    "hedges_g",
    "glass_delta",
    "cliff_delta",
    "MultipleComparisonResult",
    "bonferroni_correction",
    "holm_correction",
    "benjamini_hochberg_correction",
    "apply_correction",
    "PowerAnalysisResult",
    "power_analysis_two_sample",
    "minimum_detectable_effect",
    "ComprehensiveComparison",
    "compare_conditions",
    "DescriptiveStats",
    "describe",
    # Contamination detection
    "MemorizationResult",
    "ContaminationReport",
    "MemorizationDetector",
    "TestCaseVariant",
    "DynamicTestGenerator",
    "VerbatimMatch",
    "detect_verbatim_reproduction",
    "TestSetFingerprint",
    "create_test_set_fingerprint",
    "check_fingerprint_in_output",
    # Calibration metrics
    "ParsedConfidence",
    "parse_confidence_from_response",
    "get_confidence_prompt",
    "CONFIDENCE_ELICITATION_PROMPTS",
    "CalibrationBin",
    "CalibrationResult",
    "expected_calibration_error",
    "static_calibration_error",
    "TemperatureScalingResult",
    "find_optimal_temperature",
    "BrierScoreResult",
    "brier_score",
    "ReliabilityDiagramData",
    "get_reliability_diagram_data",
    "plot_reliability_diagram_ascii",
    "AggregateCalibrationAnalysis",
    "compare_calibration",
    # Robustness evaluation
    "introduce_typos",
    "replace_with_synonyms",
    "reorder_sentences",
    "random_case_change",
    "add_whitespace_noise",
    "add_punctuation_noise",
    "PERTURBATIONS",
    "get_perturbation",
    "apply_perturbation",
    "PerturbationResult",
    "RobustnessResult",
    "RobustnessEvaluator",
    "ConsistencyResult",
    "ConsistencyTester",
    "AdversarialResult",
    "test_adversarial_robustness",
    "ComprehensiveRobustnessAnalysis",
    "comprehensive_robustness_analysis",
    # LLM-as-judge
    "RubricLevel",
    "RubricCriterion",
    "EvaluationRubric",
    "JudgmentScore",
    "JudgmentResult",
    "LLMJudge",
    "PairwiseComparisonResult",
    "PairwiseJudge",
    "InterJudgeAgreement",
    "compute_inter_judge_agreement",
    "EnsembleJudgmentResult",
    "EnsembleJudge",
    "JudgeCalibrationResult",
    "calibrate_judge",
    "create_default_judge",
    "quick_evaluate",
    # Experiment tracking
    "MLflowTracker",
    "WandBTracker",
    "LocalTracker",
    "RunMetadata",
    "create_tracker",
    "RunComparison",
    "compare_experiments",
    "generate_comparison_report",
    "TrackedExperiment",
    "track_pipeline_run",
    # Selective prediction
    "SelectivePredictionResult",
    "selective_prediction_metrics",
    "RiskCoverageCurve",
    "compute_risk_coverage_curve",
    "plot_risk_coverage_ascii",
    "ThresholdOptimizationResult",
    "optimize_threshold",
    "AbstentionStrategy",
    "ConfidenceThresholdStrategy",
    "AdaptiveThresholdStrategy",
    "UncertaintyBasedStrategy",
    "EnsembleDisagreementStrategy",
    "SelectiveEvaluationResult",
    "evaluate_selective_prediction",
    "CostSensitiveResult",
    "cost_sensitive_selective_prediction",
    # Test set diversity
    "DistributionAnalysis",
    "analyze_distribution",
    "DiversityReport",
    "analyze_test_set_diversity",
    "CoverageCell",
    "CrossFeatureCoverage",
    "analyze_cross_feature_coverage",
    "StratificationResult",
    "analyze_stratification",
    "ContentDiversityResult",
    "analyze_content_diversity",
    "DifficultyAnalysis",
    "analyze_difficulty_distribution",
    "ComprehensiveDiversityAnalysis",
    "comprehensive_diversity_analysis",
    # Cost-performance Pareto analysis
    "ModelResult",
    "ParetoPoint",
    "ParetoAnalysis",
    "compute_pareto_frontier",
    "MultiObjectiveParetoResult",
    "multi_objective_pareto",
    "CostEffectivenessMetrics",
    "compute_cost_effectiveness",
    "ROIAnalysis",
    "compute_upgrade_roi",
    "ScalingAnalysis",
    "analyze_cost_scaling",
    "CostPerformanceReport",
    "generate_cost_performance_report",
    "print_pareto_report",
    # Prompt sensitivity
    "PROMPT_VARIATION_TEMPLATES",
    "get_prompt_variations",
    "VariationResult",
    "SensitivityResult",
    "PromptSensitivityAnalyzer",
    "InstructionFollowingResult",
    "analyze_instruction_following",
    "FormatSensitivityResult",
    "analyze_format_sensitivity",
    "OptimizedPrompt",
    "optimize_prompt_simple",
    "ComprehensivePromptAnalysis",
    "comprehensive_prompt_analysis",
]
