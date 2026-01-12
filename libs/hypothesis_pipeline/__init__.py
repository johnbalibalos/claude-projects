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
# Base classes
from .base import (
    ContextBuilder,
    Evaluator,
    ModelClient,
    PromptStrategy,
    RAGProvider,
    ToolRegistry,
)

# Calibration metrics
from .calibration import (
    CONFIDENCE_ELICITATION_PROMPTS,
    AggregateCalibrationAnalysis,
    BrierScoreResult,
    CalibrationBin,
    CalibrationResult,
    ParsedConfidence,
    ReliabilityDiagramData,
    TemperatureScalingResult,
    brier_score,
    compare_calibration,
    expected_calibration_error,
    find_optimal_temperature,
    get_confidence_prompt,
    get_reliability_diagram_data,
    parse_confidence_from_response,
    plot_reliability_diagram_ascii,
    static_calibration_error,
)

# CLI-based judges (for Max subscription)
from .cli_judge import (
    CLIJudgeConfig,
    create_cli_ensemble_judge,
    create_cli_judge,
    create_cli_pairwise_judge,
)

# Config
from .config import (
    ConfigLoader,
    PipelineConfig,
    create_ablation_config,
    create_full_config,
    create_minimal_config,
    parse_cli_overrides,
)

# Contamination detection
from .contamination import (
    ContaminationReport,
    DynamicTestGenerator,
    MemorizationDetector,
    MemorizationResult,
    TestCaseVariant,
    TestSetFingerprint,
    VerbatimMatch,
    check_fingerprint_in_output,
    create_test_set_fingerprint,
    detect_verbatim_reproduction,
)

# Context Builders
from .context import (
    CallbackContextBuilder,
    ComposableContextBuilder,
    MinimalContextBuilder,
    OracleContextBuilder,
    RichContextBuilder,
    StandardContextBuilder,
    get_context_builder,
)

# Cost estimation
from .cost import (
    CostEstimate,
    ModelPricing,
    confirm_experiment_cost,
    estimate_experiment_cost,
    get_model_pricing_table,
)

# Test set diversity
from .diversity import (
    ComprehensiveDiversityAnalysis,
    ContentDiversityResult,
    CoverageCell,
    CrossFeatureCoverage,
    DifficultyAnalysis,
    DistributionAnalysis,
    DiversityReport,
    StratificationResult,
    analyze_content_diversity,
    analyze_cross_feature_coverage,
    analyze_difficulty_distribution,
    analyze_distribution,
    analyze_stratification,
    analyze_test_set_diversity,
    comprehensive_diversity_analysis,
)

# Experiment tracking
from .experiment_tracking import (
    LocalTracker,
    MLflowTracker,
    RunComparison,
    RunMetadata,
    TrackedExperiment,
    WandBTracker,
    compare_experiments,
    create_tracker,
    generate_comparison_report,
    track_pipeline_run,
)

# LLM-as-judge
from .llm_judge import (
    EnsembleJudge,
    EnsembleJudgmentResult,
    EvaluationRubric,
    InterJudgeAgreement,
    JudgeCalibrationResult,
    JudgmentResult,
    JudgmentScore,
    LLMJudge,
    PairwiseComparisonResult,
    PairwiseJudge,
    PluggableJudge,
    PluggableJudgeConfig,
    PluggableJudgeResult,
    RubricCriterion,
    RubricLevel,
    calibrate_judge,
    compute_inter_judge_agreement,
    create_default_judge,
    quick_evaluate,
)
from .models import (
    ContextLevel,
    DataSource,
    ExperimentResults,
    HypothesisCondition,
    RAGMode,
    ReasoningType,
    ToolConfig,
    TrialInput,
    TrialResult,
)

# Cost-performance Pareto analysis
from .pareto import (
    CostEffectivenessMetrics,
    CostPerformanceReport,
    ModelResult,
    MultiObjectiveParetoResult,
    ParetoAnalysis,
    ParetoPoint,
    ROIAnalysis,
    ScalingAnalysis,
    analyze_cost_scaling,
    compute_cost_effectiveness,
    compute_pareto_frontier,
    compute_upgrade_roi,
    generate_cost_performance_report,
    multi_objective_pareto,
)
from .pareto import (
    print_report as print_pareto_report,
)

# Pipeline (use config.PipelineConfig for full features)
from .pipeline import (
    HypothesisPipeline,
    create_simple_pipeline,
)

# Prompt sensitivity
from .prompt_sensitivity import (
    PROMPT_VARIATION_TEMPLATES,
    ComprehensivePromptAnalysis,
    FormatSensitivityResult,
    InstructionFollowingResult,
    OptimizedPrompt,
    PromptSensitivityAnalyzer,
    SensitivityResult,
    VariationResult,
    analyze_format_sensitivity,
    analyze_instruction_following,
    comprehensive_prompt_analysis,
    get_prompt_variations,
    optimize_prompt_simple,
)

# RAG Providers
from .rag import (
    CallbackRAGProvider,
    HybridRAGProvider,
    NegativeRAGProvider,
    NoRAGProvider,
    OracleRAGProvider,
    RAGRegistry,
    VectorRAGProvider,
    get_provider_for_mode,
    get_rag_provider,
    register_rag_provider,
)

# Robustness evaluation
from .robustness import (
    PERTURBATIONS,
    AdversarialResult,
    ComprehensiveRobustnessAnalysis,
    ConsistencyResult,
    ConsistencyTester,
    PerturbationResult,
    RobustnessEvaluator,
    RobustnessResult,
    add_punctuation_noise,
    add_whitespace_noise,
    apply_perturbation,
    comprehensive_robustness_analysis,
    get_perturbation,
    introduce_typos,
    random_case_change,
    reorder_sentences,
    replace_with_synonyms,
    test_adversarial_robustness,
)

# Selective prediction
from .selective_prediction import (
    AbstentionStrategy,
    AdaptiveThresholdStrategy,
    ConfidenceThresholdStrategy,
    CostSensitiveResult,
    EnsembleDisagreementStrategy,
    RiskCoverageCurve,
    SelectiveEvaluationResult,
    SelectivePredictionResult,
    ThresholdOptimizationResult,
    UncertaintyBasedStrategy,
    compute_risk_coverage_curve,
    cost_sensitive_selective_prediction,
    evaluate_selective_prediction,
    optimize_threshold,
    plot_risk_coverage_ascii,
    selective_prediction_metrics,
)

# Statistical rigor
from .statistics import (
    BootstrapResult,
    ComprehensiveComparison,
    DescriptiveStats,
    EffectSizeResult,
    MultipleComparisonResult,
    PermutationTestResult,
    PowerAnalysisResult,
    apply_correction,
    benjamini_hochberg_correction,
    bonferroni_correction,
    bootstrap_ci,
    bootstrap_compare,
    cliff_delta,
    cohens_d,
    compare_conditions,
    describe,
    glass_delta,
    hedges_g,
    holm_correction,
    independent_permutation_test,
    minimum_detectable_effect,
    paired_permutation_test,
    power_analysis_two_sample,
)

# Strategies
from .strategies import (
    COT,
    DIRECT,
    WOT,
    ChainOfThoughtStrategy,
    DirectStrategy,
    FewShotStrategy,
    ReActStrategy,
    SelfConsistencyStrategy,
    WebOfThoughtStrategy,
    get_strategy,
)

# Tracker
from .tracker import (
    ExperimentConclusion,
    ExperimentMetadata,
    ExperimentRecord,
    ExperimentTracker,
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
    # Pluggable judge (for domain-specific prompts)
    "PluggableJudge",
    "PluggableJudgeConfig",
    "PluggableJudgeResult",
    # CLI-based judges (for Max subscription)
    "CLIJudgeConfig",
    "create_cli_judge",
    "create_cli_pairwise_judge",
    "create_cli_ensemble_judge",
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
