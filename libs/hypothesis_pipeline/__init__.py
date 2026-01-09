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

__all__ = [
    # Models
    "ReasoningType",
    "ContextLevel",
    "RAGMode",
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
]
