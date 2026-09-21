"""
Shared data models for NeuroPilots.

This module defines the common data contracts exchanged between the
different layers of the NeuroPilots system:

    Artifact Ingestion
            ↓
    Workload & Model Engine
            ↓
    Performance & Diagnostics
            ↓
    AI Research Advisor
            ↓
    Experiment Execution
            ↓
    Feedback & Evaluation

The models are implemented using Pydantic so that data exchanged between
components is validated consistently.

Important design distinction:
    - domain  = application domain, e.g. "automotive"
    - workload = ML workload, e.g. "computer_vision", "nlp", "tabular"

This distinction is important because automotive is a use case/domain,
not one of the seven ML workloads defined by the NeuroPilots architecture.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# LAYER 1 — ARTIFACT INGESTION
# ============================================================================

class DatasetAnalysis(BaseModel):
    """
    Analysis results produced by the Dataset Analyzer.

    The Dataset Analyzer is responsible for extracting information such as:
        - Dataset dimensions
        - Feature information
        - Target information
        - Class distribution
        - Missing values
        - General data-quality signals

    The model intentionally keeps some diagnostic information generic because
    different workloads expose different dataset characteristics.
    """

    name: Optional[str] = Field(
        default=None,
        description="Name or identifier of the dataset.",
    )

    rows: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of samples/rows in the dataset.",
    )

    columns: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of features/columns in the dataset.",
    )

    features: List[str] = Field(
        default_factory=list,
        description="Feature or input variable names.",
    )

    target: Optional[str] = Field(
        default=None,
        description="Target/label column when applicable.",
    )

    class_distribution: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Class distribution information used for detecting "
            "class imbalance and related data issues."
        ),
    )

    missing_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Missing-value statistics for the dataset.",
    )

    quality_signals: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional data-quality indicators such as duplicates, "
            "outliers, invalid values, or potential leakage signals."
        ),
    )


class ModelAnalysis(BaseModel):
    """
    Analysis results produced by the Model Analyzer.

    This represents information about the model architecture and configuration
    that may influence its performance.
    """

    name: Optional[str] = Field(
        default=None,
        description="Model name or identifier.",
    )

    architecture: Optional[str] = Field(
        default=None,
        description="Model architecture or model family.",
    )

    parameter_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total number of trainable/non-trainable parameters.",
    )

    embeddings: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Embedding-related information when applicable, "
            "primarily for NLP or Generative AI workloads."
        ),
    )

    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Model configuration such as layers, dimensions, "
            "activation functions, tokenizer configuration, etc."
        ),
    )


class TrainingLogAnalysis(BaseModel):
    """
    Analysis results produced from training logs.

    Training logs are important for identifying failure modes such as:
        - Overfitting
        - Underfitting
        - Training instability
        - Poor learning-rate schedules
        - Early convergence
        - Divergence
    """

    epochs: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of training epochs completed.",
    )

    metrics: Dict[str, List[float]] = Field(
        default_factory=dict,
        description=(
            "Training/validation metric history. "
            "Example: {'train_loss': [...], 'val_loss': [...]}"
        ),
    )

    learning_rate: List[float] = Field(
        default_factory=list,
        description="Learning-rate history across training steps/epochs.",
    )

    checkpoints: List[str] = Field(
        default_factory=list,
        description="Paths or identifiers of available model checkpoints.",
    )

    training_signals: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Derived training signals such as convergence, instability, "
            "loss divergence, or early stopping information."
        ),
    )


class ExperimentArtifacts(BaseModel):
    """
    Unified representation of all artifacts extracted from an experiment.

    This object acts as the primary hand-off contract between the ingestion
    layer and the downstream diagnostic/advisor layers.
    """

    dataset: Optional[DatasetAnalysis] = Field(
        default=None,
        description="Dataset analysis results.",
    )

    model: Optional[ModelAnalysis] = Field(
        default=None,
        description="Model analysis results.",
    )

    training_logs: Optional[TrainingLogAnalysis] = Field(
        default=None,
        description="Training-log analysis results.",
    )

    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters used for the experiment.",
    )


# ============================================================================
# LAYER 3 — PERFORMANCE ANALYSIS
# ============================================================================

class PerformanceResult(BaseModel):
    """
    Performance analysis produced by the Performance Analyzer.

    This model captures both quantitative metrics and supporting evidence.
    The evidence is important because the AI Research Advisor should explain
    recommendations using observable experiment information rather than
    generating unsupported suggestions.
    """

    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Evaluation metrics such as accuracy, F1, RMSE, MAE, "
            "precision, recall, latency, etc."
        ),
    )

    resource_usage: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Resource measurements such as training time, "
            "GPU/CPU utilization, memory consumption, or inference latency."
        ),
    )

    comparisons: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Comparison with previous/baseline experiments."
        ),
    )

    evidence: List[str] = Field(
        default_factory=list,
        description=(
            "Evidence extracted from metrics, logs, artifacts, "
            "or experiment comparisons."
        ),
    )


# ============================================================================
# LAYER 3 — ROOT-CAUSE ANALYSIS
# ============================================================================

class RootCause(BaseModel):
    """
    Represents one diagnosed root cause of model-performance problems.

    Examples from the NeuroPilots architecture include:
        - Overfitting
        - Underfitting
        - Data leakage
        - Class imbalance
        - Domain shift
        - Temporal leakage
        - Noise
        - Hallucination
        - Poor retrieval
        - Training instability
    """

    name: str = Field(
        min_length=1,
        description="Name of the identified failure mode/root cause.",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score assigned to the root-cause diagnosis, "
            "between 0 and 1."
        ),
    )

    evidence: List[str] = Field(
        default_factory=list,
        description="Observed evidence supporting this root cause.",
    )

    explanation: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of the root cause.",
    )


class Diagnosis(BaseModel):
    """
    Complete diagnostic result generated by the diagnostics layer.

    The system can identify one primary root cause and multiple additional
    contributing causes.
    """

    primary_root_cause: Optional[RootCause] = Field(
        default=None,
        description="Most likely root cause of the observed problem.",
    )

    additional_root_causes: List[RootCause] = Field(
        default_factory=list,
        description="Other contributing root causes.",
    )

    performance: Optional[PerformanceResult] = Field(
        default=None,
        description="Performance analysis supporting the diagnosis.",
    )


# ============================================================================
# LAYER 4 — AI RESEARCH ADVISOR
# ============================================================================

class ExperimentRecommendation(BaseModel):
    """
    Represents the next-best experiment recommended by the AI Research
    Advisor.

    A recommendation must describe:
        1. What should change
        2. Why the change is appropriate
        3. What improvement is expected
        4. Approximate cost/effort
        5. Priority
        6. Supporting evidence
    """

    title: str = Field(
        min_length=1,
        description="Short name of the recommended experiment.",
    )

    change: str = Field(
        min_length=1,
        description="Specific change proposed for the experiment.",
    )

    reason: str = Field(
        min_length=1,
        description="Reason the experiment is being recommended.",
    )

    expected_improvement: Optional[str] = Field(
        default=None,
        description="Expected performance improvement.",
    )

    cost: Optional[str] = Field(
        default=None,
        description="Estimated computational or engineering cost.",
    )

    priority: Optional[str] = Field(
        default=None,
        description="Priority of the recommendation.",
    )

    evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting the recommendation.",
    )


# ============================================================================
# LAYER 5 — EXPERIMENT EXECUTION
# ============================================================================

class ExperimentResult(BaseModel):
    """
    Result produced after executing a recommended experiment.

    This model is later consumed by the Feedback Engine to determine whether
    the recommended experiment actually improved the model.
    """

    experiment_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the executed experiment.",
    )

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters/hyperparameters used by the experiment.",
    )

    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Metrics produced by the executed experiment.",
    )

    artifacts: List[str] = Field(
        default_factory=list,
        description="Paths or identifiers of generated artifacts.",
    )

    success: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the experiment completed successfully. "
            "This is execution success, not necessarily model improvement."
        ),
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if the execution failed.",
    )


# ============================================================================
# COMPLETE EXPERIMENT CONTEXT
# ============================================================================

class ExperimentContext(BaseModel):
    """
    Complete state of an experiment as it moves through NeuroPilots.

    This is the central data contract connecting the major system layers.

    Example flow:

        ExperimentContext
              ↓
        Artifact Ingestion
              ↓
        Performance Analysis
              ↓
        Root-Cause Diagnosis
              ↓
        AI Recommendation
              ↓
        Experiment Execution
              ↓
        Feedback / Evaluation

    `domain` and `workload` are intentionally separate:

        domain="automotive"
        workload="computer_vision"

    This allows the same ML workload engine to be reused across different
    application domains.
    """

    domain: Optional[str] = Field(
        default=None,
        description=(
            "Application domain, e.g. automotive, healthcare, finance, "
            "robotics, etc."
        ),
    )

    workload: str = Field(
        min_length=1,
        description=(
            "ML workload type, e.g. computer_vision, nlp, time_series, "
            "tabular, speech, generative_ai, or reinforcement_learning."
        ),
    )

    artifacts: ExperimentArtifacts = Field(
        default_factory=ExperimentArtifacts,
        description="Artifacts extracted from the experiment.",
    )

    performance: Optional[PerformanceResult] = Field(
        default=None,
        description="Performance analysis results.",
    )

    diagnosis: Optional[Diagnosis] = Field(
        default=None,
        description="Root-cause diagnosis.",
    )

    recommendation: Optional[ExperimentRecommendation] = Field(
        default=None,
        description="Next-best experiment recommendation.",
    )

    result: Optional[ExperimentResult] = Field(
        default=None,
        description="Result of the executed experiment.",
    )