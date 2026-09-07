"""
Shared constants for NeuroPilots.

This module contains controlled vocabulary used across the application.

Keeping these values in one place helps prevent:
    - Typos in workload names
    - Inconsistent failure-mode names
    - Different components using different priority labels
    - Hard-coded strings being duplicated throughout the codebase

These constants represent the terminology defined by the NeuroPilots
architecture and problem statement.
"""


# ============================================================================
# APPLICATION DOMAINS
# ============================================================================

# Application/domain is intentionally separate from ML workload.
#
# Example:
#     domain = "automotive"
#     workload = "computer_vision"
#
# The same workload can later be reused for another application domain.

DOMAIN_AUTOMOTIVE = "automotive"


# ============================================================================
# SUPPORTED ML WORKLOADS
# ============================================================================

# The NeuroPilots architecture defines seven supported ML/DL workloads.

WORKLOAD_COMPUTER_VISION = "computer_vision"
WORKLOAD_NLP = "nlp"
WORKLOAD_TIME_SERIES = "time_series"
WORKLOAD_TABULAR = "tabular"
WORKLOAD_SPEECH = "speech"
WORKLOAD_GENERATIVE_AI = "generative_ai"
WORKLOAD_REINFORCEMENT_LEARNING = "reinforcement_learning"


# Complete list of workloads supported by the system.
#
# Use this collection for validation, routing, and UI selection instead of
# duplicating the workload names in multiple modules.

SUPPORTED_WORKLOADS = (
    WORKLOAD_COMPUTER_VISION,
    WORKLOAD_NLP,
    WORKLOAD_TIME_SERIES,
    WORKLOAD_TABULAR,
    WORKLOAD_SPEECH,
    WORKLOAD_GENERATIVE_AI,
    WORKLOAD_REINFORCEMENT_LEARNING,
)


# ============================================================================
# DEFAULT MODELS BY WORKLOAD
# ============================================================================

# These are the model families specified in the NeuroPilots architecture.
#
# They are represented as strings here because this constants module should
# not import heavyweight ML frameworks or model implementations.

DEFAULT_MODEL_BY_WORKLOAD = {
    WORKLOAD_COMPUTER_VISION: "ConvNeXt-Tiny",
    WORKLOAD_NLP: "ModernBERT-base",
    WORKLOAD_TIME_SERIES: "Chronos-2",
    WORKLOAD_TABULAR: "XGBoost",
    WORKLOAD_SPEECH: "Whisper-large-v3-turbo",
    WORKLOAD_GENERATIVE_AI: "Llama 3.2 3B + RAG",
    WORKLOAD_REINFORCEMENT_LEARNING: "PPO",
}


# ============================================================================
# ROOT-CAUSE / FAILURE MODES
# ============================================================================

# Failure modes defined in the NeuroPilots architecture.
#
# These values will later be used by the Root-Cause Engine and the LLM
# Research Advisor.

FAILURE_MODE_OVERFITTING = "overfitting"
FAILURE_MODE_UNDERFITTING = "underfitting"
FAILURE_MODE_DATA_LEAKAGE = "data_leakage"
FAILURE_MODE_CLASS_IMBALANCE = "class_imbalance"
FAILURE_MODE_DOMAIN_SHIFT = "domain_shift"
FAILURE_MODE_TEMPORAL_LEAKAGE = "temporal_leakage"
FAILURE_MODE_NOISE = "noise"
FAILURE_MODE_HALLUCINATION = "hallucination"
FAILURE_MODE_POOR_RETRIEVAL = "poor_retrieval"
FAILURE_MODE_TRAINING_INSTABILITY = "training_instability"


# Complete list of supported failure modes.

SUPPORTED_FAILURE_MODES = (
    FAILURE_MODE_OVERFITTING,
    FAILURE_MODE_UNDERFITTING,
    FAILURE_MODE_DATA_LEAKAGE,
    FAILURE_MODE_CLASS_IMBALANCE,
    FAILURE_MODE_DOMAIN_SHIFT,
    FAILURE_MODE_TEMPORAL_LEAKAGE,
    FAILURE_MODE_NOISE,
    FAILURE_MODE_HALLUCINATION,
    FAILURE_MODE_POOR_RETRIEVAL,
    FAILURE_MODE_TRAINING_INSTABILITY,
)


# ============================================================================
# RECOMMENDATION PRIORITIES
# ============================================================================

# Priority levels used by the Next-Best Experiment component.

PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"


SUPPORTED_PRIORITIES = (
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,
)


# ============================================================================
# EXPERIMENT STATUS
# ============================================================================

# These statuses describe the lifecycle of an experiment.
#
# They are intentionally independent of the success field in ExperimentResult.
# An experiment can, for example, complete successfully but fail to improve
# model performance.

EXPERIMENT_STATUS_PENDING = "pending"
EXPERIMENT_STATUS_RUNNING = "running"
EXPERIMENT_STATUS_COMPLETED = "completed"
EXPERIMENT_STATUS_FAILED = "failed"


SUPPORTED_EXPERIMENT_STATUSES = (
    EXPERIMENT_STATUS_PENDING,
    EXPERIMENT_STATUS_RUNNING,
    EXPERIMENT_STATUS_COMPLETED,
    EXPERIMENT_STATUS_FAILED,
)


# ============================================================================
# FEEDBACK OUTCOMES
# ============================================================================

# Used later by the Feedback Engine to determine whether a recommendation
# actually helped the model.

FEEDBACK_IMPROVED = "improved"
FEEDBACK_NO_IMPROVEMENT = "no_improvement"
FEEDBACK_DEGRADED = "degraded"
FEEDBACK_INCONCLUSIVE = "inconclusive"


SUPPORTED_FEEDBACK_OUTCOMES = (
    FEEDBACK_IMPROVED,
    FEEDBACK_NO_IMPROVEMENT,
    FEEDBACK_DEGRADED,
    FEEDBACK_INCONCLUSIVE,
)