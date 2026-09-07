"""
Failure mode definitions for NeuroPilots diagnostics.

This module provides the centralized diagnostic vocabulary used by
the NeuroPilots Performance Analyzer and Root-Cause Engine.

The failure modes are derived from the NeuroPilots architecture:

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

This module does not determine which failure mode is actually present.
It only defines the known failure modes and the evidence that can be
used by later diagnostic components.

Actual diagnosis is performed by:
    diagnostics.performance_analyzer
    diagnostics.root_cause_engine
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from utils.constants import (
    FAILURE_MODE_CLASS_IMBALANCE,
    FAILURE_MODE_DATA_LEAKAGE,
    FAILURE_MODE_DOMAIN_SHIFT,
    FAILURE_MODE_HALLUCINATION,
    FAILURE_MODE_NOISE,
    FAILURE_MODE_OVERFITTING,
    FAILURE_MODE_POOR_RETRIEVAL,
    FAILURE_MODE_TEMPORAL_LEAKAGE,
    FAILURE_MODE_TRAINING_INSTABILITY,
    FAILURE_MODE_UNDERFITTING,
    SUPPORTED_FAILURE_MODES,
)
from utils.logger import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Failure mode metadata
# ---------------------------------------------------------------------------
#
# Each failure mode contains:
#
#   description:
#       Human-readable explanation of the failure mode.
#
#   evidence_signals:
#       Observable signals that may provide evidence for the failure.
#
#   affected_workloads:
#       Workloads where this failure mode is particularly relevant.
#
#   diagnostic_questions:
#       Questions the Root-Cause Engine should consider.
#
#   severity:
#       Default diagnostic importance. This is NOT an experiment-specific
#       severity; the actual severity should be determined later from evidence.
#
# ---------------------------------------------------------------------------

_FAILURE_MODE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    FAILURE_MODE_OVERFITTING: {
        "name": FAILURE_MODE_OVERFITTING,
        "description": (
            "The model performs well on training data but generalizes "
            "poorly to validation or evaluation data."
        ),
        "evidence_signals": [
            "training_metric_improving",
            "validation_metric_degrading",
            "train_validation_metric_gap",
            "validation_loss_increasing",
            "training_loss_decreasing",
        ],
        "affected_workloads": [
            "computer_vision",
            "nlp",
            "time_series",
            "tabular",
            "speech",
            "generative_ai",
            "reinforcement_learning",
        ],
        "diagnostic_questions": [
            "Is the training metric substantially better than validation?",
            "Is validation performance degrading while training performance improves?",
            "Is the train-validation gap increasing over time?",
            "Does the model have substantially more capacity than the available data supports?",
        ],
        "severity": "medium",
    },
    FAILURE_MODE_UNDERFITTING: {
        "name": FAILURE_MODE_UNDERFITTING,
        "description": (
            "The model fails to adequately learn the underlying patterns "
            "in the training data, resulting in poor training and evaluation performance."
        ),
        "evidence_signals": [
            "poor_training_metric",
            "poor_validation_metric",
            "training_loss_remains_high",
            "training_metric_plateau",
            "validation_metric_plateau",
        ],
        "affected_workloads": [
            "computer_vision",
            "nlp",
            "time_series",
            "tabular",
            "speech",
            "generative_ai",
            "reinforcement_learning",
        ],
        "diagnostic_questions": [
            "Is training performance itself poor?",
            "Is training loss remaining high?",
            "Has the training metric plateaued at an unsatisfactory level?",
            "Is the model capacity insufficient for the task?",
        ],
        "severity": "medium",
    },
    FAILURE_MODE_DATA_LEAKAGE: {
        "name": FAILURE_MODE_DATA_LEAKAGE,
        "description": (
            "Information that should be unavailable to the model during "
            "training is unintentionally exposed through the training data "
            "or preprocessing pipeline."
        ),
        "evidence_signals": [
            "unexpectedly_high_validation_metric",
            "duplicate_samples_across_splits",
            "target_information_in_features",
            "preprocessing_before_split",
            "suspicious_feature_correlation",
            "train_test_overlap",
        ],
        "affected_workloads": [
            "computer_vision",
            "nlp",
            "tabular",
            "speech",
        ],
        "diagnostic_questions": [
            "Are samples duplicated across training and validation sets?",
            "Do any input features contain target information?",
            "Was preprocessing performed before dataset splitting?",
            "Is validation performance unexpectedly close to or better than training performance?",
        ],
        "severity": "high",
    },
    FAILURE_MODE_CLASS_IMBALANCE: {
        "name": FAILURE_MODE_CLASS_IMBALANCE,
        "description": (
            "The target classes are distributed unevenly enough that "
            "standard training or evaluation may favor majority classes."
        ),
        "evidence_signals": [
            "class_distribution",
            "minority_class_percentage",
            "majority_minority_ratio",
            "per_class_metric_gap",
            "low_minority_class_recall",
            "low_minority_class_precision",
        ],
        "affected_workloads": [
            "computer_vision",
            "nlp",
            "tabular",
            "speech",
        ],
        "diagnostic_questions": [
            "Is one class substantially more frequent than the others?",
            "Is the minority-class representation too small?",
            "Are aggregate metrics hiding poor minority-class performance?",
            "Is the model biased toward the majority class?",
        ],
        "severity": "medium",
    },
    FAILURE_MODE_DOMAIN_SHIFT: {
        "name": FAILURE_MODE_DOMAIN_SHIFT,
        "description": (
            "The distribution of data encountered during evaluation or "
            "deployment differs from the distribution used during training."
        ),
        "evidence_signals": [
            "feature_distribution_shift",
            "label_distribution_shift",
            "training_evaluation_metric_gap",
            "covariate_shift",
            "deployment_data_difference",
        ],
        "affected_workloads": [
            "computer_vision",
            "nlp",
            "time_series",
            "tabular",
            "speech",
            "generative_ai",
        ],
        "diagnostic_questions": [
            "Does evaluation data differ from training data?",
            "Have important input distributions changed?",
            "Does performance degrade substantially on deployment-like data?",
            "Are environmental or operating conditions different?",
        ],
        "severity": "high",
    },
    FAILURE_MODE_TEMPORAL_LEAKAGE: {
        "name": FAILURE_MODE_TEMPORAL_LEAKAGE,
        "description": (
            "Future information is unintentionally used when training or "
            "evaluating a model for a time-dependent prediction task."
        ),
        "evidence_signals": [
            "future_information_in_features",
            "random_split_on_temporal_data",
            "timestamp_order_violation",
            "future_target_information",
            "unexpectedly_high_validation_metric",
        ],
        "affected_workloads": [
            "time_series",
        ],
        "diagnostic_questions": [
            "Does training use information from the future relative to prediction time?",
            "Was a random split used for inherently temporal data?",
            "Are timestamps ordered correctly?",
            "Could a feature contain information that would not be available at inference time?",
        ],
        "severity": "high",
    },
    FAILURE_MODE_NOISE: {
        "name": FAILURE_MODE_NOISE,
        "description": (
            "The dataset or training signal contains irrelevant, corrupted, "
            "inconsistent, or highly variable information that makes learning harder."
        ),
        "evidence_signals": [
            "missing_values",
            "duplicate_samples",
            "outliers",
            "label_inconsistency",
            "high_feature_variance",
            "corrupted_inputs",
            "high_reward_variance",
        ],
        "affected_workloads": [
            "computer_vision",
            "nlp",
            "time_series",
            "tabular",
            "speech",
            "generative_ai",
            "reinforcement_learning",
        ],
        "diagnostic_questions": [
            "Does the dataset contain corrupted or inconsistent samples?",
            "Are there significant outliers?",
            "Are labels or targets noisy?",
            "Are training signals highly variable or unstable?",
        ],
        "severity": "medium",
    },
    FAILURE_MODE_HALLUCINATION: {
        "name": FAILURE_MODE_HALLUCINATION,
        "description": (
            "A Generative AI system produces information that is unsupported, "
            "incorrect, or not grounded in the available source information."
        ),
        "evidence_signals": [
            "hallucination_rate",
            "groundedness",
            "faithfulness",
            "unsupported_claims",
            "source_answer_mismatch",
            "factual_error_rate",
        ],
        "affected_workloads": [
            "generative_ai",
        ],
        "diagnostic_questions": [
            "Are generated answers supported by retrieved source documents?",
            "Does the response contain unsupported claims?",
            "Is groundedness or faithfulness below the expected threshold?",
            "Does hallucination increase when retrieval quality decreases?",
        ],
        "severity": "high",
    },
    FAILURE_MODE_POOR_RETRIEVAL: {
        "name": FAILURE_MODE_POOR_RETRIEVAL,
        "description": (
            "A Retrieval-Augmented Generation system fails to retrieve "
            "relevant or sufficient information for the user's query."
        ),
        "evidence_signals": [
            "retrieval_precision",
            "retrieval_recall",
            "retrieval_relevance",
            "retrieved_document_count",
            "similarity_score",
            "irrelevant_documents",
        ],
        "affected_workloads": [
            "generative_ai",
        ],
        "diagnostic_questions": [
            "Are relevant documents being retrieved?",
            "Is retrieval recall too low?",
            "Are retrieved documents relevant to the query?",
            "Is the retrieval configuration too restrictive?",
            "Could chunk size, overlap, embedding model, or top-k be limiting retrieval quality?",
        ],
        "severity": "high",
    },
    FAILURE_MODE_TRAINING_INSTABILITY: {
        "name": FAILURE_MODE_TRAINING_INSTABILITY,
        "description": (
            "Training metrics or optimization behavior fluctuate, diverge, "
            "or fail to converge reliably."
        ),
        "evidence_signals": [
            "loss_oscillation",
            "loss_divergence",
            "metric_oscillation",
            "learning_rate_instability",
            "gradient_instability",
            "high_reward_variance",
            "policy_loss_instability",
            "value_loss_instability",
            "high_approx_kl",
        ],
        "affected_workloads": [
            "computer_vision",
            "nlp",
            "time_series",
            "tabular",
            "speech",
            "generative_ai",
            "reinforcement_learning",
        ],
        "diagnostic_questions": [
            "Are training metrics oscillating rather than converging?",
            "Is the loss diverging?",
            "Is the learning rate causing unstable optimization?",
            "Are gradients unstable?",
            "For RL, are policy or value losses unstable?",
            "For RL, is KL divergence becoming excessively large?",
        ],
        "severity": "high",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_failure_mode_definition(failure_mode: str) -> Dict[str, Any]:
    """
    Return the definition for a single failure mode.

    Args:
        failure_mode: Failure mode identifier.

    Returns:
        A copy of the failure mode definition.

    Raises:
        ValueError:
            If the failure mode is not supported.
    """

    if failure_mode not in SUPPORTED_FAILURE_MODES:
        raise ValueError(
            f"Unsupported failure mode: {failure_mode!r}. "
            f"Supported modes: {SUPPORTED_FAILURE_MODES}"
        )

    return dict(_FAILURE_MODE_DEFINITIONS[failure_mode])


def get_all_failure_mode_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Return definitions for all supported failure modes.

    Returns:
        Dictionary mapping failure mode identifiers to their definitions.
    """

    return {
        failure_mode: dict(definition)
        for failure_mode, definition in _FAILURE_MODE_DEFINITIONS.items()
    }


def get_evidence_signals(failure_mode: str) -> Tuple[str, ...]:
    """
    Return observable evidence signals associated with a failure mode.

    Args:
        failure_mode: Failure mode identifier.

    Returns:
        Tuple of evidence signal names.
    """

    definition = get_failure_mode_definition(failure_mode)

    return tuple(definition["evidence_signals"])


def get_affected_workloads(failure_mode: str) -> Tuple[str, ...]:
    """
    Return workloads for which a failure mode is relevant.

    Args:
        failure_mode: Failure mode identifier.

    Returns:
        Tuple of workload names.
    """

    definition = get_failure_mode_definition(failure_mode)

    return tuple(definition["affected_workloads"])


def get_diagnostic_questions(failure_mode: str) -> Tuple[str, ...]:
    """
    Return diagnostic questions associated with a failure mode.

    Args:
        failure_mode: Failure mode identifier.

    Returns:
        Tuple of diagnostic questions.
    """

    definition = get_failure_mode_definition(failure_mode)

    return tuple(definition["diagnostic_questions"])


def is_failure_mode_supported(failure_mode: str) -> bool:
    """
    Check whether a failure mode is supported by NeuroPilots.

    Args:
        failure_mode: Failure mode identifier.

    Returns:
        True when supported, otherwise False.
    """

    return failure_mode in SUPPORTED_FAILURE_MODES


def get_failure_modes_for_workload(workload: str) -> Tuple[str, ...]:
    """
    Return failure modes relevant to a specific workload.

    Args:
        workload: Workload identifier.

    Returns:
        Tuple containing relevant failure mode identifiers.
    """

    return tuple(
        failure_mode
        for failure_mode, definition in _FAILURE_MODE_DEFINITIONS.items()
        if workload in definition["affected_workloads"]
    )


def validate_failure_mode_definitions() -> None:
    """
    Validate consistency between the global failure-mode constants and
    the detailed definitions in this module.

    Raises:
        ValueError:
            If a supported failure mode is missing or malformed.
    """

    supported_modes = set(SUPPORTED_FAILURE_MODES)
    defined_modes = set(_FAILURE_MODE_DEFINITIONS)

    missing_definitions = supported_modes - defined_modes
    unexpected_definitions = defined_modes - supported_modes

    if missing_definitions:
        raise ValueError(
            "Missing failure mode definitions: "
            f"{sorted(missing_definitions)}"
        )

    if unexpected_definitions:
        raise ValueError(
            "Failure mode definitions contain unsupported modes: "
            f"{sorted(unexpected_definitions)}"
        )

    required_fields = {
        "name",
        "description",
        "evidence_signals",
        "affected_workloads",
        "diagnostic_questions",
        "severity",
    }

    for failure_mode, definition in _FAILURE_MODE_DEFINITIONS.items():
        missing_fields = required_fields - set(definition)

        if missing_fields:
            raise ValueError(
                f"Failure mode {failure_mode!r} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if definition["name"] != failure_mode:
            raise ValueError(
                f"Failure mode name mismatch for {failure_mode!r}."
            )

        if not definition["evidence_signals"]:
            raise ValueError(
                f"Failure mode {failure_mode!r} has no evidence signals."
            )

        if not definition["affected_workloads"]:
            raise ValueError(
                f"Failure mode {failure_mode!r} has no affected workloads."
            )

        if not definition["diagnostic_questions"]:
            raise ValueError(
                f"Failure mode {failure_mode!r} has no diagnostic questions."
            )

    logger.debug(
        "Validated %d failure mode definitions successfully.",
        len(_FAILURE_MODE_DEFINITIONS),
    )


# Validate the module definitions once when the module is imported.
validate_failure_mode_definitions()


__all__ = [
    "get_failure_mode_definition",
    "get_all_failure_mode_definitions",
    "get_evidence_signals",
    "get_affected_workloads",
    "get_diagnostic_questions",
    "is_failure_mode_supported",
    "get_failure_modes_for_workload",
    "validate_failure_mode_definitions",
]