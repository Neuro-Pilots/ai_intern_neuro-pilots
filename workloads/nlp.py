"""
Natural Language Processing workload engine for NeuroPilots.

This module defines the workload-specific behavior for NLP
experiments.

The current responsibility of this layer is to:
    - Identify the NLP workload.
    - Expose ModernBERT-base as the default model.
    - Validate NLP experiment artifacts.
    - Describe NLP-specific experiment requirements.

Actual model loading, tokenization, training, evaluation, and
experiment execution are handled by later layers.
"""

from __future__ import annotations

from typing import Any, Dict

from utils.constants import WORKLOAD_NLP
from utils.logger import get_logger
from utils.models import ExperimentArtifacts

from workloads.base import BaseWorkloadEngine


logger = get_logger(__name__)


class NLPWorkload(BaseWorkloadEngine):
    """
    Workload engine for Natural Language Processing experiments.

    NeuroPilots uses ModernBERT-base as the default NLP model according
    to the system architecture.

    This class provides workload metadata and validation only. It does
    not instantiate or download the actual NLP model.
    """

    # Canonical workload identifier used throughout NeuroPilots.
    workload_name = WORKLOAD_NLP

    # ------------------------------------------------------------------
    # Model specification
    # ------------------------------------------------------------------

    def get_model_spec(self) -> Dict[str, Any]:
        """
        Return the NLP model specification.

        Returns
        -------
        dict
            NLP model metadata used by downstream components.
        """

        return {
            "workload": self.workload_name,
            "model_name": self.default_model_name,
            "model_family": "ModernBERT",
            "model_variant": "base",
            "task_types": [
                "text_classification",
                "sequence_classification",
                "token_classification",
            ],
            "input_type": "text",
            "framework_agnostic": True,
        }

    # ------------------------------------------------------------------
    # Artifact validation
    # ------------------------------------------------------------------

    def _validate_workload_specific_artifacts(
        self,
        artifacts: ExperimentArtifacts,
    ) -> None:
        """
        Validate artifacts required by the NLP workload.

        At this stage we validate the existence of the core dataset and
        model artifacts. Detailed text/tokenization validation belongs
        to the NLP-specific ingestion and execution logic.

        Parameters
        ----------
        artifacts:
            ExperimentArtifacts produced by the ingestion layer.

        Raises
        ------
        ValueError
            If required NLP artifacts are missing.
        """

        if artifacts.dataset is None:
            raise ValueError(
                "NLP workload requires dataset artifacts."
            )

        if artifacts.model is None:
            raise ValueError(
                "NLP workload requires model artifacts."
            )

        logger.debug(
            "NLP artifacts validated successfully."
        )

    # ------------------------------------------------------------------
    # Experiment requirements
    # ------------------------------------------------------------------

    def get_experiment_requirements(self) -> Dict[str, Any]:
        """
        Return the information expected from an NLP experiment.

        These requirements describe the signals that downstream
        components should consider when analyzing an NLP experiment.
        """

        return {
            "dataset": {
                "required": True,
                "expected_input_type": "text",
                "important_signals": [
                    "dataset_size",
                    "text_length_distribution",
                    "class_distribution",
                    "missing_values",
                    "duplicate_samples",
                    "data_quality",
                ],
            },
            "model": {
                "required": True,
                "default_model": self.default_model_name,
                "important_signals": [
                    "architecture",
                    "parameter_count",
                    "configuration",
                    "tokenizer_configuration",
                    "maximum_sequence_length",
                ],
            },
            "training_logs": {
                "required": False,
                "important_signals": [
                    "train_loss",
                    "validation_loss",
                    "train_accuracy",
                    "validation_accuracy",
                    "learning_rate",
                    "epoch_progression",
                ],
            },
            "hyperparameters": {
                "important_parameters": [
                    "learning_rate",
                    "batch_size",
                    "weight_decay",
                    "optimizer",
                    "scheduler",
                    "warmup_steps",
                    "maximum_sequence_length",
                    "epochs",
                ],
            },
            "text_processing": {
                "important_parameters": [
                    "tokenizer",
                    "tokenization_strategy",
                    "sequence_length",
                    "truncation",
                    "padding",
                ],
            },
        }

    # ------------------------------------------------------------------
    # Workload description
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """
        Return a complete description of the NLP workload.

        Combines common workload metadata with NLP-specific experiment
        requirements.
        """

        description = super().describe()

        description.update(
            {
                "domain": "natural_language_processing",
                "experiment_requirements": self.get_experiment_requirements(),
            }
        )

        return description