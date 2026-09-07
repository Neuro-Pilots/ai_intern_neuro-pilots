"""
Computer Vision workload engine for NeuroPilots.

This module defines the workload-specific behavior for Computer Vision
experiments.

The current responsibility of this layer is to describe and validate
the workload and its model configuration. Actual model loading,
training, evaluation, and experiment execution are handled by the
execution layer.
"""

from __future__ import annotations

from typing import Any, Dict

from utils.constants import WORKLOAD_COMPUTER_VISION
from utils.logger import get_logger
from utils.models import ExperimentArtifacts

from workloads.base import BaseWorkloadEngine


logger = get_logger(__name__)


class ComputerVisionWorkload(BaseWorkloadEngine):
    """
    Workload engine for Computer Vision experiments.

    NeuroPilots uses ConvNeXt-Tiny as the default Computer Vision model
    according to the system architecture.

    This class intentionally does not instantiate the actual model.
    Keeping model execution outside the workload-definition layer
    allows the system to remain framework-agnostic and makes it easier
    to support different execution backends later.
    """

    # Canonical workload identifier used throughout NeuroPilots.
    workload_name = WORKLOAD_COMPUTER_VISION

    # ------------------------------------------------------------------
    # Model specification
    # ------------------------------------------------------------------

    def get_model_spec(self) -> Dict[str, Any]:
        """
        Return the Computer Vision model specification.

        The specification is metadata used by downstream components
        such as the experiment engine, diagnostics engine, advisor,
        and Streamlit UI.

        Returns
        -------
        dict
            Computer Vision model specification.
        """

        return {
            "workload": self.workload_name,
            "model_name": self.default_model_name,
            "model_family": "ConvNeXt",
            "model_variant": "Tiny",
            "task_types": [
                "image_classification",
            ],
            "input_type": "image",
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
        Validate artifacts required by the Computer Vision workload.

        At the workload-engine stage we only validate the presence and
        structure of the artifacts. We do not attempt to inspect image
        files, instantiate neural networks, or perform training here.

        Parameters
        ----------
        artifacts:
            ExperimentArtifacts produced by the ingestion layer.

        Raises
        ------
        ValueError
            If required Computer Vision artifacts are missing or
            inconsistent.
        """

        # A Computer Vision experiment requires dataset information.
        if artifacts.dataset is None:
            raise ValueError(
                "Computer Vision workload requires dataset artifacts."
            )

        # A model or model configuration is required so that the
        # workload engine can reason about the experiment architecture.
        if artifacts.model is None:
            raise ValueError(
                "Computer Vision workload requires model artifacts."
            )

        logger.debug(
            "Computer Vision artifacts validated successfully."
        )

    # ------------------------------------------------------------------
    # Workload-specific experiment information
    # ------------------------------------------------------------------

    def get_experiment_requirements(self) -> Dict[str, Any]:
        """
        Return the information expected from a Computer Vision
        experiment.

        This metadata can later be consumed by the ingestion layer,
        diagnostics engine, advisor, and Streamlit interface.
        """

        return {
            "dataset": {
                "required": True,
                "expected_input_type": "image",
                "important_signals": [
                    "dataset_size",
                    "image_dimensions",
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
                    "input_configuration",
                    "pretrained_configuration",
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
                    "augmentation",
                    "epochs",
                ],
            },
        }

    # ------------------------------------------------------------------
    # Workload description
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """
        Return a complete description of the Computer Vision workload.

        The description combines the common workload metadata with
        Computer Vision-specific requirements.
        """

        description = super().describe()

        description.update(
            {
                "domain": "computer_vision",
                "experiment_requirements": self.get_experiment_requirements(),
            }
        )

        return description