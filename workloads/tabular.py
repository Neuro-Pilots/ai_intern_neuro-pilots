"""
Tabular ML workload engine for NeuroPilots.

This module defines the workload-specific behavior for tabular
machine-learning experiments.

The current responsibility of this layer is to:
    - Identify the Tabular ML workload.
    - Expose XGBoost as the default model.
    - Validate tabular experiment artifacts.
    - Describe tabular-specific experiment requirements.

Actual model training, preprocessing, evaluation, and experiment
execution are handled by later layers.
"""

from __future__ import annotations

from typing import Any, Dict

from utils.constants import WORKLOAD_TABULAR
from utils.logger import get_logger
from utils.models import ExperimentArtifacts

from workloads.base import BaseWorkloadEngine


logger = get_logger(__name__)


class TabularWorkload(BaseWorkloadEngine):
    """
    Workload engine for Tabular Machine Learning experiments.

    NeuroPilots uses XGBoost as the default Tabular ML model according
    to the system architecture.

    This class provides workload metadata and artifact validation only.
    It does not instantiate, train, or evaluate the actual XGBoost
    model.
    """

    # Canonical workload identifier used throughout NeuroPilots.
    workload_name = WORKLOAD_TABULAR

    # ------------------------------------------------------------------
    # Model specification
    # ------------------------------------------------------------------

    def get_model_spec(self) -> Dict[str, Any]:
        """
        Return the Tabular ML model specification.

        Returns
        -------
        dict
            Tabular model metadata used by downstream components.
        """

        return {
            "workload": self.workload_name,
            "model_name": self.default_model_name,
            "model_family": "XGBoost",
            "model_variant": "gradient_boosted_trees",
            "task_types": [
                "classification",
                "regression",
            ],
            "input_type": "tabular",
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
        Validate artifacts required by the Tabular ML workload.

        The tabular workload requires both dataset and model
        information. Detailed feature preprocessing and target
        validation are handled by the appropriate ingestion/execution
        components.

        Parameters
        ----------
        artifacts:
            ExperimentArtifacts produced by the ingestion layer.

        Raises
        ------
        ValueError
            If required Tabular ML artifacts are missing.
        """

        if artifacts.dataset is None:
            raise ValueError(
                "Tabular workload requires dataset artifacts."
            )

        if artifacts.model is None:
            raise ValueError(
                "Tabular workload requires model artifacts."
            )

        logger.debug(
            "Tabular workload artifacts validated successfully."
        )

    # ------------------------------------------------------------------
    # Experiment requirements
    # ------------------------------------------------------------------

    def get_experiment_requirements(self) -> Dict[str, Any]:
        """
        Return the information expected from a Tabular ML experiment.

        These requirements describe the signals that downstream
        components should consider when analyzing a tabular experiment.
        """

        return {
            "dataset": {
                "required": True,
                "expected_input_type": "tabular",
                "important_signals": [
                    "dataset_size",
                    "feature_count",
                    "numerical_features",
                    "categorical_features",
                    "class_distribution",
                    "missing_values",
                    "duplicate_rows",
                    "constant_features",
                    "outliers",
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
                    "feature_importance",
                ],
            },
            "training_logs": {
                "required": False,
                "important_signals": [
                    "train_loss",
                    "validation_loss",
                    "train_metric",
                    "validation_metric",
                    "learning_rate",
                    "epoch_progression",
                ],
            },
            "hyperparameters": {
                "important_parameters": [
                    "learning_rate",
                    "max_depth",
                    "n_estimators",
                    "subsample",
                    "colsample_bytree",
                    "min_child_weight",
                    "gamma",
                    "reg_alpha",
                    "reg_lambda",
                ],
            },
            "preprocessing": {
                "important_parameters": [
                    "missing_value_strategy",
                    "categorical_encoding",
                    "numerical_scaling",
                    "feature_selection",
                    "feature_engineering",
                ],
            },
        }

    # ------------------------------------------------------------------
    # Workload description
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """
        Return a complete description of the Tabular ML workload.

        Combines common workload metadata with Tabular ML-specific
        experiment requirements.
        """

        description = super().describe()

        description.update(
            {
                "domain": "tabular_machine_learning",
                "experiment_requirements": self.get_experiment_requirements(),
            }
        )

        return description