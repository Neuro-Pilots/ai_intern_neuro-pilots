"""
Time Series workload engine for NeuroPilots.

This module defines the workload-specific behavior for Time Series
experiments.

The current responsibility of this layer is to:
    - Identify the Time Series workload.
    - Expose Chronos-2 as the default model.
    - Validate Time Series experiment artifacts.
    - Describe Time Series-specific experiment requirements.

Actual model loading, forecasting, training, evaluation, and
experiment execution are handled by later layers.
"""

from __future__ import annotations

from typing import Any, Dict

from utils.constants import WORKLOAD_TIME_SERIES
from utils.logger import get_logger
from utils.models import ExperimentArtifacts

from workloads.base import BaseWorkloadEngine


logger = get_logger(__name__)


class TimeSeriesWorkload(BaseWorkloadEngine):
    """
    Workload engine for Time Series experiments.

    NeuroPilots uses Chronos-2 as the default Time Series model
    according to the system architecture.

    This class provides workload metadata and artifact validation only.
    It does not instantiate, download, train, or evaluate the actual
    forecasting model.
    """

    # Canonical workload identifier used throughout NeuroPilots.
    workload_name = WORKLOAD_TIME_SERIES

    # ------------------------------------------------------------------
    # Model specification
    # ------------------------------------------------------------------

    def get_model_spec(self) -> Dict[str, Any]:
        """
        Return the Time Series model specification.

        Returns
        -------
        dict
            Time Series model metadata used by downstream components.
        """

        return {
            "workload": self.workload_name,
            "model_name": self.default_model_name,
            "model_family": "Chronos",
            "model_variant": "2",
            "task_types": [
                "forecasting",
                "time_series_prediction",
            ],
            "input_type": "time_series",
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
        Validate artifacts required by the Time Series workload.

        At this stage we validate the existence of the core dataset
        and model artifacts. Detailed temporal validation belongs to
        the workload-specific data and execution logic.

        Parameters
        ----------
        artifacts:
            ExperimentArtifacts produced by the ingestion layer.

        Raises
        ------
        ValueError
            If required Time Series artifacts are missing.
        """

        if artifacts.dataset is None:
            raise ValueError(
                "Time Series workload requires dataset artifacts."
            )

        if artifacts.model is None:
            raise ValueError(
                "Time Series workload requires model artifacts."
            )

        logger.debug(
            "Time Series artifacts validated successfully."
        )

    # ------------------------------------------------------------------
    # Experiment requirements
    # ------------------------------------------------------------------

    def get_experiment_requirements(self) -> Dict[str, Any]:
        """
        Return the information expected from a Time Series experiment.

        These requirements describe the signals that downstream
        components should consider when analyzing a forecasting
        experiment.
        """

        return {
            "dataset": {
                "required": True,
                "expected_input_type": "time_series",
                "important_signals": [
                    "dataset_size",
                    "timestamp_column",
                    "sampling_frequency",
                    "missing_values",
                    "duplicate_timestamps",
                    "outliers",
                    "trend",
                    "seasonality",
                    "stationarity",
                    "data_quality",
                ],
            },
            "model": {
                "required": True,
                "default_model": self.default_model_name,
                "important_signals": [
                    "architecture",
                    "parameter_count",
                    "context_length",
                    "prediction_length",
                    "configuration",
                ],
            },
            "training_logs": {
                "required": False,
                "important_signals": [
                    "train_loss",
                    "validation_loss",
                    "forecasting_error",
                    "learning_rate",
                    "epoch_progression",
                ],
            },
            "hyperparameters": {
                "important_parameters": [
                    "learning_rate",
                    "batch_size",
                    "context_length",
                    "prediction_length",
                    "forecast_horizon",
                    "epochs",
                ],
            },
            "temporal_processing": {
                "important_parameters": [
                    "timestamp_column",
                    "sampling_frequency",
                    "forecast_horizon",
                    "train_validation_split",
                    "temporal_order",
                ],
            },
        }

    # ------------------------------------------------------------------
    # Workload description
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """
        Return a complete description of the Time Series workload.

        Combines common workload metadata with Time Series-specific
        experiment requirements.
        """

        description = super().describe()

        description.update(
            {
                "domain": "time_series",
                "experiment_requirements": self.get_experiment_requirements(),
            }
        )

        return description