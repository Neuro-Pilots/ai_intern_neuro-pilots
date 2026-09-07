"""
MLflow experiment tracking for NeuroPilots.

This module provides a centralized interface for logging experiments,
parameters, metrics, diagnostics, recommendations, and artifacts to MLflow.

The tracker is intentionally independent from the experiment execution
logic. ExperimentExecutor produces results; MLflowTracker records them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import mlflow

from config.settings import MLFLOW_TRACKING_URI
from utils.logger import get_logger
from utils.models import (
    Diagnosis,
    ExperimentRecommendation,
    ExperimentResult,
)

LOGGER = get_logger(__name__)


DEFAULT_EXPERIMENT_NAME = "NeuroPilots"


@dataclass
class MLflowRunResult:
    """
    Result returned after an MLflow run is created and logged.
    """

    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    run_name: Optional[str] = None
    status: str = "failed"

    logged_parameters: list[str] = field(default_factory=list)
    logged_metrics: list[str] = field(default_factory=list)
    logged_artifacts: list[str] = field(default_factory=list)

    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return True when the MLflow run was successfully logged."""
        return self.status == "completed" and self.run_id is not None


class MLflowTracker:
    """
    Centralized MLflow tracking interface for NeuroPilots.

    Responsibilities:
    - Configure the MLflow tracking URI.
    - Create/select the NeuroPilots experiment.
    - Start and finish MLflow runs.
    - Log experiment parameters.
    - Log experiment metrics.
    - Log diagnosis and recommendation metadata.
    - Log generated artifacts.
    - Return a structured tracking result.

    The tracker does not perform model training or evaluation.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    ) -> None:
        """
        Initialize the MLflow tracker.

        Args:
            tracking_uri:
                MLflow tracking server URI. If omitted, the value from
                config.settings is used.

            experiment_name:
                MLflow experiment under which runs are stored.
        """
        if not experiment_name or not experiment_name.strip():
            raise ValueError("experiment_name must not be empty.")

        self.tracking_uri = (
            tracking_uri.strip()
            if tracking_uri
            else MLFLOW_TRACKING_URI
        )
        self.experiment_name = experiment_name.strip()

        mlflow.set_tracking_uri(self.tracking_uri)

        self.experiment_id = self._get_or_create_experiment()

        LOGGER.info(
            "MLflowTracker initialized: uri=%s, experiment=%s, "
            "experiment_id=%s",
            self.tracking_uri,
            self.experiment_name,
            self.experiment_id,
        )

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def _get_or_create_experiment(self) -> str:
        """
        Get an existing MLflow experiment or create it.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is not None:
            return experiment.experiment_id

        experiment_id = mlflow.create_experiment(self.experiment_name)

        LOGGER.info(
            "Created MLflow experiment: name=%s, experiment_id=%s",
            self.experiment_name,
            experiment_id,
        )

        return experiment_id

    # ------------------------------------------------------------------
    # Run logging
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        experiment_result: ExperimentResult,
        *,
        workload: Optional[str] = None,
        domain: Optional[str] = None,
        diagnosis: Optional[Diagnosis] = None,
        recommendation: Optional[ExperimentRecommendation] = None,
        run_name: Optional[str] = None,
        extra_parameters: Optional[Mapping[str, Any]] = None,
        extra_metrics: Optional[Mapping[str, float]] = None,
        tags: Optional[Mapping[str, str]] = None,
    ) -> MLflowRunResult:
        """
        Log a completed NeuroPilots experiment to MLflow.

        Args:
            experiment_result:
                Result produced by ExperimentExecutor.

            workload:
                Workload type, for example "computer_vision".

            domain:
                Domain/use case, for example "automotive".

            diagnosis:
                Root-cause diagnosis produced by RootCauseEngine.

            recommendation:
                Recommendation produced by RecommendationEngine.

            run_name:
                Optional human-readable MLflow run name.

            extra_parameters:
                Additional parameters to log.

            extra_metrics:
                Additional numeric metrics to log.

            tags:
                Additional MLflow tags.

        Returns:
            MLflowRunResult describing the tracking operation.
        """
        if not isinstance(experiment_result, ExperimentResult):
            raise TypeError(
                "experiment_result must be an ExperimentResult instance."
            )

        resolved_run_name = (
            run_name.strip()
            if run_name and run_name.strip()
            else experiment_result.experiment_id or "neuropilots-experiment"
        )

        result = MLflowRunResult(
            run_name=resolved_run_name,
        )

        LOGGER.info(
            "Logging experiment to MLflow: experiment_id=%s, run_name=%s",
            experiment_result.experiment_id,
            resolved_run_name,
        )

        try:
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=resolved_run_name,
            ) as run:
                result.run_id = run.info.run_id
                result.experiment_id = run.info.experiment_id

                self._log_experiment_parameters(
                    experiment_result=experiment_result,
                    workload=workload,
                    domain=domain,
                    extra_parameters=extra_parameters,
                    result=result,
                )

                self._log_experiment_metrics(
                    experiment_result=experiment_result,
                    extra_metrics=extra_metrics,
                    result=result,
                )

                self._log_diagnosis(
                    diagnosis=diagnosis,
                )

                self._log_recommendation(
                    recommendation=recommendation,
                )

                self._log_tags(
                    experiment_result=experiment_result,
                    workload=workload,
                    domain=domain,
                    diagnosis=diagnosis,
                    recommendation=recommendation,
                    tags=tags,
                )

                self._log_artifacts(
                    experiment_result=experiment_result,
                    result=result,
                )

                result.status = "completed"

                LOGGER.info(
                    "MLflow experiment logged successfully: "
                    "run_id=%s, experiment_id=%s",
                    result.run_id,
                    result.experiment_id,
                )

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)

            LOGGER.exception(
                "Failed to log experiment to MLflow: %s",
                exc,
            )

        return result

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _log_experiment_parameters(
        self,
        *,
        experiment_result: ExperimentResult,
        workload: Optional[str],
        domain: Optional[str],
        extra_parameters: Optional[Mapping[str, Any]],
        result: MLflowRunResult,
    ) -> None:
        """
        Log experiment configuration as MLflow parameters.
        """
        parameters: dict[str, Any] = {}

        if experiment_result.experiment_id:
            parameters["experiment_id"] = experiment_result.experiment_id

        if workload:
            parameters["workload"] = workload

        if domain:
            parameters["domain"] = domain

        parameters.update(experiment_result.parameters)

        if extra_parameters:
            parameters.update(dict(extra_parameters))

        normalized_parameters = self._normalize_parameters(parameters)

        if normalized_parameters:
            mlflow.log_params(normalized_parameters)
            result.logged_parameters.extend(
                normalized_parameters.keys()
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _log_experiment_metrics(
        self,
        *,
        experiment_result: ExperimentResult,
        extra_metrics: Optional[Mapping[str, float]],
        result: MLflowRunResult,
    ) -> None:
        """
        Log numeric experiment metrics to MLflow.
        """
        metrics: dict[str, float] = {}

        metrics.update(experiment_result.metrics)

        if extra_metrics:
            metrics.update(dict(extra_metrics))

        normalized_metrics = self._normalize_metrics(metrics)

        if normalized_metrics:
            mlflow.log_metrics(normalized_metrics)
            result.logged_metrics.extend(
                normalized_metrics.keys()
            )

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    def _log_diagnosis(
        self,
        *,
        diagnosis: Optional[Diagnosis],
    ) -> None:
        """
        Log root-cause diagnosis information as MLflow tags.
        """
        if diagnosis is None:
            return

        if diagnosis.primary_root_cause is not None:
            mlflow.set_tag(
                "primary_root_cause",
                diagnosis.primary_root_cause.name,
            )

            mlflow.set_tag(
                "root_cause_confidence",
                str(diagnosis.primary_root_cause.confidence),
            )

        if diagnosis.additional_root_causes:
            additional_causes = ",".join(
                cause.name
                for cause in diagnosis.additional_root_causes
            )

            mlflow.set_tag(
                "additional_root_causes",
                additional_causes,
            )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def _log_recommendation(
        self,
        *,
        recommendation: Optional[ExperimentRecommendation],
    ) -> None:
        """
        Log AI Research Advisor recommendation metadata.
        """
        if recommendation is None:
            return

        mlflow.set_tag(
            "recommendation_title",
            self._safe_tag_value(recommendation.title),
        )

        mlflow.set_tag(
            "recommendation_priority",
            self._safe_tag_value(recommendation.priority),
        )

        mlflow.set_tag(
            "recommendation_cost",
            self._safe_tag_value(recommendation.cost),
        )

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def _log_tags(
        self,
        *,
        experiment_result: ExperimentResult,
        workload: Optional[str],
        domain: Optional[str],
        diagnosis: Optional[Diagnosis],
        recommendation: Optional[ExperimentRecommendation],
        tags: Optional[Mapping[str, str]],
    ) -> None:
        """
        Log common metadata tags.
        """
        common_tags: dict[str, str] = {
            "application": "NeuroPilots",
        }

        if workload:
            common_tags["workload"] = workload

        if domain:
            common_tags["domain"] = domain

        if experiment_result.success is not None:
            common_tags["experiment_success"] = str(
                experiment_result.success
            )

        if diagnosis is not None:
            common_tags["diagnosis_available"] = "true"

        if recommendation is not None:
            common_tags["recommendation_available"] = "true"

        if tags:
            common_tags.update(
                {
                    str(key): self._safe_tag_value(value)
                    for key, value in tags.items()
                }
            )

        normalized_tags = {
            str(key): self._safe_tag_value(value)
            for key, value in common_tags.items()
        }

        if normalized_tags:
            mlflow.set_tags(normalized_tags)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def _log_artifacts(
        self,
        *,
        experiment_result: ExperimentResult,
        result: MLflowRunResult,
    ) -> None:
        """
        Log experiment artifact files when they exist.

        Non-file artifact references are intentionally ignored here.
        They can still be retained in ExperimentResult for downstream use.
        """
        for artifact in experiment_result.artifacts:
            if not artifact:
                continue

            artifact_path = Path(str(artifact))

            if not artifact_path.exists():
                LOGGER.warning(
                    "Experiment artifact does not exist, skipping: %s",
                    artifact_path,
                )
                continue

            if not artifact_path.is_file():
                LOGGER.warning(
                    "Experiment artifact is not a file, skipping: %s",
                    artifact_path,
                )
                continue

            mlflow.log_artifact(str(artifact_path))
            result.logged_artifacts.append(str(artifact_path))

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_parameters(
        parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Convert parameter values into MLflow-compatible scalar values.
        """
        normalized: dict[str, Any] = {}

        for key, value in parameters.items():
            if value is None:
                continue

            normalized_key = str(key).strip()

            if not normalized_key:
                continue

            if isinstance(value, (str, int, float, bool)):
                normalized[normalized_key] = value
            else:
                normalized[normalized_key] = str(value)

        return normalized

    @staticmethod
    def _normalize_metrics(
        metrics: Mapping[str, Any],
    ) -> dict[str, float]:
        """
        Keep only finite numeric metric values.
        """
        normalized: dict[str, float] = {}

        for key, value in metrics.items():
            if value is None:
                continue

            try:
                metric_value = float(value)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Skipping non-numeric MLflow metric: %s=%r",
                    key,
                    value,
                )
                continue

            if metric_value != metric_value:
                LOGGER.warning(
                    "Skipping NaN MLflow metric: %s",
                    key,
                )
                continue

            if metric_value in (float("inf"), float("-inf")):
                LOGGER.warning(
                    "Skipping infinite MLflow metric: %s",
                    key,
                )
                continue

            normalized_key = str(key).strip()

            if not normalized_key:
                continue

            normalized[normalized_key] = metric_value

        return normalized

    @staticmethod
    def _safe_tag_value(value: Any) -> str:
        """
        Convert optional values into safe MLflow tag strings.
        """
        if value is None:
            return "unknown"

        return str(value).strip() or "unknown"

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: Optional[str] = None,
    ) -> Any:
        """
        Start a manual MLflow run.

        This is provided for workflows that need to log incrementally
        rather than using log_experiment().
        """
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
        )

    @staticmethod
    def log_parameters(
        parameters: Mapping[str, Any],
    ) -> None:
        """Log parameters to the currently active MLflow run."""
        normalized = MLflowTracker._normalize_parameters(parameters)

        if normalized:
            mlflow.log_params(normalized)

    @staticmethod
    def log_metrics(
        metrics: Mapping[str, Any],
    ) -> None:
        """Log metrics to the currently active MLflow run."""
        normalized = MLflowTracker._normalize_metrics(metrics)

        if normalized:
            mlflow.log_metrics(normalized)

    @staticmethod
    def log_artifact(
        artifact_path: str,
    ) -> None:
        """Log one artifact file to the currently active MLflow run."""
        path = Path(artifact_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Artifact does not exist: {artifact_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Artifact path is not a file: {artifact_path}"
            )

        mlflow.log_artifact(str(path))

    @staticmethod
    def end_run() -> None:
        """End the currently active MLflow run."""
        mlflow.end_run()

    def __repr__(self) -> str:
        """Return a concise tracker representation."""
        return (
            f"MLflowTracker("
            f"tracking_uri={self.tracking_uri!r}, "
            f"experiment_name={self.experiment_name!r}, "
            f"experiment_id={self.experiment_id!r})"
        )