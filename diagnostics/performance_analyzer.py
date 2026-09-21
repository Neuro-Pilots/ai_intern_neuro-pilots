"""
Performance analysis for NeuroPilots.

This module analyzes experiment artifacts and produces a structured
PerformanceResult.

Responsibilities:
    - Summarize available training and validation metrics.
    - Calculate metric trends.
    - Calculate train/validation gaps where applicable.
    - Compare metrics against an optional baseline.
    - Preserve available resource-usage information.
    - Produce evidence that can later be consumed by the Root-Cause Engine.

This module does NOT determine the root cause of poor performance.

For example, a large train/validation gap is reported as evidence.
The Root-Cause Engine is responsible for deciding whether that evidence
supports overfitting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.helpers import calculate_absolute_change, calculate_relative_change
from utils.logger import get_logger
from utils.models import ExperimentArtifacts, PerformanceResult


logger = get_logger(__name__)


class PerformanceAnalyzer:
    """
    Analyze performance information contained in experiment artifacts.

    The analyzer is framework-agnostic. It does not train models or run
    evaluation itself. It works with metrics already available through
    TrainingLogAnalysis and optional externally supplied evaluation/resource
    information.
    """

    def __init__(
        self,
        primary_metric: Optional[str] = None,
        higher_is_better: Optional[bool] = None,
    ) -> None:
        """
        Initialize the performance analyzer.

        Args:
            primary_metric:
                Optional metric to treat as the primary metric.

            higher_is_better:
                Whether a higher value represents better performance.
                When omitted, the analyzer does not assume direction.
        """

        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        artifacts: ExperimentArtifacts,
        *,
        evaluation_metrics: Optional[Dict[str, Any]] = None,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        resource_usage: Optional[Dict[str, Any]] = None,
    ) -> PerformanceResult:
        """
        Analyze the available experiment performance information.

        Args:
            artifacts:
                Experiment artifacts containing training-log information.

            evaluation_metrics:
                Optional metrics produced by an evaluation run.

            baseline_metrics:
                Optional metrics from a previous/baseline experiment.

            resource_usage:
                Optional CPU/GPU/memory/latency information.

        Returns:
            PerformanceResult containing metrics, comparisons,
            resource usage, and evidence.

        Raises:
            ValueError:
                If artifacts is not a valid ExperimentArtifacts instance.
        """

        if not isinstance(artifacts, ExperimentArtifacts):
            raise ValueError(
                "artifacts must be an ExperimentArtifacts instance."
            )

        evaluation_metrics = evaluation_metrics or {}
        baseline_metrics = baseline_metrics or {}
        resource_usage = resource_usage or {}

        metrics = self._collect_metrics(
            artifacts=artifacts,
            evaluation_metrics=evaluation_metrics,
        )

        comparisons = self._build_comparisons(
            artifacts=artifacts,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
        )

        evidence = self._build_evidence(
            artifacts=artifacts,
            metrics=metrics,
            comparisons=comparisons,
            resource_usage=resource_usage,
        )

        result = PerformanceResult(
            metrics=metrics,
            resource_usage=resource_usage,
            comparisons=comparisons,
            evidence=evidence,
        )

        logger.info(
            "Performance analysis completed: %d metrics, %d evidence items.",
            len(metrics),
            len(evidence),
        )

        return result

    # ------------------------------------------------------------------
    # Metric collection
    # ------------------------------------------------------------------

    def _collect_metrics(
        self,
        artifacts: ExperimentArtifacts,
        evaluation_metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Collect final/best metric summaries from training logs and evaluation.

        Metric keys are normalized into descriptive names where possible.

        Training-log series are summarized using:
            - final value
            - best value
            - first value
            - trend

        Evaluation metrics take precedence over training summaries when
        they use the same metric name because evaluation represents the
        externally supplied evaluation result.
        """

        metrics: Dict[str, float] = {}

        training_logs = artifacts.training_logs

        if training_logs is not None:
            for metric_name, values in training_logs.metrics.items():
                numeric_values = self._numeric_values(values)

                if not numeric_values:
                    continue

                normalized_name = self._normalize_metric_name(metric_name)

                metrics[f"{normalized_name}_final"] = numeric_values[-1]
                metrics[f"{normalized_name}_best"] = self._best_value(
                    normalized_name,
                    numeric_values,
                )
                metrics[f"{normalized_name}_initial"] = numeric_values[0]

                trend = self._calculate_trend(numeric_values)

                if trend is not None:
                    metrics[f"{normalized_name}_trend"] = trend

        for metric_name, value in evaluation_metrics.items():
            numeric_value = self._to_float(value)

            if numeric_value is None:
                continue

            normalized_name = self._normalize_metric_name(metric_name)
            metrics[normalized_name] = numeric_value

        return metrics

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def _build_comparisons(
        self,
        artifacts: ExperimentArtifacts,
        metrics: Dict[str, float],
        baseline_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build comparisons between available performance measurements.

        Comparisons include:
            - train/validation relationships
            - baseline changes
            - primary metric information
        """

        comparisons: Dict[str, Any] = {}

        train_validation = self._compare_train_validation(metrics)

        if train_validation:
            comparisons["train_validation"] = train_validation

        baseline_comparison = self._compare_baseline(
            metrics,
            baseline_metrics,
        )

        if baseline_comparison:
            comparisons["baseline"] = baseline_comparison

        primary_metric_comparison = self._get_primary_metric_comparison(
            metrics
        )

        if primary_metric_comparison:
            comparisons["primary_metric"] = primary_metric_comparison

        if artifacts.training_logs is not None:
            epoch_count = artifacts.training_logs.epochs

            if epoch_count is not None:
                comparisons["training_progress"] = {
                    "epochs": epoch_count,
                }

        return comparisons

    def _compare_train_validation(
        self,
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Compare matching training and validation metrics.

        The comparison is descriptive. It does not label the experiment
        as overfitting or underfitting.
        """

        comparisons: Dict[str, Any] = {}

        candidate_pairs = [
            ("loss", "loss"),
            ("accuracy", "accuracy"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1"),
            ("auc", "auc"),
            ("wer", "wer"),
            ("cer", "cer"),
            ("reward", "reward"),
        ]

        for train_name, validation_name in candidate_pairs:
            train_key = f"train_{train_name}_final"
            validation_key = f"validation_{validation_name}_final"

            if train_key not in metrics or validation_key not in metrics:
                continue

            train_value = metrics[train_key]
            validation_value = metrics[validation_key]

            absolute_gap = calculate_absolute_change(
                train_value,
                validation_value,
            )

            relative_gap = calculate_relative_change(
                train_value,
                validation_value,
            )

            comparisons[train_name] = {
                "train_final": train_value,
                "validation_final": validation_value,
                "absolute_gap": absolute_gap,
                "relative_gap": relative_gap,
            }

        return comparisons

    def _compare_baseline(
        self,
        metrics: Dict[str, float],
        baseline_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare current metrics with baseline metrics.

        Only metrics available in both experiments are compared.
        """

        if not baseline_metrics:
            return {}

        comparison: Dict[str, Any] = {}

        for metric_name, current_value in metrics.items():
            baseline_value = self._find_baseline_value(
                metric_name,
                baseline_metrics,
            )

            if baseline_value is None:
                continue

            comparison[metric_name] = {
                "current": current_value,
                "baseline": baseline_value,
                "absolute_change": calculate_absolute_change(
                    baseline_value,
                    current_value,
                ),
                "relative_change": calculate_relative_change(
                    baseline_value,
                    current_value,
                ),
            }

        return comparison

    def _get_primary_metric_comparison(
        self,
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Return primary-metric information when a primary metric is configured.
        """

        if not self.primary_metric:
            return {}

        metric_name = self._normalize_metric_name(self.primary_metric)

        candidates = [
            metric_name,
            f"{metric_name}_final",
            f"evaluation_{metric_name}",
        ]

        for candidate in candidates:
            if candidate in metrics:
                result: Dict[str, Any] = {
                    "metric": candidate,
                    "value": metrics[candidate],
                }

                if self.higher_is_better is not None:
                    result["higher_is_better"] = self.higher_is_better

                return result

        return {}

    # ------------------------------------------------------------------
    # Evidence generation
    # ------------------------------------------------------------------

    def _build_evidence(
        self,
        artifacts: ExperimentArtifacts,
        metrics: Dict[str, float],
        comparisons: Dict[str, Any],
        resource_usage: Dict[str, Any],
    ) -> List[str]:
        """
        Build human-readable evidence statements.

        These statements describe observations only. They intentionally
        avoid assigning root-cause labels.
        """

        evidence: List[str] = []

        training_logs = artifacts.training_logs

        if training_logs is not None:
            if training_logs.epochs is not None:
                evidence.append(
                    f"Training logs contain {training_logs.epochs} epochs."
                )

            for metric_name, values in training_logs.metrics.items():
                numeric_values = self._numeric_values(values)

                if len(numeric_values) < 2:
                    continue

                trend = self._calculate_trend(numeric_values)

                if trend is None:
                    continue

                direction = (
                    "increased"
                    if trend > 0
                    else "decreased"
                    if trend < 0
                    else "remained approximately stable"
                )

                evidence.append(
                    f"{metric_name} {direction} from "
                    f"{numeric_values[0]:.6g} to "
                    f"{numeric_values[-1]:.6g}."
                )

        train_validation = comparisons.get("train_validation", {})

        for metric_name, comparison in train_validation.items():
            train_value = comparison["train_final"]
            validation_value = comparison["validation_final"]
            relative_gap = comparison["relative_gap"]

            evidence.append(
                f"Final train {metric_name}={train_value:.6g}; "
                f"validation {metric_name}={validation_value:.6g}; "
                f"relative difference={relative_gap:.6g}."
            )

        baseline = comparisons.get("baseline", {})

        for metric_name, comparison in baseline.items():
            evidence.append(
                f"Metric {metric_name} changed from baseline "
                f"{comparison['baseline']:.6g} to "
                f"{comparison['current']:.6g} "
                f"(relative change={comparison['relative_change']:.6g})."
            )

        if resource_usage:
            for resource_name, resource_value in resource_usage.items():
                evidence.append(
                    f"Resource usage reported for {resource_name}: "
                    f"{resource_value}."
                )

        if not evidence:
            evidence.append(
                "No sufficient numeric performance evidence was available "
                "from the supplied artifacts."
            )

        return evidence

    # ------------------------------------------------------------------
    # Numeric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_values(values: Sequence[Any]) -> List[float]:
        """
        Convert a metric sequence into finite float values.
        """

        numeric_values: List[float] = []

        for value in values:
            numeric_value = PerformanceAnalyzer._to_float(value)

            if numeric_value is not None:
                numeric_values.append(numeric_value)

        return numeric_values

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """
        Safely convert a value to float.

        Non-numeric, NaN, and infinite values are ignored.
        """

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if numeric_value != numeric_value:
            return None

        if numeric_value in (float("inf"), float("-inf")):
            return None

        return numeric_value

    @staticmethod
    def _calculate_trend(
        values: Sequence[float],
    ) -> Optional[float]:
        """
        Calculate relative change from the first to the final value.

        Returns:
            Relative change, or None when insufficient information exists.
        """

        if len(values) < 2:
            return None

        return calculate_relative_change(
            values[0],
            values[-1],
        )

    @staticmethod
    def _best_value(
        metric_name: str,
        values: Sequence[float],
    ) -> float:
        """
        Return the best value for a metric.

        Metrics where lower is normally better:
            loss, error, wer, cer

        Other metrics default to higher-is-better.

        This is only a summary heuristic. The Root-Cause Engine should
        use workload-specific context when interpreting the result.
        """

        lower_is_better_tokens = (
            "loss",
            "error",
            "wer",
            "cer",
            "latency",
        )

        metric_name_lower = metric_name.lower()

        if any(
            token in metric_name_lower
            for token in lower_is_better_tokens
        ):
            return min(values)

        return max(values)

    @staticmethod
    def _normalize_metric_name(metric_name: str) -> str:
        """
        Normalize a metric name into a stable snake_case representation.
        """

        normalized = str(metric_name).strip().lower()

        replacements = {
            " ": "_",
            "-": "_",
            "/": "_",
            ".": "_",
        }

        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        while "__" in normalized:
            normalized = normalized.replace("__", "_")

        return normalized

    @staticmethod
    def _find_baseline_value(
        metric_name: str,
        baseline_metrics: Dict[str, Any],
    ) -> Optional[float]:
        """
        Find a matching numeric baseline metric.

        Exact match is preferred, followed by normalized-name matching.
        """

        if metric_name in baseline_metrics:
            return PerformanceAnalyzer._to_float(
                baseline_metrics[metric_name]
            )

        normalized_target = PerformanceAnalyzer._normalize_metric_name(
            metric_name
        )

        for key, value in baseline_metrics.items():
            normalized_key = PerformanceAnalyzer._normalize_metric_name(
                str(key)
            )

            if normalized_key == normalized_target:
                return PerformanceAnalyzer._to_float(value)

        return None


__all__ = ["PerformanceAnalyzer"]