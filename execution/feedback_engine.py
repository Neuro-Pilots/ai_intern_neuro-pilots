"""
Feedback engine for NeuroPilots.

The FeedbackEngine closes the experiment loop by comparing the baseline
experiment with the result of a newly executed experiment.

Flow:

    Baseline metrics
          ↓
    Executed experiment
          ↓
    Metric comparison
          ↓
    Improved / No improvement / Degraded / Inconclusive
          ↓
    Feedback for the next research iteration

Design principles:
    - Recommendation generation remains the responsibility of the LLM.
    - Experiment execution remains the responsibility of ExperimentExecutor.
    - This module only evaluates the observed outcome.
    - Metric direction is explicit: higher-is-better or lower-is-better.
    - No diagnosis-to-recommendation mapping is hardcoded.
    - Missing or ambiguous metrics produce an inconclusive result rather
      than an invented conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from utils.constants import (
    FEEDBACK_DEGRADED,
    FEEDBACK_IMPROVED,
    FEEDBACK_INCONCLUSIVE,
    FEEDBACK_NO_IMPROVEMENT,
)
from utils.helpers import (
    calculate_absolute_change,
    calculate_relative_change,
)
from utils.logger import get_logger
from utils.models import ExperimentResult


LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default comparison configuration
# ---------------------------------------------------------------------------

DEFAULT_IMPROVEMENT_TOLERANCE = 0.0


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class MetricComparison:
    """
    Comparison between a baseline metric and an experiment metric.

    Attributes:
        metric:
            Metric name.

        baseline:
            Baseline value.

        experiment:
            Experiment value.

        absolute_change:
            Experiment minus baseline.

        relative_change:
            Relative change.

        higher_is_better:
            Whether a larger value represents better performance.

        improved:
            Whether this metric improved beyond the configured tolerance.

        degraded:
            Whether this metric degraded beyond the configured tolerance.

        within_tolerance:
            Whether the change is too small to classify as meaningful.

        explanation:
            Human-readable interpretation.
    """

    metric: str
    baseline: float
    experiment: float
    absolute_change: float
    relative_change: float
    higher_is_better: bool
    improved: bool
    degraded: bool
    within_tolerance: bool
    explanation: str


@dataclass
class FeedbackResult:
    """
    Result produced by FeedbackEngine.

    Attributes:
        outcome:
            One of:
                improved
                no_improvement
                degraded
                inconclusive

        experiment_id:
            Identifier of the evaluated experiment.

        primary_metric:
            Primary metric used for the final decision.

        baseline_value:
            Baseline primary metric value.

        experiment_value:
            Experiment primary metric value.

        absolute_change:
            Absolute change in the primary metric.

        relative_change:
            Relative change in the primary metric.

        metric_comparisons:
            Detailed comparisons for all evaluated metrics.

        summary:
            Human-readable outcome summary.

        recommendations:
            Generic next-step guidance based on the observed outcome.

        metadata:
            Additional feedback information.
    """

    outcome: str
    experiment_id: Optional[str] = None
    primary_metric: Optional[str] = None
    baseline_value: Optional[float] = None
    experiment_value: Optional[float] = None
    absolute_change: Optional[float] = None
    relative_change: Optional[float] = None
    metric_comparisons: Dict[str, MetricComparison] = field(
        default_factory=dict
    )
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feedback Engine
# ---------------------------------------------------------------------------


class FeedbackEngine:
    """
    Compare experiment results against a baseline.

    The engine requires an explicit metric direction. This prevents incorrect
    conclusions such as treating a reduction in validation loss as a failure.

    Example:

        engine = FeedbackEngine(
            primary_metric="validation_accuracy",
            higher_is_better=True,
        )

        feedback = engine.evaluate(
            baseline={"validation_accuracy": 0.68},
            experiment={"validation_accuracy": 0.74},
        )
    """

    def __init__(
        self,
        primary_metric: str = "validation_accuracy",
        higher_is_better: bool = True,
        improvement_tolerance: float = DEFAULT_IMPROVEMENT_TOLERANCE,
    ) -> None:
        if not isinstance(primary_metric, str):
            raise TypeError(
                "primary_metric must be a string."
            )

        if not primary_metric.strip():
            raise ValueError(
                "primary_metric cannot be empty."
            )

        if improvement_tolerance < 0:
            raise ValueError(
                "improvement_tolerance cannot be negative."
            )

        self.primary_metric = primary_metric.strip()
        self.higher_is_better = higher_is_better
        self.improvement_tolerance = improvement_tolerance

        LOGGER.info(
            "FeedbackEngine initialized: primary_metric=%s, "
            "higher_is_better=%s, tolerance=%.6f",
            self.primary_metric,
            self.higher_is_better,
            self.improvement_tolerance,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        baseline: Mapping[str, Any] | ExperimentResult,
        experiment: Mapping[str, Any] | ExperimentResult,
        metric_directions: Optional[Mapping[str, bool]] = None,
        experiment_id: Optional[str] = None,
    ) -> FeedbackResult:
        """
        Compare baseline and experiment metrics.

        Args:
            baseline:
                Baseline metrics or ExperimentResult.

            experiment:
                Experiment metrics or ExperimentResult.

            metric_directions:
                Optional per-metric direction mapping.

                Example:
                    {
                        "validation_accuracy": True,
                        "validation_loss": False,
                    }

            experiment_id:
                Optional experiment identifier when experiment is supplied
                as a plain mapping.

        Returns:
            FeedbackResult.

        Notes:
            If the primary metric is missing from either side, the result is
            inconclusive. The engine never substitutes a different metric
            without explicit configuration.
        """
        baseline_metrics = self._extract_metrics(baseline)
        experiment_metrics = self._extract_metrics(experiment)

        resolved_experiment_id = (
            experiment_id
            if experiment_id is not None
            else self._extract_experiment_id(experiment)
        )

        directions = self._resolve_metric_directions(
            metric_directions
        )

        LOGGER.info(
            "Evaluating experiment feedback: experiment_id=%s",
            resolved_experiment_id,
        )

        comparisons: Dict[str, MetricComparison] = {}

        common_metrics = sorted(
            set(baseline_metrics).intersection(experiment_metrics)
        )

        for metric in common_metrics:
            higher_is_better = directions.get(
                metric,
                self.higher_is_better,
            )

            comparison = self.compare_metric(
                metric=metric,
                baseline_value=baseline_metrics[metric],
                experiment_value=experiment_metrics[metric],
                higher_is_better=higher_is_better,
            )

            comparisons[metric] = comparison

        # ---------------------------------------------------------------
        # Primary metric is required for a definitive result.
        # ---------------------------------------------------------------

        if self.primary_metric not in baseline_metrics:
            return self._build_inconclusive_result(
                experiment_id=resolved_experiment_id,
                comparisons=comparisons,
                reason=(
                    f"Primary metric '{self.primary_metric}' is missing "
                    "from the baseline result."
                ),
            )

        if self.primary_metric not in experiment_metrics:
            return self._build_inconclusive_result(
                experiment_id=resolved_experiment_id,
                comparisons=comparisons,
                reason=(
                    f"Primary metric '{self.primary_metric}' is missing "
                    "from the experiment result."
                ),
            )

        primary_direction = directions.get(
            self.primary_metric,
            self.higher_is_better,
        )

        primary_comparison = comparisons[self.primary_metric]

        outcome = self._determine_outcome(
            primary_comparison
        )

        summary = self._build_summary(
            outcome=outcome,
            comparison=primary_comparison,
        )

        recommendations = self._build_next_steps(
            outcome=outcome,
            comparison=primary_comparison,
        )

        result = FeedbackResult(
            outcome=outcome,
            experiment_id=resolved_experiment_id,
            primary_metric=self.primary_metric,
            baseline_value=baseline_metrics[self.primary_metric],
            experiment_value=experiment_metrics[self.primary_metric],
            absolute_change=calculate_absolute_change(
                baseline_metrics[self.primary_metric],
                experiment_metrics[self.primary_metric],
            ),
            relative_change=calculate_relative_change(
                baseline_metrics[self.primary_metric],
                experiment_metrics[self.primary_metric],
            ),
            metric_comparisons=comparisons,
            summary=summary,
            recommendations=recommendations,
            metadata={
                "higher_is_better": primary_direction,
                "improvement_tolerance": self.improvement_tolerance,
                "common_metrics": common_metrics,
            },
        )

        LOGGER.info(
            "Experiment feedback completed: outcome=%s, metric=%s",
            result.outcome,
            result.primary_metric,
        )

        return result

    # -----------------------------------------------------------------------
    # Metric comparison
    # -----------------------------------------------------------------------

    def compare_metric(
        self,
        metric: str,
        baseline_value: float,
        experiment_value: float,
        higher_is_better: bool,
    ) -> MetricComparison:
        """
        Compare one metric between baseline and experiment.

        Args:
            metric:
                Metric name.

            baseline_value:
                Baseline metric value.

            experiment_value:
                Experiment metric value.

            higher_is_better:
                Direction of improvement.

        Returns:
            MetricComparison.
        """
        baseline = self._to_float(
            baseline_value,
            metric,
        )

        experiment = self._to_float(
            experiment_value,
            metric,
        )

        absolute_change = calculate_absolute_change(
            baseline,
            experiment,
        )

        relative_change = calculate_relative_change(
            baseline,
            experiment,
        )

        meaningful_change = (
            abs(absolute_change) > self.improvement_tolerance
        )

        if not meaningful_change:
            improved = False
            degraded = False
            within_tolerance = True
            explanation = (
                f"{metric} changed by {absolute_change:.6g}, "
                "which is within the configured tolerance."
            )

        elif higher_is_better:
            improved = experiment > (
                baseline + self.improvement_tolerance
            )
            degraded = experiment < (
                baseline - self.improvement_tolerance
            )
            within_tolerance = False

            if improved:
                explanation = (
                    f"{metric} improved because it increased from "
                    f"{baseline:.6g} to {experiment:.6g}."
                )
            else:
                explanation = (
                    f"{metric} degraded because it decreased from "
                    f"{baseline:.6g} to {experiment:.6g}."
                )

        else:
            improved = experiment < (
                baseline - self.improvement_tolerance
            )
            degraded = experiment > (
                baseline + self.improvement_tolerance
            )
            within_tolerance = False

            if improved:
                explanation = (
                    f"{metric} improved because it decreased from "
                    f"{baseline:.6g} to {experiment:.6g}."
                )
            else:
                explanation = (
                    f"{metric} degraded because it increased from "
                    f"{baseline:.6g} to {experiment:.6g}."
                )

        return MetricComparison(
            metric=metric,
            baseline=baseline,
            experiment=experiment,
            absolute_change=absolute_change,
            relative_change=relative_change,
            higher_is_better=higher_is_better,
            improved=improved,
            degraded=degraded,
            within_tolerance=within_tolerance,
            explanation=explanation,
        )

    # -----------------------------------------------------------------------
    # Outcome determination
    # -----------------------------------------------------------------------

    @staticmethod
    def _determine_outcome(
        comparison: MetricComparison,
    ) -> str:
        """
        Determine feedback outcome from the primary metric.
        """
        if comparison.improved:
            return FEEDBACK_IMPROVED

        if comparison.degraded:
            return FEEDBACK_DEGRADED

        if comparison.within_tolerance:
            return FEEDBACK_NO_IMPROVEMENT

        return FEEDBACK_NO_IMPROVEMENT

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        outcome: str,
        comparison: MetricComparison,
    ) -> str:
        """
        Build a concise human-readable outcome summary.
        """
        if outcome == FEEDBACK_IMPROVED:
            return (
                f"Experiment improved {comparison.metric} from "
                f"{comparison.baseline:.6g} to "
                f"{comparison.experiment:.6g}."
            )

        if outcome == FEEDBACK_DEGRADED:
            return (
                f"Experiment degraded {comparison.metric} from "
                f"{comparison.baseline:.6g} to "
                f"{comparison.experiment:.6g}."
            )

        if outcome == FEEDBACK_NO_IMPROVEMENT:
            return (
                f"Experiment did not produce a meaningful improvement in "
                f"{comparison.metric}: "
                f"{comparison.baseline:.6g} → "
                f"{comparison.experiment:.6g}."
            )

        return (
            f"Experiment outcome for {comparison.metric} is inconclusive."
        )

    # -----------------------------------------------------------------------
    # Next steps
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_next_steps(
        outcome: str,
        comparison: MetricComparison,
    ) -> List[str]:
        """
        Generate generic next-step guidance.

        This does not recommend a specific ML technique.
        """
        if outcome == FEEDBACK_IMPROVED:
            return [
                "Record the experiment as a successful candidate.",
                "Use the improved result as evidence for the next research iteration.",
            ]

        if outcome == FEEDBACK_DEGRADED:
            return [
                "Do not promote the experiment result over the baseline.",
                "Investigate the degradation before trying another experiment.",
            ]

        if outcome == FEEDBACK_NO_IMPROVEMENT:
            return [
                "Do not treat the experiment as an improvement.",
                "Consider a different experiment or additional evidence.",
            ]

        return [
            "Collect the missing or ambiguous evaluation evidence.",
            "Re-run evaluation before making a promotion decision.",
        ]

    # -----------------------------------------------------------------------
    # Inconclusive result
    # -----------------------------------------------------------------------

    def _build_inconclusive_result(
        self,
        experiment_id: Optional[str],
        comparisons: Dict[str, MetricComparison],
        reason: str,
    ) -> FeedbackResult:
        """
        Build an explicit inconclusive result.
        """
        LOGGER.warning(
            "Experiment feedback is inconclusive: %s",
            reason,
        )

        return FeedbackResult(
            outcome=FEEDBACK_INCONCLUSIVE,
            experiment_id=experiment_id,
            primary_metric=self.primary_metric,
            metric_comparisons=comparisons,
            summary=reason,
            recommendations=self._build_next_steps(
                FEEDBACK_INCONCLUSIVE,
                MetricComparison(
                    metric=self.primary_metric,
                    baseline=0.0,
                    experiment=0.0,
                    absolute_change=0.0,
                    relative_change=0.0,
                    higher_is_better=self.higher_is_better,
                    improved=False,
                    degraded=False,
                    within_tolerance=True,
                    explanation=reason,
                ),
            ),
            metadata={
                "higher_is_better": self.higher_is_better,
                "improvement_tolerance": self.improvement_tolerance,
            },
        )

    # -----------------------------------------------------------------------
    # Input extraction
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_metrics(
        result: Mapping[str, Any] | ExperimentResult,
    ) -> Dict[str, float]:
        """
        Extract metrics from a mapping or ExperimentResult.
        """
        if isinstance(result, ExperimentResult):
            metrics = result.metrics

        elif isinstance(result, Mapping):
            if "metrics" in result:
                metrics = result["metrics"]

                if not isinstance(metrics, Mapping):
                    raise TypeError(
                        "'metrics' must be a mapping."
                    )
            else:
                metrics = result

        else:
            raise TypeError(
                "Result must be a mapping or ExperimentResult."
            )

        normalized: Dict[str, float] = {}

        for name, value in metrics.items():
            try:
                normalized[str(name)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Metric '{name}' has non-numeric value: {value!r}"
                ) from exc

        return normalized

    @staticmethod
    def _extract_experiment_id(
        result: Mapping[str, Any] | ExperimentResult,
    ) -> Optional[str]:
        """
        Extract experiment ID when available.
        """
        if isinstance(result, ExperimentResult):
            return result.experiment_id

        if isinstance(result, Mapping):
            value = result.get("experiment_id")

            if value is None:
                return None

            return str(value)

        return None

    def _resolve_metric_directions(
        self,
        metric_directions: Optional[Mapping[str, bool]],
    ) -> Dict[str, bool]:
        """
        Resolve per-metric direction overrides.
        """
        if metric_directions is None:
            return {}

        directions: Dict[str, bool] = {}

        for metric, direction in metric_directions.items():
            if not isinstance(direction, bool):
                raise TypeError(
                    f"Metric direction for '{metric}' must be boolean."
                )

            directions[str(metric)] = direction

        return directions

    # -----------------------------------------------------------------------
    # Numeric validation
    # -----------------------------------------------------------------------

    @staticmethod
    def _to_float(
        value: Any,
        metric: str,
    ) -> float:
        """
        Convert and validate a metric value.
        """
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Metric '{metric}' must contain a numeric value."
            ) from exc

        if not (
            result == result
            and result not in (float("inf"), float("-inf"))
        ):
            raise ValueError(
                f"Metric '{metric}' must be finite."
            )

        return result

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "FeedbackEngine("
            f"primary_metric={self.primary_metric!r}, "
            f"higher_is_better={self.higher_is_better}, "
            f"improvement_tolerance={self.improvement_tolerance}"
            ")"
        )