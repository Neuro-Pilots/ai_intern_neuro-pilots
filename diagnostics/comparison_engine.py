"""Evidence-based comparison of baseline and candidate experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from utils.models import ExperimentResult


LOWER_IS_BETTER = ("loss", "error", "latency", "time")


@dataclass
class ExperimentComparison:
    baseline_id: str | None
    candidate_id: str | None
    improvements: dict[str, float] = field(default_factory=dict)
    regressions: dict[str, float] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)
    per_class: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "improvements": self.improvements,
            "regressions": self.regressions,
            "unchanged": self.unchanged,
            "per_class": self.per_class,
        }


class ComparisonEngine:
    """Compare measured values without a third-party diff service."""

    def compare(self, baseline: ExperimentResult, candidate: ExperimentResult) -> ExperimentComparison:
        output = ExperimentComparison(baseline.experiment_id, candidate.experiment_id)
        for metric, candidate_value in candidate.metrics.items():
            if metric not in baseline.metrics:
                continue
            baseline_value = baseline.metrics[metric]
            delta = float(candidate_value) - float(baseline_value)
            if abs(delta) < 1e-12:
                output.unchanged.append(metric)
            elif self._is_improvement(metric, delta):
                output.improvements[metric] = delta
            else:
                output.regressions[metric] = delta
        output.per_class = self._per_class(candidate.parameters, baseline.parameters)
        return output

    @staticmethod
    def _is_improvement(metric: str, delta: float) -> bool:
        return delta < 0 if any(token in metric.lower() for token in LOWER_IS_BETTER) else delta > 0

    @staticmethod
    def _per_class(candidate_parameters: Mapping[str, Any], baseline_parameters: Mapping[str, Any]) -> dict[str, Any]:
        candidate = candidate_parameters.get("per_class_metrics", {})
        baseline = baseline_parameters.get("per_class_metrics", {})
        result: dict[str, Any] = {}
        for label, current in candidate.items():
            previous = baseline.get(label)
            if isinstance(current, Mapping) and isinstance(previous, Mapping):
                result[str(label)] = {key: float(current[key]) - float(previous[key]) for key in current.keys() & previous.keys()
                                      if isinstance(current[key], (int, float)) and isinstance(previous[key], (int, float))}
        return result
