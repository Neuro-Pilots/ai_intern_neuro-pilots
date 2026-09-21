"""Shared contract for workloads scheduled for later implementation.

These workloads can participate in UI selection and artifact validation now,
but they deliberately do not expose a training or evaluation implementation.
"""

from __future__ import annotations

from typing import Any, Dict

from utils.models import ExperimentArtifacts
from workloads.base import BaseWorkloadEngine


class DeferredWorkloadEngine(BaseWorkloadEngine):
    """Metadata-only workload contract for a planned execution pipeline."""

    model_family: str = "planned"
    input_type: str = "unknown"
    task_types: tuple[str, ...] = ()

    def get_model_spec(self) -> Dict[str, Any]:
        return {
            "workload": self.workload_name,
            "model_name": self.default_model_name,
            "model_family": self.model_family,
            "task_types": list(self.task_types),
            "input_type": self.input_type,
            "execution_status": "planned",
        }

    def _validate_workload_specific_artifacts(
        self,
        artifacts: ExperimentArtifacts,
    ) -> None:
        if artifacts.dataset is None:
            raise ValueError(
                f"{self.workload_name} workload requires dataset artifacts."
            )

        if artifacts.model is None:
            raise ValueError(
                f"{self.workload_name} workload requires model artifacts."
            )

    def get_experiment_requirements(self) -> Dict[str, Any]:
        return {
            "execution_status": "planned",
            "dataset": {"required": True, "expected_input_type": self.input_type},
            "model": {"required": True, "default_model": self.default_model_name},
            "training_logs": {"required": False},
        }

    def describe(self) -> Dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "execution_status": "planned",
                "experiment_requirements": self.get_experiment_requirements(),
            }
        )
        return description
