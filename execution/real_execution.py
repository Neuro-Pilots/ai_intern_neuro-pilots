"""Route real NeuroPilots workloads to their concrete execution engines."""

from __future__ import annotations

import uuid
from typing import Any

from execution.cv_yolo import IDDYOLOExecutor
from execution.nlp_modernbert import AmazonAutomotiveModernBERTExecutor
from utils.constants import WORKLOAD_COMPUTER_VISION, WORKLOAD_NLP
from utils.models import ExperimentResult


class RealExecutionService:
    """Dispatch real CV/NLP experiments while rejecting unsupported workloads."""

    def execute(self, workload: str, configuration: dict[str, Any]) -> ExperimentResult:
        experiment_id = f"real_{uuid.uuid4().hex[:12]}"
        if workload == WORKLOAD_COMPUTER_VISION:
            return IDDYOLOExecutor().run(configuration, experiment_id)
        if workload == WORKLOAD_NLP:
            return AmazonAutomotiveModernBERTExecutor().run(configuration, experiment_id)
        return ExperimentResult(
            experiment_id=experiment_id,
            success=False,
            error=f"Real execution has not yet been implemented for '{workload}'.",
        )
