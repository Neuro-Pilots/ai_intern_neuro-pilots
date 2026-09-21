"""Small, bounded hyperparameter search for real NeuroPilots runs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

from execution.real_execution import RealExecutionService
from utils.models import ExperimentResult


@dataclass
class HPOResult:
    trials: list[ExperimentResult]
    best_experiment_id: str | None
    primary_metric: str


class SimpleHPO:
    """Run an explicit, bounded grid of real experiments and retain all results."""

    def run(self, workload: str, base_configuration: dict[str, Any], search_space: dict[str, Iterable[Any]],
            primary_metric: str, max_trials: int = 10) -> HPOResult:
        keys = list(search_space)
        combinations = product(*(list(search_space[key]) for key in keys))
        trials: list[ExperimentResult] = []
        for values in combinations:
            if len(trials) >= max_trials:
                break
            configuration = copy.deepcopy(base_configuration)
            configuration.update(dict(zip(keys, values)))
            trials.append(RealExecutionService().execute(workload, configuration))
        completed = [trial for trial in trials if trial.success and primary_metric in trial.metrics]
        best = max(completed, key=lambda trial: trial.metrics[primary_metric], default=None)
        return HPOResult(trials=trials, best_experiment_id=best.experiment_id if best else None, primary_metric=primary_metric)
