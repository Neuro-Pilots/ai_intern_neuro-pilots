"""Tabular-ML workload contract; real execution is planned."""

from workloads.deferred import DeferredWorkloadEngine
from utils.constants import WORKLOAD_TABULAR


class TabularWorkload(DeferredWorkloadEngine):
    workload_name = WORKLOAD_TABULAR
    model_family = "gradient_boosting"
    input_type = "tabular"
    task_types = ("classification", "regression")
