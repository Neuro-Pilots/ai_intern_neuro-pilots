"""Time-series workload contract; real execution is planned."""

from workloads.deferred import DeferredWorkloadEngine
from utils.constants import WORKLOAD_TIME_SERIES


class TimeSeriesWorkload(DeferredWorkloadEngine):
    workload_name = WORKLOAD_TIME_SERIES
    model_family = "Chronos"
    input_type = "time_series"
    task_types = ("forecasting",)
