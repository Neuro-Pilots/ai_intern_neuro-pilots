"""Reinforcement-learning workload contract; real execution is planned."""

from workloads.deferred import DeferredWorkloadEngine
from utils.constants import WORKLOAD_REINFORCEMENT_LEARNING


class ReinforcementLearningWorkload(DeferredWorkloadEngine):
    workload_name = WORKLOAD_REINFORCEMENT_LEARNING
    model_family = "PPO"
    input_type = "environment"
    task_types = ("policy_optimisation",)
