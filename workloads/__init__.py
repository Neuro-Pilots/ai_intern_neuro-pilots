"""Workload contracts available to the NeuroPilots application."""

from workloads.computer_vision import ComputerVisionWorkload
from workloads.generative_ai import GenerativeAIWorkload
from workloads.nlp import NLPWorkload
from workloads.reinforcement_learning import ReinforcementLearningWorkload
from workloads.speech import SpeechWorkload
from workloads.tabular import TabularWorkload
from workloads.time_series import TimeSeriesWorkload

__all__ = [
    "ComputerVisionWorkload",
    "GenerativeAIWorkload",
    "NLPWorkload",
    "ReinforcementLearningWorkload",
    "SpeechWorkload",
    "TabularWorkload",
    "TimeSeriesWorkload",
]
