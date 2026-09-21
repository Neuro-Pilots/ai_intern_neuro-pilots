"""Generative-AI workload contract; real execution is planned."""

from workloads.deferred import DeferredWorkloadEngine
from utils.constants import WORKLOAD_GENERATIVE_AI


class GenerativeAIWorkload(DeferredWorkloadEngine):
    workload_name = WORKLOAD_GENERATIVE_AI
    model_family = "LLM_RAG"
    input_type = "documents"
    task_types = ("retrieval_augmented_generation",)
