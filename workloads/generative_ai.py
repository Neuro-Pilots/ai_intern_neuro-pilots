"""
Generative AI workload engine for NeuroPilots.

This module defines the workload-specific behavior for Generative AI
experiments, including Retrieval-Augmented Generation (RAG).

The current responsibility of this layer is to:
    - Identify the Generative AI workload.
    - Expose Llama 3.2 3B + RAG as the default configuration.
    - Validate Generative AI experiment artifacts.
    - Describe generation and retrieval requirements.

Actual model loading, embedding generation, vector-store access,
retrieval, generation, evaluation, and experiment execution are
handled by later layers.
"""

from __future__ import annotations

from typing import Any, Dict

from utils.constants import WORKLOAD_GENERATIVE_AI
from utils.logger import get_logger
from utils.models import ExperimentArtifacts

from workloads.base import BaseWorkloadEngine


logger = get_logger(__name__)


class GenerativeAIWorkload(BaseWorkloadEngine):
    """
    Workload engine for Generative AI and RAG experiments.

    NeuroPilots uses Llama 3.2 3B with RAG as the default Generative AI
    configuration according to the system architecture.

    This class provides workload metadata and artifact validation only.
    It does not instantiate the LLM, create embeddings, query a vector
    database, or perform generation.
    """

    # Canonical workload identifier used throughout NeuroPilots.
    workload_name = WORKLOAD_GENERATIVE_AI

    # ------------------------------------------------------------------
    # Model specification
    # ------------------------------------------------------------------

    def get_model_spec(self) -> Dict[str, Any]:
        """
        Return the Generative AI model specification.

        Returns
        -------
        dict
            Generative AI and RAG metadata used by downstream
            components.
        """

        return {
            "workload": self.workload_name,
            "model_name": self.default_model_name,
            "model_family": "Llama",
            "model_variant": "3.2 3B",
            "task_types": [
                "text_generation",
                "question_answering",
                "retrieval_augmented_generation",
            ],
            "input_type": "text",
            "rag_enabled": True,
            "framework_agnostic": True,
        }

    # ------------------------------------------------------------------
    # Artifact validation
    # ------------------------------------------------------------------

    def _validate_workload_specific_artifacts(
        self,
        artifacts: ExperimentArtifacts,
    ) -> None:
        """
        Validate artifacts required by the Generative AI workload.

        A Generative AI experiment requires dataset and model
        information. RAG-specific configuration is optional at this
        stage because ExperimentArtifacts can be incrementally
        enriched by later ingestion components.

        Parameters
        ----------
        artifacts:
            ExperimentArtifacts produced by the ingestion layer.

        Raises
        ------
        ValueError
            If required Generative AI artifacts are missing.
        """

        if artifacts.dataset is None:
            raise ValueError(
                "Generative AI workload requires dataset artifacts."
            )

        if artifacts.model is None:
            raise ValueError(
                "Generative AI workload requires model artifacts."
            )

        logger.debug(
            "Generative AI artifacts validated successfully."
        )

    # ------------------------------------------------------------------
    # Experiment requirements
    # ------------------------------------------------------------------

    def get_experiment_requirements(self) -> Dict[str, Any]:
        """
        Return the information expected from a Generative AI/RAG
        experiment.

        These requirements describe the signals that downstream
        components should consider when analyzing both retrieval and
        generation quality.
        """

        return {
            "dataset": {
                "required": True,
                "expected_input_type": "text",
                "important_signals": [
                    "dataset_size",
                    "document_count",
                    "text_length_distribution",
                    "duplicate_documents",
                    "missing_values",
                    "data_quality",
                ],
            },
            "model": {
                "required": True,
                "default_model": self.default_model_name,
                "important_signals": [
                    "architecture",
                    "parameter_count",
                    "configuration",
                    "context_length",
                    "generation_configuration",
                ],
            },
            "training_logs": {
                "required": False,
                "important_signals": [
                    "train_loss",
                    "validation_loss",
                    "learning_rate",
                    "epoch_progression",
                ],
            },
            "hyperparameters": {
                "important_parameters": [
                    "temperature",
                    "top_p",
                    "top_k",
                    "max_new_tokens",
                    "repetition_penalty",
                    "learning_rate",
                    "batch_size",
                ],
            },
            "retrieval": {
                "important_parameters": [
                    "embedding_model",
                    "vector_store",
                    "chunk_size",
                    "chunk_overlap",
                    "top_k",
                    "similarity_threshold",
                    "retrieval_strategy",
                ],
                "important_signals": [
                    "retrieval_precision",
                    "retrieval_recall",
                    "retrieval_relevance",
                    "retrieved_document_count",
                ],
            },
            "generation": {
                "important_parameters": [
                    "temperature",
                    "top_p",
                    "max_new_tokens",
                    "prompt_template",
                    "system_prompt",
                ],
                "important_signals": [
                    "answer_quality",
                    "groundedness",
                    "faithfulness",
                    "hallucination_rate",
                    "response_latency",
                ],
            },
        }

    # ------------------------------------------------------------------
    # Workload description
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """
        Return a complete description of the Generative AI workload.

        Combines common workload metadata with Generative AI and
        retrieval-specific experiment requirements.
        """

        description = super().describe()

        description.update(
            {
                "domain": "generative_ai",
                "experiment_requirements": self.get_experiment_requirements(),
            }
        )

        return description