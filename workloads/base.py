"""
Base interface for NeuroPilots workload engines.

This module defines the common contract shared by all supported ML/DL
workloads.

The workload layer is responsible for:
    - Identifying the workload.
    - Providing the default model associated with the workload.
    - Validating artifacts at the workload level.
    - Building a common ExperimentContext.

Actual model training, experiment execution, MLflow tracking, and
performance diagnosis are handled by later layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from utils.constants import DEFAULT_MODEL_BY_WORKLOAD, SUPPORTED_WORKLOADS
from utils.logger import get_logger
from utils.models import ExperimentArtifacts, ExperimentContext


logger = get_logger(__name__)


class BaseWorkloadEngine(ABC):
    """
    Abstract base class for all NeuroPilots workload engines.

    Every supported workload must provide:
        - A workload identifier.
        - A default model.
        - Workload-specific artifact validation.
        - A workload-specific model specification.

    Concrete implementations include:
        - ComputerVisionWorkload
        - NLPWorkload
        - TimeSeriesWorkload
        - TabularWorkload
        - SpeechWorkload
        - GenerativeAIWorkload
        - ReinforcementLearningWorkload
    """

    # ------------------------------------------------------------------
    # Class-level configuration
    # ------------------------------------------------------------------

    #: Identifier used internally to route the workload.
    workload_name: str

    def __init__(self) -> None:
        """
        Initialize the workload engine.

        The workload name is validated against the centrally maintained
        list of supported workloads. This prevents individual workload
        implementations from silently using unsupported identifiers.
        """

        if not self.workload_name:
            raise ValueError(
                f"{self.__class__.__name__} must define 'workload_name'."
            )

        if self.workload_name not in SUPPORTED_WORKLOADS:
            raise ValueError(
                f"Unsupported workload '{self.workload_name}'. "
                f"Supported workloads: {SUPPORTED_WORKLOADS}"
            )

        logger.debug(
            "Initialized workload engine: %s",
            self.workload_name,
        )

    # ------------------------------------------------------------------
    # Common workload metadata
    # ------------------------------------------------------------------

    @property
    def default_model_name(self) -> str:
        """
        Return the default model associated with this workload.

        The mapping is maintained centrally in utils.constants.py so
        that model selection remains consistent throughout the system.
        """

        try:
            return DEFAULT_MODEL_BY_WORKLOAD[self.workload_name]
        except KeyError as exc:
            raise ValueError(
                f"No default model configured for workload "
                f"'{self.workload_name}'."
            ) from exc

    def get_workload_name(self) -> str:
        """
        Return the canonical workload identifier.
        """

        return self.workload_name

    def get_model_spec(self) -> Dict[str, Any]:
        """
        Return the standard model specification for this workload.

        Concrete workload engines can override this method when they
        need to expose additional model-specific metadata.

        This method intentionally returns metadata only. It does not
        instantiate or download the model.
        """

        return {
            "workload": self.workload_name,
            "model_name": self.default_model_name,
        }

    # ------------------------------------------------------------------
    # Artifact validation
    # ------------------------------------------------------------------

    def validate_artifacts(self, artifacts: ExperimentArtifacts) -> None:
        """
        Validate artifacts before workload-specific processing.

        The base implementation performs common validation and then
        delegates workload-specific checks to the concrete engine.

        Parameters
        ----------
        artifacts:
            Experiment artifacts produced by the ingestion layer.

        Raises
        ------
        TypeError
            If the supplied object is not ExperimentArtifacts.

        ValueError
            If the workload-specific validation fails.
        """

        if not isinstance(artifacts, ExperimentArtifacts):
            raise TypeError(
                "artifacts must be an instance of ExperimentArtifacts."
            )

        logger.debug(
            "Validating artifacts for workload '%s'.",
            self.workload_name,
        )

        self._validate_workload_specific_artifacts(artifacts)

    @abstractmethod
    def _validate_workload_specific_artifacts(
        self,
        artifacts: ExperimentArtifacts,
    ) -> None:
        """
        Validate artifacts according to workload-specific requirements.

        Concrete workload engines must implement this method.

        Examples:
            - Computer vision may validate image-related metadata.
            - NLP may validate text/tokenization-related metadata.
            - Time series may validate temporal information.
            - Tabular ML may validate feature/target information.
            - Speech may validate audio-related metadata.
            - Generative AI may validate retrieval/RAG configuration.
            - RL may validate environment-related configuration.
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Experiment context
    # ------------------------------------------------------------------

    def build_context(
        self,
        artifacts: ExperimentArtifacts,
        domain: str | None = None,
    ) -> ExperimentContext:
        """
        Build the shared ExperimentContext for downstream layers.

        The context is the object passed between:
            ingestion
                ↓
            workload engine
                ↓
            diagnostics
                ↓
            advisor
                ↓
            execution
                ↓
            feedback

        Parameters
        ----------
        artifacts:
            Artifacts produced by the ingestion layer.

        domain:
            Optional application domain, for example "automotive".

        Returns
        -------
        ExperimentContext
            Context containing the workload and ingested artifacts.
        """

        self.validate_artifacts(artifacts)

        context = ExperimentContext(
            domain=domain,
            workload=self.workload_name,
            artifacts=artifacts,
        )

        logger.info(
            "Built experiment context for workload '%s'.",
            self.workload_name,
        )

        return context

    # ------------------------------------------------------------------
    # Human-readable description
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """
        Return a concise description of the workload engine.

        This is useful for:
            - Streamlit UI
            - logging
            - debugging
            - LLM context construction
            - workload routing
        """

        return {
            "workload": self.workload_name,
            "default_model": self.default_model_name,
            "model_spec": self.get_model_spec(),
        }

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Return a useful developer-facing representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"workload_name='{self.workload_name}', "
            f"default_model='{self.default_model_name}'"
            f")"
        )