"""
Experiment execution orchestrator for NeuroPilots.

This module coordinates the execution of a selected experiment.

Execution flow:

    Recommendation
        ↓
    Experiment Executor
        ↓
    Data / Model experiment configuration
        ↓
    Training callback
        ↓
    Evaluation callback
        ↓
    ExperimentResult

Design principles:
    - The LLM generates the recommendation.
    - ExperimentSelector determines whether it is executable.
    - DataImprovementEngine handles data-side changes.
    - ModelExperimentEngine handles model-side configuration changes.
    - This executor orchestrates execution; it does not invent experiments.
    - Actual workload-specific training is injected through callbacks.
    - Failed experiments are represented explicitly rather than hidden.
    - The original configuration and experiment inputs are preserved.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional

from advisor.experiment_selector import ExperimentSelectionResult
from execution.data_improvement import (
    DataImprovementEngine,
    DataImprovementResult,
)
from execution.model_experiment import (
    ModelExperimentEngine,
    ModelExperimentResult,
)
from utils.constants import (
    EXPERIMENT_STATUS_COMPLETED,
    EXPERIMENT_STATUS_FAILED,
    EXPERIMENT_STATUS_PENDING,
    EXPERIMENT_STATUS_RUNNING,
)
from utils.logger import get_logger
from utils.models import (
    ExperimentContext,
    ExperimentRecommendation,
    ExperimentResult,
)


LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Callback types
# ---------------------------------------------------------------------------

TrainingCallback = Callable[
    [ExperimentContext, Dict[str, Any]],
    Any,
]

EvaluationCallback = Callable[
    [Any, ExperimentContext, Dict[str, Any]],
    Mapping[str, float],
]


# ---------------------------------------------------------------------------
# Execution configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentExecutionConfig:
    """
    Configuration controlling experiment execution.

    Attributes:
        experiment_id:
            Optional externally supplied experiment identifier.

        random_seed:
            Optional seed used by downstream training implementations.

        timeout_seconds:
            Optional execution timeout metadata.

        metadata:
            Additional execution configuration.
    """

    experiment_id: Optional[str] = None
    random_seed: Optional[int] = 42
    timeout_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal execution state
# ---------------------------------------------------------------------------


@dataclass
class ExperimentExecutionState:
    """
    Internal state captured during execution.
    """

    status: str = EXPERIMENT_STATUS_PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Experiment Executor
# ---------------------------------------------------------------------------


class ExperimentExecutor:
    """
    Execute a selected experiment using injected training/evaluation logic.

    The executor is framework-agnostic.

    It does not assume PyTorch, TensorFlow, scikit-learn, XGBoost, or any
    other particular training framework.

    Instead, workload-specific implementations provide:

        training_callback(context, configuration)

    and optionally:

        evaluation_callback(trained_model, context, configuration)

    This keeps the execution layer reusable across the seven NeuroPilots
    workloads.

    Example:

        executor = ExperimentExecutor()

        result = executor.execute(
            recommendation,
            context,
            selection=selection,
            training_callback=train_model,
            evaluation_callback=evaluate_model,
        )
    """

    def __init__(
        self,
        data_engine: Optional[DataImprovementEngine] = None,
        model_engine: Optional[ModelExperimentEngine] = None,
        execution_config: Optional[ExperimentExecutionConfig] = None,
    ) -> None:
        self.data_engine = (
            data_engine
            if data_engine is not None
            else DataImprovementEngine()
        )

        self.model_engine = (
            model_engine
            if model_engine is not None
            else ModelExperimentEngine()
        )

        self.execution_config = (
            execution_config
            if execution_config is not None
            else ExperimentExecutionConfig()
        )

        LOGGER.info(
            "ExperimentExecutor initialized."
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def execute(
        self,
        recommendation: ExperimentRecommendation,
        context: ExperimentContext,
        selection: Optional[ExperimentSelectionResult] = None,
        training_callback: Optional[TrainingCallback] = None,
        evaluation_callback: Optional[EvaluationCallback] = None,
        experiment_configuration: Optional[Mapping[str, Any]] = None,
    ) -> ExperimentResult:
        """
        Execute a selected experiment.

        Args:
            recommendation:
                Structured recommendation generated by RecommendationEngine.

            context:
                Current experiment context.

            selection:
                Optional result produced by ExperimentSelector.

            training_callback:
                Workload-specific training function.

            evaluation_callback:
                Optional workload-specific evaluation function.

            experiment_configuration:
                Explicit execution configuration.

                When supplied, it takes precedence over automatic extraction.
                This is the preferred path for production execution because
                it avoids guessing parameters from natural language.

        Returns:
            ExperimentResult.

        Important:
            If no training callback is supplied, the executor does NOT pretend
            that training occurred. It returns a failed result explaining that
            a training implementation is required.
        """
        self._validate_inputs(
            recommendation=recommendation,
            context=context,
        )

        experiment_id = self._create_experiment_id()
        state = ExperimentExecutionState()

        LOGGER.info(
            "Starting experiment execution: id=%s, title=%s",
            experiment_id,
            recommendation.title,
        )

        state.status = EXPERIMENT_STATUS_RUNNING
        state.started_at = self._utc_now()

        start_time = time.perf_counter()

        try:
            # ---------------------------------------------------------------
            # Selection validation
            # ---------------------------------------------------------------

            if selection is not None and not selection.selected:
                state.status = EXPERIMENT_STATUS_FAILED

                state.error = (
                    "Experiment was not selected for execution."
                )

                return self._build_result(
                    experiment_id=experiment_id,
                    recommendation=recommendation,
                    configuration={},
                    metrics={},
                    state=state,
                    artifacts=[],
                )

            # ---------------------------------------------------------------
            # Build experiment configuration
            # ---------------------------------------------------------------

            configuration = self._prepare_experiment_configuration(
                recommendation=recommendation,
                context=context,
                explicit_configuration=experiment_configuration,
            )

            LOGGER.info(
                "Experiment configuration prepared: id=%s, parameters=%d",
                experiment_id,
                len(configuration),
            )

            # ---------------------------------------------------------------
            # Training implementation is required for actual execution.
            # ---------------------------------------------------------------

            if training_callback is None:
                state.status = EXPERIMENT_STATUS_FAILED

                state.error = (
                    "No training_callback was supplied. "
                    "A workload-specific training implementation is required "
                    "for actual experiment execution."
                )

                return self._build_result(
                    experiment_id=experiment_id,
                    recommendation=recommendation,
                    configuration=configuration,
                    metrics={},
                    state=state,
                    artifacts=[],
                )

            # ---------------------------------------------------------------
            # Execute training
            # ---------------------------------------------------------------

            LOGGER.info(
                "Executing training callback: id=%s",
                experiment_id,
            )

            training_output = training_callback(
                copy.deepcopy(context),
                copy.deepcopy(configuration),
            )

            # ---------------------------------------------------------------
            # Evaluate trained output
            # ---------------------------------------------------------------

            metrics: Dict[str, float] = {}
            artifacts = []

            if evaluation_callback is not None:
                LOGGER.info(
                    "Executing evaluation callback: id=%s",
                    experiment_id,
                )

                evaluation_output = evaluation_callback(
                    training_output,
                    copy.deepcopy(context),
                    copy.deepcopy(configuration),
                )

                metrics = self._normalize_metrics(
                    evaluation_output
                )

            elif isinstance(training_output, Mapping):
                # Allow a training implementation to directly return metrics.
                metrics = self._extract_metrics_from_mapping(
                    training_output
                )

            # ---------------------------------------------------------------
            # Successful completion
            # ---------------------------------------------------------------

            state.status = EXPERIMENT_STATUS_COMPLETED

            result = self._build_result(
                experiment_id=experiment_id,
                recommendation=recommendation,
                configuration=configuration,
                metrics=metrics,
                state=state,
                artifacts=artifacts,
            )

            LOGGER.info(
                "Experiment execution completed: id=%s, metrics=%d",
                experiment_id,
                len(result.metrics),
            )

            return result

        except Exception as exc:
            LOGGER.exception(
                "Experiment execution failed: id=%s",
                experiment_id,
            )

            state.status = EXPERIMENT_STATUS_FAILED
            state.error = str(exc)

            return self._build_result(
                experiment_id=experiment_id,
                recommendation=recommendation,
                configuration=locals().get(
                    "configuration",
                    {},
                ),
                metrics={},
                state=state,
                artifacts=[],
            )

        finally:
            state.completed_at = self._utc_now()
            state.duration_seconds = (
                time.perf_counter() - start_time
            )

            LOGGER.info(
                "Experiment execution finished: id=%s, status=%s, "
                "duration=%.4fs",
                experiment_id,
                state.status,
                state.duration_seconds,
            )

    # -----------------------------------------------------------------------
    # Configuration preparation
    # -----------------------------------------------------------------------

    def _prepare_experiment_configuration(
        self,
        recommendation: ExperimentRecommendation,
        context: ExperimentContext,
        explicit_configuration: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Prepare the configuration passed to the training callback.

        Explicit configuration is preferred.

        If explicit configuration is unavailable, the executor performs only
        conservative extraction from the recommendation text.

        It never invents missing values.
        """
        if explicit_configuration is not None:
            if not isinstance(
                explicit_configuration,
                Mapping,
            ):
                raise TypeError(
                    "experiment_configuration must be a mapping."
                )

            configuration = copy.deepcopy(
                dict(explicit_configuration)
            )

            validation_errors = (
                self.model_engine.validate_configuration(
                    configuration
                )
            )

            if validation_errors:
                raise ValueError(
                    "Invalid experiment configuration: "
                    + "; ".join(validation_errors)
                )

            return configuration

        configuration: Dict[str, Any] = {}

        # Start with the existing model/training configuration if available.
        if context.artifacts is not None:
            configuration.update(
                copy.deepcopy(
                    context.artifacts.hyperparameters
                )
            )

            if context.artifacts.model is not None:
                configuration.update(
                    copy.deepcopy(
                        context.artifacts.model.configuration
                    )
                )

        # Conservatively extract explicitly stated parameter values.
        extracted_parameters = (
            self.model_engine.extract_parameter_changes(
                recommendation.change
            )
        )

        configuration.update(extracted_parameters)

        return configuration

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        recommendation: ExperimentRecommendation,
        context: ExperimentContext,
    ) -> None:
        """
        Validate required execution inputs.
        """
        if not isinstance(
            recommendation,
            ExperimentRecommendation,
        ):
            raise TypeError(
                "recommendation must be an ExperimentRecommendation."
            )

        if not isinstance(
            context,
            ExperimentContext,
        ):
            raise TypeError(
                "context must be an ExperimentContext."
            )

    # -----------------------------------------------------------------------
    # Result construction
    # -----------------------------------------------------------------------

    def _build_result(
        self,
        experiment_id: str,
        recommendation: ExperimentRecommendation,
        configuration: Mapping[str, Any],
        metrics: Mapping[str, float],
        state: ExperimentExecutionState,
        artifacts: list[str],
    ) -> ExperimentResult:
        """
        Convert execution state into the shared ExperimentResult contract.
        """
        metadata = {
            "status": state.status,
            "started_at": state.started_at,
            "completed_at": state.completed_at,
            "duration_seconds": state.duration_seconds,
            "error": state.error,
            "recommendation_title": recommendation.title,
            "recommendation_priority": recommendation.priority,
            "execution_metadata": copy.deepcopy(
                self.execution_config.metadata
            ),
        }

        if self.execution_config.random_seed is not None:
            metadata["random_seed"] = (
                self.execution_config.random_seed
            )

        return ExperimentResult(
            experiment_id=experiment_id,
            parameters=copy.deepcopy(dict(configuration)),
            metrics=self._normalize_metrics(metrics),
            artifacts=list(artifacts),
            success=(
                state.status == EXPERIMENT_STATUS_COMPLETED
                and state.error is None
            ),
        )

    # -----------------------------------------------------------------------
    # Metric handling
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_metrics(
        metrics: Optional[Mapping[str, Any]],
    ) -> Dict[str, float]:
        """
        Convert metric values into JSON/Pydantic-safe floats.
        """
        if metrics is None:
            return {}

        if not isinstance(metrics, Mapping):
            raise TypeError(
                "Evaluation metrics must be a mapping."
            )

        normalized: Dict[str, float] = {}

        for name, value in metrics.items():
            if value is None:
                continue

            try:
                normalized[str(name)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Metric '{name}' has non-numeric value: {value!r}"
                ) from exc

        return normalized

    def _extract_metrics_from_mapping(
        self,
        training_output: Mapping[str, Any],
    ) -> Dict[str, float]:
        """
        Extract metrics when the training callback directly returns a mapping.

        Supported patterns include:

            {"metrics": {"accuracy": 0.91}}

        or:

            {"accuracy": 0.91, "loss": 0.32}

        Non-numeric metadata is ignored.
        """
        if "metrics" in training_output:
            metrics = training_output["metrics"]

            if not isinstance(metrics, Mapping):
                raise TypeError(
                    "training_output['metrics'] must be a mapping."
                )

            return self._normalize_metrics(metrics)

        numeric_values: Dict[str, Any] = {}

        for key, value in training_output.items():
            if isinstance(value, bool):
                continue

            try:
                float(value)
                numeric_values[str(key)] = value
            except (TypeError, ValueError):
                continue

        return self._normalize_metrics(numeric_values)

    # -----------------------------------------------------------------------
    # IDs and timestamps
    # -----------------------------------------------------------------------

    def _create_experiment_id(self) -> str:
        """
        Create an experiment identifier.
        """
        if self.execution_config.experiment_id:
            return self.execution_config.experiment_id

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d%H%M%S")

        unique_suffix = uuid.uuid4().hex[:8]

        return f"exp_{timestamp}_{unique_suffix}"

    @staticmethod
    def _utc_now() -> str:
        """
        Return an ISO-8601 UTC timestamp.
        """
        return datetime.now(
            timezone.utc
        ).isoformat()

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "ExperimentExecutor("
            f"random_seed={self.execution_config.random_seed}, "
            f"timeout_seconds={self.execution_config.timeout_seconds}"
            ")"
        )