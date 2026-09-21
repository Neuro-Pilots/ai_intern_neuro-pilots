"""
Artifact Ingestor for NeuroPilots.

Layer:
    Layer 1 — Artifact Ingestion

Responsibility:
    Provide one unified entry point for ingesting the artifacts associated
    with an ML experiment.

The ingestor coordinates:

    DatasetAnalyzer
    ModelAnalyzer
    TrainingLogAnalyzer

and combines their outputs into:

    ExperimentArtifacts

Architecture:

    Raw Experiment Artifacts
            │
            ├── Dataset
            │      ↓
            │  DatasetAnalyzer
            │
            ├── Model
            │      ↓
            │  ModelAnalyzer
            │
            └── Training Logs
                   ↓
             TrainingLogAnalyzer
            │
            ▼
    ExperimentArtifacts

This component performs orchestration only. It does not contain the
individual dataset/model/training-log analysis algorithms.

Design goals:
    - Single entry point for Layer 1 ingestion.
    - Keep analyzers independently testable.
    - Support partially available artifacts.
    - Preserve explicit hyperparameters.
    - Fail clearly when a supplied artifact cannot be processed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ingestion.dataset_analyzer import DatasetAnalyzer
from ingestion.model_analyzer import ModelAnalyzer
from ingestion.training_log_analyzer import TrainingLogAnalyzer

from utils.logger import get_logger
from utils.models import ExperimentArtifacts


# ============================================================================
# LOGGER
# ============================================================================

logger = get_logger(__name__)


# ============================================================================
# ARTIFACT INGESTOR
# ============================================================================

class ArtifactIngestor:
    """
    Coordinate ingestion of all available experiment artifacts.

    An experiment does not necessarily have to provide every artifact.

    For example, a user may initially provide only:

        dataset + training logs

    or:

        model + training logs

    The ingestor therefore treats each artifact as optional.

    However, if an artifact path/object is explicitly supplied and cannot
    be processed, the error is propagated rather than silently ignored.
    Silent failures would make downstream AI recommendations unreliable.
    """

    def __init__(
        self,
        workload: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> None:
        """
        Initialize the artifact ingestor.

        Parameters
        ----------
        workload:
            NeuroPilots workload identifier.

        target_column:
            Optional dataset target/label column.
        """

        self.workload = workload
        self.target_column = target_column

        # Each analyzer remains an independent component.
        self.dataset_analyzer = DatasetAnalyzer(
            target_column=target_column,
            workload=workload,
        )

        self.model_analyzer = ModelAnalyzer(
            workload=workload,
        )

        self.training_log_analyzer = TrainingLogAnalyzer()

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def ingest(
        self,
        dataset_path: Optional[str | Path] = None,
        model: Optional[Any] = None,
        model_config_path: Optional[str | Path] = None,
        training_log_path: Optional[str | Path] = None,
        training_logs: Optional[Any] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
        architecture: Optional[str] = None,
        model_configuration: Optional[Dict[str, Any]] = None,
        embeddings: Optional[Dict[str, Any]] = None,
    ) -> ExperimentArtifacts:
        """
        Ingest all supplied experiment artifacts.

        Parameters
        ----------
        dataset_path:
            Optional path to a CSV, JSON, or Parquet dataset.

        model:
            Optional Python model object or model metadata dictionary.

        model_config_path:
            Optional JSON model configuration path.

            This is used when a model object is not directly available.

        training_log_path:
            Optional path to a JSON or CSV training log.

        training_logs:
            Optional in-memory training log structure.

            This can be a DataFrame, list of dictionaries, or dictionary.

        hyperparameters:
            Optional experiment hyperparameters.

        dataset_name:
            Optional human-readable dataset name.

        model_name:
            Optional explicit model name.

        architecture:
            Optional explicit model architecture.

        model_configuration:
            Optional explicit model configuration.

        embeddings:
            Optional embedding metadata.

        Returns
        -------
        ExperimentArtifacts
            Unified representation of all successfully ingested artifacts.

        Raises
        ------
        ValueError
            If conflicting inputs are supplied or an artifact is invalid.

        FileNotFoundError
            If a supplied artifact path does not exist.
        """

        logger.info(
            "Starting artifact ingestion for workload=%s",
            self.workload or "unspecified",
        )

        # --------------------------------------------------------------------
        # Validate mutually exclusive inputs.
        # --------------------------------------------------------------------

        if model is not None and model_config_path is not None:
            raise ValueError(
                "Provide either 'model' or 'model_config_path', "
                "not both."
            )

        if training_log_path is not None and training_logs is not None:
            raise ValueError(
                "Provide either 'training_log_path' or 'training_logs', "
                "not both."
            )

        # --------------------------------------------------------------------
        # Dataset
        # --------------------------------------------------------------------

        dataset_analysis = None

        if dataset_path is not None:
            logger.info(
                "Ingesting dataset artifact: %s",
                dataset_path,
            )

            dataset_analysis = self.dataset_analyzer.analyze(
                file_path=dataset_path,
                dataset_name=dataset_name,
            )

        # --------------------------------------------------------------------
        # Model
        # --------------------------------------------------------------------

        model_analysis = None

        if model is not None:
            logger.info(
                "Ingesting model artifact.",
            )

            model_analysis = self.model_analyzer.analyze(
                model=model,
                model_name=model_name,
                architecture=architecture,
                configuration=model_configuration,
                embeddings=embeddings,
            )

        elif model_config_path is not None:
            logger.info(
                "Ingesting model configuration: %s",
                model_config_path,
            )

            model_analysis = self.model_analyzer.analyze_config(
                file_path=model_config_path,
            )

        # --------------------------------------------------------------------
        # Training Logs
        # --------------------------------------------------------------------

        training_log_analysis = None

        if training_log_path is not None:
            logger.info(
                "Ingesting training-log artifact: %s",
                training_log_path,
            )

            training_log_analysis = self.training_log_analyzer.analyze(
                file_path=training_log_path,
            )

        elif training_logs is not None:
            logger.info(
                "Ingesting in-memory training logs.",
            )

            training_log_analysis = (
                self.training_log_analyzer.analyze_records(
                    training_logs,
                )
            )

        # --------------------------------------------------------------------
        # Hyperparameters
        # --------------------------------------------------------------------

        resolved_hyperparameters = (
            hyperparameters.copy()
            if hyperparameters is not None
            else {}
        )

        # --------------------------------------------------------------------
        # Build Unified Artifact Contract
        # --------------------------------------------------------------------

        artifacts = ExperimentArtifacts(
            dataset=dataset_analysis,
            model=model_analysis,
            training_logs=training_log_analysis,
            hyperparameters=resolved_hyperparameters,
        )

        logger.info(
            "Artifact ingestion completed: "
            "dataset=%s model=%s training_logs=%s",
            dataset_analysis is not None,
            model_analysis is not None,
            training_log_analysis is not None,
        )

        return artifacts