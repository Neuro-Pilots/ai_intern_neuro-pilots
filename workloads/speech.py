"""
Speech workload engine for NeuroPilots.

This module defines the workload-specific behavior for Speech
experiments.

The current responsibility of this layer is to:
    - Identify the Speech workload.
    - Expose Whisper-large-v3-turbo as the default model.
    - Validate Speech experiment artifacts.
    - Describe Speech-specific experiment requirements.

Actual model loading, audio preprocessing, training, transcription,
evaluation, and experiment execution are handled by later layers.
"""

from __future__ import annotations

from typing import Any, Dict

from utils.constants import WORKLOAD_SPEECH
from utils.logger import get_logger
from utils.models import ExperimentArtifacts

from workloads.base import BaseWorkloadEngine


logger = get_logger(__name__)


class SpeechWorkload(BaseWorkloadEngine):
    """
    Workload engine for Speech Machine Learning experiments.

    NeuroPilots uses Whisper-large-v3-turbo as the default Speech model
    according to the system architecture.

    This class provides workload metadata and artifact validation only.
    It does not instantiate, download, train, or evaluate the actual
    speech model.
    """

    # Canonical workload identifier used throughout NeuroPilots.
    workload_name = WORKLOAD_SPEECH

    # ------------------------------------------------------------------
    # Model specification
    # ------------------------------------------------------------------

    def get_model_spec(self) -> Dict[str, Any]:
        """
        Return the Speech model specification.

        Returns
        -------
        dict
            Speech model metadata used by downstream components.
        """

        return {
            "workload": self.workload_name,
            "model_name": self.default_model_name,
            "model_family": "Whisper",
            "model_variant": "large-v3-turbo",
            "task_types": [
                "automatic_speech_recognition",
                "speech_transcription",
            ],
            "input_type": "audio",
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
        Validate artifacts required by the Speech workload.

        At this stage we validate the presence of the core dataset and
        model artifacts. Detailed audio-format, sampling-rate, and
        transcription validation belongs to later processing layers.

        Parameters
        ----------
        artifacts:
            ExperimentArtifacts produced by the ingestion layer.

        Raises
        ------
        ValueError
            If required Speech artifacts are missing.
        """

        if artifacts.dataset is None:
            raise ValueError(
                "Speech workload requires dataset artifacts."
            )

        if artifacts.model is None:
            raise ValueError(
                "Speech workload requires model artifacts."
            )

        logger.debug(
            "Speech workload artifacts validated successfully."
        )

    # ------------------------------------------------------------------
    # Experiment requirements
    # ------------------------------------------------------------------

    def get_experiment_requirements(self) -> Dict[str, Any]:
        """
        Return the information expected from a Speech experiment.

        These requirements describe the signals that downstream
        components should consider when analyzing a speech model.
        """

        return {
            "dataset": {
                "required": True,
                "expected_input_type": "audio",
                "important_signals": [
                    "dataset_size",
                    "audio_duration_distribution",
                    "sampling_rate",
                    "audio_format",
                    "missing_values",
                    "duplicate_samples",
                    "language_distribution",
                    "transcription_quality",
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
                    "processor_configuration",
                    "language_configuration",
                ],
            },
            "training_logs": {
                "required": False,
                "important_signals": [
                    "train_loss",
                    "validation_loss",
                    "word_error_rate",
                    "character_error_rate",
                    "learning_rate",
                    "epoch_progression",
                ],
            },
            "hyperparameters": {
                "important_parameters": [
                    "learning_rate",
                    "batch_size",
                    "gradient_accumulation_steps",
                    "warmup_steps",
                    "weight_decay",
                    "epochs",
                ],
            },
            "audio_processing": {
                "important_parameters": [
                    "sampling_rate",
                    "audio_duration",
                    "normalization",
                    "noise_reduction",
                    "voice_activity_detection",
                    "language",
                ],
            },
        }

    # ------------------------------------------------------------------
    # Workload description
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """
        Return a complete description of the Speech workload.

        Combines common workload metadata with Speech-specific
        experiment requirements.
        """

        description = super().describe()

        description.update(
            {
                "domain": "speech_processing",
                "experiment_requirements": self.get_experiment_requirements(),
            }
        )

        return description