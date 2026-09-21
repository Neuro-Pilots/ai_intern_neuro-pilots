"""
Model Analyzer for NeuroPilots.

Layer:
    Layer 1 — Artifact Ingestion

Responsibility:
    Extract structured information about an ML/DL model and convert it into
    the shared ModelAnalysis contract.

The analyzer is intentionally framework-agnostic.

It can currently analyze:
    - Model name
    - Architecture
    - Parameter count
    - Embedding information
    - Model configuration

The analyzer accepts either:
    1. A Python object representing a model.
    2. A dictionary containing model metadata/configuration.
    3. A JSON model configuration file.

Framework-specific model inspection is kept optional so that NeuroPilots
does not become tightly coupled to one ML framework.

Future workload-specific adapters can provide richer information for:

    Computer Vision
    NLP
    Time Series
    Tabular ML
    Speech
    Generative AI
    Reinforcement Learning
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from utils.constants import SUPPORTED_WORKLOADS
from utils.logger import get_logger
from utils.models import ModelAnalysis


# ============================================================================
# LOGGER
# ============================================================================

logger = get_logger(__name__)


# ============================================================================
# SUPPORTED CONFIGURATION FORMATS
# ============================================================================

SUPPORTED_MODEL_CONFIG_EXTENSIONS = {
    "json",
}


# ============================================================================
# MODEL ANALYZER
# ============================================================================

class ModelAnalyzer:
    """
    Analyze model metadata and produce a validated ModelAnalysis object.

    The class does not train, evaluate, or modify a model.

    Its responsibility is limited to extracting model information that can
    later be consumed by:

        Performance Analyzer
                ↓
        Root-Cause Engine
                ↓
        AI Research Advisor
    """

    def __init__(
        self,
        workload: Optional[str] = None,
    ) -> None:
        """
        Initialize the Model Analyzer.

        Parameters
        ----------
        workload:
            Optional NeuroPilots workload identifier.

        Raises
        ------
        ValueError
            If an unsupported workload is provided.
        """

        if workload is not None and workload not in SUPPORTED_WORKLOADS:
            raise ValueError(
                f"Unsupported workload '{workload}'. "
                f"Supported workloads: {SUPPORTED_WORKLOADS}"
            )

        self.workload = workload

    # ========================================================================
    # PUBLIC API — PYTHON MODEL
    # ========================================================================

    def analyze(
        self,
        model: Any,
        model_name: Optional[str] = None,
        architecture: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None,
        embeddings: Optional[Dict[str, Any]] = None,
    ) -> ModelAnalysis:
        """
        Analyze a model object or model metadata.

        Parameters
        ----------
        model:
            Model object or metadata dictionary.

        model_name:
            Optional explicit model name.

        architecture:
            Optional explicit architecture name.

        configuration:
            Optional model configuration.

        embeddings:
            Optional embedding-related information.

        Returns
        -------
        ModelAnalysis
            Structured model-analysis result.

        Notes
        -----
        Explicit metadata takes precedence over automatically extracted
        information. This is intentional because a caller may have more
        reliable information than what can be inferred from a generic
        Python object.
        """

        logger.info(
            "Starting model analysis for workload=%s",
            self.workload or "unspecified",
        )

        if model is None:
            raise ValueError(
                "Model cannot be None."
            )

        # --------------------------------------------------------------------
        # Model Name
        # --------------------------------------------------------------------

        resolved_name = (
            model_name
            or self._extract_model_name(model)
        )

        # --------------------------------------------------------------------
        # Architecture
        # --------------------------------------------------------------------

        resolved_architecture = (
            architecture
            or self._extract_architecture(model)
        )

        # --------------------------------------------------------------------
        # Parameter Count
        # --------------------------------------------------------------------

        parameter_count = self._extract_parameter_count(model)

        # --------------------------------------------------------------------
        # Configuration
        # --------------------------------------------------------------------

        resolved_configuration = (
            configuration.copy()
            if configuration is not None
            else self._extract_configuration(model)
        )

        # --------------------------------------------------------------------
        # Embeddings
        # --------------------------------------------------------------------

        resolved_embeddings = (
            embeddings.copy()
            if embeddings is not None
            else self._extract_embeddings(model)
        )

        analysis = ModelAnalysis(
            name=resolved_name,
            architecture=resolved_architecture,
            parameter_count=parameter_count,
            embeddings=resolved_embeddings,
            configuration=resolved_configuration,
        )

        logger.info(
            "Model analysis completed: name=%s architecture=%s "
            "parameters=%s",
            analysis.name or "unknown",
            analysis.architecture or "unknown",
            analysis.parameter_count
            if analysis.parameter_count is not None
            else "unknown",
        )

        return analysis

    # ========================================================================
    # PUBLIC API — JSON CONFIGURATION
    # ========================================================================

    def analyze_config(
        self,
        file_path: str | Path,
    ) -> ModelAnalysis:
        """
        Analyze a model configuration stored in a JSON file.

        Expected JSON structure can contain fields such as:

            {
                "name": "ConvNeXt-Tiny",
                "architecture": "ConvNeXt",
                "parameter_count": 28589128,
                "configuration": {
                    "input_size": 224
                },
                "embeddings": {}
            }

        Additional fields are preserved inside ``configuration`` when
        appropriate.

        Parameters
        ----------
        file_path:
            Path to the JSON configuration.

        Returns
        -------
        ModelAnalysis
            Structured model-analysis result.
        """

        path = Path(file_path)

        logger.info(
            "Starting model configuration analysis: %s",
            path,
        )

        self._validate_config_file(path)

        data = self._load_json(path)

        if not data:
            raise ValueError(
                f"Model configuration is empty: {path}"
            )

        configuration = data.get(
            "configuration",
            {},
        )

        if not isinstance(configuration, dict):
            raise ValueError(
                "The 'configuration' field must be a JSON object."
            )

        embeddings = data.get(
            "embeddings",
            {},
        )

        if not isinstance(embeddings, dict):
            raise ValueError(
                "The 'embeddings' field must be a JSON object."
            )

        analysis = ModelAnalysis(
            name=self._optional_string(
                data.get("name")
            ),
            architecture=self._optional_string(
                data.get("architecture")
            ),
            parameter_count=self._safe_parameter_count(
                data.get("parameter_count")
            ),
            embeddings=embeddings,
            configuration=configuration,
        )

        logger.info(
            "Model configuration analysis completed: %s",
            path,
        )

        return analysis

    # ========================================================================
    # MODEL NAME EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_model_name(
        model: Any,
    ) -> Optional[str]:
        """
        Extract a model name from a model object.

        The analyzer checks common metadata conventions without assuming a
        particular ML framework.
        """

        if isinstance(model, dict):
            for key in (
                "name",
                "model_name",
                "model_id",
            ):
                value = model.get(key)

                if value is not None:
                    return str(value).strip() or None

            return None

        for attribute in (
            "model_name",
            "name",
            "model_id",
        ):
            value = getattr(
                model,
                attribute,
                None,
            )

            if value is not None:
                value_string = str(value).strip()

                if value_string:
                    return value_string

        # A Python class name is a useful fallback.
        class_name = model.__class__.__name__

        return class_name or None

    # ========================================================================
    # ARCHITECTURE EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_architecture(
        model: Any,
    ) -> Optional[str]:
        """
        Extract architecture information when available.

        The method checks common metadata fields but does not make aggressive
        guesses based on class names. Accurate architecture information is
        preferable to an incorrect inferred architecture.
        """

        if isinstance(model, dict):
            for key in (
                "architecture",
                "architecture_name",
                "model_type",
            ):
                value = model.get(key)

                if value is not None:
                    return str(value).strip() or None

            return None

        for attribute in (
            "architecture",
            "architecture_name",
            "model_type",
        ):
            value = getattr(
                model,
                attribute,
                None,
            )

            if value is not None:
                value_string = str(value).strip()

                if value_string:
                    return value_string

        # Some model libraries expose a configuration object containing
        # architecture/model-type metadata.
        config = getattr(
            model,
            "config",
            None,
        )

        if config is not None:
            for attribute in (
                "architecture",
                "architectures",
                "model_type",
            ):
                value = getattr(
                    config,
                    attribute,
                    None,
                )

                if value is None:
                    continue

                if isinstance(value, (list, tuple)):
                    if value:
                        return str(value[0]).strip() or None

                value_string = str(value).strip()

                if value_string:
                    return value_string

        return None

    # ========================================================================
    # PARAMETER COUNT EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_parameter_count(
        model: Any,
    ) -> Optional[int]:
        """
        Extract the number of model parameters.

        The method supports common conventions without importing framework
        packages.

        For example, PyTorch-like models commonly expose:

            model.parameters()

        while metadata dictionaries may contain:

            parameter_count

        If the information cannot be determined safely, None is returned.
        """

        if isinstance(model, dict):
            for key in (
                "parameter_count",
                "num_parameters",
                "parameters",
            ):
                if key in model:
                    return ModelAnalyzer._safe_parameter_count(
                        model[key]
                    )

            return None

        # Common direct attributes.
        for attribute in (
            "parameter_count",
            "num_parameters",
        ):
            value = getattr(
                model,
                attribute,
                None,
            )

            if value is not None:
                result = ModelAnalyzer._safe_parameter_count(
                    value
                )

                if result is not None:
                    return result

        # Generic support for model objects exposing parameters().
        parameters_method = getattr(
            model,
            "parameters",
            None,
        )

        if callable(parameters_method):
            try:
                total_parameters = 0

                for parameter in parameters_method():
                    numel_method = getattr(
                        parameter,
                        "numel",
                        None,
                    )

                    if callable(numel_method):
                        total_parameters += int(
                            numel_method()
                        )

                return total_parameters

            except (TypeError, ValueError, RuntimeError):
                logger.warning(
                    "Unable to determine parameter count from model object."
                )

        return None

    # ========================================================================
    # CONFIGURATION EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_configuration(
        model: Any,
    ) -> Dict[str, Any]:
        """
        Extract serializable configuration information from a model.

        The method avoids blindly serializing an arbitrary model object,
        which can result in huge or non-serializable structures.
        """

        if isinstance(model, dict):
            configuration = model.get(
                "configuration",
                {},
            )

            if isinstance(configuration, dict):
                return configuration.copy()

            return {}

        config = getattr(
            model,
            "config",
            None,
        )

        if config is None:
            return {}

        # Pydantic-style configuration.
        model_dump = getattr(
            config,
            "model_dump",
            None,
        )

        if callable(model_dump):
            try:
                result = model_dump()

                if isinstance(result, dict):
                    return result

            except Exception:
                logger.debug(
                    "Unable to extract configuration using model_dump().",
                    exc_info=True,
                )

        # Hugging Face-style configuration.
        to_dict = getattr(
            config,
            "to_dict",
            None,
        )

        if callable(to_dict):
            try:
                result = to_dict()

                if isinstance(result, dict):
                    return result

            except Exception:
                logger.debug(
                    "Unable to extract configuration using to_dict().",
                    exc_info=True,
                )

        return {}

    # ========================================================================
    # EMBEDDING EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_embeddings(
        model: Any,
    ) -> Dict[str, Any]:
        """
        Extract embedding metadata when available.

        Embedding analysis is particularly relevant to NLP and Generative AI,
        but the method remains generic so it can support future workloads.
        """

        if isinstance(model, dict):
            embeddings = model.get(
                "embeddings",
                {},
            )

            if isinstance(embeddings, dict):
                return embeddings.copy()

            return {}

        embeddings = getattr(
            model,
            "embeddings",
            None,
        )

        if isinstance(embeddings, dict):
            return embeddings.copy()

        return {}

    # ========================================================================
    # CONFIGURATION FILE HELPERS
    # ========================================================================

    @staticmethod
    def _validate_config_file(
        path: Path,
    ) -> None:
        """
        Validate a model configuration file before loading it.
        """

        if not str(path).strip():
            raise ValueError(
                "Model configuration path cannot be empty."
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Model configuration file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Model configuration path is not a file: {path}"
            )

        extension = path.suffix.lower().lstrip(".")

        if extension not in SUPPORTED_MODEL_CONFIG_EXTENSIONS:
            raise ValueError(
                f"Unsupported model configuration format '.{extension}'. "
                "Supported formats: json"
            )

    @staticmethod
    def _load_json(
        path: Path,
    ) -> Dict[str, Any]:
        """
        Load and validate a JSON model configuration.
        """

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in model configuration: {path}"
            ) from exc

        except OSError as exc:
            raise ValueError(
                f"Unable to read model configuration: {path}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Model configuration JSON must contain an object."
            )

        return data

    # ========================================================================
    # VALIDATION HELPERS
    # ========================================================================

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> Optional[str]:
        """
        Convert a value into an optional clean string.
        """

        if value is None:
            return None

        result = str(value).strip()

        return result or None

    @staticmethod
    def _safe_parameter_count(
        value: Any,
    ) -> Optional[int]:
        """
        Safely convert parameter-count metadata into a non-negative integer.
        """

        if value is None:
            return None

        # Some metadata may contain a human-readable value such as
        # "28.6M". We deliberately do not guess its numeric meaning here.
        if isinstance(value, str):
            value = value.strip()

            if not value.isdigit():
                return None

        try:
            result = int(value)
        except (TypeError, ValueError):
            return None

        if result < 0:
            return None

        return result