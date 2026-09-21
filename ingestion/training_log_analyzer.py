"""
Training Log Analyzer for NeuroPilots.

Layer:
    Layer 1 — Artifact Ingestion

Responsibility:
    Parse training-log artifacts and convert them into the shared
    TrainingLogAnalysis data contract.

The analyzer extracts:

    - Number of epochs
    - Training and validation metrics
    - Learning-rate history
    - Checkpoint information
    - Basic training signals

The analyzer focuses on OBSERVATION rather than diagnosis.

For example:

    Observed:
        training loss decreases
        validation loss increases

    Not diagnosed here:
        "The model is overfitting."

That diagnosis belongs to the Root-Cause Engine.

This separation keeps the architecture clean:

    Training Logs
          ↓
    TrainingLogAnalyzer
          ↓
    TrainingLogAnalysis
          ↓
    Performance Analyzer
          ↓
    Root-Cause Engine
          ↓
    AI Research Advisor


Supported input formats:

    - JSON
    - CSV

The analyzer also supports direct Python dictionaries so that tests and
other internal components do not need to create temporary files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.helpers import get_file_extension
from utils.logger import get_logger
from utils.models import TrainingLogAnalysis


# ============================================================================
# LOGGER
# ============================================================================

logger = get_logger(__name__)


# ============================================================================
# SUPPORTED FILE FORMATS
# ============================================================================

SUPPORTED_LOG_EXTENSIONS = {
    "json",
    "csv",
}


# ============================================================================
# COMMON METRIC NAMES
# ============================================================================

# These names cover common ML training-log conventions.
#
# The analyzer is not restricted to these names. Any numeric metric supplied
# in a JSON/CSV log can be preserved.

TRAINING_METRIC_NAMES = {
    "loss",
    "train_loss",
    "training_loss",
    "val_loss",
    "validation_loss",
    "accuracy",
    "train_accuracy",
    "val_accuracy",
    "validation_accuracy",
    "precision",
    "recall",
    "f1",
    "f1_score",
    "mae",
    "mse",
    "rmse",
}


# ============================================================================
# TRAINING LOG ANALYZER
# ============================================================================

class TrainingLogAnalyzer:
    """
    Analyze training logs and produce a validated TrainingLogAnalysis.

    The class supports both file-based and in-memory training logs.

    It deliberately does not depend on a specific training framework such as
    PyTorch, TensorFlow, Keras, or Hugging Face.

    This makes it suitable for the seven NeuroPilots workloads.
    """

    # ========================================================================
    # PUBLIC API — FILE
    # ========================================================================

    def analyze(
        self,
        file_path: str | Path,
    ) -> TrainingLogAnalysis:
        """
        Analyze a training-log file.

        Parameters
        ----------
        file_path:
            Path to a JSON or CSV training-log file.

        Returns
        -------
        TrainingLogAnalysis
            Structured training-log analysis.

        Raises
        ------
        FileNotFoundError
            If the log file does not exist.

        ValueError
            If the format is unsupported or the log cannot be parsed.
        """

        path = Path(file_path)

        logger.info(
            "Starting training log analysis: %s",
            path,
        )

        self._validate_file(path)

        extension = get_file_extension(path)

        if extension == "json":
            raw_data = self._load_json(path)

            result = self.analyze_records(raw_data)

        elif extension == "csv":
            dataframe = self._load_csv(path)

            result = self.analyze_dataframe(dataframe)

        else:
            # This should be unreachable because _validate_file() already
            # checks the extension.
            raise ValueError(
                f"Unsupported training log format: {extension}"
            )

        logger.info(
            "Training log analysis completed: epochs=%s metrics=%s",
            result.epochs,
            list(result.metrics.keys()),
        )

        return result

    # ========================================================================
    # PUBLIC API — IN-MEMORY RECORDS
    # ========================================================================

    def analyze_records(
        self,
        records: Any,
    ) -> TrainingLogAnalysis:
        """
        Analyze training logs represented as Python data.

        Supported structures include:

            [
                {
                    "epoch": 1,
                    "train_loss": 0.80,
                    "val_loss": 0.85,
                    "learning_rate": 0.001
                },
                {
                    "epoch": 2,
                    "train_loss": 0.60,
                    "val_loss": 0.70,
                    "learning_rate": 0.0005
                }
            ]

        or:

            {
                "epoch": [1, 2],
                "train_loss": [0.80, 0.60],
                "val_loss": [0.85, 0.70]
            }

        Parameters
        ----------
        records:
            Training-log data.

        Returns
        -------
        TrainingLogAnalysis
            Structured analysis.
        """

        dataframe = self._records_to_dataframe(records)

        return self.analyze_dataframe(dataframe)

    # ========================================================================
    # PUBLIC API — DATAFRAME
    # ========================================================================

    def analyze_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> TrainingLogAnalysis:
        """
        Analyze a pandas DataFrame containing training history.

        The analyzer searches for common columns such as:

            epoch
            step
            train_loss
            val_loss
            learning_rate

        All numeric columns other than known control columns can also be
        retained as metrics.

        Parameters
        ----------
        dataframe:
            Training-history DataFrame.

        Returns
        -------
        TrainingLogAnalysis
            Structured training-log analysis.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "analyze_dataframe expects a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Training log is empty."
            )

        dataframe = dataframe.copy()

        # Normalize column names so logs such as "Train Loss" and
        # "train_loss" can be handled consistently.
        dataframe.columns = [
            self._normalize_column_name(column)
            for column in dataframe.columns
        ]

        epochs = self._extract_epoch_count(dataframe)

        metrics = self._extract_metrics(dataframe)

        learning_rate = self._extract_learning_rate(dataframe)

        checkpoints = self._extract_checkpoints(dataframe)

        training_signals = self._derive_training_signals(
            metrics=metrics,
            learning_rate=learning_rate,
            epochs=epochs,
        )

        return TrainingLogAnalysis(
            epochs=epochs,
            metrics=metrics,
            learning_rate=learning_rate,
            checkpoints=checkpoints,
            training_signals=training_signals,
        )

    # ========================================================================
    # FILE VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_file(
        path: Path,
    ) -> None:
        """
        Validate the supplied training-log path.
        """

        if not str(path).strip():
            raise ValueError(
                "Training log path cannot be empty."
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Training log does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Training log path is not a file: {path}"
            )

        extension = get_file_extension(path)

        if extension not in SUPPORTED_LOG_EXTENSIONS:
            supported = ", ".join(
                sorted(SUPPORTED_LOG_EXTENSIONS)
            )

            raise ValueError(
                f"Unsupported training log format '.{extension}'. "
                f"Supported formats: {supported}"
            )

    # ========================================================================
    # FILE LOADERS
    # ========================================================================

    @staticmethod
    def _load_json(
        path: Path,
    ) -> Any:
        """
        Load JSON training-log data from disk.
        """

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON training log: {path}"
            ) from exc

        except OSError as exc:
            raise ValueError(
                f"Unable to read training log: {path}"
            ) from exc

    @staticmethod
    def _load_csv(
        path: Path,
    ) -> pd.DataFrame:
        """
        Load CSV training history.
        """

        try:
            dataframe = pd.read_csv(path)

        except Exception as exc:
            logger.exception(
                "Failed to load training CSV: %s",
                path,
            )

            raise ValueError(
                f"Unable to load training log: {path}"
            ) from exc

        return dataframe

    # ========================================================================
    # DATA NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_column_name(
        column: Any,
    ) -> str:
        """
        Normalize a training-log column name.

        Examples:

            "Train Loss"       → "train_loss"
            "Validation Loss"  → "validation_loss"
            "Learning Rate"    → "learning_rate"
        """

        normalized = str(column).strip().lower()

        normalized = normalized.replace("-", "_")
        normalized = normalized.replace(" ", "_")

        while "__" in normalized:
            normalized = normalized.replace(
                "__",
                "_",
            )

        return normalized

    # ========================================================================
    # RECORD CONVERSION
    # ========================================================================

    @staticmethod
    def _records_to_dataframe(
        records: Any,
    ) -> pd.DataFrame:
        """
        Convert supported in-memory training-log structures into a DataFrame.
        """

        if isinstance(records, pd.DataFrame):
            return records.copy()

        if isinstance(records, list):
            if not records:
                raise ValueError(
                    "Training log records cannot be empty."
                )

            if not all(
                isinstance(record, dict)
                for record in records
            ):
                raise ValueError(
                    "List-based training logs must contain dictionaries."
                )

            return pd.DataFrame(records)

        if isinstance(records, dict):
            if not records:
                raise ValueError(
                    "Training log records cannot be empty."
                )

            # Case 1:
            # A dictionary containing a list of epoch records.
            #
            # Example:
            # {
            #     "history": [
            #         {"epoch": 1, "loss": 0.8},
            #         {"epoch": 2, "loss": 0.6}
            #     ]
            # }
            for key in (
                "history",
                "records",
                "logs",
            ):
                value = records.get(key)

                if isinstance(value, list):
                    return TrainingLogAnalyzer._records_to_dataframe(
                        value
                    )

            # Case 2:
            # Dictionary of metric histories.
            #
            # Example:
            # {
            #     "epoch": [1, 2, 3],
            #     "loss": [0.8, 0.6, 0.4]
            # }
            if all(
                isinstance(value, (list, tuple))
                for value in records.values()
            ):
                return pd.DataFrame(records)

            # Case 3:
            # A single training record.
            return pd.DataFrame([records])

        raise TypeError(
            "Training logs must be a DataFrame, list of dictionaries, "
            "or dictionary."
        )

    # ========================================================================
    # EPOCH EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_epoch_count(
        dataframe: pd.DataFrame,
    ) -> Optional[int]:
        """
        Determine the number of epochs represented by the log.

        Preferred behavior:

            1. Use the epoch column when available.
            2. Otherwise use the number of rows as a fallback.

        The fallback is an observation about log records, not a claim that
        every record necessarily represents exactly one epoch.
        """

        for column in (
            "epoch",
            "epochs",
        ):
            if column not in dataframe.columns:
                continue

            numeric_values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).dropna()

            if numeric_values.empty:
                continue

            return int(numeric_values.max())

        return int(len(dataframe))

    # ========================================================================
    # METRIC EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_metrics(
        dataframe: pd.DataFrame,
    ) -> Dict[str, List[float]]:
        """
        Extract numeric training metrics.

        Control columns such as epoch, step, and learning rate are excluded.

        Any remaining numeric column is considered a potential metric.
        This is intentionally permissive because different workloads use
        different evaluation metrics.
        """

        excluded_columns = {
            "epoch",
            "epochs",
            "step",
            "steps",
            "global_step",
            "learning_rate",
            "lr",
        }

        metrics: Dict[str, List[float]] = {}

        for column in dataframe.columns:
            if column in excluded_columns:
                continue

            numeric_values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            if numeric_values.notna().sum() == 0:
                continue

            values: List[float] = []

            for value in numeric_values:
                if pd.isna(value):
                    continue

                values.append(float(value))

            if values:
                metrics[column] = values

        return metrics

    # ========================================================================
    # LEARNING-RATE EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_learning_rate(
        dataframe: pd.DataFrame,
    ) -> List[float]:
        """
        Extract learning-rate history.

        Both ``learning_rate`` and ``lr`` are supported.
        """

        for column in (
            "learning_rate",
            "lr",
        ):
            if column not in dataframe.columns:
                continue

            numeric_values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).dropna()

            return [
                float(value)
                for value in numeric_values
            ]

        return []

    # ========================================================================
    # CHECKPOINT EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_checkpoints(
        dataframe: pd.DataFrame,
    ) -> List[str]:
        """
        Extract checkpoint identifiers when a checkpoint column exists.
        """

        for column in (
            "checkpoint",
            "checkpoint_path",
            "checkpoint_id",
        ):
            if column not in dataframe.columns:
                continue

            values = dataframe[column].dropna()

            return [
                str(value)
                for value in values
            ]

        return []

    # ========================================================================
    # TRAINING SIGNALS
    # ========================================================================

    @staticmethod
    def _derive_training_signals(
        metrics: Dict[str, List[float]],
        learning_rate: List[float],
        epochs: Optional[int],
    ) -> Dict[str, Any]:
        """
        Derive simple factual signals from the training history.

        These are NOT root-cause diagnoses.

        Example:

            "train_loss_trend": "decreasing"

        is an observation.

            "root_cause": "overfitting"

        would be a diagnosis and therefore belongs in the diagnostics layer.
        """

        signals: Dict[str, Any] = {}

        signals["epochs_available"] = epochs is not None

        # --------------------------------------------------------------------
        # Loss Trends
        # --------------------------------------------------------------------

        train_loss = TrainingLogAnalyzer._find_metric(
            metrics,
            (
                "train_loss",
                "training_loss",
                "loss",
            ),
        )

        validation_loss = TrainingLogAnalyzer._find_metric(
            metrics,
            (
                "val_loss",
                "validation_loss",
            ),
        )

        if train_loss:
            signals["train_loss_trend"] = (
                TrainingLogAnalyzer._calculate_trend(
                    train_loss
                )
            )

        if validation_loss:
            signals["validation_loss_trend"] = (
                TrainingLogAnalyzer._calculate_trend(
                    validation_loss
                )
            )

        # --------------------------------------------------------------------
        # Metric Trends
        # --------------------------------------------------------------------

        train_accuracy = TrainingLogAnalyzer._find_metric(
            metrics,
            (
                "train_accuracy",
            ),
        )

        validation_accuracy = TrainingLogAnalyzer._find_metric(
            metrics,
            (
                "val_accuracy",
                "validation_accuracy",
            ),
        )

        if train_accuracy:
            signals["train_accuracy_trend"] = (
                TrainingLogAnalyzer._calculate_trend(
                    train_accuracy
                )
            )

        if validation_accuracy:
            signals["validation_accuracy_trend"] = (
                TrainingLogAnalyzer._calculate_trend(
                    validation_accuracy
                )
            )

        # --------------------------------------------------------------------
        # Learning Rate
        # --------------------------------------------------------------------

        if learning_rate:
            signals["learning_rate_changes"] = (
                len(learning_rate) - 1
            )

            signals["initial_learning_rate"] = (
                learning_rate[0]
            )

            signals["final_learning_rate"] = (
                learning_rate[-1]
            )

        return signals

    # ========================================================================
    # METRIC LOOKUP
    # ========================================================================

    @staticmethod
    def _find_metric(
        metrics: Dict[str, List[float]],
        names: tuple[str, ...],
    ) -> Optional[List[float]]:
        """
        Find the first available metric from a list of possible names.
        """

        for name in names:
            if name in metrics:
                return metrics[name]

        return None

    # ========================================================================
    # TREND ANALYSIS
    # ========================================================================

    @staticmethod
    def _calculate_trend(
        values: List[float],
    ) -> str:
        """
        Calculate a simple trend classification.

        The classification is intentionally conservative:

            increasing
            decreasing
            stable

        A trend is calculated using the first and last values.

        This is a descriptive signal only and should not be treated as a
        statistical significance test.
        """

        if len(values) < 2:
            return "insufficient_data"

        first = values[0]
        last = values[-1]

        if first == 0:
            difference = abs(last - first)

            if difference < 1e-12:
                return "stable"

            return "increasing" if last > first else "decreasing"

        relative_change = (
            (last - first)
            / abs(first)
        )

        # Small changes are treated as stable to avoid over-interpreting
        # floating-point noise.
        if abs(relative_change) < 0.01:
            return "stable"

        if relative_change > 0:
            return "increasing"

        return "decreasing"