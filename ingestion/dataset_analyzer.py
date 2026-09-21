"""
Dataset Analyzer for NeuroPilots.

Layer:
    Layer 1 — Artifact Ingestion

Responsibility:
    Inspect an input dataset and convert its raw structure into the shared
    DatasetAnalysis data contract.

The analyzer extracts:

    - Dataset name
    - Number of rows and columns
    - Input feature names
    - Target column
    - Class distribution
    - Missing-value statistics
    - Duplicate-row statistics
    - Constant columns
    - Numerical feature statistics
    - Categorical feature statistics
    - Basic data-quality signals

Important architectural principle:

    This component OBSERVES the dataset.

    It does not diagnose the cause of model-performance problems and does
    not generate recommendations.

The downstream architecture is:

    Dataset
       ↓
    DatasetAnalyzer
       ↓
    DatasetAnalysis
       ↓
    Performance Analyzer
       ↓
    Root-Cause Engine
       ↓
    AI Research Advisor

Supported input formats:

    - CSV
    - JSON
    - Parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from utils.constants import SUPPORTED_WORKLOADS
from utils.helpers import get_file_extension
from utils.logger import get_logger
from utils.models import DatasetAnalysis


# ============================================================================
# LOGGER
# ============================================================================

logger = get_logger(__name__)


# ============================================================================
# SUPPORTED DATASET FORMATS
# ============================================================================

SUPPORTED_DATASET_EXTENSIONS = {
    "csv",
    "json",
    "parquet",
}


# ============================================================================
# DATASET ANALYZER
# ============================================================================

class DatasetAnalyzer:
    """
    Analyze datasets and produce a validated DatasetAnalysis object.

    The analyzer is intentionally independent of:

        - Streamlit
        - MLflow
        - LLM providers
        - Model training
        - Workload-specific model implementations

    This keeps dataset inspection reusable and independently testable.
    """

    def __init__(
        self,
        target_column: Optional[str] = None,
        workload: Optional[str] = None,
    ) -> None:
        """
        Initialize the Dataset Analyzer.

        Parameters
        ----------
        target_column:
            Optional target/label column.

            If provided, the target is explicitly separated from the model
            input features.

        workload:
            Optional NeuroPilots workload identifier.

        Raises
        ------
        ValueError
            If an unsupported workload is supplied.
        """

        self.target_column = target_column

        if workload is not None and workload not in SUPPORTED_WORKLOADS:
            raise ValueError(
                f"Unsupported workload '{workload}'. "
                f"Supported workloads: {SUPPORTED_WORKLOADS}"
            )

        self.workload = workload

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def analyze(
        self,
        file_path: str | Path,
        dataset_name: Optional[str] = None,
    ) -> DatasetAnalysis:
        """
        Analyze a dataset from disk.

        Parameters
        ----------
        file_path:
            Path to a CSV, JSON, or Parquet dataset.

        dataset_name:
            Optional human-readable dataset name.

            If omitted, the filename without its extension is used.

        Returns
        -------
        DatasetAnalysis
            Structured and validated dataset analysis.

        Raises
        ------
        FileNotFoundError
            If the dataset does not exist.

        ValueError
            If the dataset format is unsupported, unreadable, empty,
            or the configured target column does not exist.
        """

        path = Path(file_path)

        logger.info(
            "Starting dataset analysis: %s",
            path,
        )

        self._validate_file(path)

        dataframe = self._load_dataset(path)

        if dataframe.empty:
            raise ValueError(
                f"Dataset is empty: {path}"
            )

        resolved_name = dataset_name or path.stem

        analysis = DatasetAnalysis(
            name=resolved_name,
            rows=int(dataframe.shape[0]),
            columns=int(dataframe.shape[1]),
            features=self._extract_feature_columns(dataframe),
            target=self._resolve_target(dataframe),
            class_distribution=self._analyze_class_distribution(
                dataframe
            ),
            missing_values=self._analyze_missing_values(
                dataframe
            ),
            quality_signals=self._analyze_quality(
                dataframe
            ),
        )

        logger.info(
            "Dataset analysis completed: name=%s rows=%d columns=%d",
            resolved_name,
            analysis.rows or 0,
            analysis.columns or 0,
        )

        return analysis

    # ========================================================================
    # FILE VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_file(
        path: Path,
    ) -> None:
        """
        Validate that the supplied dataset path is usable.
        """

        if not str(path).strip():
            raise ValueError(
                "Dataset path cannot be empty."
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Dataset path is not a file: {path}"
            )

        extension = get_file_extension(path)

        if extension not in SUPPORTED_DATASET_EXTENSIONS:
            supported = ", ".join(
                sorted(SUPPORTED_DATASET_EXTENSIONS)
            )

            raise ValueError(
                f"Unsupported dataset format '.{extension}'. "
                f"Supported formats: {supported}"
            )

    # ========================================================================
    # DATASET LOADING
    # ========================================================================

    @staticmethod
    def _load_dataset(
        path: Path,
    ) -> pd.DataFrame:
        """
        Load a supported dataset into a pandas DataFrame.
        """

        extension = get_file_extension(path)

        try:
            if extension == "csv":
                dataframe = pd.read_csv(path)

            elif extension == "json":
                dataframe = pd.read_json(path)

            elif extension == "parquet":
                dataframe = pd.read_parquet(path)

            else:
                # Defensive check. _validate_file() should already prevent
                # this branch.
                raise ValueError(
                    f"Unsupported dataset format: {extension}"
                )

        except ValueError:
            raise

        except Exception as exc:
            logger.exception(
                "Failed to load dataset: %s",
                path,
            )

            raise ValueError(
                f"Unable to load dataset '{path}'."
            ) from exc

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError(
                f"Dataset '{path}' could not be represented as a "
                "tabular DataFrame."
            )

        return dataframe

    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================

    def _extract_feature_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Extract model input feature columns.

        The explicitly configured target column is excluded.

        Example:

            Dataset columns:
                speed
                temperature
                vehicle_type
                target

            target_column:
                target

            Result:
                speed
                temperature
                vehicle_type

        This distinction is important because the target is the value the
        model is trying to predict and must not be treated as an input
        feature.
        """

        return [
            str(column)
            for column in dataframe.columns
            if column != self.target_column
        ]

    # ========================================================================
    # TARGET RESOLUTION
    # ========================================================================

    def _resolve_target(
        self,
        dataframe: pd.DataFrame,
    ) -> Optional[str]:
        """
        Resolve the explicitly configured target column.

        NeuroPilots deliberately does not guess the target automatically.

        Automatically assuming that the last column, "label", or "target"
        is the target can produce incorrect downstream diagnostics.

        Therefore:

            target configured + exists
                → use it

            target configured + does not exist
                → raise an error

            target not configured
                → return None
        """

        if self.target_column is None:
            return None

        if self.target_column not in dataframe.columns:
            raise ValueError(
                f"Configured target column '{self.target_column}' "
                "does not exist in the dataset."
            )

        return self.target_column

    # ========================================================================
    # MISSING-VALUE ANALYSIS
    # ========================================================================

    @staticmethod
    def _analyze_missing_values(
        dataframe: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Analyze missing values across all dataset columns.

        Returns:

            - Total missing values
            - Columns containing missing values
            - Missing count per column
            - Missing percentage per column
            - Number of affected columns
        """

        missing_counts = dataframe.isna().sum()

        total_rows = len(dataframe)

        missing_columns: Dict[str, Dict[str, Any]] = {}

        for column, count in missing_counts.items():
            count_int = int(count)

            if count_int == 0:
                continue

            percentage = (
                (count_int / total_rows) * 100
                if total_rows > 0
                else 0.0
            )

            missing_columns[str(column)] = {
                "count": count_int,
                "percentage": round(
                    percentage,
                    4,
                ),
            }

        total_missing = int(
            missing_counts.sum()
        )

        return {
            "total_missing_values": total_missing,
            "columns_with_missing_values": missing_columns,
            "columns_affected": len(missing_columns),
        }

    # ========================================================================
    # CLASS-DISTRIBUTION ANALYSIS
    # ========================================================================

    def _analyze_class_distribution(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Analyze target-class distribution when a target is configured.

        For continuous regression targets, a class distribution is not
        meaningful and therefore is not generated.

        This method provides observations only. It does not determine whether
        the class distribution is problematic. Class-imbalance diagnosis
        belongs to the Root-Cause Engine.
        """

        if self.target_column is None:
            return {
                "available": False,
                "reason": "No target column configured.",
            }

        target = dataframe[self.target_column]

        unique_count = int(
            target.nunique(
                dropna=True
            )
        )

        # A numeric target with many unique values is treated as likely
        # continuous for descriptive purposes.
        #
        # This does NOT classify the overall ML problem as regression.
        if pd.api.types.is_numeric_dtype(target):
            if unique_count > 20:
                return {
                    "available": False,
                    "reason": (
                        "Target appears continuous; "
                        "class distribution is not applicable."
                    ),
                }

        value_counts = target.value_counts(
            dropna=False,
        )

        total = len(target)

        classes: Dict[str, Dict[str, Any]] = {}

        for class_value, count in value_counts.items():
            count_int = int(count)

            if pd.isna(class_value):
                class_name = "<missing>"
            else:
                class_name = str(class_value)

            percentage = (
                (count_int / total) * 100
                if total > 0
                else 0.0
            )

            classes[class_name] = {
                "count": count_int,
                "percentage": round(
                    percentage,
                    4,
                ),
            }

        return {
            "available": True,
            "number_of_classes": len(classes),
            "classes": classes,
        }

    # ========================================================================
    # DATA-QUALITY ANALYSIS
    # ========================================================================

    def _analyze_quality(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Calculate general data-quality signals for INPUT FEATURES.

        The configured target is deliberately excluded from:

            - numerical_columns
            - categorical_columns
            - numerical_summary
            - categorical_summary

        The target is analyzed separately by the target/class-distribution
        logic.

        This prevents downstream components from accidentally treating the
        prediction target as an input feature.
        """

        row_count = len(dataframe)

        # --------------------------------------------------------------------
        # Duplicate rows
        # --------------------------------------------------------------------

        duplicate_rows = int(
            dataframe.duplicated().sum()
        )

        duplicate_percentage = (
            (duplicate_rows / row_count) * 100
            if row_count > 0
            else 0.0
        )

        # --------------------------------------------------------------------
        # Remove target from feature-level quality analysis
        # --------------------------------------------------------------------

        feature_dataframe = dataframe.drop(
            columns=[self.target_column],
            errors="ignore",
        )

        # --------------------------------------------------------------------
        # Constant features
        # --------------------------------------------------------------------

        constant_columns = [
            str(column)
            for column in feature_dataframe.columns
            if feature_dataframe[column].nunique(
                dropna=False
            ) <= 1
        ]

        # --------------------------------------------------------------------
        # Numerical and categorical feature detection
        # --------------------------------------------------------------------

        numerical_columns = feature_dataframe.select_dtypes(
            include="number"
        ).columns

        categorical_columns = feature_dataframe.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns

        # --------------------------------------------------------------------
        # Numerical feature statistics
        # --------------------------------------------------------------------

        numerical_summary: Dict[str, Any] = {}

        if len(numerical_columns) > 0:
            describe = feature_dataframe[
                numerical_columns
            ].describe()

            for column in numerical_columns:
                column_summary: Dict[str, Any] = {}

                for statistic in describe.index:
                    value = describe.loc[
                        statistic,
                        column,
                    ]

                    if pd.isna(value):
                        continue

                    # Convert numpy scalar types into native Python values
                    # so that the result can safely be serialized to JSON,
                    # passed through Pydantic, or sent to MLflow later.
                    if hasattr(value, "item"):
                        value = value.item()

                    column_summary[str(statistic)] = value

                numerical_summary[str(column)] = column_summary

        # --------------------------------------------------------------------
        # Categorical feature statistics
        # --------------------------------------------------------------------

        categorical_summary: Dict[str, Any] = {}

        for column in categorical_columns:
            categorical_summary[str(column)] = {
                "unique_values": int(
                    feature_dataframe[column].nunique(
                        dropna=True
                    )
                ),
                "missing_values": int(
                    feature_dataframe[column].isna().sum()
                ),
            }

        return {
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": round(
                duplicate_percentage,
                4,
            ),
            "constant_columns": constant_columns,
            "numerical_columns": [
                str(column)
                for column in numerical_columns
            ],
            "categorical_columns": [
                str(column)
                for column in categorical_columns
            ],
            "numerical_summary": numerical_summary,
            "categorical_summary": categorical_summary,
        }