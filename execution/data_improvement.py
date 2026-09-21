"""
Data improvement and transformation engine for NeuroPilots.

This module provides execution-time data transformations that can be used
when an ExperimentRecommendation requires a change to the training data
pipeline.

Design principles:
    - Recommendations come from the AI Research Advisor.
    - This module executes supported data-side transformations.
    - It does not decide which experiment should be recommended.
    - Transformations are deterministic where possible.
    - Original input data is never modified in-place.
    - Every transformation produces an auditable result.

The module currently supports generic tabular transformations and a
configuration-based image augmentation representation. Actual model-specific
augmentation execution can be connected later when the corresponding
workload/model training pipeline is implemented.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger


LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Supported transformation names
# ---------------------------------------------------------------------------

TRANSFORM_NORMALIZE_NUMERIC = "normalize_numeric"
TRANSFORM_FILL_MISSING = "fill_missing"
TRANSFORM_DROP_DUPLICATES = "drop_duplicates"
TRANSFORM_IMAGE_AUGMENTATION = "image_augmentation"

SUPPORTED_TRANSFORMATIONS = (
    TRANSFORM_NORMALIZE_NUMERIC,
    TRANSFORM_FILL_MISSING,
    TRANSFORM_DROP_DUPLICATES,
    TRANSFORM_IMAGE_AUGMENTATION,
)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class DataImprovementResult:
    """
    Result of a data improvement operation.

    Attributes:
        success:
            Whether the transformation completed successfully.

        transformation:
            Name of the transformation that was applied.

        data:
            Transformed data when the transformation operates directly on
            tabular data.

        configuration:
            Configuration describing the transformation.

        changes:
            Human-readable description of modifications.

        rows_before:
            Number of rows before transformation when applicable.

        rows_after:
            Number of rows after transformation when applicable.

        columns_before:
            Number of columns before transformation when applicable.

        columns_after:
            Number of columns after transformation when applicable.

        metadata:
            Additional execution information.

        error:
            Error message if the transformation failed.
    """

    success: bool
    transformation: str
    data: Optional[pd.DataFrame] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    changes: List[str] = field(default_factory=list)
    rows_before: Optional[int] = None
    rows_after: Optional[int] = None
    columns_before: Optional[int] = None
    columns_after: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Data Improvement Engine
# ---------------------------------------------------------------------------


class DataImprovementEngine:
    """
    Apply supported data-side experiment transformations.

    The engine intentionally operates on explicit transformation names.
    Higher-level components such as ExperimentExecutor are responsible for
    translating a structured ExperimentRecommendation into an executable
    transformation.

    Args:
        copy_input:
            Whether to copy input DataFrames before modification.

    Example:
        engine = DataImprovementEngine()

        result = engine.apply(
            data,
            TRANSFORM_FILL_MISSING,
        )
    """

    def __init__(self, copy_input: bool = True) -> None:
        self.copy_input = copy_input

        self._transformations: Dict[
            str,
            Callable[..., DataImprovementResult],
        ] = {
            TRANSFORM_NORMALIZE_NUMERIC: self._normalize_numeric,
            TRANSFORM_FILL_MISSING: self._fill_missing,
            TRANSFORM_DROP_DUPLICATES: self._drop_duplicates,
            TRANSFORM_IMAGE_AUGMENTATION: self._configure_image_augmentation,
        }

        LOGGER.info(
            "DataImprovementEngine initialized with %d transformations.",
            len(self._transformations),
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def apply(
        self,
        data: Optional[pd.DataFrame],
        transformation: str,
        **kwargs: Any,
    ) -> DataImprovementResult:
        """
        Apply a supported data transformation.

        Args:
            data:
                Input DataFrame. Some configuration-only transformations may
                accept None.

            transformation:
                Name of the transformation to execute.

            **kwargs:
                Transformation-specific parameters.

        Returns:
            DataImprovementResult.

        Raises:
            ValueError:
                If the transformation is unsupported.
            TypeError:
                If tabular data is required but not provided.
        """
        normalized_transformation = self._normalize_transformation_name(
            transformation
        )

        if normalized_transformation not in self._transformations:
            raise ValueError(
                f"Unsupported transformation '{transformation}'. "
                f"Supported transformations: {SUPPORTED_TRANSFORMATIONS}"
            )

        handler = self._transformations[normalized_transformation]

        LOGGER.info(
            "Applying data transformation: %s",
            normalized_transformation,
        )

        try:
            return handler(data, **kwargs)
        except Exception as exc:
            LOGGER.exception(
                "Data transformation failed: %s",
                normalized_transformation,
            )

            return DataImprovementResult(
                success=False,
                transformation=normalized_transformation,
                error=str(exc),
            )

    def supports(self, transformation: str) -> bool:
        """
        Return whether a transformation is supported.
        """
        normalized = self._normalize_transformation_name(transformation)
        return normalized in self._transformations

    def supported_transformations(self) -> List[str]:
        """
        Return all supported transformation names.
        """
        return list(self._transformations.keys())

    # -----------------------------------------------------------------------
    # Numeric normalization
    # -----------------------------------------------------------------------

    def _normalize_numeric(
        self,
        data: Optional[pd.DataFrame],
        method: str = "standard",
        columns: Optional[List[str]] = None,
        **_: Any,
    ) -> DataImprovementResult:
        """
        Normalize numeric columns.

        Supported methods:
            standard:
                z-score normalization.

            minmax:
                scale values to [0, 1].

        The original DataFrame is not modified.
        """
        frame = self._prepare_dataframe(data)

        numeric_columns = frame.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if columns is not None:
            unknown_columns = [
                column for column in columns
                if column not in frame.columns
            ]

            if unknown_columns:
                raise ValueError(
                    f"Unknown columns requested for normalization: "
                    f"{unknown_columns}"
                )

            numeric_columns = [
                column
                for column in columns
                if pd.api.types.is_numeric_dtype(frame[column])
            ]

        if not numeric_columns:
            return DataImprovementResult(
                success=False,
                transformation=TRANSFORM_NORMALIZE_NUMERIC,
                data=frame,
                rows_before=len(frame),
                rows_after=len(frame),
                columns_before=len(frame.columns),
                columns_after=len(frame.columns),
                error="No numeric columns available for normalization.",
            )

        if method not in {"standard", "minmax"}:
            raise ValueError(
                "Normalization method must be 'standard' or 'minmax'."
            )

        for column in numeric_columns:
            values = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            if method == "standard":
                mean = values.mean()
                std = values.std()

                if pd.isna(std) or std == 0:
                    frame[column] = values - mean
                else:
                    frame[column] = (values - mean) / std

            else:
                minimum = values.min()
                maximum = values.max()

                if pd.isna(minimum) or pd.isna(maximum):
                    continue

                difference = maximum - minimum

                if difference == 0:
                    frame[column] = 0.0
                else:
                    frame[column] = (
                        values - minimum
                    ) / difference

        return DataImprovementResult(
            success=True,
            transformation=TRANSFORM_NORMALIZE_NUMERIC,
            data=frame,
            rows_before=len(data),
            rows_after=len(frame),
            columns_before=len(data.columns),
            columns_after=len(frame.columns),
            changes=[
                f"Normalized {len(numeric_columns)} numeric column(s).",
                f"Normalization method: {method}.",
            ],
            configuration={
                "method": method,
                "columns": numeric_columns,
            },
        )

    # -----------------------------------------------------------------------
    # Missing value handling
    # -----------------------------------------------------------------------

    def _fill_missing(
        self,
        data: Optional[pd.DataFrame],
        strategy: str = "median",
        columns: Optional[List[str]] = None,
        fill_value: Any = None,
        **_: Any,
    ) -> DataImprovementResult:
        """
        Fill missing values without modifying the original DataFrame.

        Numeric columns support:
            - mean
            - median
            - zero

        Categorical/object columns support:
            - mode
            - constant
        """
        frame = self._prepare_dataframe(data)

        selected_columns = (
            columns
            if columns is not None
            else frame.columns.tolist()
        )

        unknown_columns = [
            column
            for column in selected_columns
            if column not in frame.columns
        ]

        if unknown_columns:
            raise ValueError(
                f"Unknown columns requested for missing-value handling: "
                f"{unknown_columns}"
            )

        changes: List[str] = []
        filled_columns: List[str] = []

        for column in selected_columns:
            missing_count = int(frame[column].isna().sum())

            if missing_count == 0:
                continue

            series = frame[column]

            if pd.api.types.is_numeric_dtype(series):
                if strategy == "mean":
                    replacement = series.mean()
                elif strategy == "median":
                    replacement = series.median()
                elif strategy == "zero":
                    replacement = 0
                elif strategy == "constant":
                    if fill_value is None:
                        raise ValueError(
                            "fill_value is required when strategy='constant'."
                        )
                    replacement = fill_value
                else:
                    raise ValueError(
                        "Numeric missing-value strategy must be "
                        "'mean', 'median', 'zero', or 'constant'."
                    )
            else:
                if strategy == "mode":
                    modes = series.mode(dropna=True)

                    if modes.empty:
                        replacement = "unknown"
                    else:
                        replacement = modes.iloc[0]

                elif strategy == "constant":
                    replacement = (
                        "unknown"
                        if fill_value is None
                        else fill_value
                    )

                else:
                    raise ValueError(
                        "Categorical missing-value strategy must be "
                        "'mode' or 'constant'."
                    )

            frame[column] = series.fillna(replacement)

            filled_columns.append(column)

            changes.append(
                f"Filled {missing_count} missing value(s) in '{column}'."
            )

        if not filled_columns:
            changes.append("No missing values required modification.")

        return DataImprovementResult(
            success=True,
            transformation=TRANSFORM_FILL_MISSING,
            data=frame,
            rows_before=len(data),
            rows_after=len(frame),
            columns_before=len(data.columns),
            columns_after=len(frame.columns),
            changes=changes,
            configuration={
                "strategy": strategy,
                "columns": selected_columns,
                "fill_value": fill_value,
            },
        )

    # -----------------------------------------------------------------------
    # Duplicate removal
    # -----------------------------------------------------------------------

    def _drop_duplicates(
        self,
        data: Optional[pd.DataFrame],
        subset: Optional[List[str]] = None,
        keep: str = "first",
        **_: Any,
    ) -> DataImprovementResult:
        """
        Remove duplicate rows.
        """
        frame = self._prepare_dataframe(data)

        rows_before = len(frame)

        frame = frame.drop_duplicates(
            subset=subset,
            keep=keep,
        ).reset_index(drop=True)

        rows_removed = rows_before - len(frame)

        return DataImprovementResult(
            success=True,
            transformation=TRANSFORM_DROP_DUPLICATES,
            data=frame,
            rows_before=rows_before,
            rows_after=len(frame),
            columns_before=len(data.columns),
            columns_after=len(frame.columns),
            changes=[
                f"Removed {rows_removed} duplicate row(s)."
            ],
            configuration={
                "subset": subset,
                "keep": keep,
            },
        )

    # -----------------------------------------------------------------------
    # Image augmentation configuration
    # -----------------------------------------------------------------------

    def _configure_image_augmentation(
        self,
        data: Optional[pd.DataFrame],
        horizontal_flip_probability: float = 0.5,
        rotation_degrees: float = 15.0,
        crop_scale_min: float = 0.8,
        crop_scale_max: float = 1.0,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        **_: Any,
    ) -> DataImprovementResult:
        """
        Build an executable image-augmentation configuration.

        This method intentionally does not depend on a specific deep-learning
        framework yet. The configuration can later be consumed by the CV
        training pipeline, for example by a PyTorch/Torchvision data loader.

        This is important because the current NeuroPilots architecture
        separates experiment recommendation from workload-specific execution.
        """
        probabilities = {
            "horizontal_flip_probability": horizontal_flip_probability,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
        }

        for name, value in probabilities.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{name} must be between 0.0 and 1.0."
                )

        if rotation_degrees < 0:
            raise ValueError(
                "rotation_degrees cannot be negative."
            )

        if not 0.0 < crop_scale_min <= crop_scale_max <= 1.0:
            raise ValueError(
                "Crop scale values must satisfy "
                "0 < crop_scale_min <= crop_scale_max <= 1."
            )

        configuration = {
            "horizontal_flip": {
                "enabled": horizontal_flip_probability > 0,
                "probability": horizontal_flip_probability,
            },
            "rotation": {
                "enabled": rotation_degrees > 0,
                "degrees": rotation_degrees,
            },
            "random_resized_crop": {
                "enabled": True,
                "scale_min": crop_scale_min,
                "scale_max": crop_scale_max,
            },
            "color_jitter": {
                "enabled": True,
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "hue": hue,
            },
        }

        return DataImprovementResult(
            success=True,
            transformation=TRANSFORM_IMAGE_AUGMENTATION,
            data=copy.deepcopy(data) if data is not None else None,
            configuration=configuration,
            changes=[
                "Created an image augmentation configuration.",
                "Configuration is ready for the computer-vision training pipeline.",
            ],
            rows_before=len(data) if data is not None else None,
            rows_after=len(data) if data is not None else None,
            columns_before=(
                len(data.columns)
                if data is not None
                else None
            ),
            columns_after=(
                len(data.columns)
                if data is not None
                else None
            ),
        )

    # -----------------------------------------------------------------------
    # Validation / preparation
    # -----------------------------------------------------------------------

    def _prepare_dataframe(
        self,
        data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Validate and optionally copy a DataFrame.
        """
        if data is None:
            raise TypeError(
                "This transformation requires a pandas DataFrame."
            )

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if self.copy_input:
            return data.copy(deep=True)

        return data

    @staticmethod
    def _normalize_transformation_name(
        transformation: str,
    ) -> str:
        """
        Normalize transformation names for safe lookup.
        """
        if not isinstance(transformation, str):
            raise TypeError(
                "transformation must be a string."
            )

        return transformation.strip().lower().replace("-", "_").replace(
            " ",
            "_",
        )

    # -----------------------------------------------------------------------
    # Recommendation-oriented helpers
    # -----------------------------------------------------------------------

    def infer_transformation(
        self,
        recommendation_text: str,
    ) -> Optional[str]:
        """
        Infer a supported transformation category from recommendation text.

        This helper is deliberately conservative.

        It does NOT map a root cause to a recommendation. It only recognizes
        explicit transformation terminology that may already be present in a
        generated recommendation.

        If no reliable transformation is detected, None is returned so that
        the ExperimentExecutor can request/require explicit execution
        configuration instead of guessing.
        """
        if not recommendation_text:
            return None

        text = recommendation_text.lower()

        keyword_groups = {
            TRANSFORM_IMAGE_AUGMENTATION: (
                "image augmentation",
                "image-augmentation",
                "data augmentation",
                "augmentation pipeline",
            ),
            TRANSFORM_NORMALIZE_NUMERIC: (
                "normalize numeric",
                "numeric normalization",
                "standardize numeric",
                "standard scaling",
            ),
            TRANSFORM_FILL_MISSING: (
                "fill missing",
                "missing values",
                "impute missing",
                "missing-value imputation",
            ),
            TRANSFORM_DROP_DUPLICATES: (
                "drop duplicates",
                "remove duplicates",
                "duplicate rows",
            ),
        }

        for transformation, keywords in keyword_groups.items():
            if any(keyword in text for keyword in keywords):
                return transformation

        return None

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "DataImprovementEngine("
            f"supported_transformations={len(self._transformations)}, "
            f"copy_input={self.copy_input}"
            ")"
        )