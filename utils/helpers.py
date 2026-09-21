"""
General-purpose helper functions for NeuroPilots.

This module contains small, reusable utilities that are shared across
multiple components.

Design principles:
    - Keep helpers independent of business logic.
    - Avoid importing workload-specific or LLM-specific modules here.
    - Keep functions deterministic wherever possible.
    - Validate inputs at the boundary.
    - Return predictable Python types.

The goal is to prevent common utility logic from being duplicated across
ingestion, diagnostics, advisor, execution, and Streamlit layers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ============================================================================
# FILESYSTEM HELPERS
# ============================================================================

def ensure_directory(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        Normalized Path object representing the directory.

    Raises
    ------
    ValueError
        If the supplied path is empty.

    Examples
    --------
    >>> ensure_directory("artifacts")
    PosixPath('artifacts')
    """

    if path is None:
        raise ValueError("Directory path cannot be None.")

    directory = Path(path)

    if not str(directory).strip():
        raise ValueError("Directory path cannot be empty.")

    directory.mkdir(parents=True, exist_ok=True)

    return directory


def get_file_extension(path: str | Path) -> str:
    """
    Return a file extension in lowercase without the leading dot.

    Examples
    --------
    >>> get_file_extension("data/experiment.csv")
    'csv'

    >>> get_file_extension("MODEL.JSON")
    'json'
    """

    file_path = Path(path)

    return file_path.suffix.lower().lstrip(".")


def is_supported_file(
    path: str | Path,
    supported_extensions: Iterable[str],
) -> bool:
    """
    Check whether a file has a supported extension.

    Parameters
    ----------
    path:
        File path to inspect.

    supported_extensions:
        Iterable containing supported extensions.

        Both ``"csv"`` and ``".csv"`` are accepted.

    Returns
    -------
    bool
        True when the file extension is supported.
    """

    extension = get_file_extension(path)

    normalized_extensions = {
        str(item).lower().lstrip(".")
        for item in supported_extensions
    }

    return extension in normalized_extensions


# ============================================================================
# JSON HELPERS
# ============================================================================

def load_json_file(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON object from disk.

    This helper is intentionally strict: the root JSON value must be an
    object/dictionary because NeuroPilots uses dictionaries for configuration,
    metadata, and structured artifact information.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    ValueError
        If the JSON is invalid or does not contain an object.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"JSON file does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Expected a JSON file but received: {file_path}"
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in file: {file_path}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object in {file_path}, "
            f"but received {type(data).__name__}."
        )

    return data


def save_json_file(
    data: Dict[str, Any],
    path: str | Path,
) -> Path:
    """
    Save a dictionary as a formatted JSON file.

    Parent directories are created automatically.

    Parameters
    ----------
    data:
        Dictionary to serialize.

    path:
        Destination JSON file.

    Returns
    -------
    Path
        Path to the saved file.
    """

    if not isinstance(data, dict):
        raise TypeError(
            "save_json_file expects data to be a dictionary."
        )

    file_path = Path(path)

    if not str(file_path).strip():
        raise ValueError("Output path cannot be empty.")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )
    except TypeError as exc:
        raise TypeError(
            "Data contains values that cannot be serialized to JSON."
        ) from exc

    return file_path


# ============================================================================
# DATA NORMALIZATION HELPERS
# ============================================================================

def safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Convert a value to a finite float safely.

    NaN and infinite values are rejected because they can cause problems
    when metrics are passed to diagnostics, plotting, or LLM prompts.

    Parameters
    ----------
    value:
        Value to convert.

    default:
        Value returned when conversion fails.

    Returns
    -------
    float | None
        Finite floating-point value or the supplied default.
    """

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def safe_int(
    value: Any,
    default: Optional[int] = None,
) -> Optional[int]:
    """
    Convert a value to an integer safely.

    Returns the supplied default if conversion fails.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_string(
    value: Any,
    default: str = "",
) -> str:
    """
    Normalize a value into a clean string.

    Leading/trailing whitespace is removed.

    Parameters
    ----------
    value:
        Input value.

    default:
        Value returned when the input is None or empty.
    """

    if value is None:
        return default

    normalized = str(value).strip()

    return normalized if normalized else default


# ============================================================================
# METRIC HELPERS
# ============================================================================

def calculate_relative_change(
    baseline: float,
    current: float,
) -> Optional[float]:
    """
    Calculate relative change between two metric values.

    Formula
    -------
    (current - baseline) / abs(baseline)

    Example
    -------
    A metric changing from 0.80 to 0.84 produces:

        (0.84 - 0.80) / 0.80 = 0.05

    which corresponds to a 5% relative improvement.

    Returns None when the baseline is zero because relative change would
    otherwise be undefined.
    """

    baseline_value = safe_float(baseline)
    current_value = safe_float(current)

    if baseline_value is None or current_value is None:
        return None

    if baseline_value == 0:
        return None

    return (
        current_value - baseline_value
    ) / abs(baseline_value)


def calculate_absolute_change(
    baseline: float,
    current: float,
) -> Optional[float]:
    """
    Calculate the absolute change between two metric values.

    Formula
    -------
    current - baseline
    """

    baseline_value = safe_float(baseline)
    current_value = safe_float(current)

    if baseline_value is None or current_value is None:
        return None

    return current_value - baseline_value


def is_improvement(
    baseline: float,
    current: float,
    higher_is_better: bool = True,
) -> Optional[bool]:
    """
    Determine whether a metric improved.

    Parameters
    ----------
    baseline:
        Metric value from the baseline experiment.

    current:
        Metric value from the new experiment.

    higher_is_better:
        Whether larger metric values indicate better performance.

        Examples:
            accuracy -> True
            F1        -> True
            RMSE      -> False
            MAE       -> False
            latency   -> False

    Returns
    -------
    bool | None
        True if improved, False if not improved, or None if the values
        could not be interpreted as finite numbers.
    """

    baseline_value = safe_float(baseline)
    current_value = safe_float(current)

    if baseline_value is None or current_value is None:
        return None

    if higher_is_better:
        return current_value > baseline_value

    return current_value < baseline_value


# ============================================================================
# COLLECTION HELPERS
# ============================================================================

def first_or_none(values: Iterable[Any]) -> Any:
    """
    Return the first item from an iterable.

    Returns None when the iterable is empty.

    This is useful when dealing with optional experiment artifacts where a
    component may or may not have produced a result.
    """

    for value in values:
        return value

    return None


def flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Example
    -------
    Input:

        {
            "model": {
                "architecture": "ConvNeXt-Tiny"
            }
        }

    Output:

        {
            "model.architecture": "ConvNeXt-Tiny"
        }

    This is useful when preparing structured artifact information for
    logging, comparison, or MLflow parameters.
    """

    flattened: Dict[str, Any] = {}

    for key, value in data.items():
        current_key = (
            f"{parent_key}{separator}{key}"
            if parent_key
            else str(key)
        )

        if isinstance(value, dict):
            flattened.update(
                flatten_dict(
                    value,
                    parent_key=current_key,
                    separator=separator,
                )
            )
        else:
            flattened[current_key] = value

    return flattened


def unique_preserve_order(values: Iterable[Any]) -> List[Any]:
    """
    Return unique values while preserving their original order.

    This avoids using ``set`` directly, which would not guarantee the
    intended ordering for downstream display or diagnostics.
    """

    result: List[Any] = []
    seen = set()

    for value in values:
        try:
            marker = value

            if marker in seen:
                continue

            seen.add(marker)
            result.append(value)

        except TypeError:
            # Unhashable values cannot be stored in a set.
            # Fall back to equality-based duplicate detection.
            if value not in result:
                result.append(value)

    return result