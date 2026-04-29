from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


SUPPORTED_FEATURE_COLUMNS = {"close", "return", "log_return", "zscore_20"}
SUPPORTED_NORMALIZATIONS = {"none", "zscore", "minmax"}


def required_close_points(sequence_length: int, feature_column: str) -> int:
    feature_column = feature_column.strip().lower()
    if feature_column == "close":
        return sequence_length
    if feature_column in {"return", "log_return"}:
        return sequence_length + 1
    if feature_column == "zscore_20":
        return sequence_length + 19
    raise ValueError(
        f"Unsupported feature column '{feature_column}'. "
        f"Supported values: {sorted(SUPPORTED_FEATURE_COLUMNS)}"
    )


def build_feature_series(
    closes: Iterable[float] | pd.Series,
    feature_column: str,
) -> pd.Series:
    feature_column = feature_column.strip().lower()
    close_series = pd.Series(closes, dtype=np.float32)

    if feature_column == "close":
        return close_series.dropna().astype(np.float32)
    if feature_column == "return":
        return close_series.pct_change().dropna().astype(np.float32)
    if feature_column == "log_return":
        return np.log(close_series / close_series.shift(1)).dropna().astype(np.float32)
    if feature_column == "zscore_20":
        rolling_mean = close_series.rolling(window=20).mean()
        rolling_std = close_series.rolling(window=20).std().replace(0, np.nan)
        return ((close_series - rolling_mean) / rolling_std).dropna().astype(np.float32)

    raise ValueError(
        f"Unsupported feature column '{feature_column}'. "
        f"Supported values: {sorted(SUPPORTED_FEATURE_COLUMNS)}"
    )


def normalize_sequence(values: np.ndarray, normalization: str) -> np.ndarray:
    normalization = normalization.strip().lower()
    sequence = np.asarray(values, dtype=np.float32)

    if normalization == "none":
        return np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if normalization == "zscore":
        mean = float(sequence.mean())
        std = float(sequence.std())
        if std < 1e-8:
            return np.zeros_like(sequence, dtype=np.float32)
        normalized = (sequence - mean) / std
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if normalization == "minmax":
        min_value = float(sequence.min())
        max_value = float(sequence.max())
        scale = max_value - min_value
        if scale < 1e-8:
            return np.zeros_like(sequence, dtype=np.float32)
        normalized = (sequence - min_value) / scale
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    raise ValueError(
        f"Unsupported sequence normalization '{normalization}'. "
        f"Supported values: {sorted(SUPPORTED_NORMALIZATIONS)}"
    )


def build_feature_window(
    closes: Iterable[float] | pd.Series,
    sequence_length: int,
    feature_column: str,
    normalization: str,
) -> np.ndarray:
    feature_series = build_feature_series(closes, feature_column=feature_column)
    if len(feature_series) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} feature rows to build an LSTM window, "
            f"but only {len(feature_series)} are available."
        )

    window = feature_series.tail(sequence_length).to_numpy(dtype=np.float32)
    normalized_window = normalize_sequence(window, normalization=normalization)
    return normalized_window.reshape(sequence_length, 1).astype(np.float32)
