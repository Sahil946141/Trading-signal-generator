import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from app.config import config
from app.utils.lstm_features import (
    build_feature_series,
    normalize_sequence,
    required_close_points,
)

logger = logging.getLogger(__name__)

LABEL_MAP = {"SELL": 0, "HOLD": 1, "BUY": 2}
LABEL_NAMES = {value: key for key, value in LABEL_MAP.items()}
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
FEATURE_COLUMNS = ["close", "return", "log_return", "zscore_20"]


def _get_yfinance():
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "yfinance is required for dataset preparation. Install it with `pip install yfinance`."
        ) from exc
    cache_dir = Path(config.DATA_DIR) / "yfinance-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(os.fspath(cache_dir))
    return yf


def resolve_symbols(symbol: Optional[str] = None, symbols: Optional[Iterable[str] | str] = None) -> list[str]:
    if symbols:
        raw_symbols = symbols.split(",") if isinstance(symbols, str) else list(symbols)
    elif symbol:
        raw_symbols = [symbol]
    else:
        raw_symbols = [config.DEFAULT_STOCK_SYMBOL]

    cleaned = []
    seen = set()
    for item in raw_symbols:
        value = str(item).strip().upper()
        if not value or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)

    if not cleaned:
        raise ValueError("At least one stock symbol is required.")

    return cleaned


def download_stock_history(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """Download stock OHLCV history using NSE scraping for daily data."""
    if timeframe == "1d" and requests is not None:
        try:
            return _download_stock_history_from_nse(symbol, start_date, end_date)
        except Exception as exc:
            logger.warning(f"NSE scraping failed for {symbol}: {exc}. No fallback available.")
            raise RuntimeError(f"Failed to download stock history for {symbol}. NSE scraping failed: {exc}") from exc
    else:
        raise ValueError(f"Timeframe {timeframe} not supported. Only '1d' is supported via NSE scraping.")

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    history = history.reset_index().copy()
    date_column = history.columns[0]
    history.rename(columns={date_column: "Date"}, inplace=True)

    if "Adj Close" in history.columns and history["Adj Close"].notna().any():
        history["Close"] = history["Adj Close"]

    for column in OHLCV_COLUMNS:
        if column not in history.columns:
            history[column] = 0.0 if column == "Volume" else np.nan

    history = history[["Date", *OHLCV_COLUMNS]].copy()
    history["Date"] = pd.to_datetime(history["Date"])
    for column in OHLCV_COLUMNS:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.dropna(subset=["Close"]).reset_index(drop=True)
    history["Volume"] = history["Volume"].fillna(0.0)
    history["Symbol"] = symbol.upper()
    return history


def _download_stock_history_from_nse(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if requests is None:
        raise RuntimeError("NSE fallback requires the requests package. Install it with `pip install requests`.")

    base_symbol = symbol.split(".")[0].upper()
    end_date_obj = datetime.now().date() if end_date is None else pd.to_datetime(end_date).date()
    start_date_obj = pd.to_datetime(start_date).date()
    if end_date_obj < start_date_obj:
        raise ValueError("end_date must be after start_date")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        " (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/",
        "Origin": "https://www.nseindia.com",
    })
    session.get("https://www.nseindia.com", timeout=config.PKNSE_REQUEST_TIMEOUT)

    from_date = start_date_obj.strftime("%d-%m-%Y")
    to_date = end_date_obj.strftime("%d-%m-%Y")
    url = (
        f"https://www.nseindia.com/api/historical/cm/equity?symbol={base_symbol}"
        f"&series=[%22EQ%22]&fromDate={from_date}&toDate={to_date}"
    )
    response = session.get(url, timeout=config.PKNSE_REQUEST_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(
            f"NSE historical data request failed with status {response.status_code}"
        )

    payload = response.json()
    data = payload.get("data", [])
    if not data:
        raise RuntimeError("NSE historical data response contained no records.")

    df = pd.DataFrame(data)
    if "DATE" in df.columns:
        df["Date"] = pd.to_datetime(df["DATE"], format="%d-%b-%Y", errors="coerce")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
    else:
        raise RuntimeError("Unexpected NSE response fields.")

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in df.columns:
            if column == "Volume" and "tradedQuantity" in df.columns:
                df["Volume"] = df["tradedQuantity"]
            else:
                raise RuntimeError(f"Missing expected column {column} in NSE response.")

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Symbol"] = symbol.upper()
    return df


def build_feature_frame(df: pd.DataFrame, feature_column: str = None) -> pd.DataFrame:
    """Create return-based features for the stock LSTM."""
    feature_column = feature_column or config.LSTM_FEATURE_COLUMN
    frame = df.copy()

    if "Symbol" not in frame.columns:
        frame["Symbol"] = config.DEFAULT_STOCK_SYMBOL

    frame["close"] = frame["Close"]
    frame["return"] = build_feature_series(frame["Close"], feature_column="return")
    frame["log_return"] = build_feature_series(frame["Close"], feature_column="log_return")
    frame["zscore_20"] = build_feature_series(frame["Close"], feature_column="zscore_20")

    if feature_column not in frame.columns:
        raise ValueError(f"Unsupported feature column '{feature_column}'.")

    frame = frame.dropna(subset=[feature_column]).reset_index(drop=True)
    return frame


def generate_forward_labels(
    df: pd.DataFrame,
    lookahead: int = None,
    threshold: float = None,
) -> pd.DataFrame:
    """Assign SELL/HOLD/BUY labels using the close price `lookahead` candles ahead."""
    lookahead = lookahead or config.LABEL_LOOKAHEAD
    threshold = threshold if threshold is not None else config.LABEL_THRESHOLD

    labeled = df.copy()
    labeled["future_close"] = labeled["Close"].shift(-lookahead)
    labeled["future_return"] = (labeled["future_close"] / labeled["Close"]) - 1

    labeled["Label"] = np.select(
        [
            labeled["future_return"] <= -threshold,
            labeled["future_return"].between(-threshold, threshold, inclusive="neither")
            | labeled["future_return"].eq(0),
            labeled["future_return"] >= threshold,
        ],
        ["SELL", "HOLD", "BUY"],
        default=None,
    )

    labeled = labeled.dropna(subset=["future_close", "Label"]).reset_index(drop=True)
    labeled["LabelEncoded"] = labeled["Label"].map(LABEL_MAP)
    return labeled


def build_raw_sequences(
    df: pd.DataFrame,
    sequence_length: int = None,
    feature_column: str = None,
    normalization: str = None,
):
    """Build raw return sequences and aligned labels."""
    sequence_length = sequence_length or config.LSTM_SEQUENCE_LENGTH
    feature_column = feature_column or config.LSTM_FEATURE_COLUMN
    normalization = normalization or config.LSTM_SEQUENCE_NORMALIZATION

    frame = df.copy()
    if feature_column not in frame.columns:
        frame = build_feature_frame(frame, feature_column=feature_column)
    if "LabelEncoded" not in frame.columns:
        raise ValueError("LabelEncoded column is required before creating sequences.")

    feature_values = frame[[feature_column]].astype(np.float32).values
    labels = frame["LabelEncoded"].astype(np.int64).values
    dates = pd.to_datetime(frame["Date"])

    sequences = []
    targets = []
    target_dates = []

    for index in range(len(frame) - sequence_length):
        raw_sequence = feature_values[index:index + sequence_length]
        sequences.append(normalize_sequence(raw_sequence, normalization=normalization))
        targets.append(labels[index + sequence_length])
        target_dates.append(dates.iloc[index + sequence_length].isoformat())

    if not sequences:
        raise ValueError("Not enough history to build any stock LSTM sequences.")

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.int64), target_dates


def prepare_lstm_dataset(
    output_dir: str,
    symbol: str = None,
    symbols: Optional[Iterable[str] | str] = None,
    start_date: str = None,
    end_date: Optional[str] = None,
    timeframe: str = None,
    sequence_length: int = None,
    lookahead: int = None,
    threshold: float = None,
    train_split: float = None,
    feature_column: str = None,
    normalization: str = None,
) -> Dict[str, object]:
    """Download, label, split, and save a stock LSTM dataset for one or more symbols."""
    summary, arrays, labeled_df = _prepare_symbol_datasets(
        symbol=symbol,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        sequence_length=sequence_length,
        lookahead=lookahead,
        threshold=threshold,
        train_split=train_split,
        feature_column=feature_column,
        normalization=normalization,
    )

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "X_train.npy", arrays["X_train"])
    np.save(target_dir / "y_train.npy", arrays["y_train"])
    np.save(target_dir / "X_val.npy", arrays["X_val"])
    np.save(target_dir / "y_val.npy", arrays["y_val"])
    labeled_df.to_csv(target_dir / "stocks_labeled.csv", index=False)

    summary["output_dir"] = str(target_dir)
    summary["model_path"] = config.LSTM_MODEL_PATH

    with open(target_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Prepared stock LSTM dataset at %s", target_dir)
    return summary


def preview_lstm_dataset(
    symbol: str = None,
    symbols: Optional[Iterable[str] | str] = None,
    start_date: str = None,
    end_date: Optional[str] = None,
    timeframe: str = None,
    sequence_length: int = None,
    lookahead: int = None,
    threshold: float = None,
    train_split: float = None,
    feature_column: str = None,
    normalization: str = None,
) -> Dict[str, object]:
    summary, _, _ = _prepare_symbol_datasets(
        symbol=symbol,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        sequence_length=sequence_length,
        lookahead=lookahead,
        threshold=threshold,
        train_split=train_split,
        feature_column=feature_column,
        normalization=normalization,
    )
    return summary


def load_prepared_dataset(dataset_dir: str):
    """Load persisted numpy arrays for stock LSTM training."""
    base_path = Path(dataset_dir)
    return (
        np.load(base_path / "X_train.npy"),
        np.load(base_path / "y_train.npy"),
    ), (
        np.load(base_path / "X_val.npy"),
        np.load(base_path / "y_val.npy"),
    )


def _prepare_symbol_datasets(
    symbol: str = None,
    symbols: Optional[Iterable[str] | str] = None,
    start_date: str = None,
    end_date: Optional[str] = None,
    timeframe: str = None,
    sequence_length: int = None,
    lookahead: int = None,
    threshold: float = None,
    train_split: float = None,
    feature_column: str = None,
    normalization: str = None,
):
    start_date = start_date or config.STOCK_TRAINING_START
    timeframe = timeframe or config.DEFAULT_TRAINING_TIMEFRAME
    sequence_length = sequence_length or config.LSTM_SEQUENCE_LENGTH
    lookahead = lookahead or config.LABEL_LOOKAHEAD
    threshold = threshold if threshold is not None else config.LABEL_THRESHOLD
    train_split = train_split or config.TRAIN_SPLIT
    feature_column = feature_column or config.LSTM_FEATURE_COLUMN
    normalization = normalization or config.LSTM_SEQUENCE_NORMALIZATION
    target_symbols = resolve_symbols(symbol=symbol, symbols=symbols)

    train_sequences = []
    train_labels = []
    val_sequences = []
    val_labels = []
    train_dates = []
    val_dates = []
    labeled_frames = []
    symbol_breakdown = {}

    for current_symbol in target_symbols:
        history = download_stock_history(
            symbol=current_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
        )
        feature_frame = build_feature_frame(history, feature_column=feature_column)
        labeled = generate_forward_labels(feature_frame, lookahead=lookahead, threshold=threshold)
        sequences, labels, dates = build_raw_sequences(
            labeled,
            sequence_length=sequence_length,
            feature_column=feature_column,
            normalization=normalization,
        )

        split_index = int(len(sequences) * train_split)
        if split_index <= 0 or split_index >= len(sequences):
            raise ValueError(f"Train split produced an empty train or validation set for {current_symbol}.")

        train_sequences.append(sequences[:split_index])
        train_labels.append(labels[:split_index])
        val_sequences.append(sequences[split_index:])
        val_labels.append(labels[split_index:])
        train_dates.extend(dates[:split_index])
        val_dates.extend(dates[split_index:])
        labeled_frames.append(labeled)

        symbol_breakdown[current_symbol] = {
            "total_rows": int(len(labeled)),
            "train_sequences": int(len(sequences[:split_index])),
            "validation_sequences": int(len(sequences[split_index:])),
            "train_range": {
                "from": dates[0] if split_index > 0 else None,
                "to": dates[split_index - 1] if split_index > 0 else None,
            },
            "validation_range": {
                "from": dates[split_index] if split_index < len(dates) else None,
                "to": dates[-1] if dates else None,
            },
            "class_distribution": {
                "overall": labeled["Label"].value_counts().to_dict(),
                "train": _class_distribution(labels[:split_index]),
                "validation": _class_distribution(labels[split_index:]),
            },
        }

    arrays = {
        "X_train": np.concatenate(train_sequences, axis=0).astype(np.float32),
        "y_train": np.concatenate(train_labels, axis=0).astype(np.int64),
        "X_val": np.concatenate(val_sequences, axis=0).astype(np.float32),
        "y_val": np.concatenate(val_labels, axis=0).astype(np.int64),
    }
    labeled_df = pd.concat(labeled_frames, ignore_index=True)

    summary = {
        "symbols": target_symbols,
        "timeframe": timeframe,
        "feature_column": feature_column,
        "sequence_normalization": normalization,
        "start_date": start_date,
        "end_date": end_date,
        "sequence_length": sequence_length,
        "lookahead": lookahead,
        "threshold_pct": round(threshold * 100, 2),
        "train_split": train_split,
        "total_labeled_rows": int(len(labeled_df)),
        "train_sequences": int(len(arrays["y_train"])),
        "validation_sequences": int(len(arrays["y_val"])),
        "train_range": {
            "from": train_dates[0] if train_dates else None,
            "to": train_dates[-1] if train_dates else None,
        },
        "validation_range": {
            "from": val_dates[0] if val_dates else None,
            "to": val_dates[-1] if val_dates else None,
        },
        "class_distribution": {
            "train": _class_distribution(arrays["y_train"]),
            "validation": _class_distribution(arrays["y_val"]),
            "overall": labeled_df["Label"].value_counts().to_dict(),
        },
        "symbol_breakdown": symbol_breakdown,
    }

    return summary, arrays, labeled_df


def _class_distribution(labels: np.ndarray) -> Dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    distribution = {LABEL_NAMES[int(label)]: int(count) for label, count in zip(unique, counts)}
    for label_name in LABEL_MAP:
        distribution.setdefault(label_name, 0)
    return distribution
