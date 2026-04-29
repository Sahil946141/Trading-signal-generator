import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.config import config

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)


def _get_yfinance():
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "yfinance is required for stock market data. Install it with `pip install yfinance`."
        ) from exc
    cache_dir = Path(config.DATA_DIR) / "yfinance-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(os.fspath(cache_dir))
    return yf


class StockDataService:
    """Fetch recent stock OHLCV data from yfinance and prepare stock-LSTM features."""

    TIMEFRAME_CONFIG = {
        "1m": {"interval": "1m", "period": "7d"},
        "5m": {"interval": "5m", "period": "60d"},
        "15m": {"interval": "15m", "period": "60d"},
        "1h": {"interval": "60m", "period": "730d"},
        "1d": {"interval": "1d", "period": "max"},
    }
    IST = ZoneInfo("Asia/Kolkata")

    def __init__(self):
        self.last_fetch_source = "yfinance"
        self._cache_dir = Path(config.DATA_DIR) / "ohlcv-cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 60,
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[str]]:
        if timeframe not in config.VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. Valid values: {config.VALID_TIMEFRAMES}"
            )

        return await asyncio.to_thread(self._fetch_ohlcv_sync, symbol, timeframe, limit)

    def _fetch_ohlcv_sync(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[str]]:
        cache_path = self._cache_dir / self._cache_filename(symbol, timeframe, limit)
        
        # Try NSE scraping first
        if requests is not None:
            try:
                if timeframe == "1d":
                    result = self._fetch_nse_ohlcv(symbol, limit)
                else:
                    result = self._fetch_nse_intraday(symbol, timeframe, limit)
                # Save to cache
                self._save_ohlcv_cache(cache_path, *result)
                return result
            except Exception as exc:
                logger.warning(
                    "NSE scraping failed for %s %s: %s. Attempting fallback.",
                    symbol,
                    timeframe,
                    exc,
                )
                # For intraday, try yfinance as fallback
                if timeframe != "1d":
                    try:
                        result = self._fetch_yfinance_ohlcv(symbol, timeframe, limit)
                        self._save_ohlcv_cache(cache_path, *result)
                        return result
                    except Exception as yf_exc:
                        logger.warning(
                            "YFinance fallback failed for %s %s: %s",
                            symbol,
                            timeframe,
                            yf_exc,
                        )
        
        # Fallback to cache
        cached = self._load_ohlcv_cache(cache_path)
        if cached is not None:
            self.last_fetch_source = "ohlcv_cache"
            return cached
        
        raise RuntimeError(f"Failed to fetch stock data for {symbol} on {timeframe}. No data sources available and no cache found.")

    def _download_history_with_retry(self, ticker, cfg, retries: int = 3, backoff: int = 2):
        attempt = 0
        while True:
            try:
                return ticker.history(
                    period=cfg["period"],
                    interval=cfg["interval"],
                    auto_adjust=False,
                    actions=False,
                )
            except Exception as exc:
                attempt += 1
                message = str(exc).lower()
                if attempt >= retries or "too many requests" not in message:
                    raise
                wait = backoff * attempt
                logger.warning(
                    "YFinance rate limit encountered; retrying in %s seconds (attempt %s/%s)",
                    wait,
                    attempt,
                    retries,
                )
                time.sleep(wait)

    def _cache_filename(self, symbol: str, timeframe: str, limit: int) -> str:
        safe_symbol = symbol.strip().upper().replace("/", "_")
        return f"{safe_symbol}_{timeframe}_{limit}.json"

    def _save_ohlcv_cache(
        self,
        cache_path: Path,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        times: List[str],
    ):
        data = {
            "opens": opens,
            "highs": highs,
            "lows": lows,
            "closes": closes,
            "volumes": volumes,
            "times": times,
        }
        try:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except Exception:
            logger.debug("Unable to write OHLCV cache to %s", cache_path)

    def _load_ohlcv_cache(self, cache_path: Path):
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            return (
                cached["opens"],
                cached["highs"],
                cached["lows"],
                cached["closes"],
                cached["volumes"],
                cached["times"],
            )
        except Exception as exc:
            logger.warning("Failed to read cached OHLCV file %s: %s", cache_path, exc)
            return None

    def prepare_lstm_features(self, closes: List[float]) -> np.ndarray:
        """Create a `(sequence_length, 1)` return tensor for the stock LSTM."""
        close_series = pd.Series(closes, dtype=np.float32)
        returns = close_series.pct_change().dropna().astype(np.float32)
        if len(returns) < config.LSTM_SEQUENCE_LENGTH:
            raise ValueError(
                f"Need at least {config.LSTM_SEQUENCE_LENGTH + 1} closes to build a "
                f"{config.LSTM_SEQUENCE_LENGTH}-step return sequence."
            )

        sequence = returns.tail(config.LSTM_SEQUENCE_LENGTH).to_numpy().reshape(-1, 1)
        return np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def trim_candles(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        times: List[str],
        limit: int,
    ):
        return (
            opens[-limit:],
            highs[-limit:],
            lows[-limit:],
            closes[-limit:],
            volumes[-limit:],
            times[-limit:],
        )

    def required_points_for_lstm(self) -> int:
        """Return the number of closing prices required to build the configured LSTM input window."""
        from app.utils.lstm_features import required_close_points

        return required_close_points(
            config.LSTM_SEQUENCE_LENGTH,
            config.LSTM_FEATURE_COLUMN,
        )

    def _symbol_base_and_exchange(self, symbol: str) -> tuple[str, str]:
        value = symbol.strip().upper()
        if "." in value:
            base, exchange = value.split(".", 1)
        else:
            base, exchange = value, "NSE"

        exchange_map = {
            "NS": "NSE",
            "NSE": "NSE",
            "BSE": "BSE",
            "XNSE": "NSE",
            "XBOM": "BSE",
        }
        return base, exchange_map.get(exchange, exchange)

    def _fetch_nse_ohlcv(self, symbol: str, limit: int) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[str]]:
        if requests is None:
            raise RuntimeError(
                "NSE fallback requires the requests package. Install it with `pip install requests`."
            )

        base_symbol, exchange = self._symbol_base_and_exchange(symbol)
        if exchange != "NSE":
            raise RuntimeError("NSE fallback only supports NSE symbols.")

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=max(limit * 3, 30))
        from_date = start_date.strftime("%d-%m-%Y")
        to_date = end_date.strftime("%d-%m-%Y")

        self._prepare_nse_session()
        url = (
            f"https://www.nseindia.com/api/historical/cm/equity?symbol={base_symbol}"
            f"&series=[%22EQ%22]&fromDate={from_date}&toDate={to_date}"
        )
        response = self._nse_session.get(url, timeout=config.PKNSE_REQUEST_TIMEOUT)
        if response.status_code != 200:
            raise RuntimeError(
                f"NSE historical data request failed with status {response.status_code}."
            )

        payload = response.json()
        data = payload.get("data", [])
        if not data:
            raise RuntimeError("NSE historical data response contained no OHLC records.")

        df = pd.DataFrame(data)
        if "DATE" in df.columns:
            df["date"] = pd.to_datetime(df["DATE"], format="%d-%b-%Y", errors="coerce")
        elif "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
        else:
            raise RuntimeError("Unexpected NSE historical response format.")

        required_cols = [col for col in ["open", "high", "low", "close"] if col in df.columns]
        if len(required_cols) != 4:
            raise RuntimeError("Missing required OHLC columns from NSE response.")

        df = df.dropna(subset=["date"]).sort_values("date").tail(limit)
        if len(df) < limit:
            raise RuntimeError(
                f"NSE historical data returned only {len(df)} records; {limit} required."
            )

        opens = df["open"].astype(float).tolist()
        highs = df["high"].astype(float).tolist()
        lows = df["low"].astype(float).tolist()
        closes = df["close"].astype(float).tolist()
        volumes = df.get("tradedQuantity", df.get("tradedQty", df.get("volume", [0] * len(df))))
        if isinstance(volumes, pd.Series):
            volumes = volumes.astype(float).tolist()
        else:
            volumes = [float(value) for value in volumes]
        times = [date.isoformat() for date in df["date"].tolist()]

        self.last_fetch_source = "nse_scrape"
        return opens, highs, lows, closes, volumes, times

    def _fetch_nse_intraday(
        self, symbol: str, timeframe: str, limit: int
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[str]]:
        if requests is None:
            raise RuntimeError("requests is required for NSE intraday scraping.")

        base_symbol, exchange = self._symbol_base_and_exchange(symbol)
        if exchange != "NSE":
            raise RuntimeError("NSE intraday scraping only supports NSE symbols.")

        # Map timeframe to NSE interval
        interval_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "1h": "60",
        }
        if timeframe not in interval_map:
            raise RuntimeError(f"NSE intraday does not support timeframe {timeframe}.")

        interval = interval_map[timeframe]

        self._prepare_nse_session()
        # NSE chart API for intraday data
        url = f"https://www.nseindia.com/api/chart-databyindex?index={base_symbol}&preopen=true"
        response = self._nse_session.get(url, timeout=config.PKNSE_REQUEST_TIMEOUT)
        if response.status_code != 200:
            raise RuntimeError(f"NSE intraday request failed with status {response.status_code}.")

        payload = response.json()
        grapth_data = payload.get("grapthData", [])
        if not grapth_data:
            raise RuntimeError("Intraday data not available. NSE chart API returned no data, likely because the market is not open (opens at 9:15 AM IST), symbol not found, or API changes.")

        # Parse the data: grapthData is list of [timestamp, open, high, low, close, volume?]
        candles = []
        for item in grapth_data:
            if len(item) >= 5:
                timestamp = item[0]  # Unix timestamp in milliseconds
                open_price = float(item[1])
                high = float(item[2])
                low = float(item[3])
                close = float(item[4])
                volume = float(item[5]) if len(item) > 5 else 0
                candles.append({
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                })

        if not candles:
            raise RuntimeError("No valid candles found in NSE intraday response.")

        # Sort by timestamp and take last 'limit' candles
        candles.sort(key=lambda x: x["timestamp"])
        candles = candles[-limit:]

        if len(candles) < limit:
            raise RuntimeError(f"NSE intraday returned only {len(candles)} candles; {limit} required.")

        # Convert to IST and format
        ist = ZoneInfo("Asia/Kolkata")
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        times = []

        for candle in candles:
            dt = datetime.fromtimestamp(candle["timestamp"] / 1000, tz=ist)
            times.append(dt.isoformat())
            opens.append(candle["open"])
            highs.append(candle["high"])
            lows.append(candle["low"])
            closes.append(candle["close"])
            volumes.append(candle["volume"])

        self.last_fetch_source = "nse_intraday"
        return opens, highs, lows, closes, volumes, times

    def _fetch_yfinance_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[str]]:
        yf = _get_yfinance()
        ticker = yf.Ticker(symbol)
        cfg = self.TIMEFRAME_CONFIG[timeframe]
        history = self._download_history_with_retry(ticker, cfg)

        if history.empty:
            raise ValueError(f"No OHLCV history returned for {symbol} on timeframe {timeframe}.")

        if "Adj Close" in history.columns and history["Adj Close"].notna().any():
            history["Close"] = history["Adj Close"]

        frame = history[["Open", "High", "Low", "Close", "Volume"]].copy()
        frame = frame.dropna(subset=["Close"]).tail(limit)

        if len(frame) < limit:
            raise ValueError(
                f"Only {len(frame)} candles are available for {symbol} on {timeframe}; {limit} are required."
            )

        if getattr(frame.index, "tz", None) is None:
            frame.index = frame.index.tz_localize(self.IST)
        else:
            frame.index = frame.index.tz_convert(self.IST)

        self.last_fetch_source = "yfinance"
        opens = frame["Open"].astype(float).round(4).tolist()
        highs = frame["High"].astype(float).round(4).tolist()
        lows = frame["Low"].astype(float).round(4).tolist()
        closes = frame["Close"].astype(float).round(4).tolist()
        volumes = frame["Volume"].fillna(0).astype(float).round(4).tolist()
        times = [timestamp.isoformat() for timestamp in frame.index]

        return opens, highs, lows, closes, volumes, times

    def _prepare_nse_session(self):
        if getattr(self, "_nse_session", None) is not None:
            return
        if requests is None:
            raise RuntimeError("requests is required for NSE fallback.")

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
        self._nse_session = session

    def _twelvedata_interval(self, timeframe: str) -> str:
        mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "60min",
            "1d": "1day",
        }
        return mapping.get(timeframe, timeframe)

    def _alphavantage_interval(self, timeframe: str) -> str:
        mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "60min",
            "1d": "daily",
        }
        return mapping.get(timeframe, timeframe)

    def latest_candle(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        times: List[str],
    ) -> dict:
        return {
            "time": times[-1],
            "open": round(opens[-1], 2),
            "high": round(highs[-1], 2),
            "low": round(lows[-1], 2),
            "close": round(closes[-1], 2),
            "volume": round(volumes[-1], 2),
        }

    def build_candles(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        times: List[str],
    ) -> List[dict]:
        return [
            {
                "time": timestamp,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2),
            }
            for open_price, high_price, low_price, close_price, volume, timestamp in zip(
                opens, highs, lows, closes, volumes, times
            )
        ]

    def summarize_market_data(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
    ) -> dict:
        closes_arr = np.array(closes, dtype=np.float32)
        volumes_arr = np.array(volumes, dtype=np.float32)

        price_change = float(closes_arr[-1] - closes_arr[-2]) if len(closes_arr) > 1 else 0.0
        price_change_pct = (
            float((price_change / closes_arr[-2]) * 100)
            if len(closes_arr) > 1 and closes_arr[-2] != 0
            else 0.0
        )

        return {
            "current_price": round(float(closes_arr[-1]), 2),
            "price_change": round(price_change, 2),
            "price_change_pct": round(price_change_pct, 2),
            "short_term_trend": (
                "Bullish"
                if len(closes_arr) > 5 and closes_arr[-1] > closes_arr[-5]
                else "Bearish"
                if len(closes_arr) > 5 and closes_arr[-1] < closes_arr[-5]
                else "Neutral"
            ),
            "range": round(float(max(highs) - min(lows)), 2),
            "average_volume": round(float(volumes_arr.mean()), 2),
        }


NiftyDataService = StockDataService
