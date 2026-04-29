import aiohttp
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class MarketDataService:
    """
    Fetches OHLCV data from various market data providers.
    """
    
    def __init__(
        self,
        api_key: str,
        api_type: str = "binance",
        base_url: str = None
    ):
        """
        Initialize market data service.
        
        Args:
            api_key: API key for market data provider
            api_type: API type (binance, polygon, alpaca)
            base_url: Override base URL if needed
        """
        self.api_key = api_key
        self.api_type = api_type
        self.base_url = base_url or self._get_default_url(api_type)
        logger.info(f"Market data service initialized with {api_type}")
    
    def _get_default_url(self, api_type: str) -> str:
        """Get default base URL for API type."""
        urls = {
            "binance": "https://api.binance.com/api/v3",
            "polygon": "https://api.polygon.io/v1",
            "alpaca": "https://data.alpaca.markets/v1beta3"
        }
        return urls.get(api_type, "https://api.binance.com/api/v3")
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 30
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (BTCUSDT, AAPL, etc.)
            timeframe: Candle interval (5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch (default 30)
        
        Returns:
            Tuple of (opens, highs, lows, closes, volumes)
            Each is a list of floats
        
        Example:
            opens, highs, lows, closes, volumes = await service.fetch_ohlcv(
                "BTCUSDT", "5m", 30
            )
        """
        try:
            async with aiohttp.ClientSession() as session:
                if self.api_type == "binance":
                    return await self._fetch_binance(session, symbol, timeframe, limit)
                elif self.api_type == "polygon":
                    return await self._fetch_polygon(session, symbol, timeframe, limit)
                else:
                    raise ValueError(f"Unsupported API type: {self.api_type}")
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            raise
    
    async def _fetch_binance(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> Tuple:
        """Fetch from Binance API."""
        url = f"{self.base_url}/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": limit
        }
        
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"Binance API error: {resp.status}")
            
            data = await resp.json()
            
            opens = [float(candle[1]) for candle in data]    # Open
            highs = [float(candle[2]) for candle in data]    # High
            lows = [float(candle[3]) for candle in data]     # Low
            closes = [float(candle[4]) for candle in data]   # Close
            volumes = [float(candle[7]) for candle in data]  # Quote asset volume
            
            logger.debug(f"Fetched {len(opens)} candles for {symbol}")
            return opens, highs, lows, closes, volumes
    
    async def _fetch_polygon(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> Tuple:
        """Fetch from Polygon.io API."""
        # Convert timeframe: 5m → 5, 1h → 60, 1d → day
        interval_map = {
            "1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60",
            "4h": "240", "1d": "day"
        }
        interval = interval_map.get(timeframe, "5")
        
        url = f"{self.base_url}/aggs/ticker/{symbol.upper()}/range/{interval}/{interval}/prev"
        params = {"apiKey": self.api_key, "limit": limit}
        
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"Polygon API error: {resp.status}")
            
            response = await resp.json()
            data = response.get("results", [])
            
            opens = [float(candle["o"]) for candle in data]
            highs = [float(candle["h"]) for candle in data]
            lows = [float(candle["l"]) for candle in data]
            closes = [float(candle["c"]) for candle in data]
            volumes = [float(candle["v"]) for candle in data]
            
            logger.debug(f"Fetched {len(opens)} candles for {symbol}")
            return opens, highs, lows, closes, volumes
    
    def normalize_ohlcv(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        method: str = "minmax"
    ) -> np.ndarray:
        """
        Normalize OHLCV data for model input.
        
        Args:
            method: Normalization method (minmax, zscore, robust)
        
        Returns:
            np.ndarray of shape (seq_len, 5) - normalized OHLCV
        """
        ohlcv = np.array([opens, highs, lows, closes, volumes]).T  # (seq_len, 5)
        
        if method == "minmax":
            # Min-max normalization [0, 1]
            min_vals = ohlcv.min(axis=0)
            max_vals = ohlcv.max(axis=0)
            ohlcv = (ohlcv - min_vals) / (max_vals - min_vals + 1e-8)
        
        elif method == "zscore":
            # Z-score normalization
            mean_vals = ohlcv.mean(axis=0)
            std_vals = ohlcv.std(axis=0)
            ohlcv = (ohlcv - mean_vals) / (std_vals + 1e-8)
        
        elif method == "robust":
            # Robust normalization using median and IQR
            median_vals = np.median(ohlcv, axis=0)
            q75 = np.percentile(ohlcv, 75, axis=0)
            q25 = np.percentile(ohlcv, 25, axis=0)
            iqr = q75 - q25
            ohlcv = (ohlcv - median_vals) / (iqr + 1e-8)
        
        return ohlcv
    
    def prepare_enhanced_features(
        self,
        opens: List[float],
        highs: List[float], 
        lows: List[float],
        closes: List[float],
        volumes: List[float]
    ) -> np.ndarray:
        """
        Prepare enhanced features including technical indicators.
        
        Returns:
            np.ndarray of shape (seq_len, 20) - OHLCV + technical indicators
        """
        # Convert to numpy arrays
        opens = np.array(opens)
        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(closes)
        volumes = np.array(volumes)
        
        # Calculate technical indicators
        from app.models.lstm_signal_model import TechnicalIndicators
        tech = TechnicalIndicators()
        
        # Moving averages
        sma_5 = tech.calculate_sma(closes, 5)
        sma_10 = tech.calculate_sma(closes, 10)
        ema_12 = tech.calculate_ema(closes, 12)
        
        # Momentum indicators
        rsi = tech.calculate_rsi(closes)
        macd, macd_signal, macd_hist = tech.calculate_macd(closes)
        
        # Volatility indicators
        bb_upper, bb_lower = tech.calculate_bollinger_bands(closes)
        
        # Stochastic oscillator
        stoch_k, stoch_d = tech.calculate_stochastic(highs, lows, closes)
        
        # Price and volume features
        price_change = np.diff(closes, prepend=closes[0])
        volume_sma = tech.calculate_sma(volumes, 10)
        price_to_sma = closes / (sma_5 + 1e-8)
        volume_ratio = volumes / (volume_sma + 1e-8)
        
        # Combine all features
        ohlcv = np.column_stack([opens, highs, lows, closes, volumes])
        technical_features = np.column_stack([
            sma_5, sma_10, ema_12, rsi,
            macd, macd_signal, macd_hist,
            bb_upper, bb_lower,
            stoch_k, stoch_d,
            price_change, volume_sma,
            price_to_sma, volume_ratio
        ])
        
        # Combine OHLCV with technical indicators
        enhanced_features = np.concatenate([ohlcv, technical_features], axis=1)
        
        # Normalize the enhanced features
        enhanced_features = self._normalize_features(enhanced_features)
        
        return enhanced_features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using robust scaling."""
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Robust normalization
        median_vals = np.median(features, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        iqr = q75 - q25
        
        # Avoid division by zero
        iqr = np.where(iqr == 0, 1.0, iqr)
        
        normalized = (features - median_vals) / iqr
        
        # Clip extreme values
        normalized = np.clip(normalized, -5, 5)
        
        return normalized