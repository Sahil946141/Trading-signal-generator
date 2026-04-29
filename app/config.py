import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""
    
    # FastAPI
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Models
    PATTERN_MODEL_PATH = os.getenv("PATTERN_MODEL_PATH", "chart_pattern_tf_model.h5")
    LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "models/stock_lstm.h5")
    LSTM_SCALER_PATH = os.getenv("LSTM_SCALER_PATH", "models/stock_lstm_scaler.joblib")
    ATTACHED_MODEL_DIR = "."
    ATTACHED_MODEL_PATTERN = os.getenv("ATTACHED_MODEL_PATTERN", "*_lstm_model.keras")
    USE_LSTM = os.getenv("USE_LSTM", "True").lower() == "true"
    REQUIRE_PATTERN_MODEL = os.getenv("REQUIRE_PATTERN_MODEL", "False").lower() == "true"
    PATTERN_CLASS_NAMES_RAW = os.getenv("PATTERN_CLASS_NAMES", "")
    PATTERN_CLASS_NAMES = [name.strip() for name in PATTERN_CLASS_NAMES_RAW.split(',')] if PATTERN_CLASS_NAMES_RAW else []
    
    # Market Data
    DEFAULT_STOCK_SYMBOL = os.getenv("DEFAULT_STOCK_SYMBOL", "RELIANCE.NS")
    DEFAULT_INSTRUMENT = DEFAULT_STOCK_SYMBOL
    DEFAULT_TRAINING_SYMBOLS = os.getenv("DEFAULT_TRAINING_SYMBOLS", DEFAULT_STOCK_SYMBOL)
    PKNSE_REQUEST_TIMEOUT = int(os.getenv("PKNSE_REQUEST_TIMEOUT", 30))
    DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "5m")
    DEFAULT_TRAINING_TIMEFRAME = os.getenv("DEFAULT_TRAINING_TIMEFRAME", "1d")
    VALID_TIMEFRAMES = ["1m", "5m", "15m", "1h", "1d"]
    CANDLES_TO_FETCH = int(os.getenv("CANDLES_TO_FETCH", 60))
    MARKET_DATA_SOURCES = os.getenv("MARKET_DATA_SOURCES", "yfinance")
    
    # Defaults
    STOCK_TRAINING_START = os.getenv("STOCK_TRAINING_START", "2015-01-01")
    TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", 0.8))
    LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", 60))
    LABEL_LOOKAHEAD = int(os.getenv("LABEL_LOOKAHEAD", 5))
    LABEL_THRESHOLD = float(os.getenv("LABEL_THRESHOLD", 0.01))
    LSTM_FEATURE_COLUMN = os.getenv("LSTM_FEATURE_COLUMN", "return")
    LSTM_SEQUENCE_NORMALIZATION = os.getenv("LSTM_SEQUENCE_NORMALIZATION", "minmax")
    DUMMY_PATTERN_CONFIDENCE = float(os.getenv("DUMMY_PATTERN_CONFIDENCE", 0.5))
    DATA_DIR = os.getenv("DATA_DIR", "data")
    
    # Frontend
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
    DASHBOARD_DIR = os.getenv("DASHBOARD_DIR", "app/frontend")
    
    # Performance
    MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", 10))
    INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", 30))

config = Config()

if config.PATTERN_CLASS_NAMES_RAW:
    raw = config.PATTERN_CLASS_NAMES_RAW.strip()
    if raw.startswith("["):
        config.PATTERN_CLASS_NAMES = json.loads(raw)
    else:
        config.PATTERN_CLASS_NAMES = [item.strip() for item in raw.split(",") if item.strip()]
else:
    config.PATTERN_CLASS_NAMES = []
