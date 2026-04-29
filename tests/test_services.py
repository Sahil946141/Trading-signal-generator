import pytest
import numpy as np
import pandas as pd
from app.config import config
from app.services.image_preprocess import ImagePreprocessor
from app.services.nifty_data_service import StockDataService
from app.utils.label_encoder import PatternLabelEncoder
from app.utils.lstm_features import build_feature_window, required_close_points
from app.models.cnn_pattern_model import PatternDetectionModel
from app.training.dataset_preparation import build_feature_frame, generate_forward_labels, build_raw_sequences

def test_pattern_label_encoder():
    """Test pattern label encoder."""
    patterns = ["Double Bottom", "Double Top", "Head and Shoulders"]
    encoder = PatternLabelEncoder(patterns)
    
    # Test encoding
    encoded = encoder.encode("Double Bottom", 0.87)
    assert encoded.shape == (4,)  # 3 patterns + 1 confidence
    assert encoded[0] == 1.0  # First pattern should be 1
    assert encoded[-1] == 0.87  # Confidence should be last
    
    # Test decoding
    decoded = encoder.decode(encoded)
    assert decoded == "Double Bottom"

def test_image_preprocessor():
    """Test image preprocessor initialization."""
    preprocessor = ImagePreprocessor()
    assert preprocessor.target_size == (224, 224)

def test_pattern_classes():
    """Test pattern detection model classes."""
    assert len(PatternDetectionModel.PATTERN_CLASSES) >= 10

def test_attached_chart_pattern_model_loads_and_predicts():
    """The attached chart pattern model should load and return a structured prediction."""
    model = PatternDetectionModel(model_path="chart_pattern_tf_model.h5")
    preprocessor = ImagePreprocessor()
    image = preprocessor.preprocess_tensorflow("test_image.png", target_size=model.input_size)
    result = model.predict(image)
    assert "pattern" in result
    assert "pattern_confidence" in result
    assert "probabilities" in result
    assert len(result["probabilities"]) == len(model.PATTERN_CLASSES)

def test_tensorflow_chart_prediction_does_not_double_normalize():
    """TensorFlow chart inference should pass raw pixel-space images through unchanged."""
    class DummyModel:
        def __init__(self):
            self.seen_max = None

        def predict(self, batch, verbose=0):
            self.seen_max = float(batch.max())
            return np.array([[0.1, 0.9]], dtype=np.float32)

    model = PatternDetectionModel.__new__(PatternDetectionModel)
    model.model = DummyModel()

    sample = np.full((1, 128, 128, 3), 255.0, dtype=np.float32)
    result = PatternDetectionModel._predict_tensorflow_model(model, sample)

    assert model.model.seen_max == 255.0
    assert result.shape == (1, 2)

def test_forward_label_generation():
    """Forward-looking labels should create BUY and SELL classes from future close moves."""
    frame = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "Open": [100, 102, 104, 106, 107, 104, 101, 99, 97, 95],
        "High": [101, 103, 105, 107, 108, 105, 102, 100, 98, 96],
        "Low": [99, 101, 103, 105, 106, 103, 100, 98, 96, 94],
        "Close": [100, 102, 104, 106, 107, 104, 101, 99, 97, 95],
        "Volume": [10] * 10,
    })

    features = build_feature_frame(frame, feature_column="return")
    labeled = generate_forward_labels(features, lookahead=2, threshold=0.01)
    assert "BUY" in labeled["Label"].values
    assert "SELL" in labeled["Label"].values

def test_sequence_builder_uses_requested_length():
    """Sequence builder should produce 60-step compatible return windows when enough rows exist."""
    rows = 90
    frame = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=rows, freq="D"),
        "Open": np.arange(rows, dtype=float) + 100,
        "High": np.arange(rows, dtype=float) + 101,
        "Low": np.arange(rows, dtype=float) + 99,
        "Close": np.arange(rows, dtype=float) + 100.5,
        "Volume": np.arange(rows, dtype=float) + 1000,
        "LabelEncoded": [1] * rows,
    })

    features = build_feature_frame(frame, feature_column="return")
    features["LabelEncoded"] = 1
    sequences, labels, dates = build_raw_sequences(features, sequence_length=60, feature_column="return")
    expected_sequences = len(features) - 60
    assert sequences.shape == (expected_sequences, 60, 1)
    assert labels.shape == (expected_sequences,)
    assert len(dates) == expected_sequences

def test_feature_window_normalization_shapes():
    """Inference feature windows should normalize cleanly and preserve the 60x1 shape."""
    closes = np.linspace(100, 160, 61, dtype=np.float32)
    window = build_feature_window(
        closes=closes,
        sequence_length=60,
        feature_column="return",
        normalization="zscore",
    )
    assert window.shape == (60, 1)
    assert np.isfinite(window).all()

def test_stock_data_service_required_points_matches_feature_mode():
    """Live inference should request enough closes for the configured feature pipeline."""
    service = StockDataService()
    assert service.required_points_for_lstm() == required_close_points(
        config.LSTM_SEQUENCE_LENGTH,
        config.LSTM_FEATURE_COLUMN,
    )

def test_stock_data_service_symbol_mapping_helpers():
    """Alternate OHLC providers should derive provider-specific symbols safely."""
    service = StockDataService()
    assert service._symbol_base_and_exchange("RELIANCE.NS") == ("RELIANCE", "NSE")
    assert service._twelvedata_interval("5m") == "5min"
    assert service._alphavantage_interval("1h") == "60min"
