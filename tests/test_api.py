import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import config

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Trading Signal Predictor API"
    assert "version" in data
    assert data["endpoints"]["dashboard"] == "/dashboard"

def test_dashboard_endpoint():
    """Test the built-in dashboard endpoint."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["instrument"] == config.DEFAULT_STOCK_SYMBOL
    assert "pattern_model_ready" in data
    assert "market_data_sources" in data

def test_supported_stocks_endpoint():
    """Test the attached stock-model listing endpoint."""
    response = client.get("/api/stocks")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert isinstance(data["symbols"], list)
    assert config.DEFAULT_STOCK_SYMBOL in data["symbols"]

def test_models_endpoint():
    """Test the attached stock-model metadata endpoint."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["count"] >= 1
    assert isinstance(data["models"], list)
    assert data["models"][0]["feature_pipeline"]["feature_column"] == config.LSTM_FEATURE_COLUMN

def test_market_data_invalid_timeframe():
    """Timeframe validation should fail before any upstream market call."""
    response = client.get("/api/market-data?timeframe=4h")
    assert response.status_code == 400

def test_predict_endpoint_no_file():
    """Test predict endpoint without file."""
    response = client.post("/api/predict")
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_invalid_file():
    """Test predict endpoint with invalid file type."""
    response = client.post(
        "/api/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400

# Note: Add more comprehensive tests when you have actual model files
