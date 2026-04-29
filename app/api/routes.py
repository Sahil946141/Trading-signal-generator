from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import time
import logging
import numpy as np
from uuid import uuid4
from pydantic import BaseModel

# Import all services and models
from app.services.image_preprocess import ImagePreprocessor
from app.models.cnn_pattern_model import PatternDetectionModel
from app.models.lstm_signal_model import AttachedStockModelRouter
from app.services.nifty_data_service import StockDataService
from app.services.response_builder import ResponseBuilder
from app.utils.label_encoder import PatternLabelEncoder
from app.config import config
from app.training.dataset_preparation import preview_lstm_dataset, prepare_lstm_dataset
from app.training.train_models import ModelTrainer, load_or_prepare_lstm_training_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["predictions"])

# Global service instances (initialize on startup)
_services = {}


class DatasetRequest(BaseModel):
    symbol: str = getattr(config, "DEFAULT_STOCK_SYMBOL", "RELIANCE.NS")
    symbols: str | None = None
    timeframe: str = getattr(config, "DEFAULT_TRAINING_TIMEFRAME", "1d")
    feature_column: str = config.LSTM_FEATURE_COLUMN
    start_date: str = config.STOCK_TRAINING_START
    end_date: str | None = None
    sequence_length: int = config.LSTM_SEQUENCE_LENGTH
    lookahead: int = config.LABEL_LOOKAHEAD
    threshold: float = config.LABEL_THRESHOLD
    train_split: float = config.TRAIN_SPLIT
    output_dir: str = os.path.join(config.DATA_DIR, "lstm_dataset")


class TrainLSTMRequest(DatasetRequest):
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    save_path: str = config.LSTM_MODEL_PATH

def init_services():
    """Initialize all services on app startup."""
    global _services
    
    _services["image_preprocessor"] = ImagePreprocessor(target_size=(224, 224), is_training=False)
    _services["pattern_model"] = None

    if os.path.exists(config.PATTERN_MODEL_PATH):
        try:
            _services["pattern_model"] = PatternDetectionModel(model_path=config.PATTERN_MODEL_PATH)
            logger.info("Loaded chart pattern model from %s", config.PATTERN_MODEL_PATH)
        except Exception as exc:
            logger.exception("Failed to load chart pattern model: %s", exc)
            _services["pattern_model"] = None
            raise
    else:
        raise FileNotFoundError(f"Pattern model file missing: {config.PATTERN_MODEL_PATH}")
    
    _services["signal_model"] = AttachedStockModelRouter(
        models_dir=config.ATTACHED_MODEL_DIR,
        pattern=config.ATTACHED_MODEL_PATTERN,
        fallback_model_path=config.LSTM_MODEL_PATH,
    )
    
    _services["market_data_service"] = StockDataService()
    
    _services["label_encoder"] = PatternLabelEncoder(
        PatternDetectionModel.PATTERN_CLASSES
    )
    
    logger.info("All stock services initialized successfully")

@router.post("/predict")
async def predict_signal(
    file: UploadFile = File(...),
    symbol: str = Form(getattr(config, "DEFAULT_STOCK_SYMBOL", "RELIANCE.NS")),
    timeframe: str = Form(config.DEFAULT_TIMEFRAME),
    use_lstm: bool = Form(config.USE_LSTM),
):
    """
    Main prediction endpoint.
    
    Workflow:
    1. Validate and save uploaded image
    2. Detect pattern from image
    3. Fetch market data (OHLCV)
    4. Predict signal (LSTM or rule-based)
    5. Build and return response
    
    Args:
        file: Chart image file
        timeframe: Candle interval (1m, 5m, 15m, 1h, 1d)
        use_lstm: Use LSTM model (True) or rule-based (False)
    
    Returns:
        JSON response with pattern and signal predictions
    """
    request_id = str(uuid4())[:8]
    start_time = time.time()
    
    temp_file = None
    
    try:
        if timeframe not in config.VALID_TIMEFRAMES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframe '{timeframe}'. Valid values: {config.VALID_TIMEFRAMES}"
            )

        instrument = symbol.strip().upper()
        
        # Validate file
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Supported: JPG, PNG, GIF"
            )
        
        # Check file size
        contents = await file.read()
        if len(contents) > config.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {config.MAX_IMAGE_SIZE_MB}MB"
            )
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            temp_file = tmp.name
            tmp.write(contents)

        if not _services:
            init_services()

        # Get services
        img_preprocessor = _services["image_preprocessor"]
        pattern_model = _services["pattern_model"]
        signal_model = _services["signal_model"]
        market_data_service = _services["market_data_service"]
        
        logger.info(f"[{request_id}] Received symbol parameter: '{symbol}'")
        instrument = symbol.strip().upper()
        logger.info(f"[{request_id}] Processing image for {instrument} on {timeframe}")
        
        # Step 1-2: Pattern detection from uploaded image.
        predictions = []
        if pattern_model is None:
            raise HTTPException(status_code=503, detail="Chart pattern model is not loaded.")

        # Use appropriate preprocessing based on model framework
        if pattern_model.framework in {"tensorflow", "numpy-h5"}:
            image_tensor = img_preprocessor.preprocess_tensorflow(
                temp_file, target_size=pattern_model.input_size
            )
        else:
            image_tensor = img_preprocessor.preprocess(temp_file)

        standard_result = pattern_model.predict(image_tensor)
        predictions.append(standard_result)

        # Only use test-time augmentation for PyTorch models
        if pattern_model.framework not in {"tensorflow", "numpy-h5"}:
            augmented_batch = img_preprocessor.create_test_time_augmentation(
                temp_file, num_augmentations=3
            )
            for i in range(augmented_batch.shape[0]):
                aug_tensor = augmented_batch[i:i+1]
                aug_result = pattern_model.predict(aug_tensor)
                predictions.append(aug_result)

            pattern_result = _ensemble_predictions(predictions)
            logger.info(f"[{request_id}] Enhanced pattern detection: {pattern_result['pattern']} (confidence: {pattern_result['pattern_confidence']:.4f}, ensemble of {len(predictions)})")
        else:
            pattern_result = standard_result
            logger.info(f"[{request_id}] Detected pattern: {pattern_result['pattern']} (confidence: {pattern_result['pattern_confidence']:.4f})")
        
        # Step 3: Fetch stock OHLCV data and build the 60-step return window.
        try:
            opens, highs, lows, closes, volumes, times = await market_data_service.fetch_ohlcv(
                symbol=instrument,
                timeframe=timeframe,
                limit=config.CANDLES_TO_FETCH + 1
            )

            lstm_features = market_data_service.prepare_lstm_features(closes)
            opens, highs, lows, closes, volumes, times = market_data_service.trim_candles(
                opens, highs, lows, closes, volumes, times, config.CANDLES_TO_FETCH
            )

            current_price = closes[-1]
            logger.info(f"[{request_id}] LSTM features prepared: {lstm_features.shape}")
        except Exception as e:
            logger.error(f"[{request_id}] Market data fetch failed: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Unable to fetch market data for {instrument} on timeframe {timeframe}. {e}"
            )
        
        # Step 4: Enhanced signal prediction
        if use_lstm:
            try:
                signal_result = signal_model.predict(instrument, lstm_features)
                logger.info(f"[{request_id}] Stock LSTM signal: {signal_result['signal']} ({signal_result['confidence']:.2%})")
            except FileNotFoundError as e:
                logger.error(f"[{request_id}] No attached stock model for {instrument}: {e}")
                raise HTTPException(
                    status_code=404,
                    detail=str(e)
                )
            except Exception as e:
                logger.error(f"[{request_id}] Stock LSTM prediction failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Stock LSTM prediction failed for {instrument}. {e}"
                )
        else:
            # Enhanced rule-based mapping with technical analysis
            signal_result = enhanced_rule_based_mapping(pattern_result["pattern"], lstm_features)
            logger.info(f"[{request_id}] Enhanced rule-based signal: {signal_result['signal']}")
        
        # Step 5: Build response with market context.
        technical_summary = market_data_service.summarize_market_data(highs, lows, closes, volumes)
        latest_candle = market_data_service.latest_candle(opens, highs, lows, closes, volumes, times)
        
        metadata = {
            "instrument": instrument,
            "timeframe": timeframe,
            "current_price": round(current_price, 2),
            "last_updated": time.time(),
            "candles_fetched": config.CANDLES_TO_FETCH,
            "latest_candle": latest_candle,
            "market_data_source": market_data_service.last_fetch_source,
            "market_data_symbol": instrument,
            "model_enhancements": {
                "cnn_model": "Enhanced EfficientNet with attention",
                "lstm_model": "Keras stock LSTM on 60-step return sequences",
                "sequence_length": config.LSTM_SEQUENCE_LENGTH,
                "features_count": int(lstm_features.shape[1]) if lstm_features is not None else 5,
                "test_time_augmentation": len(predictions) > 1,
                "feature_column": config.LSTM_FEATURE_COLUMN,
                "training_symbols": "single-stock or multi-stock yfinance dataset",
                "attached_model_path": signal_result.get("model_path") if use_lstm else None,
                "pattern_model_status": "loaded",
            },
            "technical_analysis": technical_summary
        }
        
        processing_time = (time.time() - start_time) * 1000
        response = ResponseBuilder.build_prediction_response(
            pattern_result=pattern_result,
            signal_result=signal_result,
            metadata=metadata,
            request_id=request_id,
            processing_time_ms=processing_time
        )
        
        logger.info(f"[{request_id}] Response sent in {processing_time:.0f}ms")
        return JSONResponse(content=response, status_code=200)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Prediction error: {str(e)}")
        error_response = ResponseBuilder.build_error_response(
            error_message=str(e),
            request_id=request_id
        )
        return JSONResponse(content=error_response, status_code=500)
    
    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

def enhanced_rule_based_mapping(pattern: str, enhanced_features: np.ndarray = None) -> dict:
    """
    Enhanced rule-based signal mapping with technical analysis.
    
    Incorporates technical indicators for better accuracy.
    """
    # Base pattern rules
    base_rules = {
        "Double Bottom": {"signal": "BUY", "confidence": 0.75},
        "Double Top": {"signal": "SELL", "confidence": 0.75},
        "Head and Shoulders": {"signal": "SELL", "confidence": 0.80},
        "Inverse Head and Shoulders": {"signal": "BUY", "confidence": 0.80},
        "Triangle Ascending": {"signal": "BUY", "confidence": 0.70},
        "Triangle Descending": {"signal": "SELL", "confidence": 0.70},
        "Flag": {"signal": "BUY", "confidence": 0.65},
        "Wedge": {"signal": "HOLD", "confidence": 0.60},
        "Cup and Handle": {"signal": "BUY", "confidence": 0.75},
        "Consolidation": {"signal": "HOLD", "confidence": 0.55}
    }
    
    rule = base_rules.get(pattern, {"signal": "HOLD", "confidence": 0.50})
    
    # Enhance with technical analysis if available
    if enhanced_features is not None and enhanced_features.shape[0] > 0:
        try:
            if enhanced_features.ndim == 2 and enhanced_features.shape[1] >= 4:
                closes = enhanced_features[:, 3]
            else:
                closes = enhanced_features[:, 0]
            
            # Simple technical analysis
            recent_reference = closes[-10] if len(closes) > 10 else closes[0]
            momentum_reference = closes[-5] if len(closes) > 5 else closes[0]
            recent_trend = "bullish" if closes[-1] > recent_reference else "bearish"
            momentum = (closes[-1] - momentum_reference) / (abs(momentum_reference) + 1e-8)
            
            # Adjust confidence based on technical analysis
            if rule["signal"] == "BUY" and recent_trend == "bullish" and momentum > 0:
                rule["confidence"] = min(0.95, rule["confidence"] + 0.15)
            elif rule["signal"] == "SELL" and recent_trend == "bearish" and momentum < 0:
                rule["confidence"] = min(0.95, rule["confidence"] + 0.15)
            elif rule["signal"] == "BUY" and recent_trend == "bearish":
                rule["confidence"] = max(0.30, rule["confidence"] - 0.20)
            elif rule["signal"] == "SELL" and recent_trend == "bullish":
                rule["confidence"] = max(0.30, rule["confidence"] - 0.20)
                
        except Exception as e:
            logger.warning(f"Technical analysis enhancement failed: {e}")
    
    # Calculate probabilities
    base_prob = rule["confidence"]
    other_prob = (1 - base_prob) / 2
    
    probabilities = {
        "BUY": base_prob if rule["signal"] == "BUY" else other_prob,
        "SELL": base_prob if rule["signal"] == "SELL" else other_prob,
        "HOLD": base_prob if rule["signal"] == "HOLD" else other_prob
    }
    
    return {
        "signal": rule["signal"],
        "confidence": rule["confidence"],
        "probabilities": probabilities
    }

def _ensemble_predictions(predictions: list) -> dict:
    """
    Ensemble multiple predictions for better accuracy.
    
    Args:
        predictions: List of prediction dictionaries
    
    Returns:
        Ensembled prediction dictionary
    """
    if len(predictions) == 1:
        return predictions[0]
    
    # Average probabilities
    all_probs = []
    for pred in predictions:
        probs = list(pred["probabilities"].values())
        all_probs.append(probs)
    
    avg_probs = np.mean(all_probs, axis=0)
    
    # Get ensemble prediction
    pattern_classes = list(predictions[0]["probabilities"].keys())
    best_idx = np.argmax(avg_probs)
    best_pattern = pattern_classes[best_idx]
    best_confidence = avg_probs[best_idx]
    
    # Create ensemble probabilities dict
    ensemble_probs = {
        pattern: prob for pattern, prob in zip(pattern_classes, avg_probs)
    }
    
    # Get top 5
    top5_indices = np.argsort(avg_probs)[-5:][::-1]
    top5 = [
        {
            "pattern": pattern_classes[idx],
            "confidence": avg_probs[idx]
        }
        for idx in top5_indices
    ]
    
    return {
        "pattern": best_pattern,
        "pattern_confidence": best_confidence,
        "probabilities": ensemble_probs,
        "top_5": top5
    }

def _calculate_technical_summary_from_features(enhanced_features: np.ndarray) -> dict:
    """Calculate technical analysis summary from enhanced features."""
    if enhanced_features is None or enhanced_features.shape[0] == 0:
        return {"status": "No technical data available"}
    
    try:
        # Extract basic OHLCV (first 5 columns)
        closes = enhanced_features[:, 3]
        volumes = enhanced_features[:, 4]
        
        # Calculate basic indicators
        current_price = closes[-1]
        price_change = closes[-1] - closes[-2] if len(closes) > 1 else 0
        price_change_pct = (price_change / closes[-2] * 100) if len(closes) > 1 and closes[-2] != 0 else 0
        
        # Trend analysis
        short_trend = "Bullish" if closes[-1] > closes[-5] else "Bearish" if len(closes) > 5 else "Neutral"
        long_trend = "Bullish" if closes[-1] > closes[-20] else "Bearish" if len(closes) > 20 else "Neutral"
        
        # Volume analysis
        avg_volume = np.mean(volumes[-10:]) if len(volumes) > 10 else np.mean(volumes)
        volume_trend = "High" if volumes[-1] > avg_volume * 1.2 else "Low" if volumes[-1] < avg_volume * 0.8 else "Normal"
        
        return {
            "current_price": round(current_price, 2),
            "price_change": round(price_change, 2),
            "price_change_pct": round(price_change_pct, 2),
            "short_term_trend": short_trend,
            "long_term_trend": long_trend,
            "volume_trend": volume_trend,
            "momentum": "Positive" if price_change > 0 else "Negative" if price_change < 0 else "Neutral"
        }
    except Exception as e:
        logger.warning(f"Technical summary calculation failed: {e}")
        return {"status": "Technical analysis failed", "error": str(e)}

def rule_based_mapping(pattern: str) -> dict:
    """
    Legacy rule-based signal mapping (kept for backward compatibility).
    """
    return enhanced_rule_based_mapping(pattern, None)




@router.get("/market-data")
async def get_market_data(
    symbol: str = config.DEFAULT_STOCK_SYMBOL,
    timeframe: str = config.DEFAULT_TIMEFRAME,
):
    """Return the latest 60 stock candles plus the latest OHLC snapshot."""
    if timeframe not in config.VALID_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe '{timeframe}'. Valid values: {config.VALID_TIMEFRAMES}"
        )

    if not _services:
        init_services()

    service = _services["market_data_service"]
    try:
        opens, highs, lows, closes, volumes, times = await service.fetch_ohlcv(
            symbol=symbol.strip().upper(),
            timeframe=timeframe,
            limit=config.CANDLES_TO_FETCH
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    candles = service.build_candles(opens, highs, lows, closes, volumes, times)
    return {
        "status": "success",
        "instrument": symbol.strip().upper(),
        "market_data_source": service.last_fetch_source,
        "market_data_symbol": symbol.strip().upper(),
        "timeframe": timeframe,
        "candles_fetched": len(candles),
        "latest_candle": candles[-1],
        "technical_analysis": service.summarize_market_data(highs, lows, closes, volumes),
        "candles": candles,
    }


@router.post("/training/preview")
async def preview_training_dataset(request: DatasetRequest):
    """Preview stock label generation and chronological split statistics."""
    try:
        summary = preview_lstm_dataset(
            symbol=request.symbol,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
            sequence_length=request.sequence_length,
            lookahead=request.lookahead,
            threshold=request.threshold,
            train_split=request.train_split,
            feature_column=request.feature_column,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "success", "data": summary}


@router.post("/training/prepare")
async def prepare_training_dataset(request: DatasetRequest):
    """Persist the supervised LSTM dataset to disk."""
    try:
        summary = prepare_lstm_dataset(
            output_dir=request.output_dir,
            symbol=request.symbol,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
            sequence_length=request.sequence_length,
            lookahead=request.lookahead,
            threshold=request.threshold,
            train_split=request.train_split,
            feature_column=request.feature_column,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "success", "data": summary}


@router.post("/training/run")
async def run_lstm_training(request: TrainLSTMRequest):
    """Prepare the stock dataset and train the Keras LSTM end to end."""
    try:
        train_data, val_data, dataset_summary = load_or_prepare_lstm_training_data(
            dataset_dir=request.output_dir,
            symbol=request.symbol,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
            sequence_length=request.sequence_length,
            lookahead=request.lookahead,
            threshold=request.threshold,
            train_split=request.train_split,
            feature_column=request.feature_column,
        )

        trainer = ModelTrainer()
        training_summary = trainer.train_lstm_model(
            train_data=train_data,
            val_data=val_data,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            save_path=request.save_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "success",
        "data": {
            "dataset": dataset_summary,
            "training": {
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "best_validation_accuracy": training_summary["best_validation_accuracy"],
                "final_validation_accuracy": training_summary["final_validation_accuracy"],
                "final_validation_loss": training_summary["final_validation_loss"],
                "model_path": request.save_path,
                "history_path": training_summary["history_path"],
                "confusion_matrix_path": training_summary["confusion_matrix_path"],
                "training_curve_path": training_summary["training_curve_path"],
                "artifacts_dir": training_summary["artifacts_dir"],
            },
        },
    }



@router.get("/health")
async def health_check():
    """Enhanced health check endpoint."""
    supported_symbols = []
    if _services.get("signal_model") is not None:
        supported_symbols = _services["signal_model"].supported_symbols()

    return {
        "status": "healthy",
        "service": "stock-trading-signal-predictor",
        "instrument": config.DEFAULT_STOCK_SYMBOL,
        "services_initialized": len(_services) > 0,
        "supported_symbols": supported_symbols,
        "timeframes": config.VALID_TIMEFRAMES,
        "pattern_model_ready": _services.get("pattern_model") is not None,
        "market_data_sources": config.MARKET_DATA_SOURCES,
        "model_enhancements": {
            "cnn_model": "EfficientNet with attention mechanism",
            "lstm_model": "Attached per-stock Keras models on 60-step return sequences",
            "features": [
                "yfinance OHLCV for stocks",
                "Return-based feature engineering",
                "60-step LSTM input windows",
                "Forward close-vs-now BUY/HOLD/SELL labels",
                "Chronological train/validation split",
                "Single-stock and multi-stock dataset preparation",
                "Saved .h5 Keras model artifacts"
            ]
        },
        "version": "3.0.0"
    }


@router.get("/stocks")
async def list_supported_stocks():
    """List the symbols that have attached stock LSTM models."""
    if not _services:
        init_services()

    signal_model = _services["signal_model"]
    symbols = signal_model.supported_symbols()
    logger.info(f"Supported stocks: {symbols}")
    return {
        "status": "success",
        "symbols": symbols,
        "default_symbol": config.DEFAULT_STOCK_SYMBOL,
    }


@router.get("/models")
async def list_attached_models():
    """List metadata for attached stock LSTM models."""
    if not _services:
        init_services()

    signal_model = _services["signal_model"]
    details = signal_model.model_details()
    return {
        "status": "success",
        "count": len(details),
        "models": details,
    }
