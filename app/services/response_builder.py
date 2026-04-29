from datetime import datetime
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ResponseBuilder:
    """
    Constructs standardized API responses.
    """
    
    @staticmethod
    def build_prediction_response(
        pattern_result: Dict[str, Any],
        signal_result: Dict[str, Any],
        metadata: Dict[str, Any],
        request_id: str,
        processing_time_ms: float = None
    ) -> Dict[str, Any]:
        """
        Build complete prediction response.
        
        Args:
            pattern_result: Output from PatternDetectionModel.predict()
            signal_result: Output from HybridSignalModel.predict()
            metadata: {instrument, timeframe, current_price, etc.}
            request_id: Unique request identifier
            processing_time_ms: Total processing time
        
        Returns:
            Standardized response dictionary
        """
        response = {
            "status": "success",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "pattern": {
                    "label": pattern_result["pattern"],
                    "confidence": round(pattern_result["pattern_confidence"], 4),
                    "top_5": pattern_result.get("top_5", [])
                },
                "signal": {
                    "label": signal_result["signal"],
                    "confidence": round(signal_result["confidence"], 4),
                    "probabilities": {
                        signal: round(prob, 4)
                        for signal, prob in signal_result["probabilities"].items()
                    }
                }
            },
            "metadata": {
                "instrument": metadata.get("instrument"),
                "timeframe": metadata.get("timeframe"),
                "current_price": metadata.get("current_price"),
                "candles_fetched": metadata.get("candles_fetched"),
                "latest_candle": metadata.get("latest_candle"),
                "market_data_source": metadata.get("market_data_source"),
                "market_data_symbol": metadata.get("market_data_symbol"),
                "cache_status": metadata.get("cache_status"),
                "fetch_error": metadata.get("fetch_error"),
                "attempted_sources": metadata.get("attempted_sources"),
                "pattern_model_ready": metadata.get("pattern_model_ready"),
                "technical_analysis": metadata.get("technical_analysis"),
                "model_enhancements": metadata.get("model_enhancements"),
                "last_candle_close_time": metadata.get("latest_candle", {}).get("time"),
                "last_updated": metadata.get("last_updated"),
            }
        }
        
        if processing_time_ms:
            response["processing_time_ms"] = round(processing_time_ms, 2)
        
        return response
    
    @staticmethod
    def build_error_response(
        error_message: str,
        request_id: str = None,
        error_code: str = "PREDICTION_ERROR"
    ) -> Dict[str, Any]:
        """
        Build error response.
        
        Args:
            error_message: Description of error
            request_id: Unique request identifier
            error_code: Error code for logging
        
        Returns:
            Error response dictionary
        """
        return {
            "status": "error",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error": {
                "code": error_code,
                "message": error_message
            }
        }
    
    @staticmethod
    def to_json(response: Dict[str, Any]) -> str:
        """Convert response to JSON string."""
        return json.dumps(response, indent=2)
