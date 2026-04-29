#!/usr/bin/env python3
"""
Trading Signal Predictor - Main Runner

This script starts the FastAPI server with proper configuration.
"""

import uvicorn
from app.config import config

if __name__ == "__main__":
    print("Starting Trading Signal Predictor...")
    print(f"Server: http://{config.HOST}:{config.PORT}")
    print(f"Docs: http://{config.HOST}:{config.PORT}/docs")
    print(f"Debug mode: {config.DEBUG}")
    print(f"LSTM enabled: {config.USE_LSTM}")
    
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )
