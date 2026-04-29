from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import routes and services
from app.api.routes import router, init_services
from app.config import config

# Create FastAPI app
app = FastAPI(
    title="Trading Signal Predictor",
    description="AI-powered stock signal prediction system with stock-specific LSTM classifiers and optional chart-pattern integration",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_URL, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routes
app.include_router(router)

dashboard_dir = os.path.abspath(config.DASHBOARD_DIR)
if os.path.isdir(dashboard_dir):
    app.mount("/static", StaticFiles(directory=dashboard_dir), name="static")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on app startup."""
    logger.info("Trading Signal Predictor starting up...")
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Initialize all services
        init_services()
        logger.info("Application ready to serve stock predictions")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on app shutdown."""
    logger.info("Trading Signal Predictor shutting down...")

@app.get("/", include_in_schema=True)
async def root():
    """API info endpoint (JSON)."""
    return await api_info()


@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    """Serve the built-in frontend dashboard (alias for root)."""
    index_path = os.path.join(dashboard_dir, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Dashboard assets not found."}
        )
    return FileResponse(index_path)


@app.get("/api", include_in_schema=True)
async def api_info():
    """API documentation endpoint."""
    return {
        "message": "Trading Signal Predictor API",
        "description": "AI-powered stock trading signal prediction system",
        "version": "3.0.0",
        "endpoints": {
            "predict": "/api/predict",
            "market_data": "/api/market-data",
            "training_preview": "/api/training/preview",
            "training_prepare": "/api/training/prepare",
            "training_run": "/api/training/run",
            "dashboard": "/dashboard",
            "health": "/api/health",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "features": [
            "Stock-specific LSTM sequence analysis",
            "Optional chart-pattern module when weights are provided",
            "yfinance stock market data integration",
            "Deterministic OHLCV cache",
            "Single-stock and multi-stock training workflows",
            "Saved .h5 model artifacts",
            "Built-in dashboard"
        ]
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected errors gracefully."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc) if config.DEBUG else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )
