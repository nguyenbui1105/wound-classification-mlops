"""FastAPI server for Wound-AI model serving with structured logging.

Improvements over original:
- Structured logging with timestamps
- Request ID tracking
- Performance metrics
- Better error handling
- Environment-based configuration
"""

import logging
import os
import sys
import time
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# =============================================================================
# Configuration from Environment
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Model configuration
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", "artifacts/checkpoints/best_efficientnet.pt"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "true").lower() == "true"

# Inference configuration
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "16"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))

# Testing mode
TESTING_MODE = os.getenv("WOUND_API_TESTING", "0") == "1"

# =============================================================================
# Structured Logging Setup
# =============================================================================

# Create logs directory
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create logger
logger = logging.getLogger("wound-ai-api")
logger.setLevel(getattr(logging, LOG_LEVEL))

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, LOG_LEVEL))
console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler
log_file = LOG_DIR / "api.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(getattr(logging, LOG_LEVEL))
file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


# Custom filter to add request_id
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "no-request-id"
        return True


logger.addFilter(RequestIDFilter())

# =============================================================================
# Import Inference Functions
# =============================================================================

from src.inference import (
    load_model,
    predict,
    preprocess_image,
    run_batch_inference,
    save_batch_report,
    scan_image_directory,
)

# =============================================================================
# Global State
# =============================================================================

MODEL = None
DEVICE = None

# =============================================================================
# Pydantic Models
# =============================================================================


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction."""

    image_dir: str
    report_id: Optional[str] = None
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = NUM_WORKERS
    max_images: Optional[int] = None


class BatchPredictResponse(BaseModel):
    """Response model for batch prediction."""

    report_dir: str
    total: int
    successful: int
    failed: int
    files: dict
    failed_images: list


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    device: str
    model_loaded: bool
    checkpoint: str
    model_version: str
    uptime_seconds: float


# =============================================================================
# Helper Functions
# =============================================================================


def _dummy_predict() -> dict:
    """Dummy predictor for testing mode."""
    return {
        "top1_class": "P",
        "top1_prob": 0.8734,
        "all_probs": {
            "BG": 0.0063,
            "D": 0.0301,
            "N": 0.0234,
            "P": 0.8734,
            "S": 0.0512,
            "V": 0.0156,
        },
    }


def _dummy_batch_predict(image_files: list) -> tuple[list[dict], list]:
    """Dummy batch predictor for testing mode."""
    predictions = []
    for img_path in image_files:
        pred = _dummy_predict()
        pred["image"] = str(img_path)
        predictions.append(pred)
    failures = []
    return predictions, failures


def load_model_once():
    """Load model once at startup with logging."""
    global MODEL, DEVICE, CHECKPOINT_PATH

    # Testing mode
    if TESTING_MODE:
        logger.info("TESTING MODE ENABLED: Using dummy model", extra={"request_id": "startup"})
        DEVICE = torch.device("cpu")
        MODEL = "DUMMY_MODEL_FOR_TESTING"
        return

    # Device detection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device detected: {DEVICE}", extra={"request_id": "startup"})

    # Load model
    try:
        logger.info(f"Loading model from: {CHECKPOINT_PATH}", extra={"request_id": "startup"})
        start_time = time.time()

        MODEL = load_model(CHECKPOINT_PATH, DEVICE)

        load_time = time.time() - start_time
        logger.info(
            f"Model loaded successfully in {load_time:.2f}s", extra={"request_id": "startup"}
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}", extra={"request_id": "startup"}, exc_info=True)
        raise


# =============================================================================
# FastAPI App
# =============================================================================

# Track startup time
STARTUP_TIME = datetime.now()

app = FastAPI(
    title="Wound-AI API",
    description="Production API for wound classification inference with structured logging",
    version=MODEL_VERSION,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
)

# =============================================================================
# Middleware
# =============================================================================


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add unique request ID to all requests for tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Log incoming request
    logger.info(
        f"Incoming request: {request.method} {request.url.path}", extra={"request_id": request_id}
    )

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # Log response
    logger.info(
        f"Response: {response.status_code} in {duration:.3f}s", extra={"request_id": request_id}
    )

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration:.3f}s"

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with logging."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"Unhandled exception: {str(exc)}", extra={"request_id": request_id}, exc_info=True
    )

    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "request_id": request_id}
    )


# =============================================================================
# Startup & Shutdown Events
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("=== Wound-AI API Starting ===", extra={"request_id": "startup"})
    logger.info(f"Python version: {sys.version}", extra={"request_id": "startup"})
    logger.info(f"PyTorch version: {torch.__version__}", extra={"request_id": "startup"})
    logger.info(f"Log level: {LOG_LEVEL}", extra={"request_id": "startup"})

    load_model_once()

    logger.info("=== Wound-AI API Ready ===", extra={"request_id": "startup"})


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("=== Wound-AI API Shutting Down ===", extra={"request_id": "shutdown"})


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint with detailed status."""
    uptime = (datetime.now() - STARTUP_TIME).total_seconds()

    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "checkpoint": str(CHECKPOINT_PATH),
        "model_version": MODEL_VERSION,
        "uptime_seconds": uptime,
    }


@app.post("/predict")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    save_artifacts: bool = False,
):
    """Predict on a single uploaded image with logging."""
    request_id = request.state.request_id

    # Validate model is loaded
    if MODEL is None:
        logger.error("Model not loaded", extra={"request_id": request_id})
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    allowed_exts = [".jpg", ".jpeg", ".png", ".webp"]

    if file_ext not in allowed_exts:
        logger.warning(
            f"Invalid file type: {file_ext} for file: {file.filename}",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=400, detail=f"Invalid file type: {file_ext}. Allowed: {allowed_exts}"
        )

    logger.info(f"Processing image: {file.filename}", extra={"request_id": request_id})

    try:
        start_time = time.time()

        # Read image
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Save to temp file
        temp_dir = ROOT / "artifacts" / "api_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / file.filename
        image.save(temp_path)

        # Preprocess
        image_tensor = preprocess_image(temp_path)

        # Predict
        if TESTING_MODE:
            prediction = _dummy_predict()
        else:
            prediction = predict(MODEL, image_tensor, DEVICE)

        inference_time = time.time() - start_time

        logger.info(
            f"Prediction successful: {prediction['top1_class']} "
            f"({prediction['top1_prob']:.4f}) in {inference_time:.3f}s",
            extra={"request_id": request_id},
        )

        # Save artifacts if requested
        if save_artifacts:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = ROOT / "artifacts" / "api_predictions" / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)

            import json

            prediction_with_image = {
                "image": file.filename,
                "top1_class": prediction["top1_class"],
                "top1_prob": round(prediction["top1_prob"], 4),
                "all_probs": {k: round(v, 4) for k, v in prediction["all_probs"].items()},
            }

            output_path = output_dir / "prediction.json"
            with open(output_path, "w") as f:
                json.dump(prediction_with_image, f, indent=2)

            logger.info(f"Artifacts saved to: {output_path}", extra={"request_id": request_id})
            prediction["artifacts_saved"] = str(output_path)

        # Cleanup temp file
        temp_path.unlink(missing_ok=True)

        return {
            "top1_class": prediction["top1_class"],
            "top1_prob": round(prediction["top1_prob"], 4),
            "all_probs": {k: round(v, 4) for k, v in prediction["all_probs"].items()},
        }

    except Exception as e:
        logger.error(
            f"Prediction failed: {str(e)}", extra={"request_id": request_id}, exc_info=True
        )

        # Cleanup temp file on error
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)

        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(request: Request, req: BatchPredictRequest):
    """Run batch prediction with logging."""
    request_id = request.state.request_id

    # Validate model is loaded
    if MODEL is None:
        logger.error("Model not loaded", extra={"request_id": request_id})
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate directory
    image_dir = Path(req.image_dir)
    if not image_dir.exists():
        logger.error(f"Directory not found: {image_dir}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=404, detail=f"Image directory not found: {image_dir.absolute()}"
        )

    if not image_dir.is_dir():
        logger.error(f"Not a directory: {image_dir}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=f"Not a directory: {image_dir}")

    logger.info(f"Starting batch prediction on: {image_dir}", extra={"request_id": request_id})

    try:
        start_time = time.time()

        # Setup report directory
        if req.report_id:
            report_id = req.report_id
        else:
            report_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_dir = ROOT / "artifacts" / "inference_reports" / report_id

        # Scan for images
        try:
            image_files = scan_image_directory(image_dir, max_images=req.max_images)
            logger.info(f"Found {len(image_files)} images", extra={"request_id": request_id})
        except ValueError as e:
            logger.error(f"Scan failed: {str(e)}", extra={"request_id": request_id})
            raise HTTPException(status_code=400, detail=str(e))

        # Run batch inference
        if TESTING_MODE:
            predictions, failures = _dummy_batch_predict(image_files)
        else:
            predictions, failures = run_batch_inference(
                MODEL,
                image_files,
                DEVICE,
                batch_size=req.batch_size,
                num_workers=req.num_workers,
            )

        # Save report
        report_files = save_batch_report(predictions, report_dir)

        batch_time = time.time() - start_time

        logger.info(
            f"Batch prediction complete: {len(predictions)} successful, "
            f"{len(failures)} failed in {batch_time:.2f}s",
            extra={"request_id": request_id},
        )

        # Format failed images
        failed_images_list = [
            {"image": str(img_path), "error": str(error)} for img_path, error in failures
        ]

        return BatchPredictResponse(
            report_dir=str(report_dir),
            total=len(image_files),
            successful=len(predictions),
            failed=len(failures),
            files={
                "jsonl": str(report_files["jsonl"]),
                "csv": str(report_files["csv"]),
                "per_image_dir": str(report_files["per_image_dir"]),
            },
            failed_images=failed_images_list,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Batch prediction failed: {str(e)}", extra={"request_id": request_id}, exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Wound-AI API server...", extra={"request_id": "main"})
    logger.info(f"API docs at: http://{API_HOST}:{API_PORT}/docs", extra={"request_id": "main"})

    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level=LOG_LEVEL.lower())
