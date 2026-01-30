"""FastAPI server for Wound-AI model serving.

How to run:
  1. Install dependencies:
     pip install fastapi uvicorn python-multipart

  2. Start server from repo root:
     uvicorn src.api:app --reload

  3. Access API:
     - Docs: http://localhost:8000/docs
     - Health: http://localhost:8000/health
"""

import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Import inference functions
from src.inference import (
    load_model,
    predict,
    preprocess_image,
    run_batch_inference,
    save_batch_report,
    scan_image_directory,
)

# Global model state
MODEL = None
DEVICE = None
CHECKPOINT_PATH = None
TESTING_MODE = os.getenv("WOUND_API_TESTING", "0") == "1"


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction."""

    image_dir: str
    report_id: Optional[str] = None
    batch_size: int = 16
    num_workers: int = 4
    max_images: Optional[int] = None


class BatchPredictResponse(BaseModel):
    """Response model for batch prediction."""

    report_dir: str
    total: int
    successful: int
    failed: int
    files: dict
    failed_images: list


def _dummy_predict() -> dict:
    """Dummy predictor for testing mode. Returns deterministic outputs."""
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
    """Load model once at startup."""
    global MODEL, DEVICE, CHECKPOINT_PATH

    # Testing mode: skip real model loading
    if TESTING_MODE:
        print("[Startup] TESTING MODE: Using dummy model")
        DEVICE = torch.device("cpu")
        CHECKPOINT_PATH = Path("artifacts/checkpoints/test_dummy.pt")
        MODEL = "DUMMY_MODEL_FOR_TESTING"
        return

    # Get checkpoint path from env or use default
    CHECKPOINT_PATH = Path(
        os.getenv("CHECKPOINT_PATH", "artifacts/checkpoints/best_efficientnet.pt")
    )

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Startup] Device: {DEVICE}")

    # Load model
    print(f"[Startup] Loading model from: {CHECKPOINT_PATH}")
    MODEL = load_model(CHECKPOINT_PATH, DEVICE)
    print("[Startup] Model loaded successfully")


# Create FastAPI app
app = FastAPI(
    title="Wound-AI API",
    description="Production API for wound classification inference",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model_once()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "checkpoint": str(CHECKPOINT_PATH),
    }


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    save_artifacts: bool = False,
):
    """Predict on a single uploaded image.

    Args:
        file: Image file upload (multipart/form-data)
        save_artifacts: Whether to save prediction artifacts (default: False)

    Returns:
        JSON with top1_class, top1_prob, all_probs
    """
    # Validate model is loaded
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    allowed_exts = [".jpg", ".jpeg", ".png", ".webp"]
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400, detail=f"Invalid file type: {file_ext}. Allowed: {allowed_exts}"
        )

    try:
        # Read image from upload
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Save to temp file for preprocessing (inference.py expects Path)
        temp_dir = ROOT / "artifacts" / "api_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / file.filename
        image.save(temp_path)

        # Preprocess
        image_tensor = preprocess_image(temp_path)

        # Predict (use dummy predictor in testing mode)
        if TESTING_MODE:
            prediction = _dummy_predict()
        else:
            prediction = predict(MODEL, image_tensor, DEVICE)

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

            prediction["artifacts_saved"] = str(output_path)

        # Clean up temp file
        temp_path.unlink(missing_ok=True)

        return {
            "top1_class": prediction["top1_class"],
            "top1_prob": round(prediction["top1_prob"], 4),
            "all_probs": {k: round(v, 4) for k, v in prediction["all_probs"].items()},
        }

    except Exception as e:
        # Clean up temp file on error
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    """Run batch prediction on a directory of images.

    Args:
        request: BatchPredictRequest with image_dir, report_id, etc.

    Returns:
        BatchPredictResponse with report summary
    """
    # Validate model is loaded
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate directory
    image_dir = Path(request.image_dir)
    if not image_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"Image directory not found: {image_dir.absolute()}"
        )
    if not image_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {image_dir}")

    try:
        # Setup report directory
        if request.report_id:
            report_id = request.report_id
        else:
            report_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_dir = ROOT / "artifacts" / "inference_reports" / report_id

        # Scan for images
        try:
            image_files = scan_image_directory(image_dir, max_images=request.max_images)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Run batch inference (use dummy predictor in testing mode)
        if TESTING_MODE:
            predictions, failures = _dummy_batch_predict(image_files)
        else:
            predictions, failures = run_batch_inference(
                MODEL,
                image_files,
                DEVICE,
                batch_size=request.batch_size,
                num_workers=request.num_workers,
            )

        # Save report
        report_files = save_batch_report(predictions, report_dir)

        # Format failed images for response
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
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("Starting Wound-AI API server...")
    print("Access API docs at: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
