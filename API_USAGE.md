# Wound-AI API Usage Guide

## Installation

```bash
pip install fastapi uvicorn python-multipart
```

## Starting the Server

From the repository root:

```bash
# Development mode (auto-reload)
uvicorn src.api:app --reload

# Production mode
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Custom checkpoint
CHECKPOINT_PATH=artifacts/checkpoints/my_model.pt uvicorn src.api:app --reload
```

## API Endpoints

### 1. Health Check

Check if the server is running and model is loaded.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "device": "cpu",
  "model_loaded": true,
  "checkpoint": "artifacts/checkpoints/best_efficientnet.pt"
}
```

### 2. Single Image Prediction

Upload an image and get predictions.

```bash
# Basic prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg"

# With artifact saving
curl -X POST "http://localhost:8000/predict?save_artifacts=true" \
  -F "file=@path/to/image.jpg"
```

**Response:**
```json
{
  "top1_class": "P",
  "top1_prob": 0.8734,
  "all_probs": {
    "BG": 0.0063,
    "D": 0.0301,
    "N": 0.0234,
    "P": 0.8734,
    "S": 0.0512,
    "V": 0.0156
  }
}
```

### 3. Batch Prediction

Process a directory of images.

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_dir": "assets/sample_images",
    "report_id": "test_run",
    "batch_size": 16,
    "max_images": 100
  }'
```

**Response:**
```json
{
  "report_dir": "artifacts/inference_reports/test_run",
  "total": 42,
  "successful": 41,
  "failed": 1,
  "files": {
    "jsonl": "artifacts/inference_reports/test_run/predictions.jsonl",
    "csv": "artifacts/inference_reports/test_run/summary.csv",
    "per_image_dir": "artifacts/inference_reports/test_run/per_image"
  },
  "failed_images": [
    {
      "image": "corrupted.jpg",
      "error": "Failed to load image: ..."
    }
  ]
}
```

## Interactive API Docs

Visit http://localhost:8000/docs for interactive Swagger UI documentation.

## Python Client Example

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Single prediction
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())

# Batch prediction
payload = {
    "image_dir": "assets/sample_images",
    "report_id": "my_experiment",
    "batch_size": 32
}
response = requests.post("http://localhost:8000/batch_predict", json=payload)
print(response.json())
```

## Configuration

### Environment Variables

- `CHECKPOINT_PATH`: Path to model checkpoint (default: `artifacts/checkpoints/best_efficientnet.pt`)

### Request Parameters

**Single Prediction (`/predict`):**
- `file`: Image file (multipart/form-data)
- `save_artifacts`: Save prediction JSON (query param, default: false)

**Batch Prediction (`/batch_predict`):**
- `image_dir`: Directory containing images (required)
- `report_id`: Custom report ID (optional, default: timestamp)
- `batch_size`: Inference batch size (default: 16)
- `num_workers`: Preprocessing workers (default: 4)
- `max_images`: Limit number of images (optional)

## Output Formats

### Single Prediction Artifacts
When `save_artifacts=true`:
```
artifacts/api_predictions/<timestamp>/
└── prediction.json
```

### Batch Prediction Reports
```
artifacts/inference_reports/<report_id>/
├── predictions.jsonl       # One prediction per line
├── summary.csv            # Tabular format
└── per_image/             # Individual JSON files
    ├── image1.json
    ├── image2.json
    └── ...
```

## Class Labels

- `BG`: Background
- `D`: D-type wound
- `N`: N-type wound
- `P`: P-type wound
- `S`: S-type wound
- `V`: V-type wound

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid file type, invalid directory, etc.)
- `404`: Resource not found (directory doesn't exist)
- `500`: Server error (prediction failed, model error)
- `503`: Service unavailable (model not loaded)

## Notes

- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- Model is loaded once at startup for optimal performance
- Batch inference uses optimized batching and parallel preprocessing
- All paths are relative to repository root
- CPU-only deployment (no CUDA required)
