# Testing Guide

This document explains how to run the test suite for the Wound-AI API.

## Test Structure

The test suite includes comprehensive tests for all API endpoints:

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and fixtures
├── test_health.py           # Tests for /health endpoint
├── test_predict.py          # Tests for /predict endpoint (single image)
└── test_batch_predict.py    # Tests for /batch_predict endpoint (batch)
```

## Setup

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

This installs:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting

## Running Tests

### Run all tests

```bash
# From repository root
pytest

# Or more verbose
pytest -v

# Quiet mode (minimal output)
pytest -q
```

### Run specific test files

```bash
# Test only the health endpoint
pytest tests/test_health.py

# Test only single prediction
pytest tests/test_predict.py

# Test only batch prediction
pytest tests/test_batch_predict.py
```

### Run specific test functions

```bash
# Run a single test by name
pytest tests/test_health.py::test_health_endpoint_returns_200

# Run tests matching a pattern
pytest -k "batch"  # Runs all tests with "batch" in the name
```

### Coverage reporting

```bash
# Run tests with coverage report
pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=src --cov-report=html
```

## Testing Mode

The test suite uses a special **testing mode** that avoids loading the real model:

- Environment variable `WOUND_API_TESTING=1` is set automatically by `conftest.py`
- Model loading is skipped at startup
- A lightweight dummy predictor returns deterministic outputs
- Tests run quickly without requiring the actual checkpoint file

### Testing Mode Implementation

In [src/api.py](src/api.py):
- `TESTING_MODE` flag checks for `WOUND_API_TESTING=1` environment variable
- `load_model_once()` skips real model loading in testing mode
- `_dummy_predict()` returns deterministic predictions
- `_dummy_batch_predict()` generates dummy batch predictions

This design ensures:
- Tests run fast (no heavy model loading)
- Tests are deterministic (predictable outputs)
- No dependency on checkpoint files
- Normal API operation is unaffected

## Test Coverage

### Health Endpoint (`/health`)
- ✅ Returns 200 status code
- ✅ Returns correct JSON structure
- ✅ Status is "ok"
- ✅ Model loaded flag is True
- ✅ Device is CPU in testing mode
- ✅ Checkpoint path is returned

### Single Prediction (`/predict`)
- ✅ Returns 200 for valid images
- ✅ Returns correct JSON structure
- ✅ Top1 class is valid (BG, D, N, P, S, V)
- ✅ Top1 probability is in [0, 1]
- ✅ All probabilities contain 6 classes
- ✅ All probabilities sum to ~1.0
- ✅ Rejects invalid file types (400)
- ✅ Accepts JPG, PNG, WEBP formats
- ✅ Predictions are deterministic in testing mode

### Batch Prediction (`/batch_predict`)
- ✅ Returns 200 for valid directory
- ✅ Returns correct JSON structure
- ✅ Counts are correct (total, successful, failed)
- ✅ Creates all report files (JSONL, CSV, per-image JSONs)
- ✅ Per-image directory contains JSON for each image
- ✅ JSONL format is valid
- ✅ CSV format is valid with headers
- ✅ Rejects non-existent directories (404)
- ✅ Respects custom batch_size parameter
- ✅ Respects max_images parameter
- ✅ Failed images list is empty on success
- ✅ Custom report_id appears in output path

## Windows-Specific Notes

The test suite is fully compatible with Windows PowerShell:

1. **Temporary directories**: Tests use `tempfile.TemporaryDirectory()` which handles Windows paths correctly

2. **Path separators**: All paths use `pathlib.Path` for cross-platform compatibility

3. **Running tests**:
   ```powershell
   # PowerShell
   pytest -q

   # Or with coverage
   pytest --cov=src --cov-report=term-missing
   ```

## Continuous Integration

To add these tests to CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Install dependencies
  run: |
    pip install -r requirements-dev.txt
    pip install -r requirements_docker.txt

- name: Run tests
  run: pytest --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Import errors

If you see import errors, ensure you're running from the repository root:

```bash
cd c:\Users\nguye\Documents\mlops-wound-ai
pytest
```

### Tests fail with "Model not loaded"

This means `WOUND_API_TESTING` was not set before importing the API. Check that:
1. `conftest.py` is in the `tests/` directory
2. You're running `pytest` (not `python -m pytest` which may skip conftest)

### Port already in use

Tests use `TestClient` which doesn't bind to a real port, so this shouldn't occur. If it does, check for running API instances:

```powershell
# Windows
netstat -ano | findstr :8000
```

## Adding New Tests

To add new tests:

1. Create a new test file in `tests/` following the `test_*.py` naming pattern
2. Import the `client` fixture from `conftest.py`
3. Write test functions starting with `test_`
4. Use descriptive test names that explain what is being tested

Example:

```python
def test_new_endpoint_feature(client):
    \"\"\"Test description here.\"\"\"
    response = client.get("/new_endpoint")
    assert response.status_code == 200
```

## Running Tests Without pytest

If you need to run tests without pytest:

```python
# Not recommended, but possible
import sys
sys.path.insert(0, ".")

import os
os.environ["WOUND_API_TESTING"] = "1"

from tests.conftest import client

# Now run test functions manually
```

However, using `pytest` is **strongly recommended** for proper test discovery, fixtures, and reporting.
