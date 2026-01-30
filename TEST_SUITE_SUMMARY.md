# Test Suite Implementation Summary

This document summarizes the changes made to add a comprehensive pytest test suite for the Wound-AI FastAPI server.

## Changes Made

### 1. Modified Files

#### `src/api.py` (minimal changes)

**Added:**
- `TESTING_MODE` flag that reads `WOUND_API_TESTING` environment variable
- `_dummy_predict()` function: Returns deterministic predictions for testing
- `_dummy_batch_predict()` function: Returns dummy batch predictions for testing

**Modified:**
- `load_model_once()`: Skips real model loading when `TESTING_MODE=True`
- `predict_image()` endpoint: Uses `_dummy_predict()` in testing mode
- `batch_predict()` endpoint: Uses `_dummy_batch_predict()` in testing mode

**Key feature**: Testing mode allows tests to run without loading the heavy model checkpoint.

### 2. New Files Created

#### Test Suite (`tests/`)

```
tests/
├── __init__.py                 # Package marker
├── conftest.py                 # Pytest configuration, sets WOUND_API_TESTING=1
├── test_health.py              # 6 tests for /health endpoint
├── test_predict.py             # 11 tests for /predict endpoint
└── test_batch_predict.py       # 13 tests for /batch_predict endpoint
```

**Total: 30 comprehensive tests**

#### Configuration Files

- `pytest.ini` - Pytest configuration (test discovery, output options)
- `requirements-dev.txt` - Development dependencies (pytest, pytest-cov)

#### Documentation

- `TESTING.md` - Complete testing guide with examples and troubleshooting
- `TEST_SUITE_SUMMARY.md` - This file (implementation summary)

## Test Coverage

### `/health` Endpoint (6 tests)
- ✅ Status code 200
- ✅ JSON structure validation
- ✅ Status field = "ok"
- ✅ Model loaded flag = True
- ✅ Device = CPU in testing mode
- ✅ Checkpoint path exists

### `/predict` Endpoint (11 tests)
- ✅ Status code 200 for valid images
- ✅ JSON structure (top1_class, top1_prob, all_probs)
- ✅ Top1 class is valid (BG/D/N/P/S/V)
- ✅ Top1 probability in [0, 1]
- ✅ All probs has 6 classes
- ✅ All probs sum to ~1.0
- ✅ Rejects invalid file types (400)
- ✅ Accepts JPG, PNG, WEBP
- ✅ Deterministic predictions in testing mode
- ✅ Multiple image formats supported

### `/batch_predict` Endpoint (13 tests)
- ✅ Status code 200 for valid directory
- ✅ JSON structure validation
- ✅ Correct counts (total, successful, failed)
- ✅ Creates report files (JSONL, CSV, per_image/)
- ✅ Per-image JSONs created for each image
- ✅ JSONL format validation
- ✅ CSV format validation with headers
- ✅ Rejects non-existent directory (404)
- ✅ Custom batch_size parameter
- ✅ max_images parameter
- ✅ Empty failed_images on success
- ✅ Custom report_id in output path
- ✅ Multiple images processed correctly

## Quick Start

### Installation

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
# From repository root
pytest

# Verbose output
pytest -v

# Quiet mode
pytest -q
```

### Run Specific Tests

```bash
# Test a specific endpoint
pytest tests/test_health.py

# Test a specific function
pytest tests/test_predict.py::test_predict_endpoint_returns_200

# Test by pattern
pytest -k "batch"
```

### Coverage Report

```bash
# Terminal coverage report
pytest --cov=src --cov-report=term-missing

# HTML coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

## How Testing Mode Works

### Architecture

1. **Environment Variable**: `WOUND_API_TESTING=1` is set by `conftest.py` before importing the API
2. **Conditional Loading**: `load_model_once()` detects testing mode and skips real model loading
3. **Dummy Predictors**: Lightweight functions return deterministic outputs
4. **No Side Effects**: Normal API operation unchanged (env var only set in tests)

### Benefits

- ✅ **Fast**: No model loading (~30s saved per test run)
- ✅ **Deterministic**: Predictable outputs for reliable tests
- ✅ **No Dependencies**: No checkpoint file required
- ✅ **Isolated**: Testing mode only active during test runs
- ✅ **Windows-Compatible**: Uses tempfile and pathlib for cross-platform support

### Example: Dummy Predictor

```python
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
        }
    }
```

Probabilities sum to 1.0 and are deterministic for test assertions.

## Windows PowerShell Usage

All commands work in Windows PowerShell:

```powershell
# Install dependencies
pip install -r requirements-dev.txt

# Run tests
pytest -q

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests\test_health.py
```

## CI/CD Integration Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          pip install -r requirements_docker.txt

      - name: Run tests with coverage
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
```

## File Size Impact

**Minimal footprint:**
- Modified: `src/api.py` (+45 lines)
- New tests: ~450 lines total
- New docs: ~200 lines total
- Dependencies: +2 packages (pytest, pytest-cov)

**Total addition: ~700 lines, 0 breaking changes**

## Design Principles

1. **Minimal Changes**: Only essential modifications to `src/api.py`
2. **No Refactoring**: Existing code structure preserved
3. **Clean Separation**: Testing logic isolated from production code
4. **Windows-First**: All paths and commands Windows-compatible
5. **Production-Safe**: Testing mode only active when env var set
6. **Comprehensive**: 30 tests covering all endpoints and edge cases

## Next Steps

### Optional Enhancements

1. **Integration Tests**: Test with real model checkpoint (marked with `@pytest.mark.slow`)
2. **Performance Tests**: Benchmark inference throughput
3. **Load Tests**: Use locust or pytest-benchmark for stress testing
4. **Security Tests**: SQL injection, XSS, file upload validation
5. **Docker Tests**: Test containerized API

### Example: Add Slow Tests

```python
# tests/test_integration.py
import pytest

@pytest.mark.slow
def test_real_model_prediction(client):
    \"\"\"Integration test with real model (slow).\"\"\"
    # Set WOUND_API_TESTING=0 or load real model
    # ...
```

Run slow tests:
```bash
pytest -m slow
```

Skip slow tests:
```bash
pytest -m "not slow"
```

## Troubleshooting

### "Model not loaded" Error
- Ensure `conftest.py` exists in `tests/` directory
- Run `pytest` from repository root
- Check that `WOUND_API_TESTING` is set before API import

### Import Errors
- Ensure project root in `sys.path`
- Run from repository root: `cd c:\Users\nguye\Documents\mlops-wound-ai`
- Check that `src/` directory exists with `__init__.py`

### Tests Pass but API Fails
- Testing mode uses dummy predictors
- To test real API, run with `WOUND_API_TESTING=0` or no env var
- Use integration tests with real checkpoint for E2E validation

## Summary

✅ **30 comprehensive tests** covering all API endpoints
✅ **Testing mode** avoids heavy model loading
✅ **Zero breaking changes** to existing API
✅ **Windows-compatible** using pathlib and tempfile
✅ **Production-ready** with pytest configuration
✅ **Well-documented** with TESTING.md guide

The test suite is ready to use! Run `pytest -q` to verify all tests pass.
