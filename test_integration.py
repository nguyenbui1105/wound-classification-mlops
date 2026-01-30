"""Integration tests with real model (slow, requires checkpoint).

Run with:
    pytest tests/test_integration.py -v

Skip slow tests:
    pytest -m "not slow"
"""

import io
import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

# Mark all tests in this file as slow
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def integration_client():
    """Create a test client with real model (slow to initialize)."""
    # Disable testing mode
    os.environ["WOUND_API_TESTING"] = "0"
    
    # Import after setting env var
    from fastapi.testclient import TestClient

    from src.api import app
    
    with TestClient(app) as client:
        yield client
    
    # Re-enable testing mode after tests
    os.environ["WOUND_API_TESTING"] = "1"


def create_test_image(width=224, height=224, color=(255, 0, 0)):
    """Create a test PNG image in memory."""
    img = Image.new("RGB", (width, height), color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


@pytest.mark.integration
def test_real_model_health_check(integration_client):
    """Test health endpoint with real model."""
    response = integration_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    # Model should be loaded
    assert data["model_loaded"] is True
    assert data["status"] == "ok"


@pytest.mark.integration
def test_real_model_prediction(integration_client):
    """Test prediction with real model."""
    img_bytes = create_test_image()
    
    response = integration_client.post(
        "/predict",
        files={"file": ("test_image.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check structure
    assert "top1_class" in data
    assert "top1_prob" in data
    assert "all_probs" in data
    
    # Validate predictions
    valid_classes = ["BG", "D", "N", "P", "S", "V"]
    assert data["top1_class"] in valid_classes
    assert 0.0 <= data["top1_prob"] <= 1.0
    
    # All probabilities should sum to ~1.0
    prob_sum = sum(data["all_probs"].values())
    assert abs(prob_sum - 1.0) < 1e-3


@pytest.mark.integration
def test_real_model_predictions_are_deterministic(integration_client):
    """Test that real model gives consistent predictions for same input."""
    # Create identical images
    img_bytes_1 = create_test_image(color=(128, 128, 128))
    img_bytes_2 = create_test_image(color=(128, 128, 128))
    
    response1 = integration_client.post(
        "/predict",
        files={"file": ("test1.png", img_bytes_1, "image/png")}
    )
    
    response2 = integration_client.post(
        "/predict",
        files={"file": ("test2.png", img_bytes_2, "image/png")}
    )
    
    data1 = response1.json()
    data2 = response2.json()
    
    # Predictions should be identical for identical inputs
    assert data1["top1_class"] == data2["top1_class"]
    assert abs(data1["top1_prob"] - data2["top1_prob"]) < 1e-4


@pytest.mark.integration
def test_real_model_batch_prediction(integration_client):
    """Test batch prediction with real model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create test images
        for i in range(3):
            img = Image.new("RGB", (224, 224), color=(i * 50, i * 80, i * 30))
            img.save(tmpdir_path / f"test_image_{i}.png")
        
        response = integration_client.post(
            "/batch_predict",
            json={
                "image_dir": str(tmpdir_path),
                "report_id": "integration_test",
                "batch_size": 16
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check counts
        assert data["total"] == 3
        assert data["successful"] == 3
        assert data["failed"] == 0
        
        # Check report files exist
        report_dir = Path(data["report_dir"])
        assert report_dir.exists()
        
        jsonl_path = Path(data["files"]["jsonl"])
        assert jsonl_path.exists()
        
        csv_path = Path(data["files"]["csv"])
        assert csv_path.exists()


@pytest.mark.integration
def test_real_model_inference_time(integration_client):
    """Test that inference completes within reasonable time."""
    import time
    
    img_bytes = create_test_image()
    
    start = time.time()
    response = integration_client.post(
        "/predict",
        files={"file": ("test_image.png", img_bytes, "image/png")}
    )
    duration = time.time() - start
    
    assert response.status_code == 200
    
    # Should complete in under 5 seconds on CPU
    assert duration < 5.0


@pytest.mark.integration
def test_real_model_handles_multiple_formats(integration_client):
    """Test that real model handles different image formats."""
    formats = [
        ("JPEG", "test.jpg", "image/jpeg"),
        ("PNG", "test.png", "image/png"),
        ("WEBP", "test.webp", "image/webp"),
    ]
    
    for fmt, filename, mime_type in formats:
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=fmt)
        img_bytes.seek(0)
        
        response = integration_client.post(
            "/predict",
            files={"file": (filename, img_bytes, mime_type)}
        )
        
        assert response.status_code == 200, f"Failed for format {fmt}"
        data = response.json()
        assert "top1_class" in data


@pytest.mark.integration
def test_real_model_handles_different_sizes(integration_client):
    """Test that real model handles images of different sizes."""
    sizes = [
        (224, 224),   # Exact model input size
        (512, 512),   # Larger
        (128, 128),   # Smaller
        (300, 200),   # Non-square
    ]
    
    for width, height in sizes:
        img = Image.new("RGB", (width, height), color=(100, 100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        response = integration_client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 200, f"Failed for size {width}x{height}"


@pytest.mark.integration
@pytest.mark.parametrize("color", [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 255),  # White
    (0, 0, 0),        # Black
])
def test_real_model_handles_different_colors(integration_client, color):
    """Test that real model handles images with different colors."""
    img_bytes = create_test_image(color=color)
    
    response = integration_client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # All colors should produce valid predictions
    assert data["top1_class"] in ["BG", "D", "N", "P", "S", "V"]
    assert 0.0 <= data["top1_prob"] <= 1.0


@pytest.mark.integration
def test_real_model_checkpoint_exists():
    """Test that model checkpoint file exists."""
    checkpoint_path = Path(os.getenv(
        "CHECKPOINT_PATH",
        "artifacts/checkpoints/best_efficientnet.pt"
    ))
    
    # This test will fail if checkpoint doesn't exist
    # That's expected - it guides users to either train or download the model
    if not checkpoint_path.exists():
        pytest.skip(f"Model checkpoint not found at: {checkpoint_path}")
    
    assert checkpoint_path.is_file()
    assert checkpoint_path.suffix == ".pt"
    
    # Check file size (should be at least 10MB for EfficientNet-B0)
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    assert file_size_mb > 10, f"Checkpoint file seems too small: {file_size_mb:.2f} MB"


@pytest.mark.integration
def test_real_model_memory_usage():
    """Test that model doesn't consume excessive memory."""
    import os

    import psutil
    
    # Disable testing mode temporarily
    original_mode = os.getenv("WOUND_API_TESTING", "0")
    os.environ["WOUND_API_TESTING"] = "0"
    
    try:
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Import and load model
        from src.api import load_model_once
        load_model_once()
        
        # Get memory after loading
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory
        
        # Model should not use more than 500MB on CPU
        assert memory_increase < 500, \
            f"Model uses too much memory: {memory_increase:.2f} MB"
    
    finally:
        # Restore testing mode
        os.environ["WOUND_API_TESTING"] = original_mode


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
