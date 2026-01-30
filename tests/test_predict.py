"""Tests for the /predict endpoint."""

import io

from PIL import Image


def create_test_image(width=224, height=224, color=(255, 0, 0)):
    """Create a test PNG image in memory.

    Args:
        width: Image width
        height: Image height
        color: RGB color tuple

    Returns:
        BytesIO object containing PNG image data
    """
    img = Image.new("RGB", (width, height), color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


def test_predict_endpoint_returns_200(client):
    """Test that /predict endpoint returns 200 for valid image."""
    img_bytes = create_test_image()

    response = client.post("/predict", files={"file": ("test_image.png", img_bytes, "image/png")})

    assert response.status_code == 200


def test_predict_endpoint_json_structure(client):
    """Test that /predict endpoint returns expected JSON structure."""
    img_bytes = create_test_image()

    response = client.post("/predict", files={"file": ("test_image.png", img_bytes, "image/png")})

    data = response.json()

    # Check required keys
    assert "top1_class" in data
    assert "top1_prob" in data
    assert "all_probs" in data


def test_predict_endpoint_top1_class_is_valid(client):
    """Test that top1_class is one of the valid classes."""
    img_bytes = create_test_image()

    response = client.post("/predict", files={"file": ("test_image.png", img_bytes, "image/png")})

    data = response.json()
    valid_classes = ["BG", "D", "N", "P", "S", "V"]

    assert data["top1_class"] in valid_classes


def test_predict_endpoint_top1_prob_is_valid(client):
    """Test that top1_prob is a valid probability value."""
    img_bytes = create_test_image()

    response = client.post("/predict", files={"file": ("test_image.png", img_bytes, "image/png")})

    data = response.json()

    # Probability should be between 0 and 1
    assert 0.0 <= data["top1_prob"] <= 1.0


def test_predict_endpoint_all_probs_has_six_classes(client):
    """Test that all_probs contains exactly 6 classes."""
    img_bytes = create_test_image()

    response = client.post("/predict", files={"file": ("test_image.png", img_bytes, "image/png")})

    data = response.json()
    all_probs = data["all_probs"]

    # Should have exactly 6 classes
    assert len(all_probs) == 6

    # Check all expected classes are present
    expected_classes = ["BG", "D", "N", "P", "S", "V"]
    for cls in expected_classes:
        assert cls in all_probs


def test_predict_endpoint_all_probs_sum_to_one(client):
    """Test that all_probs probabilities sum to approximately 1.0."""
    img_bytes = create_test_image()

    response = client.post("/predict", files={"file": ("test_image.png", img_bytes, "image/png")})

    data = response.json()
    all_probs = data["all_probs"]

    # Sum of probabilities should be close to 1.0
    prob_sum = sum(all_probs.values())
    assert abs(prob_sum - 1.0) < 1e-3


def test_predict_endpoint_rejects_invalid_file_type(client):
    """Test that /predict rejects non-image file types."""
    # Create a text file instead of image
    text_bytes = io.BytesIO(b"This is not an image")

    response = client.post("/predict", files={"file": ("test.txt", text_bytes, "text/plain")})

    # Should return 400 Bad Request for invalid file type
    assert response.status_code == 400


def test_predict_endpoint_accepts_jpg(client):
    """Test that /predict accepts JPG images."""
    img = Image.new("RGB", (224, 224), color=(0, 255, 0))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    response = client.post("/predict", files={"file": ("test_image.jpg", img_bytes, "image/jpeg")})

    assert response.status_code == 200


def test_predict_endpoint_accepts_webp(client):
    """Test that /predict accepts WEBP images."""
    img = Image.new("RGB", (224, 224), color=(0, 0, 255))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="WEBP")
    img_bytes.seek(0)

    response = client.post("/predict", files={"file": ("test_image.webp", img_bytes, "image/webp")})

    assert response.status_code == 200


def test_predict_endpoint_deterministic_in_testing_mode(client):
    """Test that predictions are deterministic in testing mode."""
    img_bytes_1 = create_test_image()
    img_bytes_2 = create_test_image()

    response1 = client.post("/predict", files={"file": ("test1.png", img_bytes_1, "image/png")})

    response2 = client.post("/predict", files={"file": ("test2.png", img_bytes_2, "image/png")})

    data1 = response1.json()
    data2 = response2.json()

    # In testing mode, predictions should be identical
    assert data1["top1_class"] == data2["top1_class"]
    assert data1["top1_prob"] == data2["top1_prob"]
    assert data1["all_probs"] == data2["all_probs"]
