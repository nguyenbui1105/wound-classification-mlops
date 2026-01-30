"""Tests for the /batch_predict endpoint."""

import json
import tempfile
from pathlib import Path

from PIL import Image


def create_test_images_in_dir(directory: Path, count: int = 3):
    """Create test PNG images in a directory.

    Args:
        directory: Directory to create images in
        count: Number of images to create

    Returns:
        List of created image paths
    """
    image_paths = []
    for i in range(count):
        img = Image.new("RGB", (224, 224), color=(i * 50, i * 80, i * 30))
        img_path = directory / f"test_image_{i}.png"
        img.save(img_path)
        image_paths.append(img_path)

    return image_paths


def test_batch_predict_endpoint_returns_200(client):
    """Test that /batch_predict endpoint returns 200 for valid directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_images_in_dir(tmpdir_path, count=3)

        response = client.post(
            "/batch_predict",
            json={"image_dir": str(tmpdir_path), "report_id": "pytest_run", "batch_size": 16},
        )

        assert response.status_code == 200


def test_batch_predict_endpoint_json_structure(client):
    """Test that /batch_predict endpoint returns expected JSON structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_images_in_dir(tmpdir_path, count=3)

        response = client.post(
            "/batch_predict",
            json={"image_dir": str(tmpdir_path), "report_id": "pytest_run", "batch_size": 16},
        )

        data = response.json()

        # Check required keys
        assert "report_dir" in data
        assert "total" in data
        assert "successful" in data
        assert "failed" in data
        assert "files" in data
        assert "failed_images" in data


def test_batch_predict_endpoint_counts_correct(client):
    """Test that /batch_predict returns correct counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_images_in_dir(tmpdir_path, count=3)

        response = client.post(
            "/batch_predict",
            json={"image_dir": str(tmpdir_path), "report_id": "pytest_run", "batch_size": 16},
        )

        data = response.json()

        assert data["total"] == 3
        assert data["successful"] == 3
        assert data["failed"] == 0


def test_batch_predict_endpoint_creates_report_files(client):
    """Test that /batch_predict creates all expected report files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_images_in_dir(tmpdir_path, count=3)

        response = client.post(
            "/batch_predict",
            json={"image_dir": str(tmpdir_path), "report_id": "pytest_run", "batch_size": 16},
        )

        data = response.json()
        files = data["files"]

        # Check that files dict has expected keys
        assert "jsonl" in files
        assert "csv" in files
        assert "per_image_dir" in files

        # Check that files actually exist
        report_dir = Path(data["report_dir"])
        assert report_dir.exists()

        jsonl_path = Path(files["jsonl"])
        assert jsonl_path.exists()

        csv_path = Path(files["csv"])
        assert csv_path.exists()

        per_image_dir = Path(files["per_image_dir"])
        assert per_image_dir.exists()
        assert per_image_dir.is_dir()


def test_batch_predict_endpoint_creates_per_image_jsons(client):
    """Test that /batch_predict creates individual JSON files for each image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_images_in_dir(tmpdir_path, count=3)

        response = client.post(
            "/batch_predict",
            json={"image_dir": str(tmpdir_path), "report_id": "pytest_run", "batch_size": 16},
        )

        data = response.json()
        per_image_dir = Path(data["files"]["per_image_dir"])

        # Count JSON files in per_image directory
        json_files = list(per_image_dir.glob("*.json"))
        assert len(json_files) == 3


def test_batch_predict_endpoint_jsonl_format(client):
    """Test that predictions.jsonl has correct format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_images_in_dir(tmpdir_path, count=3)

        response = client.post(
            "/batch_predict",
            json={"image_dir": str(tmpdir_path), "report_id": "pytest_run", "batch_size": 16},
        )

        data = response.json()
        jsonl_path = Path(data["files"]["jsonl"])

        # Read and parse JSONL
        with open(jsonl_path, "r") as f:
            lines = f.readlines()

        # Should have 3 lines (one per image)
        assert len(lines) == 3

        # Each line should be valid JSON
        for line in lines:
            pred = json.loads(line)
            assert "image" in pred
            assert "top1_class" in pred
            assert "top1_prob" in pred
            assert "all_probs" in pred


def test_batch_predict_endpoint_csv_format(client):
    """Test that summary.csv has correct format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_images_in_dir(tmpdir_path, count=3)

        response = client.post(
            "/batch_predict",
            json={"image_dir": str(tmpdir_path), "report_id": "pytest_run", "batch_size": 16},
        )

        data = response.json()
        csv_path = Path(data["files"]["csv"])

        # Read CSV
        with open(csv_path, "r") as f:
            lines = f.readlines()

        # Should have header + 3 data rows
        assert len(lines) == 4

        # Check header contains expected columns
        header = lines[0].strip()
        assert "image" in header
        assert "top1_class" in header
        assert "top1_prob" in header
        assert "prob_BG" in header
        assert "prob_P" in header


def test_batch_predict_endpoint_rejects_nonexistent_directory(client):
    """Test that /batch_predict rejects non-existent directories."""
    response = client.post(
        "/batch_predict",
        json={
            "image_dir": "/nonexistent/directory/path",
            "report_id": "pytest_run",
            "batch_size": 16,
        },
    )

    # Should return 404 Not Found
    assert response.status_code == 404


def test_batch_predict_endpoint_with_custom_batch_size(client):
    """Test that /batch_predict respects custom batch_size parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_images_in_dir(tmpdir_path, count=5)

        response = client.post(
            "/batch_predict",
            json={
                "image_dir": str(tmpdir_path),
                "report_id": "pytest_run",
                "batch_size": 2,  # Small batch size
            },
        )

        # Should still process all images
        data = response.json()
        assert data["total"] == 5
        assert data["successful"] == 5


def test_batch_predict_endpoint_with_max_images(client):
    """Test that /batch_predict respects max_images parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_images_in_dir(tmpdir_path, count=5)

        response = client.post(
            "/batch_predict",
            json={
                "image_dir": str(tmpdir_path),
                "report_id": "pytest_run",
                "batch_size": 16,
                "max_images": 2,  # Limit to 2 images
            },
        )

        data = response.json()
        # Should only process 2 images
        assert data["total"] == 2
        assert data["successful"] == 2


def test_batch_predict_endpoint_empty_failed_images_list(client):
    """Test that failed_images is empty when all predictions succeed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_images_in_dir(tmpdir_path, count=3)

        response = client.post(
            "/batch_predict",
            json={"image_dir": str(tmpdir_path), "report_id": "pytest_run", "batch_size": 16},
        )

        data = response.json()

        # In testing mode, all images should succeed
        assert len(data["failed_images"]) == 0


def test_batch_predict_endpoint_report_id_in_path(client):
    """Test that custom report_id appears in the report directory path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        create_test_images_in_dir(tmpdir_path, count=3)

        custom_id = "my_custom_test_id"

        response = client.post(
            "/batch_predict",
            json={"image_dir": str(tmpdir_path), "report_id": custom_id, "batch_size": 16},
        )

        data = response.json()

        # Report directory should contain the custom report_id
        assert custom_id in data["report_dir"]
