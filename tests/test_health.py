"""Tests for the /health endpoint."""


def test_health_endpoint_returns_200(client):
    """Test that /health endpoint returns 200 status code."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_json_structure(client):
    """Test that /health endpoint returns expected JSON structure."""
    response = client.get("/health")
    data = response.json()

    # Check all required keys are present
    assert "status" in data
    assert "device" in data
    assert "model_loaded" in data
    assert "checkpoint" in data


def test_health_endpoint_status_ok(client):
    """Test that /health endpoint returns status 'ok'."""
    response = client.get("/health")
    data = response.json()

    assert data["status"] == "ok"


def test_health_endpoint_model_loaded_in_testing_mode(client):
    """Test that model_loaded is True in testing mode."""
    response = client.get("/health")
    data = response.json()

    # In testing mode, model should be marked as loaded (dummy model)
    assert data["model_loaded"] is True


def test_health_endpoint_device_is_cpu(client):
    """Test that device is CPU in testing mode."""
    response = client.get("/health")
    data = response.json()

    # Testing mode uses CPU
    assert "cpu" in data["device"].lower()


def test_health_endpoint_checkpoint_path_exists(client):
    """Test that checkpoint path is returned."""
    response = client.get("/health")
    data = response.json()

    assert data["checkpoint"] is not None
    assert len(data["checkpoint"]) > 0
