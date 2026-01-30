"""Pytest configuration and fixtures.

This file sets up the testing environment before any tests run.
"""

import os
import sys
from pathlib import Path

# Enable testing mode BEFORE importing the API
os.environ["WOUND_API_TESTING"] = "1"

# Add project root to sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Now import the FastAPI app (with testing mode enabled)
import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="session")
def client():
    """FastAPI test client fixture.

    Scope is session so the app is created once for all tests.
    """
    with TestClient(app) as test_client:
        yield test_client
