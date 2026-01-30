"""Quick test script for API functionality.

Run this to verify the API can import and function correctly.
"""

import sys

# Test imports
print("Testing imports...")
try:
    from src.api import app

    print("✓ API imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test inference.py imports
try:
    from src.inference import (
        CLASS_NAMES,
    )

    print("✓ Inference imports successful")
    print(f"  Class names: {CLASS_NAMES}")
except Exception as e:
    print(f"✗ Inference import failed: {e}")
    sys.exit(1)

# Test FastAPI app creation
try:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    print("✓ FastAPI test client created")
except Exception as e:
    print(f"✗ Test client creation failed: {e}")
    print("  Note: Install fastapi first: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

print("\n" + "=" * 60)
print("All imports successful!")
print("=" * 60)
print("\nTo start the API server:")
print("  uvicorn src.api:app --reload")
print("\nTo run a quick test:")
print("  curl http://localhost:8000/health")
