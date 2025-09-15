# python/tests/conftest.py
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from python.app import app

# Base directories
ROOT = Path(__file__).resolve().parents[2]        # HGVRP Optimizer/
EXAMPLES_DIR = ROOT / "examples"


@pytest.fixture(scope="session")
def client():
    # IMPORTANT: this makes server exceptions come back as HTTP 500
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="session")
def payloads():
    """Load payloads.json for end-to-end flow tests."""
    path = EXAMPLES_DIR / "payloads.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def instances():
    """Always load instances.json from examples/ folder."""
    path = EXAMPLES_DIR / "instances.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def default_penalty(instances):
    """Get default penalty config from instances.json if defined."""
    return instances.get("defaults", {}).get("penalty", {})


