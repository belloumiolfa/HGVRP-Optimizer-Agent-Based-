# tests/test_integration_server.py
import json, time, threading
from pathlib import Path
import requests
import uvicorn
from python.app import app  # adjust import to your layout

import pytest
pytestmark = pytest.mark.skip(reason="Skip live Uvicorn test; use TestClient-based flow instead.")

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

def test_live_server_flow():
    # Start server in a background thread
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(0.5)  # simple wait for server to start

    P = json.loads((Path(__file__).parents[2] / "examples" / "payloads.json").read_text())
    BASE = "http://127.0.0.1:8000"

    assert requests.get(f"{BASE}/health").json()["status"] == "healthy"
    assert requests.post(f"{BASE}/evaluate", json=P["evaluate"]).status_code == 200

    con = requests.post(f"{BASE}/construct", json=P["construct"]).json()
    ls_in = dict(P["localsearch"])
    ls_in["routes"] = con.get("routes", ls_in["routes"])
    assert requests.post(f"{BASE}/localsearch", json=ls_in).status_code == 200
