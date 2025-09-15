# python/tests/test_client_like_flow.py
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from python.app import app

ROOT = Path(__file__).resolve().parents[2]
PAYLOADS = ROOT / "examples" / "payloads.json"


def load_payloads_or_skip():
    if not PAYLOADS.exists():
        pytest.skip("examples/payloads.json is not present; skipping client-like flow test")
    return json.loads(PAYLOADS.read_text(encoding="utf-8"))


def test_flow_evaluate_construct_ls():
    client = TestClient(app, raise_server_exceptions=False)
    P = load_payloads_or_skip()

    # require the three top-level blocks in payloads.json
    need = {"evaluate", "construct", "localsearch"}
    if not need.issubset(P.keys()):
        pytest.skip("payloads.json lacks required keys: 'evaluate', 'construct', 'localsearch'")

    # /health
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

    # /evaluate
    r = client.post("/evaluate", json=P["evaluate"])
    assert r.status_code == 200, f"/evaluate failed: {r.status_code} {r.text}"
    eval_out = r.json()
    assert eval_out["status"] == "ok"

    # /construct
    r = client.post("/construct", json=P["construct"])
    assert r.status_code == 200, f"/construct failed: {r.status_code} {r.text}"
    con_out = r.json()
    assert con_out["status"] == "ok"
    assert "routes" in con_out

    # /localsearch — use SAME instance as construct to avoid mismatches
    ls_in = dict(P["localsearch"])
    ls_in["instance"] = P["construct"]["instance"]
    ls_in["routes"] = con_out.get("routes", ls_in.get("routes"))
    ls_in.setdefault("budget", 100)
    ls_in.setdefault("ops", ["2opt", "oropt_intra"])
    ls_in.setdefault("seed", 42)

    r = client.post("/localsearch", json=ls_in)
    assert r.status_code == 200, f"/localsearch failed: {r.status_code} {r.text}"
    ls_out = r.json()
    assert ls_out["status"] == "ok"
