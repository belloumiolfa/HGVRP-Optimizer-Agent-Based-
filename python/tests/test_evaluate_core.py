# tests/test_vehicles_eval.py
import math
import pytest

SCEN_KEYS = [
    "diesel_basic",
    "gasoline_basic",
    "cng_basic",
    "hydrogen_basic",
    "electric_range_ok",
    "electric_range_violation",
    "hybrid_basic",
    "bike_lastmile",
    "drone_basic",
    "heavy_truck_duration_and_availability",
]

@pytest.mark.parametrize("key", SCEN_KEYS)
def test_vehicle_types_evaluate(client, instances, default_penalty, key):
    scen = instances["vehicle_eval"][key]
    payload = {
        "instance": scen["instance"],
        "routes": scen["routes"],
        "penalty": default_penalty or {}
    }
    r = client.post("/evaluate", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "ok"
    m = out["metrics"]

    exp = scen.get("expectations", {})

    # Approx checks (float with tolerance)
    approx = exp.get("approx", {})
    for k, v in approx.items():
        assert k in m, f"Missing metric {k}"
        assert abs(m[k] - v) < 1e-6, f"{key}: {k} expected≈{v}, got {m[k]}"

    # Exact equals for ints or floats if you want strict
    eq = exp.get("eq", {})
    for k, v in eq.items():
        assert k in m
        assert m[k] == v, f"{key}: {k} expected=={v}, got {m[k]}"

    # Greater-than checks
    gt = exp.get("gt", {})
    for k, v in gt.items():
        assert k in m
        assert m[k] > v, f"{key}: {k} expected>{v}, got {m[k]}"

    # Zero checks (near-zero for floats)
    zero = exp.get("zero", [])
    for k in zero:
        assert k in m
        assert abs(m[k]) < 1e-12, f"{key}: {k} expected≈0, got {m[k]}"
