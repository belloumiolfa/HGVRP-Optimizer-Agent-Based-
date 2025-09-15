# tests/test_construct_and_ls.py
import itertools

def _flatten_customers(routes):
    for rte in routes:
        for node in rte:
            if node != 0:
                yield node

def test_construct_balanced_covers_all(instances, client, default_penalty):
    scen = instances["construct_sets"]["balanced_small"]
    payload = {
        "instance": scen["instance"],
        "method": scen.get("method", "balanced"),
        "penalty": default_penalty or {}
    }
    r = client.post("/construct", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "ok"
    routes = out["routes"]

    # Coverage: each customer exactly once
    cust_ids = sorted([c["id"] for c in scen["instance"]["customers"]])
    seen = sorted(_flatten_customers(routes))
    assert seen == cust_ids, f"Coverage mismatch. expected {cust_ids}, got {seen}"

    # P_coverage should be zero
    assert abs(out["metrics"]["P_coverage"]) < 1e-12

def test_construct_greedy_covers_all(instances, client, default_penalty):
    scen = instances["construct_sets"]["greedy_small"]
    payload = {
        "instance": scen["instance"],
        "method": scen.get("method", "greedy"),
        "penalty": default_penalty or {},
        "seed": scen.get("seed")
    }
    r = client.post("/construct", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "ok"
    routes = out["routes"]

    cust_ids = sorted([c["id"] for c in scen["instance"]["customers"]])
    seen = sorted(_flatten_customers(routes))
    assert seen == cust_ids
    assert abs(out["metrics"]["P_coverage"]) < 1e-12

def test_localsearch_two_opt_improves(instances, client, default_penalty):
    scen = instances["localsearch_sets"]["two_opt_square"]

    # Evaluate before
    r0 = client.post("/evaluate", json={
        "instance": scen["instance"],
        "routes": scen["routes"],
        "penalty": default_penalty or {}
    })
    assert r0.status_code == 200
    d0 = r0.json()["metrics"]["Z1_total_distance"]

    # Run LS
    r = client.post("/localsearch", json={
        "instance": scen["instance"],
        "routes": scen["routes"],
        "penalty": default_penalty or {},
        "budget": scen.get("budget", 200),
        "ops": scen.get("ops", ["2opt","oropt_intra"]),
        "seed": scen.get("seed", 42)
    })
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "ok"
    d1 = out["metrics"]["Z1_total_distance"]
    assert d1 < d0, f"Expected 2-opt to improve distance: before={d0}, after={d1}"
    
def test_localsearch_cross_improves(instances, client, default_penalty):
    scen = instances["localsearch_sets"]["two_routes_cross"]  # add in your fixtures

    # Baseline
    r0 = client.post("/evaluate", json={
        "instance": scen["instance"],
        "routes": scen["routes"],
        "penalty": default_penalty or {}
    })
    assert r0.status_code == 200
    d0 = r0.json()["metrics"]["Z1_total_distance"]

    # LS with CROSS
    r = client.post("/localsearch", json={
        "instance": scen["instance"],
        "routes": scen["routes"],
        "penalty": default_penalty or {},
        "budget": 200,
        "ops": ["cross"],
        "seed": 123
    })
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "ok"
    d1 = out["metrics"]["Z1_total_distance"]
    assert d1 < d0, f"Expected CROSS to improve distance: before={d0}, after={d1}"
