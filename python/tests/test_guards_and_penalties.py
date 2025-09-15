# tests/test_guards_and_penalties.py
# python/tests/test_guards_and_penalties.py
def test_more_routes_than_vehicles_500(client, instances, default_penalty):
    scen = instances["vehicle_eval"]["diesel_basic"]
    inst = scen["instance"]
    routes = [scen["routes"][0], [0, 1, 0]]  # 2 routes, 1 vehicle -> invalid

    r = client.post("/evaluate", json={
        "instance": inst,
        "routes": routes,
        "penalty": default_penalty or {}
    })
    assert r.status_code == 500

def test_capacity_availability_battery_mix(client, instances, default_penalty):
    # Build on electric_range_violation but also overload capacity and availability
    base = instances["vehicle_eval"]["electric_range_violation"]
    inst = base["instance"]

    # Overload capacity by making demand high
    inst2 = {
        "depot": inst["depot"],
        "vehicles": [{
            **inst["vehicles"][0],
            "availability": [0.0, 8.0]  # will still violate (distance 12 > 8 end time)
        }],
        "customers": [
            { **inst["customers"][0], "demand": 9 },
            { **inst["customers"][1], "demand": 9 }
        ]
    }
    routes = base["routes"]

    r = client.post("/evaluate", json={
        "instance": inst2,
        "routes": routes,
        "penalty": default_penalty or {}
    })
    assert r.status_code == 200
    m = r.json()["metrics"]

    assert m["P_battery"] > 0.0, "battery penalty expected"
    assert m["P_capacity"] > 0.0, "capacity penalty expected"
    assert m["P_availability"] > 0.0 or m["P_duration"] > 0.0, "time-related penalty expected"
