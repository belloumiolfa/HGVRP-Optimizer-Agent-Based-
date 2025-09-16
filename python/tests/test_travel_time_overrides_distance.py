# python/tests/test_travel_time_overrides_distance.py
from python.models import Instance, Depot, Vehicle, Customer, PenaltyConfig
from python.agents.assignment_agent import AssignmentAgent

def test_travel_time_overrides_distance():
    dep = Depot(id=0, x=0.0, y=0.0, start_time=0.0)
    cust = [
        Customer(id=1, x=3.0, y=0.0, demand=0.0, time_windows=[0, 100], service_time=0.0),
        Customer(id=2, x=6.0, y=0.0, demand=0.0, time_windows=[0, 100], service_time=0.0),
    ]
    veh = [Vehicle(id=1, capacity=999, fuel_efficiency=1.0, emission_level=0.0,
                   vehicle_type="truck", fuel_type="diesel", fixed_costs=0.0, variable_costs=0.0)]

    # Geometry: 0->1 (3), 1->2 (3), 2->0 (6) => distance = 12
    # τ doubles only 1->2 to 6 => total travel time = 3 + 6 + 6 = 15
    tt = {
        ("dc", 0, 1): 3.0,
        ("cc", 1, 2): 6.0,
        ("cd", 2, 0): 6.0
    }
    inst = Instance(depot=dep, vehicles=veh, customers=cust, travel_time=tt)
    pcfg = PenaltyConfig()
    agent = AssignmentAgent(inst)
    out = agent.evaluate(routes=[[0, 1, 2, 0]], pc=pcfg)

    # Objectives use Z1..Z6
    assert abs(out["objectives"]["Z1"] - 12.0) < 1e-9   # distance unchanged
    assert abs(out["objectives"]["Z2"] - 0.0)  < 1e-9   # waiting unchanged

    # Optional: human-readable totals
    assert abs(out["totals"]["distance"] - 12.0) < 1e-9
    assert abs(out["totals"]["waiting"]  - 0.0)  < 1e-9

    # τ affects only the elapsed time (end_time), not geometric distance
    perveh = out["perVehicle"][0]
    assert abs(perveh["end_time"] - 15.0) < 1e-9
