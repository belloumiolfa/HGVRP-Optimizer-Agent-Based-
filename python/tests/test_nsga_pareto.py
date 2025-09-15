# python/tests/test_nsga_pareto.py
import math
import random
import pytest

from python.nsga_tools import fast_non_dominated_sort, dominates, hypervolume_2d
from python.ga_pareto import NSGA2Optimizer, Individual
from python.models import Instance, Depot, Vehicle, Customer, PenaltyConfig

def tiny_instance() -> Instance:
    # 1 depot, 3 customers in a line – enough to form small trade-offs via routing and fuel
    dep = Depot(id=0, x=0.0, y=0.0, start_time=0.0)
    cust = [
        Customer(id=1, x=1.0, y=0.0, demand=1.0, time_windows=[0, 100], service_time=0.0),
        Customer(id=2, x=2.0, y=0.0, demand=1.0, time_windows=[0, 100], service_time=0.0),
        Customer(id=3, x=3.0, y=0.0, demand=1.0, time_windows=[0, 100], service_time=0.0),
    ]
    vehs = [
        Vehicle(id=1, capacity=10.0, fuel_efficiency=1.0, emission_level=0.0, vehicle_type="Diesel", fuel_type="Diesel", fixed_costs=0.0, variable_costs=0.0),
        Vehicle(id=2, capacity=10.0, fuel_efficiency=1.0, emission_level=0.0, vehicle_type="Diesel", fuel_type="Diesel", fixed_costs=0.0, variable_costs=0.0),
    ]
    return Instance(depot=dep, depots=[dep], customers=cust, vehicles=vehs)

def default_penalty() -> PenaltyConfig:
    return PenaltyConfig(
        beta_capacity=1.0, beta_duration=1.0, beta_timewindows=1.0,
        beta_availability=1.0, beta_battery=1.0, beta_coverage=1000.0, beta_supply=1.0
    )

def is_pareto_valid(payload, objective_keys):
    # Ensure no element in payload dominates another (feasible-wise then objectives)
    vecs = [tuple(item["Z"][k] for k in objective_keys) for item in payload]
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            if i == j: 
                continue
            assert not dominates(vecs[i], vecs[j])

@pytest.mark.parametrize("seed", [0, 1])
def test_pareto_validity(seed):
    inst = tiny_instance()
    opt = NSGA2Optimizer(
        instance=inst,
        penalty_cfg=default_penalty(),
        objectives=["Z1", "Z4"],  # distance vs fuel
        hard_penalty_keys=["capacity", "duration", "timewindows"]
    )
    out = opt.evolve(pop_size=20, generations=25, constructor="balanced", pc=0.9, pm=0.3)
    pareto = out["pareto"]
    assert len(pareto) >= 2
    is_pareto_valid(pareto, ["Z1", "Z4"])

def is_monotone_tradeoff(points):
    # sort by Z1 and ensure Z4 is non-increasing (typical concave front in 2D minimization)
    pts = sorted(points, key=lambda p: p[0])
    last = float("inf")
    for _, z4 in pts:
        if z4 > last + 1e-9:
            return False
        last = min(last, z4)
    return True

def test_tradeoff_monotone_front():
    inst = tiny_instance()
    opt = NSGA2Optimizer(
        instance=inst,
        penalty_cfg=default_penalty(),
        objectives=["Z1", "Z4"],
        hard_penalty_keys=["capacity", "duration", "timewindows"],
        seed=42
    )
    out = opt.evolve(pop_size=30, generations=35, constructor="balanced")
    pareto = out["pareto"]
    pts = [(p["Z"]["Z1"], p["Z"]["Z4"]) for p in pareto]
    assert is_monotone_tradeoff(pts)

def test_hypervolume_improves_over_constructor_baseline():
    inst = tiny_instance()
    # Baseline = first generation archive (seed pop only)
    opt = NSGA2Optimizer(
        instance=inst,
        penalty_cfg=default_penalty(),
        objectives=["Z1", "Z4"],
        hard_penalty_keys=["capacity", "duration", "timewindows"],
        seed=7
    )
    # Seed population only (pop then 0 gens)
    base = opt.evolve(pop_size=25, generations=0, constructor="balanced")
    base_pts = [(p["Z"]["Z1"], p["Z"]["Z4"]) for p in base["pareto"]]
    # Reference point slightly worse than worst observed
    if base_pts:
        ref = (max(z1 for z1, _ in base_pts) + 1.0, max(z4 for _, z4 in base_pts) + 1.0)
    else:
        ref = (100.0, 100.0)
    base_hv = hypervolume_2d(base_pts, ref)

    # Now evolve for a few generations and compare HV
    opt2 = NSGA2Optimizer(
        instance=inst,
        penalty_cfg=default_penalty(),
        objectives=["Z1", "Z4"],
        hard_penalty_keys=["capacity", "duration", "timewindows"],
        seed=7
    )
    out = opt2.evolve(pop_size=25, generations=30, constructor="balanced")
    pts = [(p["Z"]["Z1"], p["Z"]["Z4"]) for p in out["pareto"]]
    new_hv = hypervolume_2d(pts, ref)
    assert new_hv >= base_hv - 1e-9
    # Usually strictly better in practice
    assert new_hv > base_hv or math.isclose(new_hv, base_hv, rel_tol=1e-9)
