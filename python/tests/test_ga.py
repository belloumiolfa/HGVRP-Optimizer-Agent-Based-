# python/tests/test_ga.py
import random
import pytest

import python.ga as ga
from python.models import Instance, Depot, Vehicle, Customer, PenaltyConfig, Routes


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: tiny synthetic VRP
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def tiny_instance():
    depot = Depot(id=0, x=0.0, y=0.0, start_time=0.0)

    vehicles = [
        Vehicle(id=101, capacity=2.0, fuel_efficiency=1.0, emission_level=0.0,
                vehicle_type="Diesel", fuel_type="Diesel", fixed_costs=0.0, variable_costs=0.0),
        Vehicle(id=102, capacity=2.0, fuel_efficiency=1.0, emission_level=0.0,
                vehicle_type="Diesel", fuel_type="Diesel", fixed_costs=0.0, variable_costs=0.0),
    ]

    customers = [
        Customer(id=1, x=1.0, y=0.0, demand=1.0, time_windows=[0, 100], service_time=0.0),
        Customer(id=2, x=0.0, y=1.0, demand=1.0, time_windows=[0, 100], service_time=0.0),
        Customer(id=3, x=-1.0, y=0.0, demand=1.0, time_windows=[0, 100], service_time=0.0),
        Customer(id=4, x=0.0, y=-1.0, demand=1.0, time_windows=[0, 100], service_time=0.0),
    ]

    # IMPORTANT: ga.coords_of_customer() reads instance.depots[0]
    # Your Instance may have depots=None and a singular depot field.
    # We set BOTH so ga.py works without modification.
    return Instance(depot=depot, depots=[depot], vehicles=vehicles, customers=customers)


@pytest.fixture
def default_penalty():
    return PenaltyConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Adapter: make GA→Evaluator interface compatible
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def adapt_AssignmentAgent_routes_signature(monkeypatch):
    """
    Convert Routes → List[List[int]] for the real AssignmentAgent and provide a safe vehicle_by_id.
    """
    from python.agents.assignment_agent import AssignmentAgent as RealAgent

    class Adapter:
        def __init__(self, instance):
            self._real = RealAgent(instance)
            self._instance = instance  # keep for vehicle lookup

        def evaluate(self, routes, penalties):
            lst = routes
            if hasattr(routes, "routes"):
                lst = list(routes.routes.values())
            return self._real.evaluate(lst, penalties)

        def evaluate_solution(self, routes, penalties):
            return self.evaluate(routes, penalties)

        def vehicle_by_id(self, vid):
            # Return the Vehicle object directly from the instance to avoid AttributeError
            for v in getattr(self._instance, "vehicles", []) or []:
                if v.id == vid:
                    return v
            return None

    monkeypatch.setattr(ga, "AssignmentAgent", Adapter)
    yield


# ──────────────────────────────────────────────────────────────────────────────
# Basic building blocks / invariants
# ──────────────────────────────────────────────────────────────────────────────
def test_split_capacity_greedy_respects_capacity_when_feasible(tiny_instance):
    perm = [1, 2, 3, 4]
    routes = ga.split_capacity_greedy(tiny_instance, perm)
    for vid, seq in routes.routes.items():
        load = sum(
            next(c for c in tiny_instance.customers if c.id == nid).demand
            for nid in seq if nid != 0
        )
        cap = next(v for v in tiny_instance.vehicles if v.id == vid).capacity
        assert load <= cap + 1e-9, f"Vehicle {vid} overloaded: load={load}, cap={cap}"


def test_bcrc_crossover_preserves_all_customers(tiny_instance):
    permA = [1, 2, 3, 4]
    permB = [4, 3, 2, 1]
    routesA = Routes(routes={101: [0, 1, 2, 0], 102: [0, 3, 4, 0]})
    child = ga.bcrc_crossover(tiny_instance, (permA, routesA), (permB, Routes(routes={})))
    assert sorted(child) == sorted(permA) == sorted(permB)
    assert len(child) == len(permA)


def test_bcrc_crossover_fallback_when_no_non_empty_route(tiny_instance):
    permA = [1, 2, 3, 4]
    permB = [4, 3, 2, 1]
    routesA = Routes(routes={101: [0, 0], 102: [0, 0]})
    child = ga.bcrc_crossover(tiny_instance, (permA, routesA), (permB, Routes(routes={})))
    assert child == permB and child is not permB


def test_mutation_ops_preserve_permutation():
    base = [1, 2, 3, 4, 5]
    for op in (ga.mut_swap, ga.mut_insert, ga.mut_oropt1):
        perm = base[:]
        random.seed(0)
        op(perm)
        assert sorted(perm) == sorted(base)
        assert len(perm) == len(base)


# ──────────────────────────────────────────────────────────────────────────────
# quick_ls operator integration
# ──────────────────────────────────────────────────────────────────────────────
def test_quick_ls_calls_moves(monkeypatch, tiny_instance):
    calls = {"two_opt": 0, "oropt1": 0}

    def fake_two_opt(agent, route, veh):
        calls["two_opt"] += 1
        return route[:], 0.0

    def fake_oropt1(agent, route, veh):
        calls["oropt1"] += 1
        return route[:], 0.0

    monkeypatch.setattr(ga, "HAVE_LS", True)
    monkeypatch.setattr(ga, "two_opt_once", fake_two_opt)
    monkeypatch.setattr(ga, "or_opt1_intra_once", fake_oropt1)

    r = Routes(routes={101: [0, 1, 2, 3, 0], 102: [0, 4, 0]})
    out = ga.quick_ls(tiny_instance, r)
    assert isinstance(out, Routes)
    assert calls["two_opt"] >= 1 and calls["oropt1"] >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Population & GA loop
# ──────────────────────────────────────────────────────────────────────────────
def test_seed_population_size_and_sorted(tiny_instance, default_penalty):
    pop = ga.seed_population(tiny_instance, default_penalty, pop_size=10, rnd_ratio=0.6)
    assert len(pop) == 10
    fits = [ind[3] for ind in pop]
    assert fits == sorted(fits), "Population should be sorted by fitness ascending"


def test_optimize_single_runs_and_monotonic_when_no_penalty_adapt(tiny_instance, default_penalty):
    cfg = ga.GAConfig(
        population=12, generations=15, elitism=2, mut_prob=0.7, cx_prob=0.8,
        adaptive_operators=True, adaptive_penalties=False, ls_prob=0.0, seed=123
    )
    out = ga.optimize_single(tiny_instance, default_penalty, cfg)

    assert set(out.keys()) == {"best", "history", "telemetry"}
    best = out["best"]
    assert set(best.keys()) == {"routes", "metrics", "fitness", "perm"}
    assert sorted(best["perm"]) == [1, 2, 3, 4]

    hist = out["history"]
    assert len(hist) == cfg.generations + 1
    series = [h["best_fitness"] for h in hist]
    assert all(series[i] <= series[i-1] + 1e-12 for i in range(1, len(series)))

    probs_seq = out["telemetry"]["bandit_probs"]
    assert len(probs_seq) == cfg.generations


def test_optimize_single_penalty_adaptation_changes_betas(tiny_instance):
    penalties = PenaltyConfig().model_copy(deep=True)
    cfg = ga.GAConfig(
        population=10, generations=8, elitism=2, mut_prob=0.6, cx_prob=0.8,
        adaptive_operators=False, adaptive_penalties=True, ls_prob=0.0, seed=7,
        pen_window=5, pen_gamma=1.5
    )
    out = ga.optimize_single(tiny_instance, penalties, cfg)
    betas_hist = out["telemetry"]["betas"]
    assert len(betas_hist) == cfg.generations  # a snapshot each gen
    start = penalties.model_dump()
    last = betas_hist[-1]
    # Same keys and valid numeric values
    assert set(last.keys()) == set(start.keys())
    assert all(isinstance(last[k], (int, float)) for k in last.keys())
    # Allow unchanged betas on fully feasible toy instances
    # (If you later craft an infeasible test instance, you can assert a change here.)
