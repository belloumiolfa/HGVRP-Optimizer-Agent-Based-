from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import random

from .nsga_tools import Individual, EliteArchive, select_nsga2
from .models import Instance, PenaltyConfig
from .eval import AssignmentAgent
from .construct import construct_balanced, construct_greedy

# Optional LS if present
try:
    from .moves import two_opt_once, or_opt1_intra_once
    HAVE_LS = True
except Exception:
    HAVE_LS = False

DEFAULT_OBJECTIVES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]


# -------- Instance SHIM (does NOT mutate Pydantic model) --------
class _InstShim:
    """
    Read-through wrapper that adds derived attributes some modules expect
    (customers_sorted, customer_ids, customer_map, depots) without mutating
    the underlying Pydantic Instance (which forbids extra fields).
    """
    def __init__(self, base: Instance):
        self.__dict__["_base"] = base
        customers = getattr(base, "customers", None) or []
        try:
            self.customers_sorted = sorted(customers, key=lambda c: getattr(c, "id", 0))
        except Exception:
            self.customers_sorted = list(customers)
        self.customer_ids = [getattr(c, "id", i + 1) for i, c in enumerate(customers)]
        try:
            self.customer_map = {getattr(c, "id", i + 1): c for i, c in enumerate(customers)}
        except Exception:
            self.customer_map = {}
        depots = getattr(base, "depots", None)
        self.depots = depots if depots else ([getattr(base, "depot")] if getattr(base, "depot", None) else [])
        self.vehicles = getattr(base, "vehicles", None) or []

    def __getattr__(self, name):
        return getattr(self.__dict__["_base"], name)

    def __setattr__(self, name, value):
        self.__dict__[name] = value


# --- Canonical objective normalizer ---
def _normalize_Z(raw_Z: Dict[str, float]) -> Dict[str, float]:
    """
    Map common aliases to canonical Z1..Z6:
      Z1: distance
      Z2: waiting
      Z3: vehicles / routes
      Z4: fuel / energy
      Z5: emissions
      Z6: cost
    """
    z = {k.lower(): float(v) for k, v in (raw_Z or {}).items()}

    def pick(*names, default=0.0):
        for n in names:
            n1 = n.lower()
            if n1 in z:
                return float(z[n1])
        keys_norm = {k.replace("_","").replace("-",""): v for k, v in z.items()}
        for n in names:
            n2 = n.lower().replace("_","").replace("-","")
            if n2 in keys_norm:
                return float(keys_norm[n2])
        return float(default)

    return {
        "Z1": pick("Z1", "distance", "total_distance", "dist", "travel_distance"),
        "Z2": pick("Z2", "waiting", "total_waiting", "wait_time", "waiting_time"),
        "Z3": pick("Z3", "vehicles", "vehicle_count", "routes", "num_routes", "n_routes"),
        "Z4": pick("Z4", "fuel", "energy", "fuel_consumption", "total_fuel"),
        "Z5": pick("Z5", "emissions", "co2", "co2e", "total_emissions"),
        "Z6": pick("Z6", "cost", "total_cost", "variable_cost", "objective_cost"),
    }


# ---- Evaluation adapter (tries positional & keyword variants, normalizes Z) ----
def evaluate_routes(agent: AssignmentAgent, routes: List[List[int]], penalty_cfg: PenaltyConfig) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Supports agent APIs like:
      - evaluate(routes, penalty_cfg)                   # positional
      - evaluate(routes, pc=penalty_cfg)                # keyword
      - evaluate(routes, penalty_cfg=<...>)
      - evaluate_routes(routes, penalty_cfg)            # positional
      - evaluate_routes(routes, pc=penalty_cfg)         # keyword
      - evaluate(routes) / evaluate_routes(routes)
    Returns (Z, penalties) with Z normalized to Z1..Z6.
    """
    attempts = [
        # Try the most common signature first
        ("evaluate",        (routes, penalty_cfg), {}),           # positional
        ("evaluate",        (routes,), {"pc": penalty_cfg}),      # keyword 'pc'
        ("evaluate",        (routes,), {"penalty_cfg": penalty_cfg}),
        ("evaluate_routes", (routes, penalty_cfg), {}),           # positional
        ("evaluate_routes", (routes,), {"pc": penalty_cfg}),      # keyword 'pc'
        ("evaluate_routes", (routes,), {"penalty_cfg": penalty_cfg}),
        ("evaluate",        (routes,), {}),                       # last-resort, no pc
        ("evaluate_routes", (routes,), {}),                       # last-resort, no pc
    ]

    last_err = None
    for name, args, kwargs in attempts:
        fn = getattr(agent, name, None)
        if not callable(fn):
            continue
        try:
            out = fn(*args, **kwargs)

            # dict with Z & penalties
            if isinstance(out, dict) and "Z" in out and "penalties" in out:
                return _normalize_Z(out["Z"]), out["penalties"] or {}

            # dict with OBJECTIVES & penalties (service-layer style)
            if isinstance(out, dict) and "objectives" in out and "penalties" in out:
                return _normalize_Z(out["objectives"]), out["penalties"] or {}

            # tuple (Z, penalties)
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict) and isinstance(out[1], dict):
                return _normalize_Z(out[0]), out[1] or {}

            # dict with only Z / or only OBJECTIVES
            if isinstance(out, dict) and "Z" in out:
                return _normalize_Z(out["Z"]), {}
            if isinstance(out, dict) and "objectives" in out:
                return _normalize_Z(out["objectives"]), {}

            # dict that is Z directly
            if isinstance(out, dict):
                return _normalize_Z(out), {}

        except Exception as e:
            last_err = e
            continue

    raise TypeError(f"Could not evaluate routes with AssignmentAgent; last error: {last_err}")


def simple_crossover(p1: Individual, p2: Individual, rng: random.Random) -> List[List[List[int]]]:
    """Route-level splice. Replace with a depot-aware BCRC for better results later."""
    a = [r[:] for r in p1.routes]
    b = [r[:] for r in p2.routes]
    if not a or not b:
        return [a, b]
    cut_a = rng.randrange(len(a))
    cut_b = rng.randrange(len(b))
    child1 = a[:cut_a] + b[cut_b:]
    child2 = b[:cut_b] + a[cut_a:]
    return [child1, child2]


def simple_mutation(routes: List[List[int]], rng: random.Random) -> List[List[int]]:
    """Swap two customers in a random route (light mutation)."""
    if len(routes) == 0:
        return routes
    r = rng.randrange(len(routes))
    rt = routes[r][:]
    if len(rt) > 3:
        i = rng.randrange(1, len(rt) - 1)
        j = rng.randrange(1, len(rt) - 1)
        if i != j:
            rt[i], rt[j] = rt[j], rt[i]
    new = [rr[:] for rr in routes]
    new[r] = rt
    return new


def maybe_local_search(agent: AssignmentAgent, routes: List[List[int]], rng: random.Random, prob: float = 0.25) -> List[List[int]]:
    # Bail out early if LS disabled or we don't "roll the dice"
    if not HAVE_LS or rng.random() > prob:
        return routes

    # Try to obtain a usable vehicle (must have start_depot_id/end_depot_id or equivalent)
    veh = None
    try:
        if getattr(agent, "instance", None) and getattr(agent.instance, "vehicles", None):
            cand = agent.instance.vehicles[0] if agent.instance.vehicles else None
            # require minimal attributes used by _route_distance
            if cand is not None and hasattr(cand, "start_depot_id"):
                veh = cand
    except Exception:
        veh = None

    # If we still don't have a usable vehicle, skip LS to avoid errors
    if veh is None:
        return routes

    best = [r[:] for r in routes]
    for ridx, r in enumerate(routes):
        if len(r) <= 3:
            continue
        r2 = r[:]
        # Run light intra-route improvements
        try:
            r2, _ = two_opt_once(agent, r2, veh)
            r2, _ = or_opt1_intra_once(agent, r2, veh)
        except Exception:
            # If any LS op fails, keep original route for robustness
            r2 = r[:]
        best[ridx] = r2
    return best

class NSGA2Optimizer:
    def __init__(
        self,
        instance: Instance,
        penalty_cfg: PenaltyConfig,
        objectives: Optional[List[str]] = None,
        hard_penalty_keys: Optional[List[str]] = None,
        archive_size: int = 64,
        seed: Optional[int] = None,
    ):
        self.raw_instance = instance
        self.instance = _InstShim(instance)
        self.penalty_cfg = penalty_cfg
        self.objectives = objectives or DEFAULT_OBJECTIVES
        self.hard_penalty_keys = hard_penalty_keys or [
            "capacity", "duration", "timewindows", "availability", "battery", "coverage", "supply"
        ]
        self.rng = random.Random(seed)

        # Your AssignmentAgent takes only instance in ctor
        self.agent = AssignmentAgent(self.instance)
        self.archive = EliteArchive(archive_size, self.objectives, self.hard_penalty_keys)

    def _seed_population(self, size: int, constructor: str = "balanced") -> List[Individual]:
        pop: List[Individual] = []
        for _ in range(size):
            # IMPORTANT: your constructors use agent utils like _route_distance
            if constructor == "balanced":
                routes: List[List[int]] = construct_balanced(self.agent)
            else:
                routes = construct_greedy(self.agent)
            Z, P = evaluate_routes(self.agent, routes, self.penalty_cfg)
            pop.append(Individual(routes=routes, Z=Z, penalties=P))
        self.archive.update(pop)
        return pop

    def _variation(self, parents: List[Individual], pc: float = 0.9, pm: float = 0.3) -> List[Individual]:
        children: List[Individual] = []
        parents = parents[:]
        self.rng.shuffle(parents)
        for p1, p2 in zip(parents[::2], parents[1::2]):
            if self.rng.random() < pc:
                ch1, ch2 = simple_crossover(p1, p2, self.rng)
            else:
                ch1, ch2 = [r[:] for r in p1.routes], [r[:] for r in p2.routes]
            if self.rng.random() < pm:
                ch1 = simple_mutation(ch1, self.rng)
            if self.rng.random() < pm:
                ch2 = simple_mutation(ch2, self.rng)
            ch1 = maybe_local_search(self.agent, ch1, self.rng, prob=0.25)
            ch2 = maybe_local_search(self.agent, ch2, self.rng, prob=0.25)
            for ch in (ch1, ch2):
                Z, P = evaluate_routes(self.agent, ch, self.penalty_cfg)
                children.append(Individual(routes=ch, Z=Z, penalties=P))
        return children

    def evolve(
        self,
        pop_size: int,
        generations: int,
        constructor: str = "balanced",
        pc: float = 0.9,
        pm: float = 0.3,
    ) -> Dict[str, Any]:
        pop = self._seed_population(pop_size, constructor)
        for _ in range(generations):
            parents = pop[:]
            children = self._variation(parents, pc=pc, pm=pm)
            combined = pop + children
            pop, fronts = select_nsga2(combined, pop_size, self.objectives, self.hard_penalty_keys)
            if fronts:
                nd = [combined[i] for i in fronts[0]]
                self.archive.update(nd)
        return {
            "pareto": self.archive.as_payload(),
            "archive_stats": self.archive.stats(),
        }
