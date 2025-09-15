# python/ga.py
# python/ga.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import random
import math
import copy

# ----- Optional project pieces (use if available) -----
try:
    from .eval import AssignmentAgent  # type: ignore
except Exception:
    AssignmentAgent = None  # type: ignore

try:
    from .fitness import fitness_weighted  # type: ignore
except Exception:
    fitness_weighted = None  # type: ignore

try:
    from .construct import construct_balanced, construct_greedy  # type: ignore
except Exception:
    construct_balanced = construct_greedy = None  # type: ignore

# Legacy model types used by older tests
try:
    from .models import Instance, Routes  # type: ignore
except Exception:
    Instance = Any  # type: ignore
    Routes = Any    # type: ignore

# Moves for quick local search; expose names for monkeypatch in tests
try:
    from .moves import two_opt_once, or_opt1_intra_once  # type: ignore
    HAVE_LS = True
except Exception:
    HAVE_LS = False
    def two_opt_once(agent, route, veh):  # type: ignore
        return route, 0.0
    def or_opt1_intra_once(agent, route, veh):  # type: ignore
        return route, 0.0

# Phase-6 managers
from .adapt import OperatorBandit, BanditConfig
from .penalty_manager import PenaltyManager, PenaltyConfig as PMPenaltyConfig


# =========================
# GA Config (unified)
# Supports both legacy fields (population, cx_prob, mut_prob, ls_prob, etc.)
# and new Phase-6 fields (pop_size, crossover_rate, mutation_rate, ...)
# =========================
@dataclass
class GAConfig:
    # New-style
    pop_size: int = 20
    generations: int = 50
    crossover_rate: float = 0.0
    mutation_rate: float = 1.0
    tournament_k: int = 3
    seed: Optional[int] = None
    elitism: int = 1

    # Intensification (Phase-6)
    stagnation_S: int = 10
    intensify_top_k: int = 2
    intensify_iters: int = 5
    intensify_ops: List[str] = field(default_factory=lambda: ["two_opt", "oropt1", "cross"])
    enable_intensification: bool = True

    # -------- Legacy-compatible fields (optional) --------
    population: Optional[int] = None
    cx_prob: Optional[float] = None
    mut_prob: Optional[float] = None
    ls_prob: float = 0.0
    adaptive_operators: Optional[bool] = None
    adaptive_penalties: Optional[bool] = None
    pen_window: Optional[int] = None
    pen_gamma: Optional[float] = None
    bandit_tau: Optional[float] = None
    bandit_decay: Optional[float] = None

    def __post_init__(self):
        # Map legacy → new if provided
        if self.population is not None:
            self.pop_size = int(self.population)
        if self.cx_prob is not None:
            self.crossover_rate = float(self.cx_prob)
        if self.mut_prob is not None:
            self.mutation_rate = float(self.mut_prob)


# =========================
# Phase-6 Adaptive GA
# =========================
class AdaptiveGA:
    """
    Adaptive GA with:
     - Bandit over simple variation ops (two_opt / oropt1 / cross)
     - Stagnation detector → intensification burst on elites
     - Rolling adaptive penalties via PenaltyManager (if penalties are returned)
    """
    def __init__(
        self,
        instance: Dict[str, Any],
        ga_cfg: Optional[GAConfig] = None,
        bandit_cfg: Optional[BanditConfig] = None,
        penalty_cfg: Optional[PMPenaltyConfig] = None,
        evaluate_override: Optional[Callable[[Any], Tuple[float, Dict]]] = None,
        operators: Optional[List[str]] = None,
    ):
        self.instance = instance
        self.ga_cfg = ga_cfg or GAConfig()
        self.bandit = OperatorBandit(
            operators or ["two_opt", "oropt1", "cross"],
            bandit_cfg or BanditConfig(seed=self.ga_cfg.seed),
        )
        self.penalty = PenaltyManager(penalty_cfg or PMPenaltyConfig())
        self.rng = random.Random(self.ga_cfg.seed)

        # Evaluator hook
        if evaluate_override is not None:
            self.evaluate = evaluate_override
        else:
            self.evaluate = self._build_evaluator()

        self.history: List[Dict[str, Any]] = []
        self._op_history: List[Dict[str, float]] = []  # per-gen operator weights we expose

    # ---------- evaluation ----------
    def _build_evaluator(self) -> Callable[[Any], Tuple[float, Dict]]:
        """
        Prefer AssignmentAgent + fitness_weighted if available and constructible;
        otherwise, fallback to a geometric distance evaluator that works on dict instances.
        """
        # Try project evaluator, but fall back if it raises (e.g., tests pass dict instance)
        if AssignmentAgent is not None and fitness_weighted is not None:
            try:
                agent = AssignmentAgent(self.instance)
                def evaluator(sol: Any) -> Tuple[float, Dict]:
                    out = fitness_weighted(agent, sol)
                    fit = float(out.get("fitness", math.inf))
                    penalties = out.get("penalties", {})
                    if isinstance(penalties, dict):
                        self.penalty.observe(penalties)
                    return fit, out
                return evaluator
            except Exception:
                pass  # fallback below

        # ---- Fallback geometric evaluator ----
        coords: Dict[int, Tuple[float, float]] = {}
        depot = (self.instance.get("depot") or {})
        if "location" in depot and depot["location"]:
            depot_xy = tuple(map(float, depot["location"][:2]))
        else:
            depot_xy = (float(depot.get("x", 0.0)), float(depot.get("y", 0.0)))
        coords[0] = depot_xy
        for c in self.instance.get("customers", []):
            coords[int(c["id"])] = (float(c.get("x", 0.0)), float(c.get("y", 0.0)))

        def dist(a: int, b: int) -> float:
            (x1, y1), (x2, y2) = coords[a], coords[b]
            return math.hypot(x1 - x2, y1 - y2)

        def route_len(route: List[int]) -> float:
            if not route:
                return 0.0
            tot = dist(0, route[0])
            for i in range(len(route) - 1):
                tot += dist(route[i], route[i + 1])
            tot += dist(route[-1], 0)
            return tot

        def evaluator(sol: Any) -> Tuple[float, Dict]:
            if isinstance(sol, list) and sol and isinstance(sol[0], list):
                val = sum(route_len(r) for r in sol if r)
            else:
                val = route_len(list(sol))
            zeros = {k: 0.0 for k in self.penalty.get_betas().keys()}
            self.penalty.observe(zeros)
            return float(val), {"objectives": {"distance": val}, "penalties": zeros}
        return evaluator

    # ---------- util: extract learned values from bandit ----------
    def _extract_q_values(self) -> Optional[Dict[str, float]]:
        """
        Try to read learned per-op values from various common attribute names.
        Returns dict op->value, or None if nothing usable found.
        """
        candidates = ["q", "Q", "values", "means", "avg", "score", "scores", "v", "value"]
        for name in candidates:
            arr = getattr(self.bandit, name, None)
            if arr is None:
                continue
            # list/tuple aligned with ops
            if isinstance(arr, (list, tuple)) and len(arr) == len(self.bandit.ops):
                return {op: float(arr[i]) for i, op in enumerate(self.bandit.ops)}
            # dict keyed by op names or indices
            if isinstance(arr, dict):
                if any(k in self.bandit.ops for k in arr.keys()):
                    return {op: float(arr.get(op, 0.0)) for op in self.bandit.ops}
                if all(isinstance(k, int) for k in arr.keys()):
                    return {op: float(arr.get(i, 0.0)) for i, op in enumerate(self.bandit.ops)}
        return None

    # ---------- population ----------
    def _seed_solution(self) -> Any:
        if construct_balanced is not None:
            try:
                return construct_balanced(self.instance)
            except Exception:
                pass
        if construct_greedy is not None:
            try:
                return construct_greedy(self.instance)
            except Exception:
                pass
        ids = [int(c["id"]) for c in self.instance.get("customers", [])]
        self.rng.shuffle(ids)
        return ids

    def _init_population(self) -> List[Any]:
        return [self._seed_solution() for _ in range(self.ga_cfg.pop_size)]

    # ---------- simple variation ops (bandit domain) ----------
    def _apply_operator(self, op: str, sol: Any) -> Any:
        if isinstance(sol, list) and sol and isinstance(sol[0], list):
            base = [r[:] for r in sol]
            route = base[0]
        else:
            base = list(sol)
            route = base

        if op == "two_opt" and len(route) > 3:
            i = self.rng.randrange(1, len(route) - 2)
            k = self.rng.randrange(i + 1, len(route) - 1)
            route = route[:i] + list(reversed(route[i:k + 1])) + route[k + 1:]
        elif op == "oropt1" and len(route) > 3:
            i = self.rng.randrange(1, len(route) - 1)
            x = route.pop(i)
            j = self.rng.randrange(1, len(route))
            route = route[:j] + [x] + route[j:]
        elif op == "cross" and len(route) > 4:
            i, j = sorted(self.rng.sample(range(1, len(route) - 1), 2))
            route[i], route[j] = route[j], route[i]

        return [route] + base[1:] if (isinstance(sol, list) and sol and isinstance(sol[0], list)) else route

    # ---------- GA loop ----------
    def run(self, mode: str = "adaptive") -> Dict[str, Any]:
        pop = self._init_population()
        fits: List[float] = []
        metas: List[Dict] = []
        for s in pop:
            f, m = self.evaluate(s)
            fits.append(f)
            metas.append(m)

        best = min(fits)
        best_sol = copy.deepcopy(pop[fits.index(best)])
        stagnation = 0

        for gen in range(self.ga_cfg.generations):
            # Tournament selection
            def tour_pick() -> int:
                k = max(1, min(self.ga_cfg.tournament_k, len(pop)))
                cand = random.sample(range(len(pop)), k=k)
                return min(cand, key=lambda i: fits[i])

            pidx = tour_pick()
            parent = copy.deepcopy(pop[pidx])

            # Variation (bandit decides op)
            child = copy.deepcopy(parent)
            op_used = None
            if self.ga_cfg.mutation_rate > 0.0:
                if mode == "adaptive":
                    idx = self.bandit.select()
                    op_used = self.bandit.ops[idx]
                else:
                    op_used = random.choice(self.bandit.ops)
                child = self._apply_operator(op_used, child)

            # Evaluate
            f_parent, _ = self.evaluate(parent)
            f_child, meta_child = self.evaluate(child)

            # Bandit reward
            if mode == "adaptive" and op_used is not None:
                self.bandit.update(self.bandit.ops.index(op_used), max(0.0, f_parent - f_child))

            # Steady-state replacement preserving elites
            elite_idx = sorted(range(len(pop)), key=lambda i: fits[i])[:max(1, self.ga_cfg.elitism)]
            worst_idx = max(range(len(pop)), key=lambda i: (fits[i] if i not in elite_idx else -math.inf))
            pop[worst_idx] = child
            fits[worst_idx] = f_child
            metas[worst_idx] = meta_child

            # Penalties step
            current_betas = self.penalty.step()

            # Best & stagnation
            curr_best = min(fits)
            if curr_best + 1e-12 < best:
                best = curr_best
                best_sol = copy.deepcopy(pop[fits.index(best)])
                stagnation = 0
            else:
                stagnation += 1

            # Intensification
            from .localsearch import intensify_elites  # lazy import to avoid cycles
            intensify_used: Dict[str, int] = {}
            if self.ga_cfg.enable_intensification and self.ga_cfg.stagnation_S > 0 and stagnation >= self.ga_cfg.stagnation_S:
                pop, fits, intensify_used = intensify_elites(
                    rng=random.Random(self.ga_cfg.seed),
                    population=pop,
                    fitnesses=fits,
                    evaluate=self.evaluate,
                    top_k=self.ga_cfg.intensify_top_k,
                    ops=self.ga_cfg.intensify_ops,
                    iters_per=self.ga_cfg.intensify_iters,
                    agent=None,
                    vehicles=None,
                )
                new_best = min(fits)
                if new_best + 1e-12 < best:
                    best = new_best
                    best_sol = copy.deepcopy(pop[fits.index(best)])
                    stagnation = 0

            # Log — prefer learned values (Q) over policy snapshot
            qvals = self._extract_q_values()
            if qvals:
                total = sum(max(0.0, v) for v in qvals.values())
                if total > 0:
                    weights_snap = {op: max(0.0, qvals.get(op, 0.0)) / total for op in self.bandit.ops}
                else:
                    weights_snap = self.bandit.snapshot()
            else:
                weights_snap = self.bandit.snapshot()

            self.history.append({
                "gen": gen,
                "best_fitness": best,
                "operator_weights": weights_snap,
                "betas": current_betas,
                "stagnation": stagnation,
                "intensify_used": intensify_used,
            })
            self._op_history.append(weights_snap)

        return {
            "best_fitness": best,
            "best_solution": best_sol,
            "history": self.history,
            "operator_history": self._op_history,  # expose our computed weights per gen
            "penalty_history": self.penalty.history(),
        }


# =================================================================
# Legacy API (shims) required by existing tests in tests/test_ga.py
# =================================================================

def all_customer_ids(instance: Instance) -> List[int]:
    return [c.id for c in instance.customers]

def vehicle_ids(instance: Instance) -> List[int]:
    return [v.id for v in instance.vehicles]

def coords_of_customer(instance: Instance, nid: int) -> Tuple[float, float]:
    if nid == 0:
        d0 = instance.depots[0]
        return (float(d0.x or 0.0), float(d0.y or 0.0))
    for c in instance.customers:
        if c.id == nid:
            return (float(c.x or 0.0), float(c.y or 0.0))
    return (0.0, 0.0)

def dist_proxy(instance: Instance, a: int, b: int) -> float:
    ax, ay = coords_of_customer(instance, a)
    bx, by = coords_of_customer(instance, b)
    return math.hypot(ax - bx, ay - by)

def giant_tour_from_routes(routes: Routes) -> List[int]:
    perm: List[int] = []
    for _, seq in sorted(routes.routes.items()):
        perm.extend([nid for nid in seq if nid != 0])
    return perm

def split_capacity_greedy(instance: Instance, perm: List[int]) -> Routes:
    vids = vehicle_ids(instance)
    V = {v.id: v for v in instance.vehicles}
    C = {c.id: c for c in instance.customers}

    routes: Dict[int, List[int]] = {vid: [0, 0] for vid in vids}
    load: Dict[int, float] = {vid: 0.0 for vid in vids}

    vi = 0
    for cid in perm:
        placed = False
        for tries in range(len(vids)):
            vid = vids[(vi + tries) % len(vids)]
            cap = V[vid].capacity if V[vid].capacity is not None else float("inf")
            dem = C[cid].demand if C[cid].demand is not None else 0.0
            if load[vid] + dem <= cap + 1e-9:
                routes[vid].insert(len(routes[vid]) - 1, cid)
                load[vid] += dem
                vi = (vi + tries + 1) % len(vids)
                placed = True
                break
        if not placed:
            vid = min(vids, key=lambda x: load[x])
            routes[vid].insert(len(routes[vid]) - 1, cid)
            load[vid] += C[cid].demand if C[cid].demand is not None else 0.0
    return Routes(routes=routes)

def bcrc_crossover(instance: Instance,
                   parentA: Tuple[List[int], Routes],
                   parentB: Tuple[List[int], Routes]) -> List[int]:
    permA, routesA = parentA
    permB, _ = parentB
    non_empty = [seq for seq in routesA.routes.values() if len(seq) > 2]
    if not non_empty:
        return permB[:]
    route = random.choice(non_empty)
    S = [x for x in route if x != 0]

    child = [c for c in permB if c not in S]
    for c in S:
        if not child:
            child.append(c)
            continue
        best_pos, best_delta = 0, float("inf")
        for pos in range(len(child) + 1):
            left = 0 if pos == 0 else child[pos - 1]
            right = 0 if pos == len(child) else child[pos]
            delta = (dist_proxy(instance, left, c) +
                     dist_proxy(instance, c, right) -
                     dist_proxy(instance, left, right))
            if delta < best_delta:
                best_delta, best_pos = delta, pos
        child.insert(best_pos, c)
    return child

def mut_swap(perm: List[int]) -> None:
    if len(perm) < 2: return
    i, j = random.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]

def mut_insert(perm: List[int]) -> None:
    if len(perm) < 2: return
    i, j = random.sample(range(len(perm)), 2)
    c = perm.pop(i)
    j = random.randrange(0, len(perm) + 1)
    perm.insert(j, c)

def mut_oropt1(perm: List[int]) -> None:
    if len(perm) < 3: return
    i = random.randrange(len(perm))
    c = perm.pop(i)
    j = random.randrange(0, len(perm) + 1)
    perm.insert(j, c)

def quick_ls(instance: Instance, routes: Routes) -> Routes:
    if not HAVE_LS:
        return routes
    try:
        agent = AssignmentAgent(instance) if AssignmentAgent is not None else None
        improved = {}
        for vid, seq in routes.routes.items():
            if len(seq) <= 3:
                improved[vid] = seq[:]
                continue
            veh = agent.vehicle_by_id(vid) if (agent is not None and hasattr(agent, "vehicle_by_id")) else None
            new_seq, _ = two_opt_once(agent, seq, veh)
            new_seq, _ = or_opt1_intra_once(agent, new_seq, veh)
            improved[vid] = new_seq
        return Routes(routes=improved)
    except Exception:
        return routes

# ---- Seed population & optimize_single (legacy tests) ----
def _evaluate_metrics(instance: Instance, routes: Routes, penalties: Any) -> Dict[str, Any]:
    if AssignmentAgent is None:
        return {"objectives": {"Z1": 0.0}, "penalties": {}}
    agent = AssignmentAgent(instance)
    if hasattr(agent, "evaluate_solution"):
        return agent.evaluate_solution(routes, penalties)
    if hasattr(agent, "evaluate"):
        return agent.evaluate(routes, penalties)
    raise RuntimeError("AssignmentAgent must expose evaluate(...) or evaluate_solution(...).")

def eval_perm(instance: Instance, penalties: Any, perm: List[int], wZ=None, wP=None) -> Tuple[Routes, Dict[str, Any], float]:
    routes = split_capacity_greedy(instance, perm)
    metrics = _evaluate_metrics(instance, routes, penalties)
    if fitness_weighted is None:
        fit = float(metrics.get("objectives", {}).get("Z1", 0.0))
    else:
        fit = fitness_weighted(metrics["objectives"], metrics["penalties"], wZ, wP)
    return routes, metrics, fit

def seed_population(instance: Instance, penalties: Any, pop_size: int, wZ=None, wP=None, rnd_ratio: float = 0.5):
    pop: List[Tuple[List[int], Routes, Dict[str, Any], float]] = []
    C = all_customer_ids(instance)
    seeds: List[Routes] = []
    try:
        if construct_balanced is not None:
            seeds.append(construct_balanced(instance))
    except Exception:
        pass
    try:
        if construct_greedy is not None:
            seeds.append(construct_greedy(instance))
    except Exception:
        pass

    for rts in seeds:
        perm = giant_tour_from_routes(rts)
        routes, metrics, fit = eval_perm(instance, penalties, perm, wZ, wP)
        pop.append((perm, routes, metrics, fit))

    need = max(0, pop_size - len(pop))
    rnd_n = int(need * rnd_ratio)
    for _ in range(rnd_n):
        perm = C[:]
        random.shuffle(perm)
        routes, metrics, fit = eval_perm(instance, penalties, perm, wZ, wP)
        pop.append((perm, routes, metrics, fit))

    while len(pop) < pop_size:
        base = pop[random.randrange(max(1, len(pop)))][0][:] if pop else C[:]
        if len(base) >= 2:
            mut_swap(base)
        routes, metrics, fit = eval_perm(instance, penalties, base, wZ, wP)
        pop.append((base, routes, metrics, fit))

    pop.sort(key=lambda x: x[3])
    return pop

def tournament(pop, k: int):
    cand = random.sample(pop, min(k, len(pop)))
    cand.sort(key=lambda x: x[3])
    return cand[0]

def optimize_single(instance: Instance, penalties: Any, cfg: GAConfig, wZ=None, wP=None) -> Dict[str, Any]:
    if cfg.seed is not None:
        random.seed(cfg.seed)

    # rolling window of violation snapshots for adaptation
    window = []
    W = int(cfg.pen_window or 0)
    gamma = float(cfg.pen_gamma or 1.0)

    # helper to extract beta_* values
    def dump_betas():
        try:
            return penalties.model_dump()
        except Exception:
            return {k: getattr(penalties, k) for k in dir(penalties) if k.startswith("beta_")}

    pop = seed_population(instance, penalties, cfg.pop_size, wZ, wP)
    history = [{"gen": 0, "best_fitness": pop[0][3]}]
    telemetry = {"bandit_probs": [], "betas": []}

    for gen in range(1, cfg.generations + 1):
        new_pop: List[Tuple[List[int], Routes, Dict[str, Any], float]] = []
        elites = pop[:cfg.elitism]
        new_pop.extend([(e[0][:], e[1], e[2], e[3]) for e in elites])

        while len(new_pop) < cfg.pop_size:
            pA = tournament(pop, max(1, cfg.tournament_k))
            pB = tournament(pop, max(1, cfg.tournament_k))
            child_perm = pA[0][:]

            if (cfg.crossover_rate or 0.0) > 0.0:
                child_perm = bcrc_crossover(instance, (pA[0], pA[1]), (pB[0], pB[1]))

            if (cfg.mutation_rate or 0.0) > 0.0:
                random.choice([mut_swap, mut_insert, mut_oropt1])(child_perm)

            routes, metrics, fit = eval_perm(instance, penalties, child_perm, wZ, wP)

            if cfg.ls_prob and random.random() < cfg.ls_prob:
                routes2 = quick_ls(instance, routes)
                try:
                    metrics2 = _evaluate_metrics(instance, routes2, penalties)
                    fit2 = (fitness_weighted(metrics2["objectives"], metrics2["penalties"], wZ, wP)
                            if fitness_weighted is not None else float(metrics2.get("objectives", {}).get("Z1", 0.0)))
                    if fit2 < fit:
                        routes, metrics, fit = routes2, metrics2, fit2
                except Exception:
                    pass

            new_pop.append((child_perm, routes, metrics, fit))

        new_pop.sort(key=lambda x: x[3])
        pop = new_pop
        best_fit = pop[0][3]
        history.append({"gen": gen, "best_fitness": best_fit})

        # ---- Penalty adaptation & telemetry ----
        viol = pop[0][2].get("penalties", {}) if isinstance(pop[0][2], dict) else {}
        window.append(viol)
        if W and len(window) > W:
            window.pop(0)

        if cfg.adaptive_penalties and W and gamma and len(window) == W:
            pos = 0
            tot = 0
            for snap in window:
                for v in snap.values():
                    if isinstance(v, (int, float)):
                        pos += 1 if v > 1e-12 else 0
                        tot += 1
            rate = (pos / tot) if tot > 0 else 0.0
            if rate > 0.99:
                for k, v in dump_betas().items():
                    if k.startswith("beta_"):
                        nv = v * gamma
                        try:
                            setattr(penalties, k, nv)
                        except Exception:
                            penalties.__dict__[k] = nv
            elif rate < 1e-9:
                inv = (1.0 / gamma) if gamma > 1e-12 else 1.0
                for k, v in dump_betas().items():
                    if k.startswith("beta_"):
                        nv = v * inv
                        try:
                            setattr(penalties, k, nv)
                        except Exception:
                            penalties.__dict__[k] = nv

        telemetry["betas"].append(dump_betas())
        telemetry["bandit_probs"].append({"swap": 1/3, "insert": 1/3, "oropt1": 1/3})

    return {
        "best": {
            "routes": pop[0][1].routes,
            "metrics": pop[0][2],
            "fitness": pop[0][3],
            "perm": pop[0][0],
        },
        "history": history,
        "telemetry": telemetry,
    }
