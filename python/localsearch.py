from __future__ import annotations
from typing import List, Tuple, Callable, Dict, Optional, Any
import random
from copy import deepcopy

# Try project-optimized move ops; otherwise use light fallbacks.
try:
    from .moves import two_opt_once as _two_opt_once  # type: ignore
except Exception:
    _two_opt_once = None

try:
    from .moves import or_opt1_intra_once as _oropt1_once  # type: ignore
except Exception:
    _oropt1_once = None

try:
    from .moves import cross_exchange_once as _cross_once  # type: ignore
except Exception:
    _cross_once = None


# ---------- tiny fallbacks (structure-only) ----------
def _fallback_two_opt(route: List[int]) -> Tuple[List[int], float]:
    if len(route) <= 3:
        return route, 0.0
    # simple best-of-one reversal
    i = 1
    k = len(route) - 2
    new = route[:i] + list(reversed(route[i:k + 1])) + route[k + 1:]
    return new, 1.0 if new != route else 0.0

def _fallback_oropt1(route: List[int]) -> Tuple[List[int], float]:
    if len(route) <= 3:
        return route, 0.0
    i = max(1, min(len(route) - 2, 1))
    x = route[i]
    rest = route[:i] + route[i + 1:]
    j = min(len(rest) - 1, 1)
    new = rest[:j] + [x] + rest[j:]
    return new, 1.0 if new != route else 0.0

def _fallback_cross_two(routes: List[List[int]]) -> Tuple[List[List[int]], float]:
    if len(routes) < 2:
        return routes, 0.0
    r0, r1 = routes[0][:], routes[1][:]
    if len(r0) > 3 and len(r1) > 3:
        cut0 = max(1, len(r0)//3)
        cut1 = max(1, len(r1)//3)
        new0 = r0[:cut0] + r1[cut1:]
        new1 = r1[:cut1] + r0[cut0:]
        if (new0, new1) != (r0, r1):
            return [new0, new1] + routes[2:], 1.0
    return routes, 0.0


# ---------- thin adapters around project ops ----------
def two_opt_once(agent, route: List[int], veh=None) -> Tuple[List[int], float]:
    if _two_opt_once is not None and agent is not None and veh is not None:
        return _two_opt_once(agent, route, veh)
    return _fallback_two_opt(route)

def or_opt1_intra_once(agent, route: List[int], veh=None) -> Tuple[List[int], float]:
    if _oropt1_once is not None and agent is not None and veh is not None:
        return _oropt1_once(agent, route, veh)
    return _fallback_oropt1(route)

def cross_exchange_once(agent, routes: List[List[int]], vehicles=None) -> Tuple[List[List[int]], float]:
    if _cross_once is not None and agent is not None and vehicles is not None:
        return _cross_once(agent, routes, vehicles)
    return _fallback_cross_two(routes)


# ---------- intensification utility used by GA ----------
def intensify_elites(
    rng: random.Random,
    population: List,
    fitnesses: List[float],
    evaluate: Callable[[Any], Tuple[float, Dict]],
    top_k: int = 2,
    ops: Optional[List[str]] = None,
    iters_per: int = 5,
    agent=None,
    vehicles=None
) -> Tuple[List, List[float], Dict[str, int]]:
    if ops is None:
        ops = ["two_opt", "oropt1", "cross"]

    indices = sorted(range(len(population)), key=lambda i: fitnesses[i])[:max(1, top_k)]
    used = {"two_opt": 0, "oropt1": 0, "cross": 0}

    for idx in indices:
        sol = population[idx]
        fit, _ = evaluate(sol)
        routes = [r[:] for r in sol] if (isinstance(sol, list) and sol and isinstance(sol[0], list)) else [sol[:]]

        for _ in range(max(1, iters_per)):
            op = rng.choice(ops)
            if op == "two_opt":
                new0, _ = two_opt_once(agent, routes[0], veh=None)
                candidate = [new0] + routes[1:]
            elif op == "oropt1":
                new0, _ = or_opt1_intra_once(agent, routes[0], veh=None)
                candidate = [new0] + routes[1:]
            else:
                candidate, _ = cross_exchange_once(agent, routes, vehicles)
            cand_fit, _ = evaluate(candidate if len(candidate) > 1 else candidate[0])
            if cand_fit + 1e-12 < fit:
                sol = candidate if len(candidate) > 1 else candidate[0]
                fit = cand_fit
                routes = candidate if len(candidate) > 1 else [candidate]
                used[op] += 1

        population[idx] = sol
        fitnesses[idx] = fit

    return population, fitnesses, used


# ---------- helper: normalize various route shapes ----------
def _normalize_routes_input(routes: Any) -> Tuple[str, List[List[int]], Optional[List[int]]]:
    """
    Returns:
      kind: "list" | "dict" | "model"
      arr:  list of NON-DEPOT sequences (e.g., [[1,2],[3]])
      keys: original keys if dict/model (to rebuild), else None
    Accepts:
      - list of lists: [[0,1,0],[0,2,0]]
      - dict: {"routes": {101: [0,1,0], 102: [0,2,0]}}
      - model with .routes dict
    """
    # list-of-lists
    if isinstance(routes, list):
        arr = []
        for seq in routes:
            if not isinstance(seq, list):
                raise ValueError("Each route must be a list")
            inner = [n for n in seq if n != 0]
            arr.append(inner)
        return "list", arr, None

    # dict {"routes": {...}}
    if isinstance(routes, dict) and "routes" in routes:
        rmap = routes["routes"]
        if not isinstance(rmap, dict):
            raise ValueError("'routes' must be a dict of {id: [nodes]}")
        keys = sorted(rmap.keys())
        arr = []
        for k in keys:
            seq = rmap[k]
            inner = [n for n in seq if n != 0]
            arr.append(inner)
        return "dict", arr, keys

    # model with .routes
    if hasattr(routes, "routes"):
        rmap = routes.routes
        keys = sorted(rmap.keys())
        arr = []
        for k in keys:
            seq = rmap[k]
            inner = [n for n in seq if n != 0]
            arr.append(inner)
        return "model", arr, keys

    raise ValueError("Unsupported routes format")


def _build_for_eval(kind: str, arr: List[List[int]], keys: Optional[List[int]]) -> Any:
    """Rebuilds a structure like the input to feed into agent.evaluate."""
    if kind == "list":
        return [[0] + r + [0] for r in arr]
    cmap = {keys[i]: [0] + arr[i] + [0] for i in range(len(arr))}
    return {"routes": cmap}  # works for both dict-based and model-based agent.evaluate usages


def _build_for_output(kind: str, arr: List[List[int]], keys: Optional[List[int]]) -> Any:
    """Rebuilds improved routes in the same shape as input."""
    if kind == "list":
        return [[0] + r + [0] for r in arr]
    cmap = {keys[i]: [0] + arr[i] + [0] for i in range(len(arr))}
    if kind == "dict":
        return {"routes": cmap}
    # kind == "model": keep it simple & consistent with /evaluate by returning the dict form
    return {"routes": cmap}


# ---------- helper: greedy CROSS using evaluation ----------
def _cross_greedy_improve(
    arr: List[List[int]],
    eval_fn: Callable[[List[List[int]]], float]
) -> Tuple[List[List[int]], float, bool]:
    """
    Try all single-customer exchanges between the first two routes.
    Return (best_arr, best_fit, improved_flag).
    """
    if len(arr) < 2 or not arr[0] or not arr[1]:
        return arr, eval_fn(arr), False

    base_fit = eval_fn(arr)
    best_fit = base_fit
    best_arr = None

    r0 = arr[0]
    r1 = arr[1]

    # 1) Swap customer i in r0 with customer j in r1
    for i in range(len(r0)):
        for j in range(len(r1)):
            cand = deepcopy(arr)
            cand[0][i], cand[1][j] = cand[1][j], cand[0][i]
            f = eval_fn(cand)
            if f + 1e-12 < best_fit:
                best_fit = f
                best_arr = cand

    # 2) Move i from r0 into r1 (all positions)
    for i in range(len(r0)):
        for pos in range(len(r1) + 1):
            cand = deepcopy(arr)
            x = cand[0].pop(i)
            cand[1].insert(pos, x)
            f = eval_fn(cand)
            if f + 1e-12 < best_fit:
                best_fit = f
                best_arr = cand

    # 3) Move j from r1 into r0 (all positions)
    for j in range(len(r1)):
        for pos in range(len(r0) + 1):
            cand = deepcopy(arr)
            x = cand[1].pop(j)
            cand[0].insert(pos, x)
            f = eval_fn(cand)
            if f + 1e-12 < best_fit:
                best_fit = f
                best_arr = cand

    if best_arr is None:
        return arr, base_fit, False
    return best_arr, best_fit, True


# ---------- API-used local search ----------
def apply_localsearch(agent, routes, ops: List[str], budget: int, rng: random.Random, penalty=None):
    """
    Repeatedly apply ops (two_opt/oropt1/cross) to improve the solution.
    Handles input routes as list-of-lists, dict{'routes':...}, or model with .routes.
    Returns (routes_like_input, improved: bool).
    """
    kind, arr, keys = _normalize_routes_input(routes)

    # Eval using the SAME shape the agent already accepts (mirrors /evaluate)
    def eval_arr(a: List[List[int]]) -> float:
        shaped = _build_for_eval(kind, a, keys)
        res = agent.evaluate(shaped, penalty or {})
        if isinstance(res, dict) and "objectives" in res:
            return float(res["objectives"].get("Z1", 0.0))
        # Soft fallback: proxy by total length
        return sum(len(r) for r in a)

    best_fit = eval_arr(arr)
    improved = False

    for _ in range(max(1, budget)):
        op = rng.choice(ops or ["two_opt", "oropt1", "cross"])
        candidate = [r[:] for r in arr]

        if op == "two_opt" and candidate:
            candidate[0], _ = two_opt_once(agent, candidate[0], veh=None)
            cand_fit = eval_arr(candidate)

        elif op == "oropt1" and candidate:
            candidate[0], _ = or_opt1_intra_once(agent, candidate[0], veh=None)
            cand_fit = eval_arr(candidate)

        elif op == "cross":
            # Prefer a greedy improving cross using evaluation (robust for test fixtures)
            cand2, cand_fit, did = _cross_greedy_improve(arr, eval_arr)
            if did and cand_fit + 1e-12 < best_fit:
                arr = cand2
                best_fit = cand_fit
                improved = True
            # Skip the generic candidate replacement path for cross since we already applied if improved.
            continue

        else:
            cand_fit = eval_arr(candidate)

        if cand_fit + 1e-12 < best_fit:
            arr = candidate
            best_fit = cand_fit
            improved = True

    out_routes = _build_for_output(kind, arr, keys)
    return out_routes, improved
