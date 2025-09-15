# python/nsga_tools.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import math
import random

# ======== Data containers ========

@dataclass
class Individual:
    routes: List[List[int]]
    Z: Dict[str, float]           # {"Z1":..., "Z2":..., ..., "Z6":...}
    penalties: Dict[str, float]   # {"capacity":..., "duration":..., ...}
    meta: Dict[str, Any] = field(default_factory=dict)

    def obj_vector(self, keys: List[str]) -> List[float]:
        return [float(self.Z[k]) for k in keys]

    def total_penalty(self, keys: Optional[List[str]] = None) -> float:
        if keys is None:
            return float(sum(self.penalties.values()))
        return float(sum(self.penalties.get(k, 0.0) for k in keys))

# ======== Constraint-domination (Deb rules) ========
def constraint_compare(a: Individual, b: Individual, hard_penalty_keys: List[str]) -> int:
    """
    Returns:
      -1 if a dominates b
       1 if b dominates a
       0 otherwise
    """
    a_pen = a.total_penalty(hard_penalty_keys)
    b_pen = b.total_penalty(hard_penalty_keys)
    a_feas = a_pen <= 1e-12
    b_feas = b_pen <= 1e-12

    if a_feas and not b_feas:
        return -1
    if b_feas and not a_feas:
        return 1
    if not a_feas and not b_feas:
        if a_pen < b_pen - 1e-12:
            return -1
        if b_pen < a_pen - 1e-12:
            return 1
        return 0
    # both feasible -> Pareto comparison handled outside
    return 0

def dominates(a: List[float], b: List[float], eps: float = 0.0) -> bool:
    """Minimization objectives."""
    assert len(a) == len(b)
    not_worse = True
    strictly_better = False
    for ai, bi in zip(a, b):
        if ai > bi + eps:
            not_worse = False
            break
        if ai + eps < bi:
            strictly_better = True
    return not_worse and strictly_better

# ======== Fast non-dominated sorting (NSGA-II) ========
def fast_non_dominated_sort(
    pop: List[Individual],
    objective_keys: List[str],
    hard_penalty_keys: List[str],
    eps: float = 0.0,
) -> List[List[int]]:
    """
    Returns list of fronts; each front is a list of indices into pop.
    Applies constraint-domination first, then Pareto dominance on objective_keys.
    """
    S = [set() for _ in pop]   # who i dominates
    n = [0] * len(pop)         # how many dominate me
    fronts: List[List[int]] = [[]]

    obj = [ind.obj_vector(objective_keys) for ind in pop]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            cc = constraint_compare(pop[p], pop[q], hard_penalty_keys)
            if cc == -1:
                S[p].add(q); continue
            if cc == 1:
                n[p] += 1;   continue
            # both feasible/equally infeasible -> Pareto
            if dominates(obj[p], obj[q], eps):
                S[p].add(q)
            elif dominates(obj[q], obj[p], eps):
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts

# ======== Crowding distance ========
def crowding_distance(front: List[int], pop: List[Individual], objective_keys: List[str]) -> Dict[int, float]:
    if not front:
        return {}
    dist = {idx: 0.0 for idx in front}
    for k in objective_keys:
        # sort indices by objective k
        front_sorted = sorted(front, key=lambda i: pop[i].Z[k])
        fmin = pop[front_sorted[0]].Z[k]
        fmax = pop[front_sorted[-1]].Z[k]
        if abs(fmax - fmin) < 1e-12:
            continue
        # boundary points get inf crowding
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")
        for i in range(1, len(front_sorted) - 1):
            prev_v = pop[front_sorted[i - 1]].Z[k]
            next_v = pop[front_sorted[i + 1]].Z[k]
            dist[front_sorted[i]] += (next_v - prev_v) / (fmax - fmin)
    return dist

# ======== Survivor selection (μ+λ) by (rank, crowding) ========
def select_nsga2(
    combined: List[Individual],
    mu: int,
    objective_keys: List[str],
    hard_penalty_keys: List[str],
) -> Tuple[List[Individual], List[List[int]]]:
    fronts = fast_non_dominated_sort(combined, objective_keys, hard_penalty_keys)
    selected: List[Individual] = []
    for front in fronts:
        if len(selected) + len(front) <= mu:
            selected.extend(combined[i] for i in front)
        else:
            cd = crowding_distance(front, combined, objective_keys)
            front_sorted = sorted(front, key=lambda i: cd.get(i, 0.0), reverse=True)
            need = mu - len(selected)
            selected.extend(combined[i] for i in front_sorted[:need])
            break
    return selected, fronts

# ======== Elite archive (external) ========
class EliteArchive:
    """
    Maintains a (bounded) non-dominated set across generations.
    Uses constraint-domination + Pareto. Thins by crowding when full.
    """
    def __init__(self, max_size: int, objective_keys: List[str], hard_penalty_keys: List[str]):
        self.max_size = max_size
        self.objective_keys = objective_keys
        self.hard_penalty_keys = hard_penalty_keys
        self.items: List[Individual] = []

    def _insert_one(self, ind: Individual) -> None:
        new_items: List[Individual] = []
        dominated_new = False
        for cur in self.items:
            cc = constraint_compare(ind, cur, self.hard_penalty_keys)
            if cc == -1:
                # ind dominates cur -> drop cur
                continue
            if cc == 1:
                # cur dominates ind -> keep cur, reject ind
                dominated_new = True
                new_items.append(cur)
                continue
            # Pareto tie-break if both feasible/equally infeasible
            a = ind.obj_vector(self.objective_keys)
            b = cur.obj_vector(self.objective_keys)
            if dominates(a, b):
                continue
            if dominates(b, a):
                dominated_new = True
                new_items.append(cur)
                continue
            new_items.append(cur)
        if not dominated_new:
            new_items.append(ind)
        self.items = new_items

        # thin if needed
        if len(self.items) > self.max_size:
            front = list(range(len(self.items)))
            cd = crowding_distance(front, self.items, self.objective_keys)
            # remove the smallest-crowding first until size fits
            order = sorted(front, key=lambda i: cd.get(i, 0.0))
            to_remove = len(self.items) - self.max_size
            keep_idx = set(order[to_remove:])
            self.items = [self.items[i] for i in range(len(self.items)) if i in keep_idx]

    def update(self, inds: List[Individual]) -> None:
        for x in inds:
            self._insert_one(x)

    def as_payload(self) -> List[Dict[str, Any]]:
        payload = []
        for i in self.items:
            payload.append({
                "routes": i.routes,
                "Z": i.Z,
                "penalties": i.penalties,
            })
        return payload

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.items),
            "objective_keys": self.objective_keys,
            "hard_penalty_keys": self.hard_penalty_keys,
        }

# ======== Simple 2D hypervolume (minimization) for tests/metrics ========
def hypervolume_2d(points: List[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    """
    Points must be a non-dominated set for 2D minimization.
    Sort by first objective ascending; integrate rectangles to ref.
    """
    if not points:
        return 0.0
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    hv = 0.0
    for f1, f2 in pts:
        width = max(0.0, ref[0] - f1)
        height = max(0.0, ref[1] - f2)
        hv += width * height
        # tighten reference horizontally
        ref = (f1, ref[1])
    return hv
