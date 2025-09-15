# python/ga_pareto.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Callable
import random

from .nsga_tools import Individual, EliteArchive, select_nsga2
from .models import Instance, Routes, PenaltyConfig, Vehicle
from .eval import AssignmentAgent
from .construct import construct_balanced, construct_greedy

# Optional LS if present
try:
    from .moves import two_opt_once, or_opt1_intra_once
    HAVE_LS = True
except Exception:
    HAVE_LS = False

DEFAULT_OBJECTIVES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]

def evaluate_routes(agent: AssignmentAgent, routes: List[List[int]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns (Z, penalties). Assumes your AssignmentAgent already computes these per your Chapter 2.
    """
    # Agent should expose an evaluation that returns objectives (Z1..Z6) and penalties dict
    # If your agent's API differs, adapt here.
    out = agent.evaluate_routes(routes)
    # Expecting out like: {"Z": {...}, "penalties": {...}}
    return out["Z"], out["penalties"]

def random_selection(pop: List[Individual], k: int) -> List[Individual]:
    return random.sample(pop, k)

def simple_crossover(p1: Individual, p2: Individual) -> List[List[List[int]]]:
    """
    Placeholder crossover: route-level mix. Replace by your BCRC if available.
    """
    a = [r[:] for r in p1.routes]
    b = [r[:] for r in p2.routes]
    if not a or not b:
        return [a, b]
    cut_a = random.randrange(len(a))
    cut_b = random.randrange(len(b))
    child1 = a[:cut_a] + b[cut_b:]
    child2 = b[:cut_b] + a[cut_a:]
    return [child1, child2]

def simple_mutation(routes: List[List[int]]) -> List[List[int]]:
    """
    Very light mutation: swap two customers in a random route.
    """
    if len(routes) == 0:
        return routes
    r = random.randrange(len(routes))
    rt = routes[r][:]
    if len(rt) > 3:
        i = random.randrange(1, len(rt)-1)
        j = random.randrange(1, len(rt)-1)
        if i != j:
            rt[i], rt[j] = rt[j], rt[i]
    new = [rr[:] for rr in routes]
    new[r] = rt
    return new

def maybe_local_search(agent: AssignmentAgent, routes: List[List[int]], prob: float = 0.3) -> List[List[int]]:
    if not HAVE_LS or random.random() > prob:
        return routes
    # Intra-route quick improvement
    veh = agent.instance.vehicles[0] if agent.instance.vehicles else None
    best = [r[:] for r in routes]
    for ridx, r in enumerate(routes):
        if len(r) <= 3: 
            continue
        r2, _ = two_opt_once(agent, r, veh)
        r3, _ = or_opt1_intra_once(agent, r2, veh)
        best[ridx] = r3
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
        self.instance = instance
        self.penalty_cfg = penalty_cfg
        self.objectives = objectives or DEFAULT_OBJECTIVES
        # Treat these penalties as "hard" constraints
        self.hard_penalty_keys = hard_penalty_keys or [
            "capacity", "duration", "timewindows", "availability", "battery", "coverage", "supply"
        ]
        self.rng = random.Random(seed)
        self.agent = AssignmentAgent(instance, penalty_cfg)
        self.archive = EliteArchive(archive_size, self.objectives, self.hard_penalty_keys)

    def _seed_population(self, size: int, constructor: str = "balanced") -> List[Individual]:
        pop: List[Individual] = []
        for _ in range(size):
            if constructor == "balanced":
                routes: List[List[int]] = construct_balanced(self.instance)
            else:
                routes = construct_greedy(self.instance)
            Z, P = evaluate_routes(self.agent, routes)
            pop.append(Individual(routes=routes, Z=Z, penalties=P))
        self.archive.update(pop)
        return pop

    def _variation(self, parents: List[Individual], pc: float = 0.9, pm: float = 0.3) -> List[Individual]:
        children: List[Individual] = []
        self.rng.shuffle(parents)
        pairs = zip(parents[::2], parents[1::2])
        for p1, p2 in pairs:
            if self.rng.random() < pc:
                ch1, ch2 = simple_crossover(p1, p2)
            else:
                ch1, ch2 = [r[:] for r in p1.routes], [r[:] for r in p2.routes]
            if self.rng.random() < pm:
                ch1 = simple_mutation(ch1)
            if self.rng.random() < pm:
                ch2 = simple_mutation(ch2)
            ch1 = maybe_local_search(self.agent, ch1, prob=0.25)
            ch2 = maybe_local_search(self.agent, ch2, prob=0.25)
            for ch in (ch1, ch2):
                Z, P = evaluate_routes(self.agent, ch)
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
            # parent selection: tournament on rank+crowding by reusing selection on combined with duplicates filtered
            # Simple: uniform random parents
            parents = pop[:]  # could do tournaments for diversity
            children = self._variation(parents, pc=pc, pm=pm)
            combined = pop + children
            pop, fronts = select_nsga2(combined, pop_size, self.objectives, self.hard_penalty_keys)
            # update archive with the *first* non-dominated front of this gen
            if fronts:
                nd = [combined[i] for i in fronts[0]]
                self.archive.update(nd)

        pareto_payload = self.archive.as_payload()
        return {
            "pareto": pareto_payload,
            "archive_stats": self.archive.stats(),
        }
