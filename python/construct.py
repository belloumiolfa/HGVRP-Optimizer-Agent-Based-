# Heuristic constructors for initial VRP solutions:
# - construct_balanced: load-balances by assigning highest-demand customers first to the least-loaded feasible vehicle.
# - construct_greedy: nearest-add heuristic per vehicle (capacity-aware), with fallback to least-loaded vehicle.

from __future__ import annotations                     # Allow forward references in type hints
from typing import List, Set                            # Type hints for lists and sets
from .eval import AssignmentAgent                       # Access distances, customers, vehicles via AssignmentAgent


def construct_balanced(agent: AssignmentAgent) -> List[List[int]]:
    customers = [c.id for c in agent.customers_sorted]  # External IDs of customers in deterministic (sorted) order
    remaining: Set[int] = set(customers)                # Track customers not yet assigned
    routes: List[List[int]] = [[0, 0] for _ in agent.vehicles]  # Initialize one empty route per vehicle: [0, 0]
    loads = [0.0]*len(agent.vehicles)                   # Current load per vehicle
    demand_map = {c.id: c.demand for c in agent.customers_sorted}  # Quick lookup: customer id -> demand

    for cid in sorted(customers, key=lambda x: -demand_map[x]):    # Assign in descending demand order
        # All vehicles that can still fit this customer's demand, paired with their current load
        choices = [
            (vi, loads[vi])
            for vi, v in enumerate(agent.vehicles)
            if loads[vi] + demand_map[cid] <= v.capacity
        ]
        # Choose least-loaded feasible vehicle; if none feasible, pick the currently least-loaded vehicle anyway
        vi = min(choices, key=lambda t: t[1])[0] if choices else min(range(len(agent.vehicles)), key=lambda t: loads[t])
        routes[vi].insert(-1, cid)                      # Insert customer before the trailing depot 0
        loads[vi] += demand_map[cid]                    # Update vehicle load
        remaining.discard(cid)                          # Mark customer as assigned
    return routes                                       # Return constructed routes


def construct_greedy(agent: AssignmentAgent) -> List[List[int]]:
    remaining: Set[int] = set([c.id for c in agent.customers_sorted])  # All customers initially unassigned
    routes: List[List[int]] = [[0, 0] for _ in agent.vehicles]         # One empty route per vehicle: [0, 0]
    loads = [0.0]*len(agent.vehicles)                                   # Current load per vehicle

    while remaining:                                                    # Continue until all customers assigned
        progressed = False                                              # Track if any assignment happened this round

        for vi, veh in enumerate(agent.vehicles):                       # Iterate vehicles round-robin
            cur = routes[vi][-2]                                        # Last visited node (before trailing depot 0)
            best, best_d = None, float("inf")                           # Best next customer and its incremental distance

            for cid in list(remaining):                                 # Try each unassigned customer
                dem = agent.customers[cid].demand                       # Demand of candidate
                if loads[vi] + dem > veh.capacity: continue             # Skip if capacity would be exceeded
                d = agent._route_distance([cur, cid], veh)              # Distance from current node to candidate
                if d < best_d:                                          # Keep nearest feasible candidate
                    best, best_d = cid, d

            if best is not None:                                        # If a feasible nearest customer was found
                routes[vi].insert(-1, best)                             # Insert before trailing depot 0
                loads[vi] += agent.customers[best].demand               # Update load
                remaining.discard(best); progressed = True              # Mark assigned and note progress

        if not progressed:                                              # If no vehicle could add any customer (stuck)
            vi = min(range(len(agent.vehicles)), key=lambda t: loads[t])  # Choose least-loaded vehicle
            cid = remaining.pop()                                         # Take an arbitrary remaining customer
            routes[vi].insert(-1, cid)                                    # Insert it before trailing depot 0
            loads[vi] += agent.customers[cid].demand                      # Update load

    return routes                                                        # Return constructed routes
from typing import List, Set, Optional, Tuple  # ← extend your imports
import random

def _best_insertion_for_customer(
    agent: AssignmentAgent, veh, route: List[int], cid: int
) -> Tuple[float, int]:
    """
    Return (delta_distance, insert_pos) for inserting external customer `cid`
    into `route` (external IDs, e.g., [0, ..., 0]) for the given vehicle.
    We compute Δ = d(a,c) + d(c,b) − d(a,b) over every edge (a→b).
    """
    assert route and route[0] == 0 and route[-1] == 0
    best_delta = float("inf")
    best_pos = 1
    for pos in range(len(route) - 1):
        a = route[pos]
        b = route[pos + 1]
        base = agent._route_distance([a, b], veh)
        add  = agent._route_distance([a, cid], veh) + agent._route_distance([cid, b], veh)
        delta = add - base
        if delta < best_delta:
            best_delta, best_pos = delta, pos + 1
    return best_delta, best_pos


def construct_regret_k(
    agent: AssignmentAgent, k: int = 2, alpha: float = 0.0, seed: Optional[int] = None
) -> List[List[int]]:

    if seed is not None:
        random.seed(seed)

    # Start empty routes per vehicle: [0, 0]
    routes: List[List[int]] = [[0, 0] for _ in agent.vehicles]
    unserved: Set[int] = set(c.id for c in agent.customers_sorted)

    # Optional warm start: give each vehicle its nearest-to-depot unserved customer (unique if possible)
    used = set()
    for vi, veh in enumerate(agent.vehicles):
        best_c, best_score = None, float("inf")
        for cid in unserved:
            # simple out-and-back score to bias toward "close" customers early
            score = agent._route_distance([0, cid], veh) + agent._route_distance([cid, 0], veh)
            if score < best_score:
                best_score, best_c = score, cid
        if best_c is not None:
            routes[vi].insert(1, best_c)
            used.add(best_c)
    unserved -= used

    # Main regret loop
    while unserved:
        # For each candidate, collect best insertions across all vehicles
        # cid -> list of (delta, pos, veh_index), sorted by delta asc
        per_cand: dict[int, List[Tuple[float, int, int]]] = {}

        for cid in unserved:
            pack: List[Tuple[float, int, int]] = []
            for vi, veh in enumerate(agent.vehicles):
                d, pos = _best_insertion_for_customer(agent, veh, routes[vi], cid)
                pack.append((d, pos, vi))
            pack.sort(key=lambda t: t[0])
            per_cand[cid] = pack

        # Compute regret scores and pick the customer with largest regret
        # regret_k = sum_{i=2..k} (Δ_i − Δ_1). If fewer than k vehicles, use available.
        best_cid, best_score = None, -float("inf")
        for cid, pack in per_cand.items():
            base = pack[0][0]
            rk = 0.0
            for i in range(1, min(k, len(pack))):
                rk += (pack[i][0] - base)
            # tiny bias toward larger base cost (customers globally harder to place)
            score = rk + 1e-9 * base
            if alpha > 0.0:
                score += alpha * random.random()
            if score > best_score:
                best_score, best_cid = score, cid

        # Insert chosen customer at its best slot (first of its pack)
        delta, pos, vi = per_cand[best_cid][0]
        routes[vi].insert(pos, best_cid)
        unserved.remove(best_cid)

    return routes
