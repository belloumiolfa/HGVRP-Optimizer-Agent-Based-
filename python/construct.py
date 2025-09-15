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
