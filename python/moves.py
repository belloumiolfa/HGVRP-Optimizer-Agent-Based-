from typing import List, Tuple
from .agents.assignment_agent import AssignmentAgent
from .models import Vehicle

def two_opt_once(agent: AssignmentAgent, route: List[int], veh: Vehicle) -> Tuple[List[int], float]:
    if len(route) <= 3:
        return route, 0.0
    best_delta = 0.0
    best = None
    base = agent._route_distance(route, veh)
    for i in range(1, len(route)-2):
        for k in range(i+1, len(route)-1):
            new = route[:i] + list(reversed(route[i:k+1])) + route[k+1:]
            d = agent._route_distance(new, veh)
            delta = base - d
            if delta > best_delta + 1e-12:
                best_delta = delta
                best = new
    if best is None:
        return route, 0.0
    return best, best_delta

def or_opt1_intra_once(agent: AssignmentAgent, route: List[int], veh: Vehicle) -> Tuple[List[int], float]:
    if len(route) <= 3:
        return route, 0.0
    base = agent._route_distance(route, veh)
    best_delta = 0.0
    best = None
    for i in range(1, len(route)-1):
        if route[i] == 0: continue
        for j in range(1, len(route)):
            if j == i or j == i+1: continue
            new = route[:i] + route[i+1:]
            new = new[:j] + [route[i]] + new[j:]
            d = agent._route_distance(new, veh)
            delta = base - d
            if delta > best_delta + 1e-12:
                best_delta = delta
                best = new
    if best is None:
        return route, 0.0
    return best, best_delta

def or_opt1_inter_once(agent: AssignmentAgent, r1: List[int], v1: Vehicle, r2: List[int], v2: Vehicle) -> Tuple[List[int], List[int], float]:
    if len(r1) <= 3:
        return r1, r2, 0.0
    base = agent._route_distance(r1, v1) + agent._route_distance(r2, v2)
    best_delta = 0.0
    best_pair = None
    for i in range(1, len(r1)-1):
        if r1[i] == 0: continue
        removed = r1[i]
        new1 = r1[:i] + r1[i+1:]
        for j in range(1, len(r2)):
            new2 = r2[:j] + [removed] + r2[j:]
            d = agent._route_distance(new1, v1) + agent._route_distance(new2, v2)
            delta = base - d
            if delta > best_delta + 1e-12:
                best_delta = delta
                best_pair = (new1, new2)
    if best_pair is None:
        return r1, r2, 0.0
    return best_pair[0], best_pair[1], best_delta


# --- Generic Or-opt(k) intra: move a contiguous block of length k within the same route ---
def or_opt_k_intra_once(agent, route, veh, k: int):
    """
    Move one contiguous block of length k within the same route.
    Returns (new_route, gain). gain > 0 means improved (distance decreased).
    """
    n = len(route)
    if n <= 3 or k <= 0 or (n - 2) < k:  # need at least k customers (exclude depots)
        return route, 0.0

    base = agent._route_distance(route, veh)
    best_delta, best = 0.0, None

    # customers are in indices [1 .. n-2]
    for i in range(1, n - 1):          # start of block
        j = i + k - 1                  # end of block (inclusive)
        if j >= n - 1:
            break
        # block must not include depots
        if any(x == 0 for x in route[i:j+1]):
            continue

        block = route[i:j+1]
        rest = route[:i] + route[j+1:]

        # try inserting block at every valid position in 'rest'
        for pos in range(1, len(rest)):   # insert before last depot
            # skip reinserting exactly where it was
            if pos == i:
                continue
            new_route = rest[:pos] + block + rest[pos:]
            d = agent._route_distance(new_route, veh)
            delta = base - d
            if delta > best_delta + 1e-12:
                best_delta, best = delta, new_route

    return (best if best else route), (best_delta if best else 0.0)

def or_opt2_intra_once(agent, route, veh):
    return or_opt_k_intra_once(agent, route, veh, k=2)


def or_opt3_intra_once(agent, route, veh):
    return or_opt_k_intra_once(agent, route, veh, k=3)


def cross_exchange_once(agent: AssignmentAgent, r1: List[int], v1: Vehicle,
                        r2: List[int], v2: Vehicle) -> Tuple[List[int], List[int], float]:
    """Try swapping a random contiguous segment of r1 with one from r2."""
    if len(r1) <= 3 or len(r2) <= 3:
        return r1, r2, 0.0
    base = agent._route_distance(r1,v1)+agent._route_distance(r2,v2)
    best_delta=0.0; best=None
    for i in range(1,len(r1)-1):
        for j in range(i,len(r1)-1):
            seg1 = r1[i:j+1]
            rest1 = r1[:i]+r1[j+1:]
            for k in range(1,len(r2)-1):
                for l in range(k,len(r2)-1):
                    seg2 = r2[k:l+1]
                    rest2 = r2[:k]+r2[l+1:]
                    new1 = rest1[:i]+seg2+rest1[i:]
                    new2 = rest2[:k]+seg1+rest2[k:]
                    d = agent._route_distance(new1,v1)+agent._route_distance(new2,v2)
                    delta = base - d
                    if delta>best_delta+1e-12:
                        best_delta=delta; best=(new1,new2)
    if best is None: return r1,r2,0.0
    return best[0], best[1], best_delta
