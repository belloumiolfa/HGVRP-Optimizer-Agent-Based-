
""" AssignmentAgent takes a problem Instance and turns it into an evaluatable routing context: it wraps depots/customers into agents, 
builds a customer distance matrix, converts external route IDs into internal indices, and provides utilities to compute leg/route 
distances and a nearest-customer heuristic. Its core method evaluate(...) simulates and scores a set of routes (one per vehicle) by 
delegating per-route evaluation to VehicleAgent, then aggregates objectives (distance, waiting, vehicles used, fuel, emissions, cost) 
and penalties (capacity, duration, time windows, driver availability, battery, coverage, and depot supply). """

from __future__ import annotations            # Allow forward references in type hints (classes referenced before they are defined)
from typing import Any, List, Dict, Optional       # Type hint utilities
from collections import Counter               # Fast multiset/counting for visit tracking
import math                                   # Math helpers (e.g., hypot for Euclidean distance)

from ..models import Instance, Vehicle, PenaltyConfig  # Core data models coming from the parent package
from .customer_agents import CustomerAgent, DepotAgent # Agent wrappers for customers and depots
from .vehicle_agent import VehicleAgent               # Agent wrapper that evaluates a single vehicle’s route


class AssignmentAgent:
    def __init__(self, inst: Instance):               # Construct from a problem instance
        self.depots: List[DepotAgent] = self._get_depots(inst)  # Resolve + wrap depots into agents
        self.vehicles: List[Vehicle] = inst.vehicles            # Keep the list of vehicles from the instance

        self.customers_sorted = sorted(inst.customers, key=lambda c: c.id) # Deterministic order of customers by id
        self.customers: Dict[int, CustomerAgent] = {                        # Map external customer id -> CustomerAgent
            c.id: CustomerAgent(c) for c in self.customers_sorted
        }

        ids_in_order = [c.id for c in self.customers_sorted]   # External IDs in the sorted order
        self.id_to_idx: Dict[int, int] = {cid: i + 1 for i, cid in enumerate(ids_in_order)}  # External id -> 1..n
        self.idx_to_id: Dict[int, int] = {idx: cid for cid, idx in self.id_to_idx.items()}   # Inverse: 1..n -> external id

        self.D: List[List[float]] = self._customer_matrix()    # Precompute symmetric customer distance matrix

    def _get_depots(self, inst: Instance) -> List[DepotAgent]: # Resolve single/multi depot and wrap
        depots = (inst.depots if (getattr(inst, "depots", None) and len(inst.depots) > 0)
                  else ([inst.depot] if getattr(inst, "depot", None) else []))  # Prefer multi; fallback to single
        if not depots:                                          # Guard: must have at least one depot
            raise ValueError("No depot/depots provided")
        return [DepotAgent(d) for d in depots]                  # Wrap raw depots as DepotAgent

    def _customer_matrix(self) -> List[List[float]]:            # Build Euclidean distance matrix between customers
        nodes = [c.coords for c in self.customers_sorted]       # Extract coords in sorted order
        n = len(nodes)                                          # Number of customers
        D = [[0.0] * (n + 1) for _ in range(n + 1)]            # (n+1)x(n+1) matrix; index 0 reserved for "depot/none"
        for i in range(n):                                      # Fill upper triangle
            xi, yi = nodes[i]                                   # Coords of i
            for j in range(i + 1, n):                           # For j > i
                xj, yj = nodes[j]                               # Coords of j
                d = math.hypot(xi - xj, yi - yj)                # Euclidean distance
                D[i + 1][j + 1] = D[j + 1][i + 1] = d           # Symmetric assignment (shift by +1 for canonical index)
        return D                                                # Return matrix

    def _leg_distance(self, a: int, b: int, start_depot: DepotAgent) -> float:  # Distance between canonical nodes a->b
        if a == 0 and b == 0:                                   # Depot->Depot (or none) contributes 0
            return 0.0
        if a == 0 and b != 0:                                   # Depot -> customer b
            bx, by = self.customers[self.idx_to_id[b]].coords   # Customer b coords (convert index->external id->agent)
            dx, dy = start_depot.coords                         # Depot coords
            return math.hypot(dx - bx, dy - by)                 # Euclidean distance depot->customer
        if a != 0 and b == 0:                                   # Customer a -> depot
            ax, ay = self.customers[self.idx_to_id[a]].coords   # Customer a coords
            dx, dy = start_depot.coords                         # Depot coords
            return math.hypot(ax - dx, ay - dy)                 # Euclidean distance customer->depot
        return self.D[a][b]                                     # Customer->customer from precomputed matrix

    def _canonicalize_route(self, route_ids: List[int]) -> List[int]:  # Convert external IDs (0=depot) to 0..n indices
        out: List[int] = []                                     # Output canonical route
        for nid in route_ids:                                   # Walk given route
            if nid == 0:                                        # Keep depot separators as 0
                out.append(0)
            else:                                               # Map external id -> canonical index
                try:
                    out.append(self.id_to_idx[nid])
                except KeyError as e:                           # Unknown id guard
                    raise ValueError(f"Unknown customer id in route: {nid}") from e
        return out                                              # Canonicalized route

    def _depot_by_id(self, dep_id: Optional[int]) -> DepotAgent:  # Resolve a depot by id (or default to first)
        if dep_id is None:                                      # If not specified
            return self.depots[0]                               # Default to first depot
        for d in self.depots:                                   # Search by id
            if d.id == dep_id:
                return d
        return self.depots[0]                                   # Fallback to first if not found

    def evaluate(self, routes: List[List[int]], pc: PenaltyConfig) -> Dict[str, float]:  # Evaluate full solution
        all_ids = list(self.customers.keys())                   # All external customer ids
        counts: Counter[int] = Counter()                        # Track visits per customer (for coverage penalty)

        if len(routes) > len(self.vehicles):                    # Guard: not more routes than vehicles
            raise ValueError("More routes provided than vehicles available.")

        Z1 = Z2 = Z4 = Z5 = Z6 = 0.0                            # Objective accumulators: distance, waiting, fuel, emis, cost
        Z3 = 0                                                  # Vehicles used
        P_cap = P_dur = P_tw = P_avl = P_bat = 0.0              # Penalty accumulators: capacity, duration, TW, availability, battery
        demand_by_depot: Dict[int, float] = {}                  # Served demand per depot for supply constraint
        per_vehicle: List[Dict[str, Any]] = []

        for v_idx, route_ids in enumerate(routes):              # Iterate routes alongside vehicle index
            route = self._canonicalize_route(route_ids)         # Convert external ids -> canonical indices
            if len(route) <= 2:                                 # Skip empty/trivial (e.g., [0] or [0,0])
                continue

            vehicle = self.vehicles[v_idx]                      # Vehicle for this route
            start_depot = self._depot_by_id(vehicle.start_depot_id)  # Resolve its starting depot
            v_agent = VehicleAgent(vehicle, start_depot)        # Vehicle agent to simulate/evaluate this route

            res = v_agent.evaluate_route(route, self, pc, counts)    # Delegate fine-grained evaluation; update counts

            Z1 += res["distance"]                               # Accumulate distance (Z1)
            Z2 += res["waiting"]                                # Accumulate waiting time (Z2)
            Z4 += res["fuel"]                                   # Accumulate fuel (Z4)
            Z5 += res["emissions"]                              # Accumulate emissions (Z5)
            Z6 += res["cost"]                                   # Accumulate cost (Z6)
            Z3 += res["used"]                                   # Accumulate vehicles used (Z3)

            P_cap += res["p_cap"]                               # Capacity penalty
            P_dur += res["p_dur"]                               # Duration penalty
            P_tw  += res["p_tw"]                                # Time windows penalty
            P_avl += res["p_avl"]                               # Availability penalty
            P_bat += res["p_bat"]                               # Battery penalty

            if start_depot.supply_capacity is not None:         # If depot has a supply cap
                d_id = start_depot.id                           # Depot id
                demand_by_depot[d_id] = demand_by_depot.get(d_id, 0.0) + res["demand_served"]  # Add served demand

            # Collect per-vehicle entry
            per_vehicle.append({
                "vehicle_id": vehicle.id,
                "distance": res["distance"],
                "waiting": res["waiting"],
                "fuel": res["fuel"],
                "emissions": res["emissions"],
                "cost": res["cost"],
                "end_time": res["end_time"],
                "max_load": res["max_load"],
                "penalties": {
                    "P_capacity":   res["p_cap"],
                    "P_duration":   res["p_dur"],
                    "P_timewindows":res["p_tw"],
                    "P_availability":res["p_avl"],
                    "P_battery":    res["p_bat"],
                }
            })

        missing = [cid for cid in all_ids if counts.get(cid, 0) == 0]  # Customers never visited
        dup_over = sum(cnt - 1 for cnt in counts.values() if cnt > 1)  # Extra visits beyond the first
        P_coverage = pc.beta_coverage * (len(missing) + dup_over)       # Coverage penalty with weight

        P_supply = 0.0                                         # Initialize supply penalty
        for d in self.depots:                                  # For each depot
            if d.supply_capacity is not None:                  # If a cap is defined
                served = demand_by_depot.get(d.id, 0.0)        # How much demand was served from this depot
                if served > d.supply_capacity:                 # If exceeded
                    P_supply += pc.beta_supply * (served - d.supply_capacity)  # Penalize the overage

        # Pack into the new structured shape
        objectives = {
            "Z1": Z1,
            "Z2": Z2,
            "Z3": float(Z3),
            "Z4": Z4,
            "Z5": Z5,
            "Z6": Z6
        }
        penalties = {
            "P_capacity":    P_cap,
            "P_duration":    P_dur,
            "P_timewindows": P_tw,
            "P_availability":P_avl,
            "P_battery":     P_bat,
            "P_coverage":    P_coverage,
            "P_supply":      P_supply
        }
        totals = {
            "distance":       Z1,
            "waiting":        Z2,
            "vehicles_used":  Z3,
            "fuel":           Z4,
            "emissions":      Z5,
            "costs":          Z6,
            "penalty_total":  sum(penalties.values())
        }

        return {                                        # Return aggregated objectives + penalties
            "objectives": objectives,
            "penalties": penalties,
            "perVehicle": per_vehicle,
            "totals": totals,
        }

        """ return {                                               
            "Z1_total_distance": Z1,
            "Z2_total_waiting": Z2,
            "Z3_vehicles_used": Z3,
            "Z4_total_fuel": Z4,
            "Z5_total_emissions": Z5,
            "Z6_total_costs": Z6,
            "P_capacity": P_cap,
            "P_duration": P_dur,
            "P_timewindows": P_tw,
            "P_availability": P_avl,
            "P_battery": P_bat,
            "P_coverage": P_coverage,
            "P_supply": P_supply,
        } """

    def _route_distance(self, route_ids: List[int], veh: Vehicle) -> float:  # Compute total geometric length of a route
        start_depot = self._depot_by_id(veh.start_depot_id)    # Resolve depot for depot legs
        route = self._canonicalize_route(route_ids)            # Canonicalize external ids
        if len(route) <= 1:                                    # Empty/trivial guard
            return 0.0
        dist = 0.0                                             # Accumulator
        for i in range(len(route) - 1):                        # Sum leg-by-leg
            dist += self._leg_distance(route[i], route[i + 1], start_depot)  # Add leg distance
        return dist                                            # Total distance

    def nearest_customer(self, from_id: int, remaining: List[int], veh: Vehicle) -> Optional[int]:  # Greedy nearest pick
        if not remaining:                                      # If candidate list is empty
            return None                                        # Nothing to choose

        start_depot = self._depot_by_id(veh.start_depot_id)    # Depot for depot-origin cases
        best_c: Optional[int] = None                           # Best candidate id
        best_d = float("inf")                                  # Best distance so far

        for cid in remaining:                                  # Check all remaining candidates
            if from_id == 0:                                   # Origin is depot
                dx, dy = start_depot.coords
            else:                                              # Origin is a customer
                if from_id not in self.customers:              # Guard invalid origin id
                    raise ValueError(f"Unknown 'from_id' in nearest_customer: {from_id}")
                dx, dy = self.customers[from_id].coords

            if cid not in self.customers:                      # Guard invalid candidate
                raise ValueError(f"Unknown candidate id in nearest_customer: {cid}")

            cx, cy = self.customers[cid].coords                # Candidate coords
            d = math.hypot(dx - cx, dy - cy)                   # Euclidean distance origin -> candidate
            if d < best_d:                                     # Keep minimum
                best_d = d
                best_c = cid

        return best_c                                          # Nearest external customer id (or None)

