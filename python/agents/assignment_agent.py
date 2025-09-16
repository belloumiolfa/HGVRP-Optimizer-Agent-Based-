""" AssignmentAgent takes a problem Instance and turns it into an evaluatable routing context: it wraps depots/customers into agents,
builds a customer distance matrix, converts external route IDs into internal indices, and provides utilities to compute leg/route
distances and a nearest-customer heuristic. Its core method evaluate(...) simulates and scores a set of routes (one per vehicle) by
delegating per-route evaluation to VehicleAgent, then aggregates objectives (distance, waiting, vehicles used, fuel, emissions, cost)
and penalties (capacity, duration, time windows, driver availability, battery, coverage, and depot supply). """

from __future__ import annotations
from typing import Any, List, Dict, Optional
from collections import Counter
import math
import json  # for optional JSON-stringified travel_time keys

from ..models import Instance, Vehicle, PenaltyConfig
from .customer_agents import CustomerAgent, DepotAgent
from .vehicle_agent import VehicleAgent


class AssignmentAgent:
    def __init__(self, inst: Instance):
        # Canonical references to the instance (back-compat: both names)
        self.instance: Instance = inst
        self.inst: Instance = inst

        self.depots: List[DepotAgent] = self._get_depots(inst)
        self.vehicles: List[Vehicle] = inst.vehicles

        self.customers_sorted = sorted(inst.customers, key=lambda c: c.id)
        self.customers: Dict[int, CustomerAgent] = {
            c.id: CustomerAgent(c) for c in self.customers_sorted
        }

        ids_in_order = [c.id for c in self.customers_sorted]
        self.id_to_idx: Dict[int, int] = {cid: i + 1 for i, cid in enumerate(ids_in_order)}  # external id -> 1..n
        self.idx_to_id: Dict[int, int] = {idx: cid for cid, idx in self.id_to_idx.items()}   # 1..n -> external id

        self.D: List[List[float]] = self._customer_matrix()

    def _route_distance(self, route_ids: List[int], veh: Vehicle) -> float:

        start_depot = self._depot_by_id(getattr(veh, "start_depot_id", None))
        route = self._canonicalize_route(route_ids)
        if len(route) <= 1:
            return 0.0
        dist = 0.0
        for i in range(len(route) - 1):
            dist += self._leg_distance(route[i], route[i + 1], start_depot)
        return dist

    def _get_depots(self, inst: Instance) -> List[DepotAgent]:
        depots = (inst.depots if (getattr(inst, "depots", None) and len(inst.depots) > 0)
                  else ([inst.depot] if getattr(inst, "depot", None) else []))
        if not depots:
            raise ValueError("No depot/depots provided")
        return [DepotAgent(d) for d in depots]

    def _customer_matrix(self) -> List[List[float]]:
        nodes = [c.coords for c in self.customers_sorted]
        n = len(nodes)
        D = [[0.0] * (n + 1) for _ in range(n + 1)]  # 0 reserved for depot/none
        for i in range(n):
            xi, yi = nodes[i]
            for j in range(i + 1, n):
                xj, yj = nodes[j]
                d = math.hypot(xi - xj, yi - yj)
                D[i + 1][j + 1] = D[j + 1][i + 1] = d
        return D

    def _leg_distance(self, a: int, b: int, start_depot: DepotAgent) -> float:
        if a == 0 and b == 0:
            return 0.0
        if a == 0 and b != 0:
            bx, by = self.customers[self.idx_to_id[b]].coords
            dx, dy = start_depot.coords
            return math.hypot(dx - bx, dy - by)
        if a != 0 and b == 0:
            ax, ay = self.customers[self.idx_to_id[a]].coords
            dx, dy = start_depot.coords
            return math.hypot(ax - dx, ay - dy)
        return self.D[a][b]

    # NEW: true travel time τ support (callable or dict), fallback to distance
    def _leg_time(self, a: int, b: int, start_depot: DepotAgent) -> float:
       
        tt = getattr(self.instance, "travel_time", None)

        # external ids (depot as 0)
        a_true = 0 if a == 0 else self.idx_to_id[a]
        b_true = 0 if b == 0 else self.idx_to_id[b]

        # (A) callable τ
        if callable(tt):
            try:
                return float(tt(a_true, b_true, start_depot.id))
            except Exception:
                return self._leg_distance(a, b, start_depot)

        # helper for dict lookups; accepts tuple or JSON-stringified key
        def _get_tt(key_tuple):
            if key_tuple in tt:
                return float(tt[key_tuple])
            if isinstance(key_tuple, tuple):
                try:
                    key_str = json.dumps(list(key_tuple))
                    if key_str in tt:
                        return float(tt[key_str])
                except Exception:
                    pass
            return None

        # (B) dict τ
        if isinstance(tt, dict):
            # depot -> customer
            if a_true == 0 and b_true != 0:
                val = _get_tt(("dc", start_depot.id, b_true))
                if val is not None:
                    return val
            # customer -> depot
            if a_true != 0 and b_true == 0:
                val = _get_tt(("cd", a_true, start_depot.id))
                if val is not None:
                    return val
            # customer -> customer (symmetric fallback)
            if a_true != 0 and b_true != 0:
                val = _get_tt(("cc", a_true, b_true))
                if val is None:
                    val = _get_tt(("cc", b_true, a_true))
                if val is not None:
                    return val

        # default: unit speed (time = distance)
        return self._leg_distance(a, b, start_depot)

    def _canonicalize_route(self, route_ids: List[int]) -> List[int]:
        out: List[int] = []
        for nid in route_ids:
            if nid == 0:
                out.append(0)
            else:
                try:
                    out.append(self.id_to_idx[nid])
                except KeyError as e:
                    raise ValueError(f"Unknown customer id in route: {nid}") from e
        return out

    def _depot_by_id(self, dep_id: Optional[int]) -> DepotAgent:
        if dep_id is None:
            return self.depots[0]
        for d in self.depots:
            if d.id == dep_id:
                return d
        return self.depots[0]

    def evaluate(self, routes: List[List[int]], pc: PenaltyConfig) -> Dict[str, Any]:
        all_ids = list(self.customers.keys())
        counts: Counter[int] = Counter()

        if len(routes) > len(self.vehicles):
            raise ValueError("More routes provided than vehicles available.")

        Z1 = Z2 = Z4 = Z5 = Z6 = 0.0
        Z3 = 0
        P_cap = P_dur = P_tw = P_avl = P_bat = 0.0
        demand_by_depot: Dict[int, float] = {}
        per_vehicle: List[Dict[str, Any]] = []

        for v_idx, route_ids in enumerate(routes):
            route = self._canonicalize_route(route_ids)
            if len(route) <= 2:
                continue

            vehicle = self.vehicles[v_idx]
            start_depot = self._depot_by_id(vehicle.start_depot_id)
            v_agent = VehicleAgent(vehicle, start_depot)

            res = v_agent.evaluate_route(route, self, pc, counts)

            Z1 += res["distance"]
            Z2 += res["waiting"]
            Z4 += res["fuel"]
            Z5 += res["emissions"]
            Z6 += res["cost"]
            Z3 += res["used"]

            P_cap += res["p_cap"]
            P_dur += res["p_dur"]
            P_tw  += res["p_tw"]
            P_avl += res["p_avl"]
            P_bat += res["p_bat"]

            if start_depot.supply_capacity is not None:
                d_id = start_depot.id
                demand_by_depot[d_id] = demand_by_depot.get(d_id, 0.0) + res["demand_served"]

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
                    "P_capacity":    res["p_cap"],
                    "P_duration":    res["p_dur"],
                    "P_timewindows": res["p_tw"],
                    "P_availability":res["p_avl"],
                    "P_battery":     res["p_bat"],
                }
            })

        missing = [cid for cid in all_ids if counts.get(cid, 0) == 0]
        dup_over = sum(cnt - 1 for cnt in counts.values() if cnt > 1)
        P_coverage = pc.beta_coverage * (len(missing) + dup_over)

        P_supply = 0.0
        for d in self.depots:
            if d.supply_capacity is not None:
                served = demand_by_depot.get(d.id, 0.0)
                if served > d.supply_capacity:
                    P_supply += pc.beta_supply * (served - d.supply_capacity)

        objectives = {
            "Z1": Z1, "Z2": Z2, "Z3": float(Z3), "Z4": Z4, "Z5": Z5, "Z6": Z6
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

        return {
            "objectives": objectives,
            "penalties": penalties,
            "perVehicle": per_vehicle,
            "totals": totals,
        }

    # Optional public wrappers (nice for readability elsewhere)
    def leg_distance(self, a: int, b: int, start_depot: DepotAgent) -> float:
        return self._leg_distance(a, b, start_depot)

    def leg_time(self, a: int, b: int, start_depot: DepotAgent) -> float:
        return self._leg_time(a, b, start_depot)
