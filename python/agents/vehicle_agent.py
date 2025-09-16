""" VehicleAgent.evaluate_route(...) simulates a single vehicle’s route (with 0 as depot and 1..N as customers), computing:

Objectives: total distance, waiting time, fuel/energy use, emissions, and total cost (variable + fixed if used).
Penalties: capacity overage, route duration overage, customer time-window violations, vehicle availability violations, and battery 
overdraw for EVs.
Accounting: returns whether the vehicle was used, the maximum load reached, and total demand served; also updates a global visit 
counter to support solution-wide coverage checks.

It uses the AssignmentAgent for leg distances and ID mapping, and PenaltyConfig for penalty weights. """

from collections import Counter                      # Multiset to count visits per true customer id
from typing import Dict, List                        # Type hints for mappings and sequences
from ..models import Vehicle, PenaltyConfig          # Domain models: vehicle specs + penalty weights
from .customer_agents import DepotAgent              # Agent wrapper for depot properties (coords, start time, supply)


class VehicleAgent:                                  # Agent responsible for evaluating a single vehicle's route
    def __init__(self, v: Vehicle, start_depot: DepotAgent):  # Construct with a vehicle and its start depot
        self.v = v                                              # Store raw vehicle model
        self.start_depot = start_depot                          # Store depot agent (start time, coords, etc.)

    def evaluate_route(                               # Evaluate one canonical route and return metrics
        self,
        route: List[int],                    # internal indices: 0=depot, 1..N=customers
        assigner: "AssignmentAgent",         # for leg distances & id maps
        pc: PenaltyConfig,                   # penalty weights/configuration
        global_counts: Counter               # counts by TRUE customer id
    ) -> Dict[str, float]:
        time = self.start_depot.start_time             # Initialize clock at depot’s start time
        load = max_load = 0.0                          # Rolling load carried and max load encountered

        distance = fuel = emissions = waiting = battery_used = cost_var = 0.0  # Objective accumulators
        p_cap = p_dur = p_tw = p_avl = p_bat = 0.0                             # Penalty accumulators

        for i in range(len(route) - 1):                # Iterate over consecutive legs a->b
            a, b = route[i], route[i + 1]              # Current leg endpoints (canonical indices)
            dist = assigner._leg_distance(a, b, self.start_depot)  # Geometric leg distance (handles depot legs)
            distance += dist                            # Accumulate distance
            # time += dist                              # Advance time assuming unit speed (τ = distance)
            tau = assigner._leg_time(a, b, self.start_depot)
            time += tau

            eff = max(1e-12, self.v.fuel_efficiency)    # Guard efficiency to avoid division by zero
            f_used = dist / eff                         # Fuel/energy used on this leg
            fuel += f_used                              # Accumulate fuel consumption
            emissions += dist * self.v.emission_level   # Emissions proportional to distance
            cost_var += dist * self.v.variable_costs    # Variable cost proportional to distance

            if self.v.vehicle_type.lower() == "electric":             # Battery checks for EVs
                battery_used += f_used                                # Treat f_used as energy draw
                if self.v.battery_capacity and battery_used > self.v.battery_capacity:
                    p_bat += pc.beta_battery * (battery_used - self.v.battery_capacity)  # Penalize overdraw

            if b != 0:                                   # If the next node is a customer (not depot)
                true_id = assigner.idx_to_id[b]          # Map canonical index -> true external ID
                c = assigner.customers[true_id]           # Get CustomerAgent
                a_i, l_i = c.window                       # Customer time window [ready, due]

                if time < a_i:                            # Arrived early: wait until window opens
                    waiting += (a_i - time)               # Accumulate waiting
                    time = a_i                            # Jump time to ready time

                time += c.service_time                    # Spend service time at customer
                load += c.demand                          # Increase load by customer demand (pickup model)
                max_load = max(max_load, load)            # Track maximum load observed

                if time > l_i:                            # Late arrival (after due time)
                    p_tw += pc.beta_timewindows * (time - l_i)  # Penalize lateness

                global_counts[true_id] += 1               # Record one visit for coverage accounting

        if max_load > self.v.capacity:                    # Capacity violation across the route
            p_cap += pc.beta_capacity * (max_load - self.v.capacity)

        if time > self.start_depot.start_time + self.v.max_route_time:  # Duration violation
            over = time - (self.start_depot.start_time + self.v.max_route_time)
            p_dur += pc.beta_duration * over

        if self.v.availability:                           # Driver/vehicle availability window constraint (optional)
            a_v, l_v = self.v.availability                # Availability [start, end]
            if time > l_v:                                # Route ends after availability window closes
                p_avl += pc.beta_availability * (time - l_v)
            if self.start_depot.start_time < a_v:         # Route starts before availability window opens
                p_avl += pc.beta_availability * (a_v - self.start_depot.start_time)

        used = 1 if len(route) > 2 else 0                # Vehicle counts as used only if route contains a customer
        cost_fix = self.v.fixed_costs * used             # Add fixed cost if vehicle is used

        return {                                          # Return all route-level metrics and penalties
            "distance": distance, 
            "waiting": waiting, 
            "fuel": fuel,
            "emissions": emissions, 
            "cost": cost_var + cost_fix,
            "p_cap": p_cap, 
            "p_dur": p_dur, 
            "p_tw": p_tw, 
            "p_avl": p_avl, 
            "p_bat": p_bat,
            "max_load": max_load, 
            "demand_served": load, 
            "used": used, 
            "end_time": time   
        }
