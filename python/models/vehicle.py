# Defines the Vehicle model with attributes, constraints, and availability for the VRP optimizer.

from typing import List, Optional                     # Type hints for lists and optional values
from pydantic import BaseModel, field_validator       # Pydantic BaseModel and field-level validator


class Vehicle(BaseModel):                             # Vehicle data model
    id: int                                           # Unique vehicle identifier
    capacity: float                                   # Maximum load capacity
    fuel_efficiency: float                            # Fuel/energy efficiency (distance per unit fuel/energy)
    emission_level: float                             # Emission factor per unit distance
    vehicle_type: str                                 # Type of vehicle (e.g., "truck", "electric")
    fuel_type: str                                    # Fuel type (e.g., "diesel", "electric")
    battery_capacity: Optional[float] = None          # Battery capacity for EVs (if applicable)
    charging_time: Optional[float] = None             # Time required to fully recharge (EVs only)
    weight_class: Optional[str] = None                # Vehicle weight class (optional descriptor)
    availability: Optional[List[float]] = None        # [start, end] availability time window
    fixed_costs: float = 0.0                          # Fixed cost incurred if the vehicle is used
    variable_costs: float = 0.0                       # Variable cost per distance unit
    location: Optional[List[float]] = None            # Optional starting location [x, y]
    time_windows: Optional[List[float]] = None        # Optional time window constraints for the vehicle
    special_constraints: Optional[str] = None         # Any custom vehicle constraints
    start_depot_id: Optional[int] = None              # ID of the starting depot
    max_route_time: float = 1e9                       # Maximum allowed route duration (default large)

    @field_validator("availability", mode="before")   # Normalize availability field before assignment
    @classmethod
    def _coerce_availability(cls, v):
        if v is None:                                 # No availability provided → leave as None
            return v
        if isinstance(v, (int, float)):               # Single number → interpret as [0, value]
            return [0.0, float(v)]
        if isinstance(v, list):
            if len(v) == 1:                           # One-element list → expand to [0, value]
                return [0.0, float(v[0])]
            if len(v) == 2:                           # Two-element list → cast to floats
                return [float(v[0]), float(v[1])]
        raise ValueError("availability must be [start, end]")  # Otherwise invalid
