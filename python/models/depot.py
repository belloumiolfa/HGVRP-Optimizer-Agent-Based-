
""" The Depot Pydantic model represents a starting/ending point for vehicles in the VRP instance.

It:

Validates that coordinates are always available (via (x, y) or location=[x, y]).
Exposes .coords for unified coordinate access.
Holds scheduling (start_time) and logistical (supply_capacity) constraints.

In short: it defines the depot node in the routing problem, ensuring it has valid coordinates and optional capacity/time 
attributes for optimization. """

from typing import List, Optional, Tuple            # Type hints: lists, optional fields, coordinate tuples
from pydantic import BaseModel                      # Pydantic base class for data validation and parsing


class Depot(BaseModel):                             # Depot data model (entry point/exit for vehicles)
    id: int                                         # Unique depot identifier
    x: Optional[float] = None                       # X-coordinate (optional if location[] is used instead)
    y: Optional[float] = None                       # Y-coordinate (optional if location[] is used instead)
    location: Optional[List[float]] = None          # Alternative coordinate representation [x, y]
    start_time: float = 0.0                         # Earliest time vehicles can depart from this depot
    supply_capacity: Optional[float] = None         # Maximum supply available at depot (None = unlimited)

    @property
    def coords(self) -> Tuple[float, float]:        # Unified accessor for depot coordinates
        if self.location and len(self.location) >= 2:   # Prefer location[] if present
            return float(self.location[0]), float(self.location[1])
        if self.x is None or self.y is None:            # If both formats missing → invalid depot
            raise ValueError(f"Depot {self.id} is missing coordinates")
        return float(self.x), float(self.y)             # Otherwise return (x, y)
