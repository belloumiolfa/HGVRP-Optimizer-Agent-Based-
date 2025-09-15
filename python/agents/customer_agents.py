""" This module defines two agent wrappers:

CustomerAgent: provides a clean interface to access customer properties (id, location, demand, time window, service time).
DepotAgent: wraps depot data (id, location, start time, supply capacity).

Both classes act as bridges between raw problem data (Customer, Depot) and the agent-based simulation, ensuring consistent, 
convenient access to relevant attributes in routing/evaluation logic. """

from typing import Tuple, Optional                 # Type hints for returning (x, y) tuples and optional values
from ..models import Customer, Depot               # Import domain models representing raw customers and depots


class CustomerAgent:                               # Lightweight wrapper around a Customer entity
    def __init__(self, c: Customer):               # Initialize with a Customer object
        self.c = c                                 # Store the raw customer instance internally

    @property
    def id(self) -> int:                           # Expose the customer's id
        return self.c.id

    @property
    def coords(self) -> Tuple[float, float]:       # Return (x, y) coordinates of the customer
        return self.c.coords

    @property
    def demand(self) -> float:                     # Return demand quantity of the customer
        return self.c.demand

    @property
    def window(self) -> Tuple[float, float]:       # Return the time window (ready time, due time)
        a_i, l_i = self.c.time_windows             # Extract lower and upper bounds from raw model
        return (float(a_i), float(l_i))            # Cast them to floats and return as a tuple

    @property
    def service_time(self) -> float:               # Return service time required at this customer
        return float(self.c.service_time)


class DepotAgent:                                  # Lightweight wrapper around a Depot entity
    def __init__(self, d: Depot):                  # Initialize with a Depot object
        self.d = d                                 # Store the raw depot instance internally

    @property
    def id(self) -> int:                           # Expose the depot's id
        return self.d.id

    @property
    def coords(self) -> Tuple[float, float]:       # Return (x, y) coordinates of the depot
        return self.d.coords

    @property
    def start_time(self) -> float:                 # Return depot’s start time as float
        return float(self.d.start_time)

    @property
    def supply_capacity(self) -> Optional[float]:  # Return maximum supply available at this depot (or None if unlimited)
        return self.d.supply_capacity
