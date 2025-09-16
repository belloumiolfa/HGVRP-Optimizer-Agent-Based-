
""" The Instance model is the container for a full Vehicle Routing Problem (VRP) input.
It holds:

Depot(s): either a single depot or multiple depots.
Vehicles: list of all vehicles in the fleet.
Customers: list of customer nodes with demand and constraints.

In short: Instance is the root data structure that defines the optimization problem input — the combination of depots, vehicles, 
and customers that the algorithms will solve. """

from typing import List, Optional, Any             # Type hints for lists and optional fields
from pydantic import BaseModel                     # Pydantic base class for data validation and parsing
from .depot import Depot                           # Import Depot model (single depot or multi-depots)
from .vehicle import Vehicle                       # Import Vehicle model (fleet definition)
from .customer import Customer                     # Import Customer model (demand points)


class Instance(BaseModel):                         # Top-level model describing a full VRP instance
    depot: Optional[Depot] = None                  # Optional single depot (legacy/simple cases)
    depots: Optional[List[Depot]] = None           # Optional list of depots (multi-depot scenarios)
    vehicles: List[Vehicle]                        # Fleet of vehicles (required)
    customers: List[Customer]                      # List of customers (required)
    travel_time: Optional[Any] = None

