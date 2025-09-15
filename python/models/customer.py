""" This file defines the Customer Pydantic model, which represents a customer in the VRP instance.

It ensures:

Customers always have valid coordinates (either (x,y) or location=[x,y] must be provided).
Time windows are normalized into a [a_i, l_i] list (single numbers/lists are expanded).
Provides a .coords property for unified access to coordinates.

In short: it validates and standardizes raw customer input into a consistent, safe format for the routing/optimization algorithms. """

from typing import List, Optional, Tuple               # Type hints for lists, optional values, and coordinate tuples
from pydantic import BaseModel, field_validator, model_validator  # Pydantic base class and validation decorators


class Customer(BaseModel):                             # Customer data model (Pydantic for validation + parsing)
    id: int                                            # Unique customer ID
    x: Optional[float] = None                          # X-coordinate (optional if location[] is used)
    y: Optional[float] = None                          # Y-coordinate (optional if location[] is used)
    location: Optional[List[float]] = None             # Alternative coordinate format: [x, y]
    demand: float                                      # Customer demand (quantity to deliver/pickup)
    time_windows: List[float]                          # Time window [a_i, l_i] = earliest and latest service time
    service_time: float                                # Service duration at this customer
    special_requirements: Optional[str] = None         # Optional field for custom requirements

    @field_validator("time_windows", mode="before")    # Validate/normalize time_windows *before* assignment
    @classmethod
    def _coerce_time_windows(cls, v):
        if isinstance(v, (int, float)):                # If single number → interpret as [0, value]
            return [0.0, float(v)]
        if isinstance(v, list):
            if len(v) == 1:                            # Single-element list → expand to [0, value]
                return [0.0, float(v[0])]
            if len(v) == 2:                            # Two elements → cast to floats
                return [float(v[0]), float(v[1])]
        raise ValueError("time_windows must be [a_i, l_i]")  # Otherwise invalid format

    @model_validator(mode="after")                     # Post-init validation (after fields parsed)
    def _check_fields(self):
        has_loc = self.location is not None and len(self.location) >= 2   # Check if location[] provided
        has_xy = self.x is not None and self.y is not None                # Or if (x,y) provided
        if not (has_loc or has_xy):                                       # Must have at least one coordinate format
            raise ValueError("Provide either (x,y) or location=[x,y] for customer")
        return self

    @property
    def coords(self) -> Tuple[float, float]:            # Unified coordinate accessor
        if self.location and len(self.location) >= 2:   # Prefer location[] if provided
            return float(self.location[0]), float(self.location[1])
        if self.x is None or self.y is None:            # If neither format available → error
            raise ValueError(f"Customer {self.id} missing coordinates")
        return float(self.x), float(self.y)             # Otherwise return (x,y)
