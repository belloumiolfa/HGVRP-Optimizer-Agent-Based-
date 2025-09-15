
# Defines penalty weights for constraint violations in the VRP optimizer.

from pydantic import BaseModel          # Import Pydantic BaseModel for validation and schema support


class PenaltyConfig(BaseModel):         # Model for penalty configuration
    beta_capacity: float = 1.0          # Penalty for exceeding vehicle capacity
    beta_duration: float = 1.0          # Penalty for exceeding maximum allowed route duration
    beta_timewindows: float = 1.0       # Penalty for violating customer time windows
    beta_availability: float = 1.0      # Penalty for violating driver/vehicle availability
    beta_battery: float = 1.0           # Penalty for exceeding EV battery capacity
    beta_coverage: float = 1000.0       # Strong penalty for unserved or duplicate-served customers
    beta_supply: float = 1000.0         # Strong penalty for exceeding depot supply capacity
