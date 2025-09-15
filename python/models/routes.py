# python/models/routes.py
from pydantic import BaseModel, field_validator
from typing import Dict, List

class Routes(BaseModel):
    # vehicle_id -> route sequence [0, c1, c2, ..., 0]
    routes: Dict[int, List[int]]

    @field_validator("routes")
    @classmethod
    def _check_depots(cls, v: Dict[int, List[int]]):
        # Optional safety: ensure each route begins/ends with depot 0
        for vid, seq in v.items():
            if not seq or seq[0] != 0 or seq[-1] != 0:
                raise ValueError(f"Route for vehicle {vid} must start & end with 0 (got {seq})")
        return v
