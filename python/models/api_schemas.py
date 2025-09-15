from typing import Any, List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from .instance import Instance
from .penalty import PenaltyConfig

class EvaluateRequest(BaseModel):
    instance: Instance
    routes: List[List[int]] = Field(..., description="Routes: node ids (0=depot, others=customer IDs)")
    penalty: PenaltyConfig = PenaltyConfig()

class EvaluateResponse(BaseModel):
    status: str
    metrics: Dict[str, float]

class ConstructRequest(BaseModel):
    instance: Instance
    method: Literal["balanced","greedy"] = "greedy"
    penalty: PenaltyConfig = PenaltyConfig()
    seed: Optional[int] = None

class ConstructResponse(BaseModel):
    status: str
    routes: List[List[int]]
    metrics: Dict[str, float]

class LocalSearchRequest(BaseModel):
    instance: Instance
    routes: List[List[int]]
    penalty: PenaltyConfig = PenaltyConfig()
    budget: int = 2000
    ops: List[Literal[
        "2opt",
        "oropt_intra",
        "oropt2_intra",
        "oropt3_intra",
        "oropt_inter",
        "cross"
    ]] = ["2opt", "oropt_intra"]    
    seed: Optional[int] = None

class LocalSearchResponse(BaseModel):
    status: str
    routes: List[List[int]]
    metrics: Dict[str, float]
    improved: bool

class OptimizeRequest(BaseModel):
    instance: Instance
    penalties: PenaltyConfig
    mode: str = "single"  # "single" | "adaptive" | "pareto" (future)
    ga_config: Optional[Dict[str, Any]] = None
    wZ: Optional[Dict[str, float]] = None
    wP: Optional[Dict[str, float]] = None
    
class OptimizePayload(BaseModel):
    instance: Dict[str, Any]
    mode: str = "adaptive"
    ga: Optional[Dict[str, Any]] = None
    bandit: Optional[Dict[str, Any]] = None
    penalty: Optional[Dict[str, Any]] = None

class OptimizeParetoRequest(BaseModel):
    mode: Literal["scalar", "pareto"] = "pareto"
    instance: Instance
    penalty: PenaltyConfig
    population_size: int = 40
    generations: int = 40
    constructor: Literal["balanced", "greedy"] = "balanced"
    pc: float = 0.9
    pm: float = 0.3
    hard_penalties: Optional[List[str]] = None
    objectives: Optional[List[str]] = None
    seed: Optional[int] = None
