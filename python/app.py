from fastapi import FastAPI,Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import random

from .models import (
    EvaluateRequest, EvaluateResponse,
    ConstructRequest, ConstructResponse,
    LocalSearchRequest, LocalSearchResponse,
    OptimizeRequest
)
from .eval import AssignmentAgent
from .construct import construct_balanced, construct_greedy

# GA (legacy + adaptive)
from .ga import GAConfig as AdaptiveGACfg, AdaptiveGA, optimize_single
from .adapt import BanditConfig
from .penalty_manager import PenaltyConfig
from .ga_pareto import NSGA2Optimizer      # Phase 7

app = FastAPI(title="HGVRP Optimizer (Agent-Based)", version="0.9")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

def _flatten_metrics(pack: dict) -> dict:
    if "Z1_total_distance" in pack:
        return pack
    obj = pack.get("objectives", {})
    pen = pack.get("penalties", {})
    return {
        "Z1_total_distance":  obj.get("Z1", 0.0),
        "Z2_total_waiting":   obj.get("Z2", 0.0),
        "Z3_vehicles_used":   obj.get("Z3", 0.0),
        "Z4_total_fuel":      obj.get("Z4", 0.0),
        "Z5_total_emissions": obj.get("Z5", 0.0),
        "Z6_total_costs":     obj.get("Z6", 0.0),
        "P_capacity":         pen.get("P_capacity", 0.0),
        "P_duration":         pen.get("P_duration", 0.0),
        "P_timewindows":      pen.get("P_timewindows", 0.0),
        "P_availability":     pen.get("P_availability", 0.0),
        "P_battery":          pen.get("P_battery", 0.0),
        "P_coverage":         pen.get("P_coverage", 0.0),
        "P_supply":           pen.get("P_supply", 0.0),
    }

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}

@app.post("/evaluate", response_model=EvaluateResponse)
def endpoint_evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    agent = AssignmentAgent(req.instance)
    structured = agent.evaluate(req.routes, req.penalty)
    metrics = _flatten_metrics(structured)
    return {"status": "ok", "metrics": metrics}

@app.post("/construct", response_model=ConstructResponse)
def endpoint_construct(req: ConstructRequest) -> Dict[str, Any]:
    if req.seed is not None:
        random.seed(req.seed)
    agent = AssignmentAgent(req.instance)
    routes = construct_balanced(agent) if req.method == "balanced" else construct_greedy(agent)
    structured = agent.evaluate(routes, req.penalty)
    metrics = _flatten_metrics(structured)
    return {"status": "ok", "routes": routes, "metrics": metrics}

@app.post("/localsearch", response_model=LocalSearchResponse)
def endpoint_localsearch(req: LocalSearchRequest) -> Dict[str, Any]:
    rng = random.Random(req.seed)
    agent = AssignmentAgent(req.instance)

    # Map external → internal op names
    op_map = {
        "2opt": "two_opt",
        "oropt_intra": "oropt1",
        "oropt1": "oropt1",
        "cross": "cross",
        "two_opt": "two_opt",
    }
    ops = [op_map.get(op, op) for op in (req.ops or ["two_opt", "oropt1"])]

    # Run LS (handles list/dict/model shapes and returns same shape)
    from .localsearch import apply_localsearch
    routes, improved = apply_localsearch(agent, req.routes, ops, req.budget, rng, penalty=req.penalty)

    structured = agent.evaluate(routes, req.penalty)
    metrics = _flatten_metrics(structured)
    return {"status": "ok", "routes": routes, "metrics": metrics, "improved": improved}

# ----------- endpoint -----------
@app.post("/optimize")
def endpoint_optimize(req: OptimizeRequest = Body(...)) -> Dict[str, Any]:
    mode = req.mode or "adaptive"

    # ----- SINGLE (scalar) -----
    if mode == "single":
        cfg_dict = req.ga_config or req.ga or {}
        if req.penalty is None:
            raise HTTPException(status_code=400, detail="Missing 'penalty' for single mode.")
        cfg = AdaptiveGACfg(**cfg_dict)
        out = optimize_single(
            req.instance,
            req.penalty,
            cfg,
            req.wZ,
            req.wP,
        )
        return {"status": "ok", "mode": mode, **out}

    # ----- ADAPTIVE (bandit GA) -----
    if mode == "adaptive":
        ga_dict = req.ga or req.ga_config or {}
        bandit_dict = req.bandit or {}
        penalty_dict: Dict[str, Any] = {}
        # allow either provided object or dict
        if req.penalty is not None:
            # already a PenaltyConfig object
            penalty_cfg = req.penalty
        else:
            penalty_cfg = PenaltyConfig(**penalty_dict) if penalty_dict else PenaltyConfig()

        ga_cfg = AdaptiveGACfg(**ga_dict)
        bandit_cfg = BanditConfig(**bandit_dict)

        algo = AdaptiveGA(
            instance=req.instance,
            ga_cfg=ga_cfg,
            bandit_cfg=bandit_cfg,
            penalty_cfg=penalty_cfg,
        )
        out = algo.run(mode="adaptive")
        return {"status": "ok", "mode": mode, "result": out}

    # ----- PARETO (NSGA-II) -----
    if mode == "pareto":
        if req.penalty is None:
            # NSGA-II still needs penalties for feasibility checks
            raise HTTPException(status_code=400, detail="Missing 'penalty' for pareto mode.")
        opt = NSGA2Optimizer(
            instance=req.instance,
            penalty_cfg=req.penalty,
            objectives=req.objectives or None,
            hard_penalty_keys=req.hard_penalties or None,
            seed=req.seed,
        )
        out = opt.evolve(
            pop_size=req.population_size,
            generations=req.generations,
            constructor=req.constructor,
            pc=req.pc,
            pm=req.pm,
        )
        return {"status": "ok", "mode": mode, **out}

    # unknown mode
    raise HTTPException(status_code=400, detail=f"Unsupported mode '{mode}'. Use 'single', 'adaptive', or 'pareto'.")
