# python/penalty_manager.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Deque, Optional, Iterable
from collections import deque

@dataclass
class PenaltyConfig:
    # initial β values (defaults are typical names used in your project)
    betas: Dict[str, float] = field(default_factory=lambda: {
        "beta_capacity": 1.0,
        "beta_duration": 1.0,
        "beta_timewindows": 1.0,
        "beta_availability": 1.0,
        "beta_battery": 1.0,
        "beta_coverage": 1000.0,  # coverage often kept high
        "beta_supply": 1.0,
    })
    window: int = 10           # rolling window length
    gamma_up: float = 1.5      # scale ↑ when violation persists
    gamma_down: float = 0.8    # scale ↓ when no violation persists
    floor: float = 1e-6        # clamp
    cap: float = 1e6

class PenaltyManager:
    """
    Rolling-window adaptive penalties.
    - Call observe(penalties) each evaluation with dict of penalty values (>=0).
    - Call step() once per generation to scale β up/down when window indicates
      persistent violation or persistent feasibility, respectively.
    """
    def __init__(self, cfg: Optional[PenaltyConfig] = None):
        self.cfg = cfg or PenaltyConfig()
        self.betas: Dict[str, float] = dict(self.cfg.betas)
        self.windows: Dict[str, Deque[int]] = {
            k: deque(maxlen=self.cfg.window) for k in self.betas.keys()
        }
        self.beta_history: Dict[str, list[float]] = {k: [v] for k, v in self.betas.items()}

    def observe(self, penalties: Dict[str, float]) -> None:
        """
        penalties: mapping like {"beta_capacity": penalty_value, ...}
        We treat >0 as "violation occurred" for that constraint in this evaluation.
        """
        for k in self.windows.keys():
            v = float(penalties.get(k, 0.0))
            self.windows[k].append(1 if v > 0.0 else 0)

    def _clamp(self, x: float) -> float:
        return max(self.cfg.floor, min(self.cfg.cap, x))

    def step(self) -> Dict[str, float]:
        """
        Called once per GA generation (after many observe calls).
        Scales β if the window is full and either:
         - all 1s (violation ~100%): multiply by gamma_up
         - all 0s (no violation): multiply by gamma_down
        """
        for k, w in self.windows.items():
            if len(w) < self.cfg.window:
                continue
            s = sum(w)
            if s == self.cfg.window:
                self.betas[k] = self._clamp(self.betas[k] * self.cfg.gamma_up)
            elif s == 0:
                self.betas[k] = self._clamp(self.betas[k] * self.cfg.gamma_down)
            # else: mixed window → no change
            self.beta_history[k].append(self.betas[k])
        return dict(self.betas)

    def get_betas(self) -> Dict[str, float]:
        return dict(self.betas)

    def history(self) -> Dict[str, list[float]]:
        return {k: v[:] for k, v in self.beta_history.items()}
