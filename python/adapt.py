# adapt.py (placeholder)
# python/adapt.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math
import random

@dataclass
class BanditConfig:
    mode: str = "epsilon_greedy"   # "epsilon_greedy" | "softmax"
    epsilon: float = 0.1           # exploration for epsilon-greedy
    tau: float = 0.5               # temperature for softmax (Boltzmann)
    min_weight: float = 1e-6       # avoid zero weight collapse
    alpha: float = 0.2             # learning rate for Q-value updates
    seed: Optional[int] = None

class OperatorBandit:
    """
    Multi-armed bandit over variation/local-search operators.
    Reward = positive offspring improvement (parent_f - child_f if > 0) in a minimization setting.
    Maintains Q-values and exposes normalized weights (used as selection probs).
    """
    def __init__(self, operators: List[str], cfg: Optional[BanditConfig] = None):
        self.ops = operators[:]
        self.cfg = cfg or BanditConfig()
        self.rng = random.Random(self.cfg.seed)
        n = len(self.ops)

        self.q: List[float] = [0.0] * n
        self.counts: List[int] = [0] * n
        self._last_weights: List[float] = [1.0 / n] * n
        self.history: List[Dict[str, float]] = []  # [{"op":w, ...}, ...]

    # ---------- selection ----------
    def _softmax_probs(self) -> List[float]:
        # Stable softmax over Q/tau
        tau = max(1e-9, self.cfg.tau)
        vals = [q / tau for q in self.q]
        m = max(vals) if vals else 0.0
        exps = [math.exp(v - m) for v in vals]
        s = sum(exps) or 1.0
        probs = [max(self.cfg.min_weight, e / s) for e in exps]
        # Renormalize if clamped
        s2 = sum(probs)
        return [p / s2 for p in probs]

    def _epsilon_greedy_probs(self) -> List[float]:
        n = len(self.ops)
        if n == 0:
            return []
        eps = min(max(self.cfg.epsilon, 0.0), 1.0)
        # best arms (greedy on Q)
        best = max(self.q) if self.q else 0.0
        winners = [i for i, q in enumerate(self.q) if abs(q - best) < 1e-12]
        base = [0.0] * n
        if winners:
            share = (1.0 - eps) / len(winners)
            for i in winners:
                base[i] = share
        explore = eps / n
        probs = [max(self.cfg.min_weight, base[i] + explore) for i in range(n)]
        s = sum(probs)
        return [p / s for p in probs]

    def weights(self) -> List[float]:
        """Current selection probabilities (also returned each generation for history)."""
        if self.cfg.mode == "softmax":
            self._last_weights = self._softmax_probs()
        else:
            self._last_weights = self._epsilon_greedy_probs()
        return self._last_weights[:]

    def select(self) -> int:
        probs = self.weights()
        # Multinomial draw
        r = self.rng.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return i
        return len(probs) - 1  # fallback

    # ---------- update ----------
    def update(self, idx: int, reward: float) -> None:
        """Incremental Q update with learning rate alpha."""
        self.counts[idx] += 1
        a = self.cfg.alpha
        self.q[idx] = (1 - a) * self.q[idx] + a * float(max(0.0, reward))

    # ---------- logging ----------
    def snapshot(self) -> Dict[str, float]:
        w = self.weights()
        snap = {self.ops[i]: w[i] for i in range(len(self.ops))}
        self.history.append(snap)
        return snap

    def as_history(self) -> List[Dict[str, float]]:
        return self.history[:]
