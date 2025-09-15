# python/tests/test_adaptive_operators.py
import math
import random
import pytest

from python.ga import AdaptiveGA, GAConfig
from python.adapt import BanditConfig
from python.penalty_manager import PenaltyConfig

TINY_INSTANCE = {
    "depot": {"id": 0, "x": 0.0, "y": 0.0},
    "customers": [{"id": 1, "x": 1.0, "y": 0.0},
                  {"id": 2, "x": 2.0, "y": 0.0},
                  {"id": 3, "x": 3.0, "y": 0.0}]
}

def test_bandit_learns_non_uniform_weights():
    """
    Seeded run where 'two_opt' gives bigger improvement than others.
    Expect non-uniform operator weights after 25 generations.
    """
    seed = 123
    ga = AdaptiveGA(
        instance=TINY_INSTANCE,
        ga_cfg=GAConfig(pop_size=12, generations=25, seed=seed, mutation_rate=1.0, stagnation_S=9999),
        bandit_cfg=BanditConfig(mode="epsilon_greedy", epsilon=0.2, alpha=0.3, seed=seed),
        penalty_cfg=PenaltyConfig()
    )

    # Monkeypatch: make _apply_operator deterministically better for 'two_opt'
    def rigged_apply(op, sol):
        # pretend each op applies a known gain to fitness via evaluation override
        return sol  # don't change structure; we simulate reward via evaluator

    ga._apply_operator = rigged_apply  # type: ignore

    # Override evaluate so reward depends on the op used (captured via bandit.update)
    # We'll intercept bandit.update to inject reward:
    original_update = ga.bandit.update
    rng = random.Random(seed)

    def fake_update(idx, reward_unused):
        op = ga.bandit.ops[idx]
        # assign synthetic reward distribution
        if op == "two_opt":
            reward = 1.0
        elif op == "oropt1":
            reward = 0.2
        else:
            reward = 0.1
        return original_update(idx, reward)

    ga.bandit.update = fake_update  # type: ignore

    out = ga.run(mode="adaptive")
    last = out["operator_history"][-1]
    # Expect two_opt to dominate
    assert last["two_opt"] > last["oropt1"] > last["cross"], last

def test_stagnation_intensification_improves(monkeypatch):
    """
    Inject stagnation: evaluation returns constant fitness unless intensification runs,
    where elites get improved once. Compare best fitness with intensification on vs off.
    """
    seed = 7
    base_fit = 100.0

    def const_eval(sol):
        return base_fit, {"penalties": {}}

    ga_on = AdaptiveGA(
        instance=TINY_INSTANCE,
        ga_cfg=GAConfig(pop_size=6, generations=20, seed=seed, mutation_rate=1.0,
                        stagnation_S=5, enable_intensification=True,
                        intensify_top_k=2, intensify_iters=1),
        bandit_cfg=BanditConfig(seed=seed)
    )
    ga_on.evaluate = const_eval  # type: ignore

    # Patch intensify_elites to actually improve elites once triggered
    import python.localsearch as ls

    def fake_intensify(rng, population, fitnesses, evaluate, top_k, ops, iters_per, agent=None, vehicles=None):
        # subtract 1.0 from top-1 elite
        best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        fitnesses[best_idx] = fitnesses[best_idx] - 1.0
        return population, fitnesses, {"two_opt": 1}

    monkeypatch.setattr(ls, "intensify_elites", fake_intensify)

    out_on = ga_on.run(mode="adaptive")
    best_on = out_on["best_fitness"]

    # Same GA but intensification disabled
    ga_off = AdaptiveGA(
        instance=TINY_INSTANCE,
        ga_cfg=GAConfig(pop_size=6, generations=20, seed=seed, mutation_rate=1.0,
                        stagnation_S=5, enable_intensification=False),
        bandit_cfg=BanditConfig(seed=seed)
    )
    ga_off.evaluate = const_eval  # type: ignore
    out_off = ga_off.run(mode="adaptive")
    best_off = out_off["best_fitness"]

    assert best_on < best_off, (best_on, best_off)

def test_penalty_manager_scales_betas():
    """
    When a penalty is violated ~100% of a full window, β should scale up (gamma_up).
    When always 0 violations for a window, β should scale down (gamma_down).
    """
    from python.penalty_manager import PenaltyManager, PenaltyConfig
    pc = PenaltyConfig(betas={"beta_capacity": 1.0, "beta_duration": 2.0}, window=5, gamma_up=2.0, gamma_down=0.5)
    pm = PenaltyManager(pc)

    # Fill window with violations for capacity; none for duration
    for _ in range(pc.window):
        pm.observe({"beta_capacity": 10.0, "beta_duration": 0.0})
    betas = pm.step()
    assert math.isclose(betas["beta_capacity"], 2.0, rel_tol=1e-9)
    assert math.isclose(betas["beta_duration"], 1.0, rel_tol=1e-9)

    # Now fill window with zeros for capacity; violations for duration
    for _ in range(pc.window):
        pm.observe({"beta_capacity": 0.0, "beta_duration": 5.0})
    betas = pm.step()
    # capacity down, duration up
    assert math.isclose(betas["beta_capacity"], 1.0, rel_tol=1e-9 * 10)  # 2.0 * 0.5
    assert math.isclose(betas["beta_duration"], 2.0, rel_tol=1e-9 * 10)  # 1.0 * 2.0
