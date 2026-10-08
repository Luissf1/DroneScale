"""Tests for the PSO optimizer."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimization.pso_optimizer import PSOOptimizer, PSOConfig, PSOBounds


def sphere(x):
    return float(np.sum((x - 3.0) ** 2))


def test_pso_finds_sphere_minimum():
    config = PSOConfig(swarm_size=20, max_iterations=40, num_runs=1)
    bounds = PSOBounds(
        lower=np.full(12, -10.0),
        upper=np.full(12, 10.0),
    )
    opt = PSOOptimizer(config, bounds, sphere)
    best_pos, best_fit, conv = opt.optimize(seed=0)
    assert best_fit < 5.0
    assert len(conv) == config.max_iterations + 1


def test_pso_respects_bounds():
    config = PSOConfig(swarm_size=10, max_iterations=10, num_runs=1)
    bounds = PSOBounds()
    opt = PSOOptimizer(config, bounds, sphere)
    best_pos, _, _ = opt.optimize(seed=0)
    assert np.all(best_pos >= bounds.lower)
    assert np.all(best_pos <= bounds.upper)


def test_multiple_runs_returns_best():
    config = PSOConfig(swarm_size=10, max_iterations=15)
    bounds = PSOBounds(lower=np.full(12, -5.0), upper=np.full(12, 5.0))
    opt = PSOOptimizer(config, bounds, sphere)
    best_pos, best_fit, conv_list = opt.optimize_multiple_runs(num_runs=3)
    assert len(conv_list) == 3
    assert best_fit <= min(c[-1] for c in conv_list) + 1e-9