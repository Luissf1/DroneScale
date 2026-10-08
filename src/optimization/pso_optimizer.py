"""
Particle Swarm Optimization for PID Tuning
==========================================

Implements PSO for optimizing 12 PID gains for quadrotor control.

Author: Luis Adrián Silva Reyes
Institution: Tijuana Institute of Technology
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class PSOConfig:
    """PSO algorithm configuration."""

    swarm_size: int = 50
    max_iterations: int = 100
    inertia_max: float = 0.7
    inertia_min: float = 0.4
    cognitive_coeff: float = 1.7
    social_coeff: float = 1.7
    velocity_max: float = 0.2
    num_runs: int = 30


@dataclass
class PSOBounds:
    """Variable bounds for PID gains."""

    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.lower is None:
            self.lower = np.array([
                2.0, 0.01, 0.1,     # Altitude
                0.1, 0.001, 0.1,    # Roll
                0.1, 0.001, 0.1,    # Pitch
                0.1, 0.001, 0.1,    # Yaw
            ])
        if self.upper is None:
            self.upper = np.array([
                15.0, 2.0, 5.0,     # Altitude
                10.0, 0.1, 2.0,     # Roll
                10.0, 0.1, 2.0,     # Pitch
                10.0, 0.1, 2.0,     # Yaw
            ])


class PSOOptimizer:
    """Particle Swarm Optimization for PID gain tuning."""

    def __init__(
        self,
        config: PSOConfig,
        bounds: PSOBounds,
        fitness_func: Callable[[np.ndarray], float],
    ):
        self.config = config
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.n_vars = len(bounds.lower)

        self.best_position: Optional[np.ndarray] = None
        self.best_fitness: float = float("inf")
        self.convergence_history: List[float] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def optimize(self, seed: int = 42) -> Tuple[np.ndarray, float, List[float]]:
        """Run PSO optimization."""
        np.random.seed(seed)
        positions, velocities, pbest_pos, pbest_fit = self._init_swarm()

        best_idx = int(np.argmin(pbest_fit))
        gbest_pos = pbest_pos[best_idx].copy()
        gbest_fit = float(pbest_fit[best_idx])

        convergence: List[float] = [gbest_fit]

        for iteration in range(self.config.max_iterations):
            w = self._compute_inertia(iteration)
            for i in range(self.config.swarm_size):
                r1, r2 = np.random.rand(2)
                cognitive = self.config.cognitive_coeff * r1 * (pbest_pos[i] - positions[i])
                social = self.config.social_coeff * r2 * (gbest_pos - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social

                vel_max = self.config.velocity_max * (self.bounds.upper - self.bounds.lower)
                velocities[i] = np.clip(velocities[i], -vel_max, vel_max)

                positions[i] = np.clip(
                    positions[i] + velocities[i],
                    self.bounds.lower,
                    self.bounds.upper,
                )

                fitness = float(self.fitness_func(positions[i]))
                if fitness < pbest_fit[i]:
                    pbest_pos[i] = positions[i].copy()
                    pbest_fit[i] = fitness
                    if fitness < gbest_fit:
                        gbest_pos = positions[i].copy()
                        gbest_fit = fitness

            convergence.append(gbest_fit)

        self.best_position = gbest_pos
        self.best_fitness = gbest_fit
        self.convergence_history = convergence
        return gbest_pos, gbest_fit, convergence

    def optimize_multiple_runs(
        self, num_runs: Optional[int] = None
    ) -> Tuple[np.ndarray, float, List[List[float]]]:
        """Run PSO multiple times and return the best result."""
        if num_runs is None:
            num_runs = self.config.num_runs

        best_pos = None
        best_fit = float("inf")
        all_convergence: List[List[float]] = []

        for run in range(num_runs):
            pos, fit, conv = self.optimize(seed=42 + run)
            all_convergence.append(conv)
            if fit < best_fit:
                best_fit = fit
                best_pos = pos.copy()

        return best_pos, best_fit, all_convergence

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _init_swarm(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positions = np.zeros((self.config.swarm_size, self.n_vars))
        velocities = np.zeros((self.config.swarm_size, self.n_vars))
        pbest_pos = np.zeros((self.config.swarm_size, self.n_vars))
        pbest_fit = np.full(self.config.swarm_size, float("inf"))

        for i in range(self.config.swarm_size):
            positions[i] = np.random.uniform(self.bounds.lower, self.bounds.upper)
            fitness = float(self.fitness_func(positions[i]))
            pbest_pos[i] = positions[i].copy()
            pbest_fit[i] = fitness

        return positions, velocities, pbest_pos, pbest_fit

    def _compute_inertia(self, iteration: int) -> float:
        return (
            self.config.inertia_max
            - (self.config.inertia_max - self.config.inertia_min)
            * (iteration / max(1, self.config.max_iterations))
        )