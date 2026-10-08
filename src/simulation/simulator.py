"""
Quadrotor Simulator
===================

Runs time-domain simulations of the quadrotor under PID (or fuzzy-adaptive
PID) control, returning trajectories and performance metrics.

Author: Luis Adrián Silva Reyes
Institution: Tijuana Institute of Technology
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from src.dynamics.quadrotor_model import QuadrotorDynamics, QuadrotorParameters


@dataclass
class SimulationResult:
    """Container for simulation outputs."""

    t: np.ndarray
    X: np.ndarray                    # shape (12, N)
    metrics: dict = field(default_factory=dict)
    fitness: float = float("inf")


class Simulator:
    """
    Time-domain simulator for the quadrotor with PID control.

    Args:
        dynamics: QuadrotorDynamics instance.
        t_span: (t_start, t_end) tuple.
        n_points: Number of output points.
        tol_settling: Fractional tolerance for settling-time detection.
        cost_weights: (w_ts, w_mp, w_itse, w_iae).
    """

    def __init__(
        self,
        dynamics: QuadrotorDynamics,
        t_span: tuple = (0.0, 10.0),
        n_points: int = 500,
        tol_settling: float = 0.02,
        cost_weights: tuple = (0.3, 0.3, 0.2, 0.2),
    ):
        self.dynamics = dynamics
        self.t_span = t_span
        self.n_points = n_points
        self.tol_settling = tol_settling
        self.cost_weights = cost_weights

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def simulate(
        self,
        gains: np.ndarray,
        reference: np.ndarray,
        gain_modifier: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    ) -> SimulationResult:
        """
        Simulate the closed-loop system.

        Args:
            gains: PID gains [12].
            reference: Desired state [z, phi, theta, psi] [4].
            gain_modifier: Optional callable (t, state) -> gains that returns
                possibly time-varying gains (used for fuzzy adaptation).

        Returns:
            SimulationResult
        """
        t_eval = np.linspace(self.t_span[0], self.t_span[1], self.n_points)
        X0 = np.zeros(12)

        integral_err = np.zeros(4)

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            nonlocal integral_err
            current_gains = gains if gain_modifier is None else gain_modifier(t, state)
            # Use an approximate dt for integral update
            dt = self._approximate_dt(t)
            controls, _, integral_err = self.dynamics.compute_control_effort(
                state, current_gains, reference,
                integral_errors=integral_err,
                dt=dt,
            )
            return self.dynamics.derivatives(t, state, controls)

        sol = solve_ivp(
            rhs,
            self.t_span,
            X0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
        )

        if not sol.success:
            return SimulationResult(
                t=np.array([0.0]),
                X=np.zeros((12, 1)),
                metrics=self._default_metrics(),
                fitness=1000.0,
            )

        metrics = self._compute_metrics(sol.t, sol.y, reference)
        fitness = self._compute_fitness(metrics)

        return SimulationResult(t=sol.t, X=sol.y, metrics=metrics, fitness=fitness)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _compute_metrics(
        self, t: np.ndarray, X: np.ndarray, reference: np.ndarray
    ) -> dict:
        z = X[2, :]
        e_z = reference[0] - z

        RMSE = float(np.sqrt(np.mean(e_z ** 2)))
        IAE = float(np.trapz(np.abs(e_z), t))
        ITSE = float(np.trapz(t * e_z ** 2, t))

        tol = self.tol_settling * max(abs(reference[0]), 1e-3)
        inside = np.abs(e_z) <= tol
        t_settling = float(t[-1])
        for i in range(len(t) - 1, -1, -1):
            if inside[i]:
                # check remaining
                if np.all(inside[i:]):
                    t_settling = float(t[i])
                else:
                    break

        max_z = float(np.max(z)) if len(z) > 0 else 0.0
        overshoot = max(0.0, (max_z - reference[0]) / max(reference[0], 1e-6) * 100.0)

        return {
            "RMSE": RMSE,
            "IAE": IAE,
            "ITSE": ITSE,
            "t_settling": t_settling,
            "overshoot": overshoot,
        }

    def _compute_fitness(self, metrics: dict) -> float:
        w_ts, w_mp, w_itse, w_iae = self.cost_weights
        t_norm = min(metrics["t_settling"] / 10.0, 1.0)
        mp_norm = min(metrics["overshoot"] / 100.0, 1.0)
        itse_norm = min(metrics["ITSE"] / 50.0, 1.0)
        iae_norm = min(metrics["IAE"] / 20.0, 1.0)
        return float(
            w_ts * t_norm + w_mp * mp_norm + w_itse * itse_norm + w_iae * iae_norm
        )

    @staticmethod
    def _default_metrics() -> dict:
        return {
            "RMSE": 10.0,
            "IAE": 50.0,
            "ITSE": 100.0,
            "t_settling": 10.0,
            "overshoot": 100.0,
        }

    def _approximate_dt(self, t: float) -> float:
        return (self.t_span[1] - self.t_span[0]) / max(self.n_points, 1)