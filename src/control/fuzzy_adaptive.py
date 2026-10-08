"""
Type-1 Fuzzy Adaptive Scaling System
=====================================

Implements a Mamdani-type fuzzy inference system that adaptively
adjusts PID gains based on payload variation, turbulence level,
battery state, and error dynamics.

Author: Luis Adrián Silva Reyes
Institution: Tijuana Institute of Technology
"""

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
# Membership functions
# ---------------------------------------------------------------------- #
class FuzzyMembershipFunction:
    """Base class for membership functions."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GaussianMF(FuzzyMembershipFunction):
    """Gaussian membership function."""

    def __init__(self, mean: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mean = mean
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)

    def __repr__(self) -> str:
        return f"GaussianMF(mean={self.mean}, sigma={self.sigma})"


class TriangularMF(FuzzyMembershipFunction):
    """Triangular membership function."""

    def __init__(self, a: float, b: float, c: float):
        if not (a < b < c):
            raise ValueError("Require a < b < c.")
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        left = np.where(x <= self.b,
                        np.where(x >= self.a, (x - self.a) / (self.b - self.a), 0.0),
                        0.0)
        right = np.where(x >= self.b,
                         np.where(x <= self.c, (self.c - x) / (self.c - self.b), 0.0),
                         0.0)
        return np.maximum(left, right)

    def __repr__(self) -> str:
        return f"TriangularMF({self.a}, {self.b}, {self.c})"


# ---------------------------------------------------------------------- #
# Fuzzy adaptive scaler
# ---------------------------------------------------------------------- #
class FuzzyAdaptiveScaler:
    """
    Type-1 fuzzy system for adaptive PID gain correction.

    Inputs (all scalar):
        - payload_variation: [0, 50] percentage above nominal mass
        - turbulence_level:  [0, 1] normalized
        - battery_state:     [0, 1] normalized (1.0 = full)
        - error_dynamics:    [0, 1] normalized

    Outputs:
        - alpha_p: proportional gain correction in [0.5, 2.0]
        - alpha_i: integral gain correction in [0.5, 2.0]
        - alpha_d: derivative gain correction in [0.5, 2.0]
    """

    def __init__(self) -> None:
        self._init_membership_functions()
        self._init_rule_base()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _init_membership_functions(self) -> None:
        """Initialize input and output membership functions."""
        # Payload variation (percentage)
        self.payload_mfs: Dict[str, FuzzyMembershipFunction] = {
            "light": GaussianMF(0, 5),
            "medium": GaussianMF(15, 8),
            "heavy": GaussianMF(35, 10),
        }

        # Turbulence level (0–1)
        self.turbulence_mfs: Dict[str, FuzzyMembershipFunction] = {
            "calm": GaussianMF(0.1, 0.15),
            "moderate": GaussianMF(0.4, 0.15),
            "strong": GaussianMF(0.8, 0.15),
        }

        # Battery state (0–1)
        self.battery_mfs: Dict[str, FuzzyMembershipFunction] = {
            "low": GaussianMF(0.15, 0.10),
            "medium": GaussianMF(0.50, 0.15),
            "high": GaussianMF(0.85, 0.10),
        }

        # Error dynamics (0–1)
        self.error_mfs: Dict[str, FuzzyMembershipFunction] = {
            "small": GaussianMF(0.05, 0.08),
            "medium": GaussianMF(0.30, 0.12),
            "large": GaussianMF(0.70, 0.15),
        }

        # Output MFs (gain correction factors)
        self.output_mfs: Dict[str, TriangularMF] = {
            "low": TriangularMF(0.5, 0.7, 0.9),
            "medium": TriangularMF(0.8, 1.0, 1.2),
            "high": TriangularMF(1.1, 1.4, 1.7),
            "very_high": TriangularMF(1.5, 1.8, 2.0),
        }

    def _init_rule_base(self) -> None:
        """
        Initialize the fuzzy rule base.

        Each rule is a tuple:
            (payload, turbulence, battery, error, alpha_p, alpha_i, alpha_d)
        """
        # Explicit physically-motivated rules
        explicit_rules: List[Tuple[str, str, str, str, str, str, str]] = [
            # Nominal conditions
            ("light", "calm", "high", "small", "low", "low", "medium"),
            ("light", "calm", "high", "medium", "medium", "medium", "medium"),
            ("light", "calm", "high", "large", "high", "high", "high"),
            # Heavy payload compensation
            ("heavy", "calm", "high", "small", "high", "medium", "medium"),
            ("heavy", "calm", "high", "medium", "high", "high", "medium"),
            ("heavy", "calm", "high", "large", "very_high", "high", "high"),
            # Strong turbulence — increased damping
            ("light", "strong", "high", "small", "high", "medium", "very_high"),
            ("light", "strong", "high", "medium", "high", "high", "very_high"),
            ("medium", "strong", "high", "large", "very_high", "high", "very_high"),
            # Low battery compensation
            ("light", "calm", "low", "small", "high", "high", "medium"),
            ("medium", "calm", "low", "medium", "high", "high", "high"),
            ("heavy", "calm", "low", "large", "very_high", "very_high", "high"),
            # Combined disturbances
            ("heavy", "strong", "low", "large", "very_high", "very_high", "very_high"),
            ("medium", "moderate", "medium", "medium", "high", "high", "high"),
            ("light", "moderate", "medium", "small", "medium", "medium", "medium"),
        ]

        # Complete the rule base with defaults for any missing combination
        payload_terms = ["light", "medium", "heavy"]
        turb_terms = ["calm", "moderate", "strong"]
        batt_terms = ["low", "medium", "high"]
        err_terms = ["small", "medium", "large"]

        explicit_keys = {(r[0], r[1], r[2], r[3]) for r in explicit_rules}
        full_rules: List[Tuple[str, str, str, str, str, str, str]] = list(explicit_rules)

        for p in payload_terms:
            for t in turb_terms:
                for b in batt_terms:
                    for e in err_terms:
                        if (p, t, b, e) not in explicit_keys:
                            full_rules.append((p, t, b, e, "medium", "medium", "medium"))

        self.rules = full_rules

    # ------------------------------------------------------------------ #
    # Inference pipeline
    # ------------------------------------------------------------------ #
    def _fuzzify(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Compute membership degrees for all inputs."""
        memberships: Dict[str, Dict[str, float]] = {}

        memberships["payload"] = {
            name: float(mf(np.array([inputs["payload"]]))[0])
            for name, mf in self.payload_mfs.items()
        }
        memberships["turbulence"] = {
            name: float(mf(np.array([inputs["turbulence"]]))[0])
            for name, mf in self.turbulence_mfs.items()
        }
        memberships["battery"] = {
            name: float(mf(np.array([inputs["battery"]]))[0])
            for name, mf in self.battery_mfs.items()
        }
        memberships["error"] = {
            name: float(mf(np.array([inputs["error"]]))[0])
            for name, mf in self.error_mfs.items()
        }
        return memberships

    def _infer(
        self, memberships: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Perform fuzzy inference using the rule base."""
        alpha_p_firing = {name: 0.0 for name in self.output_mfs}
        alpha_i_firing = {name: 0.0 for name in self.output_mfs}
        alpha_d_firing = {name: 0.0 for name in self.output_mfs}

        for rule in self.rules:
            p, t, b, e, ap, ai, ad = rule
            firing = min(
                memberships["payload"][p],
                memberships["turbulence"][t],
                memberships["battery"][b],
                memberships["error"][e],
            )
            alpha_p_firing[ap] = max(alpha_p_firing[ap], firing)
            alpha_i_firing[ai] = max(alpha_i_firing[ai], firing)
            alpha_d_firing[ad] = max(alpha_d_firing[ad], firing)

        return {
            "alpha_p": alpha_p_firing,
            "alpha_i": alpha_i_firing,
            "alpha_d": alpha_d_firing,
        }

    def _defuzzify(
        self, output_memberships: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Defuzzify using the centroid method."""
        x = np.linspace(0.5, 2.0, 200)
        results: Dict[str, float] = {}

        for output_name, firing in output_memberships.items():
            aggregated = np.zeros_like(x)
            for mf_name, strength in firing.items():
                if strength > 0.0:
                    aggregated = np.maximum(aggregated, strength * self.output_mfs[mf_name](x))

            total = np.trapz(aggregated, x)
            if total > 1e-9:
                results[output_name] = float(np.trapz(x * aggregated, x) / total)
            else:
                results[output_name] = 1.0

        return results

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compute_corrections(
        self,
        payload_variation: float,
        turbulence_level: float,
        battery_state: float,
        error_dynamics: float,
    ) -> Tuple[float, float, float]:
        """
        Compute adaptive gain correction factors.

        Returns:
            (alpha_p, alpha_i, alpha_d)
        """
        inputs = {
            "payload": float(np.clip(payload_variation, 0.0, 50.0)),
            "turbulence": float(np.clip(turbulence_level, 0.0, 1.0)),
            "battery": float(np.clip(battery_state, 0.0, 1.0)),
            "error": float(np.clip(error_dynamics, 0.0, 1.0)),
        }
        memberships = self._fuzzify(inputs)
        output_memberships = self._infer(memberships)
        corrections = self._defuzzify(output_memberships)
        return corrections["alpha_p"], corrections["alpha_i"], corrections["alpha_d"]

    def apply_to_gains(
        self,
        gains: np.ndarray,
        payload_variation: float = 0.0,
        turbulence_level: float = 0.0,
        battery_state: float = 1.0,
        error_dynamics: float = 0.0,
    ) -> np.ndarray:
        """
        Apply fuzzy corrections to a 12-element PID gain vector.
        """
        alpha_p, alpha_i, alpha_d = self.compute_corrections(
            payload_variation, turbulence_level, battery_state, error_dynamics
        )
        corrected = np.asarray(gains, dtype=float).copy()
        for i in range(4):
            corrected[i * 3] *= alpha_p
            corrected[i * 3 + 1] *= alpha_i
            corrected[i * 3 + 2] *= alpha_d
        return corrected