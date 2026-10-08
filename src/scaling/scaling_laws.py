"""
Scaling Laws for PSO-Optimized PID Gains
=========================================

Implements the theoretical scaling laws derived from dimensional
analysis for transferring PID gains between quadrotor sizes.

Scaling laws:
    Kp ∝ m·l⁻¹
    Ki ∝ m·l⁻³/²
    Kd ∝ m·l⁻¹/²

Author: Luis Adrián Silva Reyes
Institution: Tijuana Institute of Technology
"""

from typing import Dict, List

import numpy as np


class ScalingLaw:
    """Implements PID gain scaling between quadrotor configurations."""

    def __init__(
        self,
        source_mass: float,
        source_arm_length: float,
        source_inertia: np.ndarray = None,
    ):
        if source_mass <= 0 or source_arm_length <= 0:
            raise ValueError("Source mass and arm length must be positive.")
        self.m0 = source_mass
        self.l0 = source_arm_length
        self.I0 = source_inertia if source_inertia is not None else np.array([0.1, 0.1, 0.2])

    # ------------------------------------------------------------------ #
    # Gain scaling
    # ------------------------------------------------------------------ #
    def scale_gains(
        self,
        gains: np.ndarray,
        target_mass: float,
        target_arm_length: float,
    ) -> np.ndarray:
        """Scale a 12-element gain vector from source to target quadrotor."""
        m_ratio = target_mass / self.m0
        l_ratio = target_arm_length / self.l0

        scaled = np.asarray(gains, dtype=float).copy()

        for axis in range(4):
            idx = axis * 3
            # Kp ∝ m·l⁻¹
            scaled[idx] *= m_ratio / l_ratio
            # Ki ∝ m·l⁻³/²
            scaled[idx + 1] *= m_ratio / (l_ratio ** 1.5)
            # Kd ∝ m·l⁻¹/²
            scaled[idx + 2] *= m_ratio / np.sqrt(l_ratio)

        return scaled

    # ------------------------------------------------------------------ #
    # Exponents from data
    # ------------------------------------------------------------------ #
    def compute_exponents(
        self,
        gains_list: List[np.ndarray],
        masses: List[float],
        lengths: List[float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute scaling exponents from experimental data.

        Fits log(K) = a + b·log(m) + c·log(l) for each gain.
        """
        masses = np.asarray(masses, dtype=float)
        lengths = np.asarray(lengths, dtype=float)
        gains_array = np.asarray(gains_list, dtype=float)

        log_m = np.log(masses)
        log_l = np.log(lengths)
        X = np.column_stack([np.ones_like(log_m), log_m, log_l])

        gain_names = [
            "Kp_z", "Ki_z", "Kd_z",
            "Kp_phi", "Ki_phi", "Kd_phi",
            "Kp_theta", "Ki_theta", "Kd_theta",
            "Kp_psi", "Ki_psi", "Kd_psi",
        ]

        results: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(gain_names):
            y = np.log(np.maximum(gains_array[:, i], 1e-12))
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ coeffs
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            results[name] = {
                "constant": float(np.exp(coeffs[0])),
                "mass_exponent": float(coeffs[1]),
                "length_exponent": float(coeffs[2]),
                "r_squared": float(r2),
            }
        return results


# ---------------------------------------------------------------------- #
# Variant generation
# ---------------------------------------------------------------------- #
def generate_variants(
    base_mass: float = 1.0,
    base_arm_length: float = 0.25,
    size_factors: List[float] = None,
) -> Dict[float, Dict[str, float]]:
    """Generate quadrotor variants based on geometric similarity."""
    if size_factors is None:
        size_factors = [0.5, 1.0, 2.0, 5.0]

    variants: Dict[float, Dict[str, float]] = {}
    for factor in size_factors:
        mass = base_mass * factor
        arm_length = base_arm_length * (factor ** (1.0 / 3.0))
        inertia_scale = factor * (factor ** (2.0 / 3.0))
        variants[factor] = {
            "mass": mass,
            "arm_length": arm_length,
            "Ixx": 0.1 * inertia_scale,
            "Iyy": 0.1 * inertia_scale,
            "Izz": 0.2 * inertia_scale,
        }
    return variants