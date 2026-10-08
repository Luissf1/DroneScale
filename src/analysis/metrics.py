"""
Metrics Utilities
=================

Functions for computing aggregate metrics from experiment outputs.

Author: Luis Adrián Silva Reyes
Institution: Tijuana Institute of Technology
"""

from typing import Dict, List

import numpy as np


def compute_transfer_metrics(
    fit_optimal: float,
    fit_scaled: float,
    fit_unscaled: float,
    fit_zn: float = None,
) -> Dict[str, float]:
    """
    Compute transfer metrics for a single source->target pair.

    Args:
        fit_optimal: Fitness of fully re-optimized gains.
        fit_scaled: Fitness of scaled transfer.
        fit_unscaled: Fitness of unscaled transfer.
        fit_zn: Optional Ziegler-Nichols baseline fitness.

    Returns:
        Dictionary with degradation and improvement metrics.
    """
    eps = 1e-12
    degradation_unscaled = (fit_unscaled - fit_optimal) / (fit_optimal + eps) * 100.0
    degradation_scaled = (fit_scaled - fit_optimal) / (fit_optimal + eps) * 100.0

    improvement = (
        (fit_unscaled - fit_scaled) / (fit_unscaled + eps) * 100.0
        if fit_unscaled > 0
        else 0.0
    )

    retained_benefit = None
    if fit_zn is not None and (fit_zn - fit_optimal) > eps:
        retained_benefit = (fit_zn - fit_scaled) / (fit_zn - fit_optimal) * 100.0

    return {
        "fit_optimal": float(fit_optimal),
        "fit_scaled": float(fit_scaled),
        "fit_unscaled": float(fit_unscaled),
        "fit_zn": float(fit_zn) if fit_zn is not None else None,
        "degradation_unscaled": float(degradation_unscaled),
        "degradation_scaled": float(degradation_scaled),
        "improvement": float(improvement),
        "retained_benefit": float(retained_benefit) if retained_benefit is not None else None,
    }


def summarize_improvements(improvements: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of improvements."""
    if not improvements:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    arr = np.asarray(improvements, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Fit y = a * x^b via log-log least squares.

    Returns:
        Dictionary with 'a', 'b', 'r_squared'.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    log_x = np.log(np.maximum(x, 1e-12))
    log_y = np.log(np.maximum(y, 1e-12))

    b, log_a = np.polyfit(log_x, log_y, 1)
    y_pred = log_a + b * log_x
    ss_res = float(np.sum((log_y - y_pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "a": float(np.exp(log_a)),
        "b": float(b),
        "r_squared": float(r2),
    }