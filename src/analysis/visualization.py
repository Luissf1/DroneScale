"""
Visualization
=============

Plotting helpers for the scaling analysis and fuzzy adaptive experiments.

Author: Luis Adrián Silva Reyes
Institution: Tijuana Institute of Technology
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .metrics import fit_power_law


# ---------------------------------------------------------------------- #
# Figure 1: Kp scaling
# ---------------------------------------------------------------------- #
def plot_kp_scaling(
    masses: List[float],
    kp_z: List[float],
    kp_attitude_avg: List[float],
    kd_z: List[float],
    output_path: str = "figures/fig1_kp_scaling.png",
) -> str:
    """Generate Figure 1: scaling of PID gains with mass."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    masses_arr = np.asarray(masses, dtype=float)

    # (a) Altitude Kp
    fit = fit_power_law(masses_arr, np.asarray(kp_z))
    axes[0].scatter(masses, kp_z, s=100, color="#1f77b4", zorder=5)
    m_fit = np.linspace(min(masses), max(masses), 100)
    axes[0].plot(
        m_fit, fit["a"] * m_fit ** fit["b"], "r-", linewidth=2,
        label=f"$K_p \\propto m^{{{fit['b']:.2f}}}$",
    )
    axes[0].set_xlabel("Mass (kg)")
    axes[0].set_ylabel("$K_p$ (Altitude)")
    axes[0].set_title("(a) Proportional Gain Scaling", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # (b) Attitude average Kp
    fit_att = fit_power_law(masses_arr, np.asarray(kp_attitude_avg))
    axes[1].scatter(masses, kp_attitude_avg, s=100, color="#ff7f0e", zorder=5)
    axes[1].plot(
        m_fit, fit_att["a"] * m_fit ** fit_att["b"], "r-", linewidth=2,
        label=f"$K_p \\propto m^{{{fit_att['b']:.2f}}}$",
    )
    axes[1].set_xlabel("Mass (kg)")
    axes[1].set_ylabel("$K_p$ (Attitude avg)")
    axes[1].set_title("(b) Attitude Gains Scaling", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # (c) Comparison of gain types
    axes[2].scatter(masses, kp_z, s=80, label="$K_p$ (altitude)", alpha=0.7)
    axes[2].scatter(masses, kd_z, s=80, label="$K_d$ (altitude)", alpha=0.7)
    axes[2].scatter(masses, kp_attitude_avg, s=80, label="$K_p$ (attitude)", alpha=0.7)
    axes[2].set_xlabel("Mass (kg)")
    axes[2].set_ylabel("Gain Value")
    axes[2].set_title("(c) Comparison of Gain Types", fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle("Figure 1: Scaling of PID Gains with Quadrotor Mass",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------------------- #
# Figure 2: Performance comparison
# ---------------------------------------------------------------------- #
def plot_performance_comparison(
    target_names: List[str],
    fit_unscaled: List[float],
    fit_scaled: List[float],
    fit_optimal: List[float],
    output_path: str = "figures/fig2_performance.png",
) -> str:
    """Generate Figure 2: performance comparison."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(target_names))
    width = 0.25

    ax1.bar(x - width, fit_unscaled, width, label="Unscaled Transfer",
            color="#d62728", alpha=0.8, edgecolor="black")
    ax1.bar(x, fit_scaled, width, label="Scaled Transfer",
            color="#2ca02c", alpha=0.8, edgecolor="black")
    ax1.bar(x + width, fit_optimal, width, label="Re-optimized",
            color="#1f77b4", alpha=0.8, edgecolor="black")

    ax1.set_xlabel("Target Quadrotor Size")
    ax1.set_ylabel("Fitness (lower is better)")
    ax1.set_title("(a) Performance Comparison", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(target_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    improvements = [
        (u - s) / u * 100.0 if u > 0 else 0.0
        for u, s in zip(fit_unscaled, fit_scaled)
    ]
    ax2.bar(x, improvements, width, color="#2ca02c", alpha=0.8, edgecolor="black")
    ax2.set_xlabel("Target Quadrotor Size")
    ax2.set_ylabel("Improvement over Unscaled (%)")
    ax2.set_title("(b) Improvement with Scaling", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(target_names, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")
    for i, imp in enumerate(improvements):
        ax2.text(i, imp + 1, f"{imp:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Figure 2: Performance of Gain Transfer from Baseline Quadrotor",
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------------------- #
# Figure 3: Scenario breakdown
# ---------------------------------------------------------------------- #
def plot_scenario_breakdown(
    scenario_names: List[str],
    fit_unscaled: List[float],
    fit_scaled: List[float],
    fit_fuzzy: List[float],
    fit_optimal: List[float],
    output_path: str = "figures/fig3_scenarios.png",
) -> str:
    """Generate Figure 3: performance by scenario."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(scenario_names))
    width = 0.2

    ax.bar(x - 1.5 * width, fit_unscaled, width, label="Unscaled",
           color="#d62728", alpha=0.8, edgecolor="black")
    ax.bar(x - 0.5 * width, fit_scaled, width, label="Scaled",
           color="#2ca02c", alpha=0.8, edgecolor="black")
    ax.bar(x + 0.5 * width, fit_fuzzy, width, label="Fuzzy Adaptive",
           color="#9467bd", alpha=0.8, edgecolor="black")
    ax.bar(x + 1.5 * width, fit_optimal, width, label="Re-optimized",
           color="#1f77b4", alpha=0.8, edgecolor="black")

    ax.set_xlabel("Flight Scenario")
    ax.set_ylabel("Fitness (lower is better)")
    ax.set_title("Figure 3: Performance by Scenario", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------------------- #
# Figure 4: Fuzzy improvements
# ---------------------------------------------------------------------- #
def plot_fuzzy_improvements(
    scenario_names: List[str],
    improvements: List[float],
    output_path: str = "figures/fig4_fuzzy_improvements.png",
) -> str:
    """Generate Figure 4: fuzzy adaptive improvements per scenario."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(scenario_names))
    colors = ["#2ca02c" if imp > 0 else "#d62728" for imp in improvements]

    ax.bar(x, improvements, color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Improvement over Static Scaling (%)")
    ax.set_title("Figure 4: Fuzzy Adaptive Scaling Improvement",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=20, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    for i, imp in enumerate(improvements):
        va = "bottom" if imp >= 0 else "top"
        offset = 0.3 if imp >= 0 else -0.3
        ax.text(i, imp + offset, f"{imp:+.1f}%", ha="center", va=va, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path