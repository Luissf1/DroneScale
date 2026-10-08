"""
Fuzzy Adaptive Scaling Experiment
==================================

Compares static scaling vs fuzzy adaptive scaling across various
disturbance scenarios and quadrotor sizes.

Usage:
    python experiments/run_fuzzy_adaptive.py
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dynamics.quadrotor_model import QuadrotorDynamics, QuadrotorParameters
from src.control.fuzzy_adaptive import FuzzyAdaptiveScaler
from src.scaling.scaling_laws import ScalingLaw, generate_variants
from src.simulation.scenarios import get_disturbance_scenarios
from src.simulation.simulator import Simulator
from src.analysis.metrics import summarize_improvements
from src.analysis.visualization import plot_fuzzy_improvements


# Baseline gains from PSO (from scaling analysis / thesis)
BASELINE_GAINS = np.array([
    12.5, 1.25, 3.125,   # Altitude
    6.25, 0.075, 1.25,   # Roll
    6.25, 0.075, 1.25,   # Pitch
    5.0, 0.05, 1.0       # Yaw
])


def build_dynamics(variant: Dict[str, float]) -> QuadrotorDynamics:
    params = QuadrotorParameters(
        mass=variant["mass"],
        arm_length=variant["arm_length"],
        Ixx=variant["Ixx"],
        Iyy=variant["Iyy"],
        Izz=variant["Izz"],
    )
    return QuadrotorDynamics(params)


def evaluate_scenario(
    dynamics: QuadrotorDynamics,
    gains: np.ndarray,
    scenario,
    cost_weights=(0.3, 0.3, 0.2, 0.2),
) -> float:
    """Evaluate fitness under a scenario (payload & battery folded into params)."""
    # Payload changes mass
    payload_mass = dynamics.params.mass * (1.0 + scenario.payload_pct / 100.0)

    # Battery reduces available thrust authority — model as friction increase
    # and slight inertia growth (approximate)
    battery_penalty = (1.0 - scenario.battery) * 0.15

    params = QuadrotorParameters(
        mass=payload_mass,
        arm_length=dynamics.params.arm_length,
        Ixx=dynamics.params.Ixx,
        Iyy=dynamics.params.Iyy,
        Izz=dynamics.params.Izz,
        friction=dynamics.params.friction + battery_penalty,
    )
    dyn = QuadrotorDynamics(params)

    # Turbulence: add disturbance to reference
    reference = scenario.reference.copy()
    if scenario.turbulence > 0:
        np.random.seed(1234)
        reference = reference + np.random.normal(0, scenario.turbulence * 0.1, 4)

    sim = Simulator(dyn, cost_weights=cost_weights)
    result = sim.simulate(gains, reference)
    return float(result.fitness)


def main():
    parser = argparse.ArgumentParser(description="Fuzzy adaptive experiment")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"fuzzy_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("FUZZY ADAPTIVE SCALING EXPERIMENT")
    print("=" * 60)

    variants = generate_variants()
    scenarios = get_disturbance_scenarios()
    fuzzy = FuzzyAdaptiveScaler()

    results: List[Dict] = []

    for target_factor in [0.5, 2.0, 5.0]:
        v = variants[target_factor]
        dyn = build_dynamics(v)
        scaler = ScalingLaw(1.0, 0.25)

        scaled_gains = scaler.scale_gains(
            BASELINE_GAINS, v["mass"], v["arm_length"]
        )

        print(f"\n[{target_factor}×] mass={v['mass']:.3f} kg")

        for sc in scenarios:
            # Static scaling
            fit_static = evaluate_scenario(dyn, scaled_gains, sc)

            # Fuzzy adaptive
            fuzzy_gains = fuzzy.apply_to_gains(
                scaled_gains,
                payload_variation=sc.payload_pct,
                turbulence_level=sc.turbulence,
                battery_state=sc.battery,
                error_dynamics=min(sc.turbulence + sc.payload_pct / 100.0, 1.0),
            )
            fit_fuzzy = evaluate_scenario(dyn, fuzzy_gains, sc)

            improvement = (
                (fit_static - fit_fuzzy) / fit_static * 100.0
                if fit_static > 0
                else 0.0
            )

            results.append({
                "target_factor": float(target_factor),
                "scenario": sc.name,
                "static_fitness": fit_static,
                "fuzzy_fitness": fit_fuzzy,
                "improvement": improvement,
            })

            print(f"   {sc.name:18s}: "
                  f"static={fit_static:.4f}, "
                  f"fuzzy={fit_fuzzy:.4f}, "
                  f"impr={improvement:+.1f}%")

    improvements = [r["improvement"] for r in results]
    summary = summarize_improvements(improvements)

    print("\n" + "=" * 60)
    print(f"Mean improvement: {summary['mean']:+.2f}%")
    print(f"Max  improvement: {summary['max']:+.2f}%")
    print("=" * 60)

    # Save JSON
    json_path = os.path.join(out_dir, "fuzzy_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "results": results,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"✅ Saved: {json_path}")

    # Figure: aggregate by scenario name (across sizes)
    scenario_names = [s.name for s in scenarios]
    mean_improvements = []
    for name in scenario_names:
        vals = [r["improvement"] for r in results if r["scenario"] == name]
        mean_improvements.append(float(np.mean(vals)) if vals else 0.0)

    fig_path = plot_fuzzy_improvements(
        scenario_names, mean_improvements,
        output_path=os.path.join(out_dir, "fig4_fuzzy_improvements.png"),
    )
    print(f"   Figure: {fig_path}")


if __name__ == "__main__":
    main()