"""
Three-Way Comparison Experiment
================================

Compares:
    1. Re-optimized PSO gains (best possible)
    2. Static scaled gains
    3. Fuzzy adaptive scaled gains

Across target sizes and disturbance scenarios.

Usage:
    python experiments/run_comparison.py
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dynamics.quadrotor_model import QuadrotorDynamics, QuadrotorParameters
from src.control.fuzzy_adaptive import FuzzyAdaptiveScaler
from src.optimization.pso_optimizer import PSOOptimizer, PSOConfig, PSOBounds
from src.scaling.scaling_laws import ScalingLaw, generate_variants
from src.simulation.scenarios import get_default_scenarios
from src.simulation.simulator import Simulator
from src.analysis.visualization import plot_scenario_breakdown


BASELINE_GAINS = np.array([
    12.5, 1.25, 3.125,
    6.25, 0.075, 1.25,
    6.25, 0.075, 1.25,
    5.0, 0.05, 1.0,
])


def main():
    parser = argparse.ArgumentParser(description="Three-way comparison")
    parser.add_argument("--rapido", action="store_true",
                        help="Fast mode for smoke testing")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    if args.rapido:
        pso_config = PSOConfig(swarm_size=8, max_iterations=8, num_runs=1)
    else:
        pso_config = PSOConfig(swarm_size=20, max_iterations=30, num_runs=3)

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"comparison_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("THREE-WAY COMPARISON")
    print("=" * 60)

    variants = generate_variants()
    scenarios = get_default_scenarios()
    fuzzy = FuzzyAdaptiveScaler()
    bounds = PSOBounds()
    cost_weights = (0.3, 0.3, 0.2, 0.2)

    # We test on 5× as the target
    target_factor = 5.0
    v = variants[target_factor]
    params = QuadrotorParameters(
        mass=v["mass"], arm_length=v["arm_length"],
        Ixx=v["Ixx"], Iyy=v["Iyy"], Izz=v["Izz"],
    )
    dyn = QuadrotorDynamics(params)
    sim = Simulator(dyn, cost_weights=cost_weights)

    scaler = ScalingLaw(1.0, 0.25)
    scaled_gains = scaler.scale_gains(
        BASELINE_GAINS, v["mass"], v["arm_length"]
    )

    print(f"\nTarget: {target_factor}× quadrotor")
    print(f"Optimizing per-scenario gains (this may take a while)...")

    # Re-optimize per scenario
    scenario_fit_opt = []
    scenario_fit_static = []
    scenario_fit_fuzzy = []
    scenario_fit_unscaled = []

    for sc in scenarios:
        print(f"\n  Scenario {sc.code}: {sc.name}")

        # Re-optimize
        def fitness(g):
            return sim.simulate(g, sc.reference).fitness

        opt = PSOOptimizer(pso_config, bounds, fitness)
        best_pos, best_fit, _ = opt.optimize_multiple_runs(pso_config.num_runs)
        scenario_fit_opt.append(best_fit)
        print(f"    Re-optimized fitness: {best_fit:.4f}")

        # Static scaling
        fit_static = sim.simulate(scaled_gains, sc.reference).fitness
        scenario_fit_static.append(fit_static)
        print(f"    Static scaled fitness: {fit_static:.4f}")

        # Fuzzy adaptive
        # Use current error as a proxy (initial)
        fuzzy_gains = fuzzy.apply_to_gains(
            scaled_gains,
            payload_variation=0.0,
            turbulence_level=0.2,
            battery_state=1.0,
            error_dynamics=0.2,
        )
        fit_fuzzy = sim.simulate(fuzzy_gains, sc.reference).fitness
        scenario_fit_fuzzy.append(fit_fuzzy)
        print(f"    Fuzzy adaptive fitness: {fit_fuzzy:.4f}")

        # Unscaled
        fit_unscaled = sim.simulate(BASELINE_GAINS, sc.reference).fitness
        scenario_fit_unscaled.append(fit_unscaled)
        print(f"    Unscaled fitness: {fit_unscaled:.4f}")

    # Save results
    json_path = os.path.join(out_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "target_factor": target_factor,
            "scenarios": [s.code for s in scenarios],
            "scenario_names": [s.name for s in scenarios],
            "fit_optimal": scenario_fit_opt,
            "fit_static": scenario_fit_static,
            "fit_fuzzy": scenario_fit_fuzzy,
            "fit_unscaled": scenario_fit_unscaled,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\n✅ Saved: {json_path}")

    # Figure
    scenario_labels = [f"{s.code}" for s in scenarios]
    fig_path = plot_scenario_breakdown(
        scenario_labels,
        scenario_fit_unscaled,
        scenario_fit_static,
        scenario_fit_fuzzy,
        scenario_fit_opt,
        output_path=os.path.join(out_dir, "fig3_scenarios.png"),
    )
    print(f"   Figure: {fig_path}")

    print("\n" + "=" * 60)
    print("✅ COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()