"""
Scaling Analysis Experiment
============================

Runs PSO optimization across all quadrotor size variants and evaluates
the scaling laws for gain transfer.

Usage:
    python experiments/run_scaling_analysis.py
    python experiments/run_scaling_analysis.py --rapido
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dynamics.quadrotor_model import QuadrotorDynamics, QuadrotorParameters
from src.optimization.pso_optimizer import PSOOptimizer, PSOConfig, PSOBounds
from src.scaling.scaling_laws import ScalingLaw, generate_variants
from src.simulation.scenarios import get_default_scenarios
from src.simulation.simulator import Simulator
from src.analysis.metrics import compute_transfer_metrics, fit_power_law
from src.analysis.visualization import plot_kp_scaling, plot_performance_comparison


def build_fitness_function(
    dynamics: QuadrotorDynamics,
    simulator: Simulator,
    reference: np.ndarray,
):
    """Return a fitness function for PSO."""
    def fitness(gains: np.ndarray) -> float:
        result = simulator.simulate(gains, reference)
        return result.fitness
    return fitness


def optimize_variant(
    params: QuadrotorParameters,
    scenarios,
    pso_config: PSOConfig,
    bounds: PSOBounds,
    cost_weights,
    num_runs: int,
):
    """Optimize PID gains for a quadrotor variant across all scenarios."""
    dynamics = QuadrotorDynamics(params)
    simulator = Simulator(dynamics, cost_weights=cost_weights)

    per_scenario_gains = []
    for scenario in scenarios:
        fitness_fn = build_fitness_function(dynamics, simulator, scenario.reference)
        optimizer = PSOOptimizer(pso_config, bounds, fitness_fn)
        best_pos, best_fit, _ = optimizer.optimize_multiple_runs(num_runs)
        per_scenario_gains.append(best_pos)
        print(f"      {scenario.code}: best fitness = {best_fit:.4f}")

    return np.mean(per_scenario_gains, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Scaling analysis experiment")
    parser.add_argument("--rapido", action="store_true",
                        help="Fast smoke test with reduced budget")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    args = parser.parse_args()

    if args.rapido:
        pso_config = PSOConfig(swarm_size=10, max_iterations=10, num_runs=2)
        print("⚡ RAPID MODE: reduced PSO budget")
    else:
        pso_config = PSOConfig()

    bounds = PSOBounds()
    cost_weights = (0.3, 0.3, 0.2, 0.2)

    variants = generate_variants()
    scenarios = get_default_scenarios()

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"scaling_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)

    optimal_gains = {}
    for factor, v in variants.items():
        print(f"\n[{factor}×] mass={v['mass']:.3f} kg, l={v['arm_length']:.3f} m")
        params = QuadrotorParameters(
            mass=v["mass"],
            arm_length=v["arm_length"],
            Ixx=v["Ixx"],
            Iyy=v["Iyy"],
            Izz=v["Izz"],
        )
        gains = optimize_variant(
            params, scenarios, pso_config, bounds, cost_weights,
            num_runs=pso_config.num_runs,
        )
        optimal_gains[factor] = gains
        print(f"   → Mean gains: Kp_z={gains[0]:.3f}, Ki_z={gains[1]:.3f}, "
              f"Kd_z={gains[2]:.3f}")

    # ------------------------------------------------------------------ #
    # Transfer evaluation
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("TRANSFER EVALUATION")
    print("=" * 60)

    transfer_results = []
    for target_factor, v_t in variants.items():
        if target_factor == 1.0:
            continue

        params_t = QuadrotorParameters(
            mass=v_t["mass"], arm_length=v_t["arm_length"],
            Ixx=v_t["Ixx"], Iyy=v_t["Iyy"], Izz=v_t["Izz"],
        )
        dyn_t = QuadrotorDynamics(params_t)
        sim_t = Simulator(dyn_t, cost_weights=cost_weights)

        # Optimal (target-native)
        fit_opt = np.mean([
            sim_t.simulate(optimal_gains[target_factor], s.reference).fitness
            for s in scenarios
        ])

        # Source = baseline 1.0
        src_gains = optimal_gains[1.0]
        src = variants[1.0]
        scaler = ScalingLaw(src["mass"], src["arm_length"])

        scaled_gains = scaler.scale_gains(
            src_gains, v_t["mass"], v_t["arm_length"]
        )

        fit_unscaled = np.mean([
            sim_t.simulate(src_gains, s.reference).fitness for s in scenarios
        ])
        fit_scaled = np.mean([
            sim_t.simulate(scaled_gains, s.reference).fitness for s in scenarios
        ])

        metrics = compute_transfer_metrics(fit_opt, fit_scaled, fit_unscaled)
        metrics["target_factor"] = float(target_factor)
        transfer_results.append(metrics)

        print(f"\n[{target_factor}×] "
              f"opt={fit_opt:.4f}, scaled={fit_scaled:.4f}, "
              f"unscaled={fit_unscaled:.4f}, "
              f"improvement={metrics['improvement']:+.1f}%")

    # ------------------------------------------------------------------ #
    # Save JSON
    # ------------------------------------------------------------------ #
    serializable = {
        "optimal_gains": {str(k): v.tolist() for k, v in optimal_gains.items()},
        "transfer": transfer_results,
        "timestamp": datetime.now().isoformat(),
    }
    json_path = os.path.join(out_dir, "scaling_results.json")
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n✅ Saved: {json_path}")

    # ------------------------------------------------------------------ #
    # Figures
    # ------------------------------------------------------------------ #
    factors_sorted = sorted(variants.keys())
    masses = [variants[f]["mass"] for f in factors_sorted]
    kp_z = [optimal_gains[f][0] for f in factors_sorted]
    kd_z = [optimal_gains[f][2] for f in factors_sorted]
    kp_att = [
        np.mean([optimal_gains[f][3], optimal_gains[f][6], optimal_gains[f][9]])
        for f in factors_sorted
    ]

    fig1 = plot_kp_scaling(
        masses, kp_z, kp_att, kd_z,
        output_path=os.path.join(out_dir, "fig1_kp_scaling.png"),
    )
    print(f"   Figure 1: {fig1}")

    target_names = []
    fit_unscaled_list = []
    fit_scaled_list = []
    fit_opt_list = []
    for r in sorted(transfer_results, key=lambda x: x["target_factor"]):
        target_names.append(f"{r['target_factor']}×")
        fit_unscaled_list.append(r["fit_unscaled"])
        fit_scaled_list.append(r["fit_scaled"])
        fit_opt_list.append(r["fit_optimal"])

    if target_names:
        fig2 = plot_performance_comparison(
            target_names, fit_unscaled_list, fit_scaled_list, fit_opt_list,
            output_path=os.path.join(out_dir, "fig2_performance.png"),
        )
        print(f"   Figure 2: {fig2}")

    print("\n" + "=" * 60)
    print("✅ SCALING ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()