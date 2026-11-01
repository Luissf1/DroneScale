"""
Regenerate Figure 1 excluding the 1.0x outlier from the fit.

Auto-discovers the newest full-budget scaling run under results/.
"""

import sys, os, glob, json
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from src.analysis.visualization import plot_kp_scaling
from src.scaling.scaling_laws import generate_variants


def find_latest_scaling_json(prefer_budget="full"):
    """Return path to the newest scaling_results.json, preferring the full-budget run."""
    candidates = glob.glob("results/scaling_*/scaling_results.json")
    if not candidates:
        raise FileNotFoundError("No scaling_results.json found under results/")

    loaded = []
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        ts = data.get("timestamp", "")
        budget = data.get("budget", "unknown")
        if ts:
            loaded.append((budget, ts, path))

    if not loaded:
        raise FileNotFoundError("No valid scaling_results.json under results/")

    # Prefer the requested budget; fall back to newest of any budget.
    preferred = [x for x in loaded if x[0] == prefer_budget]
    pool = preferred if preferred else loaded
    pool.sort(key=lambda x: x[1], reverse=True)
    budget, ts, path = pool[0]

    print(f"[figure] Using {path}")
    print(f"[figure]   budget={budget}, timestamp={ts}")
    return path


def main():
    json_path = find_latest_scaling_json(prefer_budget="full")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    variants = generate_variants()
    factors_sorted = sorted(variants.keys())
    masses = [variants[f]["mass"] for f in factors_sorted]
    gains = data["optimal_gains"]

    kp_z = [gains[str(f)][0] for f in factors_sorted]
    kd_z = [gains[str(f)][2] for f in factors_sorted]
    kp_att = [
        np.mean([gains[str(f)][3], gains[str(f)][6], gains[str(f)][9]])
        for f in factors_sorted
    ]

    out_path = "figures/fig1_kp_scaling_final.png"
    os.makedirs("figures", exist_ok=True)

    plot_kp_scaling(
        masses, kp_z, kp_att, kd_z,
        output_path=out_path,
        subtitle="Full 30-run PSO budget (1.0× outlier excluded from fit)",
    )
    print(f"[figure] Saved {out_path}")

    # Also print the exponents so you can copy them into the LaTeX table.
    from src.analysis.metrics import fit_power_law
    masses_arr = np.array(masses)
    mask = ~np.isclose(masses_arr, 1.0)

    fz = fit_power_law(masses_arr[mask], np.array(kp_z)[mask])
    fa = fit_power_law(masses_arr[mask], np.array(kp_att)[mask])
    print()
    print(f"Kp (altitude) exponent   = {fz['b']:.3f}   (R² = {fz['r_squared']:.3f})")
    print(f"Kp (attitude) exponent   = {fa['b']:.3f}   (R² = {fa['r_squared']:.3f})")
    print()
    print("Copy these into Table 3 of the LaTeX file.")


if __name__ == "__main__":
    main()