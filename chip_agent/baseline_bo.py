"""Baseline: Bayesian Optimization for NPU architecture design.

Uses Gaussian Process surrogate + Expected Improvement acquisition function
to search the NPU design space. Same constraints, scoring, and simulation
budget as the LLM agent for fair comparison.
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from typing import Any

import numpy as np

from neusim_runner import apply_chip_overrides, load_base_configs, run_simulation
from neusim.configs.chips.ChipConfig import ChipConfig

warnings.filterwarnings("ignore")

DEFAULT_UPPER_BOUNDS = {
    "hbm_bw_GBps": 600.0,
    "hbm_size_GB": 16,
    "ici_bw_GBps": 125.0,
    "dcn_bw_GBps": 12.5,
}

# ── Search space definition ──────────────────────────────────────────────────
# Continuous parameters with (min, max) bounds
PARAM_BOUNDS = {
    "num_sa":       (2, 16),
    "num_vu":       (2, 12),
    "sa_dim_idx":   (0, 2),      # index into [64, 128, 256]
    "freq_GHz":     (0.5, 2.0),
    "vmem_size_MB": (32, 512),
}
SA_DIM_VALUES = [64, 128, 256]

PARAM_NAMES = list(PARAM_BOUNDS.keys())
BOUNDS_LOW = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES])
BOUNDS_HIGH = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES])
N_DIMS = len(PARAM_NAMES)


def _vec_to_config(x: np.ndarray, upper_bounds: dict[str, float]) -> dict[str, Any]:
    """Convert a continuous vector to a chip config dict."""
    num_sa = int(round(x[0]))
    num_vu = int(round(x[1]))
    sa_dim_idx = int(round(np.clip(x[2], 0, 2)))
    freq_GHz = round(float(x[3]), 2)
    vmem_size_MB = int(round(x[4]))

    return {
        "num_sa": int(np.clip(num_sa, 2, 16)),
        "num_vu": int(np.clip(num_vu, 2, 12)),
        "num_vu_ports": int(np.clip(num_vu, 2, 12)),
        "sa_dim": SA_DIM_VALUES[sa_dim_idx],
        "freq_GHz": max(0.5, freq_GHz),
        "vmem_size_MB": int(np.clip(vmem_size_MB, 32, 512)),
        "hbm_bw_GBps": upper_bounds["hbm_bw_GBps"],
        "hbm_size_GB": upper_bounds["hbm_size_GB"],
        "ici_bw_GBps": upper_bounds["ici_bw_GBps"],
        "dcn_bw_GBps": upper_bounds["dcn_bw_GBps"],
    }


def _config_to_vec(config: dict[str, Any]) -> np.ndarray:
    """Convert a chip config dict to a continuous vector."""
    sa_dim = config.get("sa_dim", 128)
    sa_dim_idx = SA_DIM_VALUES.index(sa_dim) if sa_dim in SA_DIM_VALUES else 1
    return np.array([
        config["num_sa"],
        config["num_vu"],
        sa_dim_idx,
        config["freq_GHz"],
        config["vmem_size_MB"],
    ], dtype=float)


def _fast_power_check(
    base_config: dict[str, Any],
    overrides: dict[str, Any],
    upper_bounds: dict[str, float],
) -> float:
    """Compute peak power without running simulation."""
    merged = dict(base_config)
    merged = apply_chip_overrides(merged, overrides, upper_bounds=upper_bounds)
    chip_cfg = ChipConfig.model_validate(merged)
    return chip_cfg.total_power_peak_W


# ── Gaussian Process with Expected Improvement ──────────────────────────────

class SimpleGP:
    """Minimal GP implementation using sklearn for the BO baseline."""

    def __init__(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel

        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(N_DIMS))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
        )
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    def add_observation(self, x: np.ndarray, score: float) -> None:
        self.X.append(x.copy())
        self.y.append(score)

    def fit(self) -> None:
        if len(self.X) < 2:
            return
        X = np.array(self.X)
        y = np.array(self.y)
        # Normalize X to [0, 1] for better GP performance
        self._X_min = BOUNDS_LOW
        self._X_range = BOUNDS_HIGH - BOUNDS_LOW
        self._X_range[self._X_range == 0] = 1.0
        X_norm = (X - self._X_min) / self._X_range
        self.gp.fit(X_norm, y)

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        """Return (mean, std) prediction."""
        if len(self.X) < 2:
            return 0.0, 1.0
        x_norm = ((x - self._X_min) / self._X_range).reshape(1, -1)
        mean, std = self.gp.predict(x_norm, return_std=True)
        return float(mean[0]), float(std[0])

    def expected_improvement(self, x: np.ndarray, best_score: float, xi: float = 0.01) -> float:
        """Compute Expected Improvement at point x."""
        from scipy.stats import norm

        mean, std = self.predict(x)
        if std < 1e-8:
            return 0.0
        z = (best_score - mean - xi) / std
        ei = (best_score - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
        return float(ei)

    def suggest_next(
        self,
        best_score: float,
        base_config: dict[str, Any],
        upper_bounds: dict[str, float],
        power_budget_W: float,
        n_candidates: int = 5000,
    ) -> np.ndarray:
        """Suggest next point by maximizing EI over random candidates."""
        # Generate random candidates
        candidates = np.random.uniform(BOUNDS_LOW, BOUNDS_HIGH, size=(n_candidates, N_DIMS))

        # Filter by power constraint
        valid_candidates = []
        for c in candidates:
            config = _vec_to_config(c, upper_bounds)
            try:
                power = _fast_power_check(base_config, config, upper_bounds)
                if power <= power_budget_W:
                    valid_candidates.append(c)
            except Exception:
                continue

        if not valid_candidates:
            # If no valid candidates, just return a random one
            return candidates[np.random.randint(len(candidates))]

        valid_candidates = np.array(valid_candidates)

        # Compute EI for all valid candidates
        eis = np.array([self.expected_improvement(c, best_score) for c in valid_candidates])

        return valid_candidates[np.argmax(eis)]


# ── Main BO loop ─────────────────────────────────────────────────────────────

def run_bo_search(
    model_name: str,
    chip_version: str,
    num_chips: int = 8,
    tp: int = 4,
    pp: int = 2,
    dp: int = 1,
    ep: int = 1,
    bs: int = 16,
    pp_dcn: int = 1,
    dp_dcn: int = 1,
    power_budget_W: float = 300.0,
    output_dir: str = "design_results",
    upper_bounds: dict[str, float] | None = None,
    ttft_weight: float = 0.5,
    tpot_weight: float = 0.5,
    n_init: int = 15,
    n_iterations: int = 50,
) -> dict[str, Any]:
    if upper_bounds is None:
        upper_bounds = dict(DEFAULT_UPPER_BOUNDS)

    base_config = load_base_configs(model_name, chip_version)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "prefill" if tpot_weight == 0 else ("decode" if ttft_weight == 0 else "combined")
    run_dir = os.path.join(output_dir, f"{model_name}_bo_{label}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    focus_str = "PREFILL-only" if tpot_weight == 0 else ("DECODE-only" if ttft_weight == 0 else "combined")

    print(f"\n{'='*60}")
    print(f"Bayesian Optimization — {model_name} on {num_chips} chips (TP={tp}, PP={pp})")
    print(f"Optimization: {focus_str}")
    print(f"Power budget: {power_budget_W}W")
    print(f"Init samples: {n_init}, BO iterations: {n_iterations}")
    print(f"Total budget: {n_init + n_iterations} simulations")
    print(f"{'='*60}\n")

    gp = SimpleGP()
    history: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_score = float("inf")
    baseline_ttft: float | None = None
    baseline_tpot: float | None = None
    sim_count = 0
    seen_configs: set[str] = set()

    def _config_sig(config: dict) -> str:
        key_params = ["num_sa", "num_vu", "sa_dim", "freq_GHz", "vmem_size_MB"]
        return json.dumps({k: config.get(k) for k in key_params}, sort_keys=True)

    def _evaluate(config: dict[str, Any], iteration_label: str) -> dict[str, Any] | None:
        nonlocal sim_count, baseline_ttft, baseline_tpot, best_result, best_score

        sig = _config_sig(config)
        if sig in seen_configs:
            return None
        seen_configs.add(sig)

        # Power pre-check
        try:
            power = _fast_power_check(base_config, config, upper_bounds)
            if power > power_budget_W:
                print(f"  [{iteration_label}] POWER FAIL: {power:.0f}W > {power_budget_W}W — skipped")
                return None
        except Exception:
            return None

        sim_count += 1
        try:
            results = run_simulation(
                model_name=model_name,
                chip_version=chip_version,
                chip_overrides=config,
                num_chips=num_chips,
                tp=tp, pp=pp, dp=dp, ep=ep, bs=bs,
                pp_dcn=pp_dcn, dp_dcn=dp_dcn,
                output_dir=os.path.join(run_dir, f"sim_{sim_count:03d}"),
                upper_bounds=upper_bounds,
            )
        except Exception as e:
            print(f"  [{iteration_label}] ERROR: {e}")
            return None

        is_valid = not results["out_of_memory"] and results["total_power_peak_W"] <= power_budget_W
        status = "VALID" if is_valid else ("OOM" if results["out_of_memory"] else f"PWR={results['total_power_peak_W']:.0f}W")

        key_cfg = {k: config[k] for k in ["num_sa", "sa_dim", "freq_GHz", "vmem_size_MB"] if k in config}
        print(
            f"  [{iteration_label}] {json.dumps(key_cfg)}  →  "
            f"TTFT={results['TTFT_sec']:.4f}s  TPOT={results['TPOT_ms_request']:.4f}ms  "
            f"Pwr={results['total_power_peak_W']:.0f}W  {status}"
        )

        entry = {
            "iteration": iteration_label,
            "chip_overrides": config,
            "results": {
                "TTFT_sec": results["TTFT_sec"],
                "TPOT_ms_request": results["TPOT_ms_request"],
                "throughput_tokens_per_sec_request": results["throughput_tokens_per_sec_request"],
                "avg_power_W_prefill": results["avg_power_W_prefill"],
                "avg_power_W_decode": results["avg_power_W_decode"],
                "total_power_peak_W": results["total_power_peak_W"],
                "max_power_W_prefill": results.get("max_power_W_prefill", 0),
                "max_power_W_decode": results.get("max_power_W_decode", 0),
                "mem_footprint_GB_prefill": results["mem_footprint_GB_prefill"],
                "mem_footprint_GB_decode": results["mem_footprint_GB_decode"],
                "out_of_memory": results["out_of_memory"],
            },
        }
        history.append(entry)

        if is_valid:
            if baseline_ttft is None:
                baseline_ttft = results["TTFT_sec"]
                baseline_tpot = results["TPOT_ms_request"]

            score = 0.0
            if ttft_weight > 0 and baseline_ttft and baseline_ttft > 0:
                score += ttft_weight * results["TTFT_sec"] / baseline_ttft
            if tpot_weight > 0 and baseline_tpot and baseline_tpot > 0:
                score += tpot_weight * results["TPOT_ms_request"] / baseline_tpot
            if score == 0:
                score = float("inf")

            if score < best_score:
                best_score = score
                best_result = entry

            return {"score": score, "vec": _config_to_vec(config)}

        # Invalid configs get a penalty score
        return {"score": 10.0, "vec": _config_to_vec(config)}

    # ── Phase 1: Latin Hypercube initial sampling ────────────────────────────
    print("--- Phase 1: Initial sampling (Latin Hypercube) ---")

    # Generate LHS samples
    from scipy.stats.qmc import LatinHypercube

    sampler = LatinHypercube(d=N_DIMS, seed=42)
    lhs_samples = sampler.random(n=n_init)
    init_points = BOUNDS_LOW + lhs_samples * (BOUNDS_HIGH - BOUNDS_LOW)

    for i, x in enumerate(init_points):
        config = _vec_to_config(x, upper_bounds)
        result = _evaluate(config, f"init-{i+1}")
        if result:
            gp.add_observation(result["vec"], result["score"])

    print(f"\n  Init complete: {sim_count} simulations, best score = {best_score:.4f}")

    # ── Phase 2: BO iterations ───────────────────────────────────────────────
    print(f"\n--- Phase 2: Bayesian Optimization ({n_iterations} iterations) ---")

    for i in range(n_iterations):
        if len(gp.X) < 2:
            # Not enough data yet, sample randomly
            x_next = np.random.uniform(BOUNDS_LOW, BOUNDS_HIGH)
        else:
            gp.fit()
            x_next = gp.suggest_next(
                best_score=best_score,
                base_config=base_config,
                upper_bounds=upper_bounds,
                power_budget_W=power_budget_W,
            )

        config = _vec_to_config(x_next, upper_bounds)
        result = _evaluate(config, f"bo-{i+1}")
        if result:
            gp.add_observation(result["vec"], result["score"])

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BAYESIAN OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total simulations: {sim_count}")

    # Print iteration table
    print(f"\n{'Iter':>6}  {'TTFT(s)':>8}  {'TPOT(ms)':>9}  {'PeakPwr(W)':>11}  {'OOM':>4}  {'Valid':>5}")
    print("-" * 55)
    for h in history:
        r = h["results"]
        is_v = not r["out_of_memory"] and r["total_power_peak_W"] <= power_budget_W
        print(
            f"{str(h['iteration']):>6}  "
            f"{r['TTFT_sec']:>8.4f}  "
            f"{r['TPOT_ms_request']:>9.4f}  "
            f"{r['total_power_peak_W']:>11.1f}  "
            f"{'Y' if r['out_of_memory'] else 'N':>4}  "
            f"{'Y' if is_v else 'N':>5}"
        )

    if best_result:
        print(f"\nBest configuration ({best_result['iteration']}):")
        cfg = best_result["chip_overrides"]
        key_params = {k: cfg[k] for k in ["num_sa", "num_vu", "sa_dim", "freq_GHz", "vmem_size_MB"] if k in cfg}
        print(f"  Config: {json.dumps(key_params, indent=2)}")
        r = best_result["results"]
        print(f"  TTFT:       {r['TTFT_sec']:.4f} sec")
        print(f"  TPOT:       {r['TPOT_ms_request']:.4f} ms/request")
        print(f"  Peak Power: {r['total_power_peak_W']:.1f} W")
    else:
        print("\nNo valid configuration found within constraints.")

    summary = {
        "method": "bayesian_optimization",
        "model_name": model_name,
        "chip_version": chip_version,
        "upper_bounds": upper_bounds,
        "power_budget_W": power_budget_W,
        "ttft_weight": ttft_weight,
        "tpot_weight": tpot_weight,
        "n_init": n_init,
        "n_iterations": n_iterations,
        "total_simulations": sim_count,
        "parallelism": {"num_chips": num_chips, "tp": tp, "pp": pp, "dp": dp, "ep": ep, "bs": bs},
        "best_result": best_result,
        "history": history,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {run_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization baseline for NPU architecture design")
    parser.add_argument("model", type=str, help="Model name")
    parser.add_argument("--chip-version", type=str, default="4")
    parser.add_argument("--num-chips", type=int, default=8)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--pp-dcn", type=int, default=1)
    parser.add_argument("--dp-dcn", type=int, default=1)
    parser.add_argument("--power-budget", type=float, default=300.0)
    parser.add_argument("--output-dir", type=str, default="design_results")
    parser.add_argument("--ttft-weight", type=float, default=0.5)
    parser.add_argument("--tpot-weight", type=float, default=0.5)
    parser.add_argument("--max-hbm-bw", type=float, default=600.0)
    parser.add_argument("--max-hbm-size", type=float, default=16)
    parser.add_argument("--max-ici-bw", type=float, default=125.0)
    parser.add_argument("--max-dcn-bw", type=float, default=12.5)
    parser.add_argument("--disaggregated", action="store_true",
                        help="Run BO separately for prefill and decode")
    parser.add_argument("--n-init", type=int, default=15,
                        help="Number of initial LHS samples (default: 15)")
    parser.add_argument("--n-iterations", type=int, default=50,
                        help="Number of BO iterations after init (default: 50)")

    args = parser.parse_args()

    upper_bounds = {
        "hbm_bw_GBps": args.max_hbm_bw,
        "hbm_size_GB": args.max_hbm_size,
        "ici_bw_GBps": args.max_ici_bw,
        "dcn_bw_GBps": args.max_dcn_bw,
    }

    common = dict(
        model_name=args.model,
        chip_version=args.chip_version,
        num_chips=args.num_chips,
        tp=args.tp, pp=args.pp, dp=args.dp, ep=args.ep, bs=args.bs,
        pp_dcn=args.pp_dcn, dp_dcn=args.dp_dcn,
        power_budget_W=args.power_budget,
        output_dir=args.output_dir,
        upper_bounds=upper_bounds,
        n_init=args.n_init,
        n_iterations=args.n_iterations,
    )

    if args.disaggregated:
        print("\n>>> BO: PREFILL optimization <<<")
        prefill = run_bo_search(**common, ttft_weight=1.0, tpot_weight=0.0)
        print("\n>>> BO: DECODE optimization <<<")
        decode = run_bo_search(**common, ttft_weight=0.0, tpot_weight=1.0)

        print(f"\n{'='*60}")
        print("  BO DISAGGREGATED SUMMARY")
        print(f"{'='*60}")
        pb = prefill.get("best_result")
        db = decode.get("best_result")
        if pb:
            r = pb["results"]
            print(f"PREFILL ({pb['iteration']}): TTFT={r['TTFT_sec']:.4f}s, Pwr={r['total_power_peak_W']:.0f}W")
            print(f"  Config: {json.dumps({k: pb['chip_overrides'][k] for k in ['num_sa','sa_dim','freq_GHz','vmem_size_MB'] if k in pb['chip_overrides']})}")
        if db:
            r = db["results"]
            print(f"DECODE ({db['iteration']}): TPOT={r['TPOT_ms_request']:.4f}ms, Pwr={r['total_power_peak_W']:.0f}W")
            print(f"  Config: {json.dumps({k: db['chip_overrides'][k] for k in ['num_sa','sa_dim','freq_GHz','vmem_size_MB'] if k in db['chip_overrides']})}")

        if pb and db:
            print(f"\nCombined: TTFT={pb['results']['TTFT_sec']:.4f}s, TPOT={db['results']['TPOT_ms_request']:.4f}ms")
    else:
        run_bo_search(**common, ttft_weight=args.ttft_weight, tpot_weight=args.tpot_weight)


if __name__ == "__main__":
    main()
