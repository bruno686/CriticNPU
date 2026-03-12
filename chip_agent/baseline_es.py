"""Baseline: Evolutionary Search (CMA-ES) for NPU architecture design.

Uses Covariance Matrix Adaptation Evolution Strategy to search the NPU design
space. Same constraints, scoring, and simulation budget as the LLM agent.
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

# ── Search space (same as BO baseline) ───────────────────────────────────────
PARAM_BOUNDS = {
    "num_sa":       (2, 16),
    "num_vu":       (2, 12),
    "sa_dim_idx":   (0, 2),
    "freq_GHz":     (0.5, 2.0),
    "vmem_size_MB": (32, 512),
}
SA_DIM_VALUES = [64, 128, 256]

PARAM_NAMES = list(PARAM_BOUNDS.keys())
BOUNDS_LOW = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES])
BOUNDS_HIGH = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES])
N_DIMS = len(PARAM_NAMES)


def _vec_to_config(x: np.ndarray, upper_bounds: dict[str, float]) -> dict[str, Any]:
    """Convert continuous vector to chip config dict."""
    x_clipped = np.clip(x, BOUNDS_LOW, BOUNDS_HIGH)
    num_sa = int(round(x_clipped[0]))
    num_vu = int(round(x_clipped[1]))
    sa_dim_idx = int(round(np.clip(x_clipped[2], 0, 2)))
    freq_GHz = round(float(x_clipped[3]), 2)
    vmem_size_MB = int(round(x_clipped[4]))

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


def _fast_power_check(
    base_config: dict[str, Any],
    overrides: dict[str, Any],
    upper_bounds: dict[str, float],
) -> float:
    merged = dict(base_config)
    merged = apply_chip_overrides(merged, overrides, upper_bounds=upper_bounds)
    chip_cfg = ChipConfig.model_validate(merged)
    return chip_cfg.total_power_peak_W


# ── CMA-ES implementation ───────────────────────────────────────────────────

class SimpleCMAES:
    """Minimal CMA-ES for bounded optimization."""

    def __init__(
        self,
        x0: np.ndarray,
        sigma0: float = 0.3,
        pop_size: int = 10,
    ):
        self.n = len(x0)
        self.mean = x0.copy().astype(float)
        # Normalize sigma to [0,1] space
        self.sigma = sigma0
        self.pop_size = pop_size
        self.mu = pop_size // 2  # number of parents

        # Weights for recombination
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()

        # Adaptation parameters
        self.c_sigma = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n + 2) ** 2 + self.mu_eff))

        # State
        self.p_sigma = np.zeros(self.n)
        self.p_c = np.zeros(self.n)
        self.C = np.eye(self.n)
        self.chi_n = np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))
        self.generation = 0

    def ask(self) -> list[np.ndarray]:
        """Generate pop_size candidate solutions."""
        try:
            sqrt_C = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            sqrt_C = np.eye(self.n)

        solutions = []
        for _ in range(self.pop_size):
            z = np.random.randn(self.n)
            x = self.mean + self.sigma * (sqrt_C @ z)
            # Clip to bounds
            x = np.clip(x, BOUNDS_LOW, BOUNDS_HIGH)
            solutions.append(x)
        return solutions

    def tell(self, solutions: list[np.ndarray], scores: list[float]) -> None:
        """Update distribution based on evaluated solutions."""
        # Sort by score (lower is better)
        indices = np.argsort(scores)
        selected = [solutions[i] for i in indices[:self.mu]]

        old_mean = self.mean.copy()
        # Recombine
        self.mean = sum(w * x for w, x in zip(self.weights, selected))

        # Update evolution path for sigma
        try:
            C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.C))
        except np.linalg.LinAlgError:
            C_inv_sqrt = np.eye(self.n)

        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
            np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * \
            C_inv_sqrt @ (self.mean - old_mean) / self.sigma

        # Update evolution path for C
        h_sigma = 1.0 if np.linalg.norm(self.p_sigma) / \
            np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.generation + 1))) < \
            (1.4 + 2 / (self.n + 1)) * self.chi_n else 0.0

        self.p_c = (1 - self.c_c) * self.p_c + \
            h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * \
            (self.mean - old_mean) / self.sigma

        # Update covariance matrix
        artmp = np.array([(x - old_mean) / self.sigma for x in selected])
        self.C = (1 - self.c1 - self.c_mu) * self.C + \
            self.c1 * np.outer(self.p_c, self.p_c) + \
            self.c_mu * sum(w * np.outer(a, a) for w, a in zip(self.weights, artmp))

        # Ensure symmetry
        self.C = (self.C + self.C.T) / 2
        # Add small diagonal for numerical stability
        self.C += 1e-10 * np.eye(self.n)

        # Update sigma
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) *
                             (np.linalg.norm(self.p_sigma) / self.chi_n - 1))
        # Clamp sigma
        self.sigma = max(0.01, min(self.sigma, 2.0))

        self.generation += 1


# ── Main ES loop ─────────────────────────────────────────────────────────────

def run_es_search(
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
    max_simulations: int = 65,
    pop_size: int = 10,
) -> dict[str, Any]:
    if upper_bounds is None:
        upper_bounds = dict(DEFAULT_UPPER_BOUNDS)

    base_config = load_base_configs(model_name, chip_version)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "prefill" if tpot_weight == 0 else ("decode" if ttft_weight == 0 else "combined")
    run_dir = os.path.join(output_dir, f"{model_name}_es_{label}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    focus_str = "PREFILL-only" if tpot_weight == 0 else ("DECODE-only" if ttft_weight == 0 else "combined")

    print(f"\n{'='*60}")
    print(f"Evolutionary Search (CMA-ES) — {model_name} on {num_chips} chips (TP={tp}, PP={pp})")
    print(f"Optimization: {focus_str}")
    print(f"Power budget: {power_budget_W}W")
    print(f"Population: {pop_size}, Max simulations: {max_simulations}")
    print(f"{'='*60}\n")

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

    def _evaluate(config: dict[str, Any], iteration_label: str) -> float:
        """Evaluate a config. Returns score (lower=better), or penalty for invalid."""
        nonlocal sim_count, baseline_ttft, baseline_tpot, best_result, best_score

        sig = _config_sig(config)
        if sig in seen_configs:
            return 10.0  # penalty for duplicate
        seen_configs.add(sig)

        # Power pre-check
        try:
            power = _fast_power_check(base_config, config, upper_bounds)
            if power > power_budget_W:
                return 10.0  # penalty
        except Exception:
            return 10.0

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
            return 10.0

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

            return score

        return 10.0  # penalty for invalid

    # ── Initialize CMA-ES from a known-feasible point ──────────────────────
    # Start from a low-power config to ensure initial samples are feasible
    x0 = np.array([4.0, 4.0, 1.0, 0.8, 200.0])  # num_sa=4, num_vu=4, sa_dim=128, freq=0.8, vmem=200
    cma = SimpleCMAES(x0=x0, sigma0=0.5, pop_size=pop_size)

    generation = 0
    max_generations = max_simulations * 10  # cap to prevent infinite loop
    while sim_count < max_simulations and generation < max_generations:
        generation += 1
        candidates = cma.ask()

        solutions = []
        scores = []
        for i, x in enumerate(candidates):
            if sim_count >= max_simulations:
                break
            config = _vec_to_config(x, upper_bounds)
            score = _evaluate(config, f"g{generation}-{i+1}")
            solutions.append(x)
            scores.append(score)

        if len(solutions) >= 2:
            cma.tell(solutions, scores)

        print(f"  Generation {generation} complete: {sim_count}/{max_simulations} sims, best={best_score:.4f}")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVOLUTIONARY SEARCH (CMA-ES) COMPLETE")
    print(f"{'='*60}")
    print(f"Total simulations: {sim_count}")
    print(f"Generations: {generation}")

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
        "method": "evolutionary_search_cmaes",
        "model_name": model_name,
        "chip_version": chip_version,
        "upper_bounds": upper_bounds,
        "power_budget_W": power_budget_W,
        "ttft_weight": ttft_weight,
        "tpot_weight": tpot_weight,
        "max_simulations": max_simulations,
        "pop_size": pop_size,
        "total_simulations": sim_count,
        "generations": generation,
        "parallelism": {"num_chips": num_chips, "tp": tp, "pp": pp, "dp": dp, "ep": ep, "bs": bs},
        "best_result": best_result,
        "history": history,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {run_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evolutionary Search (CMA-ES) baseline for NPU architecture design")
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
    parser.add_argument("--disaggregated", action="store_true")
    parser.add_argument("--max-sims", type=int, default=65,
                        help="Total simulation budget (default: 65)")
    parser.add_argument("--pop-size", type=int, default=10,
                        help="CMA-ES population size (default: 10)")

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
        max_simulations=args.max_sims,
        pop_size=args.pop_size,
    )

    if args.disaggregated:
        print("\n>>> ES: PREFILL optimization <<<")
        prefill = run_es_search(**common, ttft_weight=1.0, tpot_weight=0.0)
        print("\n>>> ES: DECODE optimization <<<")
        decode = run_es_search(**common, ttft_weight=0.0, tpot_weight=1.0)

        print(f"\n{'='*60}")
        print("  ES DISAGGREGATED SUMMARY")
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
        run_es_search(**common, ttft_weight=args.ttft_weight, tpot_weight=args.tpot_weight)


if __name__ == "__main__":
    main()
