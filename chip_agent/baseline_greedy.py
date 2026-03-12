"""Baseline: Greedy Local Search for NPU architecture design.

Starts from the best seed config, then iteratively perturbs one parameter
at a time, keeping changes that improve the score. Classic hill-climbing
with restarts.
"""

import argparse
import json
import os
import random
import warnings
from datetime import datetime
from typing import Any

from neusim_runner import apply_chip_overrides, load_base_configs, run_simulation
from neusim.configs.chips.ChipConfig import ChipConfig

warnings.filterwarnings("ignore")

DEFAULT_UPPER_BOUNDS = {
    "hbm_bw_GBps": 600.0,
    "hbm_size_GB": 16,
    "ici_bw_GBps": 125.0,
    "dcn_bw_GBps": 12.5,
}

# Same seeds as the agent
SEED_CONFIGS = [
    {"name": "min-compute",    "num_sa": 2, "num_vu": 2, "num_vu_ports": 2, "sa_dim": 64,  "freq_GHz": 0.8, "vmem_size_MB": 128},
    {"name": "low-compute",    "num_sa": 4, "num_vu": 4, "num_vu_ports": 4, "sa_dim": 64,  "freq_GHz": 0.8, "vmem_size_MB": 128},
    {"name": "balanced",       "num_sa": 8, "num_vu": 6, "num_vu_ports": 6, "sa_dim": 128, "freq_GHz": 1.0, "vmem_size_MB": 256},
    {"name": "high-compute",   "num_sa": 12,"num_vu": 8, "num_vu_ports": 8, "sa_dim": 256, "freq_GHz": 1.2, "vmem_size_MB": 384},
    {"name": "max-compute",    "num_sa": 16,"num_vu": 10,"num_vu_ports":10, "sa_dim": 256, "freq_GHz": 1.5, "vmem_size_MB": 384},
    {"name": "low-freq-big",   "num_sa": 8, "num_vu": 6, "num_vu_ports": 6, "sa_dim": 256, "freq_GHz": 0.8, "vmem_size_MB": 384},
    {"name": "hi-freq-small",  "num_sa": 4, "num_vu": 4, "num_vu_ports": 4, "sa_dim": 128, "freq_GHz": 1.4, "vmem_size_MB": 128},
    {"name": "vlow-freq-tiny", "num_sa": 2, "num_vu": 2, "num_vu_ports": 2, "sa_dim": 64,  "freq_GHz": 0.5, "vmem_size_MB": 128},
    {"name": "med-freq-mid",   "num_sa": 6, "num_vu": 4, "num_vu_ports": 4, "sa_dim": 128, "freq_GHz": 0.65,"vmem_size_MB": 256},
]

# Perturbation steps for each parameter
PERTURBATIONS = {
    "num_sa":       [-2, -1, 1, 2],
    "num_vu":       [-2, -1, 1, 2],
    "sa_dim":       [-1, 1],       # index shift in [64, 128, 256]
    "freq_GHz":     [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2],
    "vmem_size_MB": [-128, -64, 64, 128],
}

SA_DIM_VALUES = [64, 128, 256]

PARAM_BOUNDS = {
    "num_sa":       (2, 16),
    "num_vu":       (2, 12),
    "freq_GHz":     (0.5, 2.0),
    "vmem_size_MB": (32, 512),
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


def _perturb_config(config: dict[str, Any]) -> dict[str, Any]:
    """Generate a neighbor by perturbing one random parameter."""
    new_config = dict(config)
    param = random.choice(list(PERTURBATIONS.keys()))
    deltas = PERTURBATIONS[param]
    delta = random.choice(deltas)

    if param == "sa_dim":
        cur_idx = SA_DIM_VALUES.index(config["sa_dim"])
        new_idx = max(0, min(2, cur_idx + delta))
        new_config["sa_dim"] = SA_DIM_VALUES[new_idx]
    elif param == "num_vu":
        new_val = config["num_vu"] + delta
        lo, hi = PARAM_BOUNDS["num_vu"]
        new_config["num_vu"] = max(lo, min(hi, new_val))
        new_config["num_vu_ports"] = new_config["num_vu"]
    else:
        new_val = config[param] + delta
        lo, hi = PARAM_BOUNDS[param]
        if isinstance(config[param], int):
            new_config[param] = int(max(lo, min(hi, new_val)))
        else:
            new_config[param] = max(lo, min(hi, round(new_val, 2)))

    return new_config


def run_greedy_search(
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
) -> dict[str, Any]:
    if upper_bounds is None:
        upper_bounds = dict(DEFAULT_UPPER_BOUNDS)

    base_config = load_base_configs(model_name, chip_version)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "prefill" if tpot_weight == 0 else ("decode" if ttft_weight == 0 else "combined")
    run_dir = os.path.join(output_dir, f"{model_name}_greedy_{label}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    focus_str = "PREFILL-only" if tpot_weight == 0 else ("DECODE-only" if ttft_weight == 0 else "combined")

    print(f"\n{'='*60}")
    print(f"Greedy Local Search — {model_name} on {num_chips} chips (TP={tp}, PP={pp})")
    print(f"Optimization: {focus_str}")
    print(f"Power budget: {power_budget_W}W")
    print(f"Max simulations: {max_simulations}")
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

    def _evaluate(config: dict[str, Any], iteration_label: str) -> float | None:
        """Returns score if valid, None if should skip."""
        nonlocal sim_count, baseline_ttft, baseline_tpot, best_result, best_score

        sig = _config_sig(config)
        if sig in seen_configs:
            return None
        seen_configs.add(sig)

        # Power pre-check
        try:
            power = _fast_power_check(base_config, config, upper_bounds)
            if power > power_budget_W:
                print(f"  [{iteration_label}] POWER: {power:.0f}W > {power_budget_W}W — skipped")
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

        if not is_valid:
            return None

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

    # ── Phase 1: Evaluate seeds ──────────────────────────────────────────────
    print("--- Phase 1: Seed evaluation ---")
    best_seed_config = None
    best_seed_score = float("inf")

    for seed in SEED_CONFIGS:
        if sim_count >= max_simulations:
            break
        config = {k: v for k, v in seed.items() if k != "name"}
        config.update({
            "hbm_bw_GBps": upper_bounds["hbm_bw_GBps"],
            "hbm_size_GB": upper_bounds["hbm_size_GB"],
            "ici_bw_GBps": upper_bounds["ici_bw_GBps"],
            "dcn_bw_GBps": upper_bounds["dcn_bw_GBps"],
        })
        score = _evaluate(config, f"seed-{seed['name']}")
        if score is not None and score < best_seed_score:
            best_seed_score = score
            best_seed_config = dict(config)

    if best_seed_config is None:
        print("No valid seed found!")
        return {"method": "greedy_local_search", "best_result": None, "history": history}

    print(f"\n  Best seed: score={best_seed_score:.4f}")

    # ── Phase 2: Greedy hill-climbing from best seed ─────────────────────────
    print(f"\n--- Phase 2: Greedy local search (budget: {max_simulations - sim_count} remaining) ---")
    current_config = dict(best_seed_config)
    current_score = best_seed_score
    no_improve_count = 0
    total_attempts = 0
    max_attempts = max_simulations * 5  # prevent infinite loop on duplicates

    while sim_count < max_simulations and total_attempts < max_attempts:
        total_attempts += 1
        # Try a random perturbation
        neighbor = _perturb_config(current_config)
        score = _evaluate(neighbor, f"local-{sim_count+1}")

        if score is not None and score < current_score:
            current_config = neighbor
            current_score = score
            no_improve_count = 0
            print(f"    ↑ Improved! score={score:.4f}")
        else:
            no_improve_count += 1

        # Random restart with multi-step perturbation if stuck
        if no_improve_count >= 10 and sim_count < max_simulations - 5:
            no_improve_count = 0
            if best_result:
                # Apply 3 random perturbations to escape local optimum
                restart_config = dict(best_result["chip_overrides"])
                for _ in range(3):
                    restart_config = _perturb_config(restart_config)
                current_config = restart_config
                current_score = best_score
                print("    Stuck — random restart (3-step perturbation)")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GREEDY LOCAL SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total simulations: {sim_count}")

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
        "method": "greedy_local_search",
        "model_name": model_name,
        "chip_version": chip_version,
        "upper_bounds": upper_bounds,
        "power_budget_W": power_budget_W,
        "ttft_weight": ttft_weight,
        "tpot_weight": tpot_weight,
        "max_simulations": max_simulations,
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
    parser = argparse.ArgumentParser(description="Greedy Local Search baseline for NPU architecture design")
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
    )

    if args.disaggregated:
        print("\n>>> Greedy: PREFILL optimization <<<")
        prefill = run_greedy_search(**common, ttft_weight=1.0, tpot_weight=0.0)
        print("\n>>> Greedy: DECODE optimization <<<")
        decode = run_greedy_search(**common, ttft_weight=0.0, tpot_weight=1.0)

        print(f"\n{'='*60}")
        print("  GREEDY DISAGGREGATED SUMMARY")
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
        run_greedy_search(**common, ttft_weight=args.ttft_weight, tpot_weight=args.tpot_weight)


if __name__ == "__main__":
    main()
