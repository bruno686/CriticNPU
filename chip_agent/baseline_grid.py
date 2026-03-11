"""Baseline: grid search over NPU architecture parameters.

Same input/output/constraints as npudesignagent.py, but uses brute-force
grid search instead of an LLM agent. Serves as a comparison baseline.
"""

import argparse
import itertools
import json
import os
from datetime import datetime
from typing import Any

from neusim_runner import format_results, run_simulation

DEFAULT_UPPER_BOUNDS = {
    "hbm_bw_GBps": 600.0,
    "hbm_size_GB": 16,
    "ici_bw_GBps": 125.0,
    "dcn_bw_GBps": 12.5,
}

# ── Search grid ─────────────────────────────────────────────────────────────
# Each param has a list of candidate values to try.
SEARCH_GRID = {
    "num_sa": [4, 6, 8, 10, 12],
    "num_vu": [4, 6, 8, 10],
    "sa_dim": [64, 128, 256],
    "freq_GHz": [0.8, 1.0, 1.2, 1.4],
    "vmem_size_MB": [128, 256, 384],
}

# Fixed: always max out constrained params (best-case for the baseline)
FIXED_PARAMS = {
    "hbm_bw_GBps": 600.0,
    "hbm_size_GB": 16,
    "ici_bw_GBps": 125.0,
    "dcn_bw_GBps": 12.5,
}


def run_grid_search(
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
) -> dict[str, Any]:
    if upper_bounds is None:
        upper_bounds = dict(DEFAULT_UPPER_BOUNDS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{model_name}_baseline_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Generate all combinations
    keys = list(SEARCH_GRID.keys())
    values = list(SEARCH_GRID.values())
    all_combos = list(itertools.product(*values))
    total = len(all_combos)

    print(f"\n{'='*60}")
    print(f"Baseline Grid Search — {model_name} on {num_chips} chips (TP={tp}, PP={pp})")
    print(f"Power budget: {power_budget_W}W")
    print(f"Weights: TTFT={ttft_weight}, TPOT={tpot_weight}")
    print(f"Grid: {' x '.join(str(len(v)) for v in values)} = {total} combinations")
    print(f"{'='*60}\n")

    history = []
    best_result = None
    best_score = float("inf")
    baseline_tpot = None
    baseline_ttft = None
    valid_count = 0
    invalid_count = 0

    for idx, combo in enumerate(all_combos):
        overrides = dict(zip(keys, combo))
        overrides["num_vu_ports"] = overrides["num_vu"]
        overrides.update(FIXED_PARAMS)

        print(f"[{idx+1}/{total}] {overrides}", end="  →  ", flush=True)

        try:
            results = run_simulation(
                model_name=model_name,
                chip_version=chip_version,
                chip_overrides=overrides,
                num_chips=num_chips,
                tp=tp, pp=pp, dp=dp, ep=ep, bs=bs,
                pp_dcn=pp_dcn, dp_dcn=dp_dcn,
                output_dir=os.path.join(run_dir, f"combo_{idx:04d}"),
                upper_bounds=upper_bounds,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            history.append({
                "index": idx, "chip_overrides": overrides,
                "error": str(e),
                "results": {"TPOT_ms_request": float("inf"), "TTFT_sec": float("inf")},
            })
            invalid_count += 1
            continue

        is_valid = (
            not results["out_of_memory"]
            and results["total_power_peak_W"] <= power_budget_W
        )

        entry = {
            "index": idx,
            "chip_overrides": overrides,
            "results": {
                "TTFT_sec": results["TTFT_sec"],
                "TPOT_ms_request": results["TPOT_ms_request"],
                "throughput_tokens_per_sec_request": results["throughput_tokens_per_sec_request"],
                "avg_power_W_prefill": results["avg_power_W_prefill"],
                "avg_power_W_decode": results["avg_power_W_decode"],
                "total_power_peak_W": results["total_power_peak_W"],
                "mem_footprint_GB_prefill": results["mem_footprint_GB_prefill"],
                "mem_footprint_GB_decode": results["mem_footprint_GB_decode"],
                "out_of_memory": results["out_of_memory"],
            },
        }
        history.append(entry)

        status = "VALID" if is_valid else "INVALID"
        if results["out_of_memory"]:
            status += "(OOM)"
        elif results["total_power_peak_W"] > power_budget_W:
            status += f"(PWR {results['total_power_peak_W']:.0f}W)"

        print(
            f"TTFT={results['TTFT_sec']:.4f}s  "
            f"TPOT={results['TPOT_ms_request']:.4f}ms  "
            f"Pwr={results['total_power_peak_W']:.0f}W  "
            f"{status}"
        )

        if is_valid:
            valid_count += 1
            if baseline_tpot is None:
                baseline_tpot = results["TPOT_ms_request"]
                baseline_ttft = results["TTFT_sec"]

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
        else:
            invalid_count += 1

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BASELINE GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {total} | Valid: {valid_count} | Invalid: {invalid_count}")

    if best_result:
        r = best_result["results"]
        print(f"\nBest configuration (combo #{best_result['index']}):")
        # Only show the search params, not the fixed ones
        search_cfg = {k: best_result["chip_overrides"][k] for k in SEARCH_GRID}
        print(f"  Config: {json.dumps(search_cfg, indent=2)}")
        print(f"  TTFT:       {r['TTFT_sec']:.4f} sec")
        print(f"  TPOT:       {r['TPOT_ms_request']:.4f} ms/request")
        print(f"  Peak Power: {r['total_power_peak_W']:.1f} W")
        print(f"  Avg Power:  {r['avg_power_W_prefill']:.1f} W (prefill), {r['avg_power_W_decode']:.2f} W (decode)")
    else:
        print("\nNo valid configuration found within constraints.")

    summary = {
        "method": "baseline_grid_search",
        "model_name": model_name,
        "chip_version": chip_version,
        "upper_bounds": upper_bounds,
        "power_budget_W": power_budget_W,
        "ttft_weight": ttft_weight,
        "tpot_weight": tpot_weight,
        "search_grid": SEARCH_GRID,
        "parallelism": {"num_chips": num_chips, "tp": tp, "pp": pp, "dp": dp, "ep": ep, "bs": bs},
        "total_combos": total,
        "valid_count": valid_count,
        "best_result": best_result,
        "history": history,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {run_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Baseline grid search for NPU architecture design")
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
                        help="Run grid search separately for prefill and decode")

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
    )

    if args.disaggregated:
        print("\n>>> Baseline: PREFILL grid search <<<")
        prefill = run_grid_search(**common, ttft_weight=1.0, tpot_weight=0.0)
        print("\n>>> Baseline: DECODE grid search <<<")
        decode = run_grid_search(**common, ttft_weight=0.0, tpot_weight=1.0)

        print(f"\n{'='*60}")
        print("  BASELINE DISAGGREGATED SUMMARY")
        print(f"{'='*60}")
        pb = prefill.get("best_result")
        db = decode.get("best_result")
        if pb:
            print(f"Prefill: TTFT={pb['results']['TTFT_sec']:.4f}s, Pwr={pb['results']['total_power_peak_W']:.0f}W")
        if db:
            print(f"Decode:  TPOT={db['results']['TPOT_ms_request']:.4f}ms, Pwr={db['results']['total_power_peak_W']:.0f}W")
    else:
        run_grid_search(**common, ttft_weight=args.ttft_weight, tpot_weight=args.tpot_weight)


if __name__ == "__main__":
    main()
