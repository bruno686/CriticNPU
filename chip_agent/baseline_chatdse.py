"""Baseline: ChatDSE — LLM as zero-shot surrogate (no simulator in the loop).

The LLM proposes candidate NPU configs AND estimates their TTFT/TPOT/power
directly from its parametric knowledge. No simulator is called during the
search; a single ground-truth simulation is run at the end on the config
the LLM predicted to be best. Sim count reported in the log is 1.

This implements the ChatDSE paradigm used as a scalar-surrogate baseline
(cf. Section 1 of the paper).
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

from neusim_runner import apply_chip_overrides, load_base_configs, run_simulation
from neusim.configs.chips.ChipConfig import ChipConfig

warnings.filterwarnings("ignore")

DEFAULT_UPPER_BOUNDS = {
    "hbm_bw_GBps": 600.0,
    "hbm_size_GB": 16,
    "ici_bw_GBps": 125.0,
    "dcn_bw_GBps": 12.5,
}

# Same discrete search space as other baselines
SA_DIM_VALUES = [64, 128, 256]
PARAM_BOUNDS = {
    "num_sa": (2, 16),
    "num_vu": (2, 12),
    "sa_dim": SA_DIM_VALUES,
    "freq_GHz": (0.5, 2.0),
    "vmem_size_MB": (32, 512),
}

SYSTEM_PROMPT = """You are an NPU chip architect estimating the inference performance of LLM workloads.

TASK: Given an LLM workload and deployment setting, propose one NPU configuration and estimate its TTFT (seconds), TPOT (ms), and peak power (W).

DESIGN SPACE:
- num_sa (systolic arrays): 2-16
- num_vu (vector units): 2-12
- sa_dim (SA side length): one of {64, 128, 256}
- freq_GHz: 0.5 - 2.0
- vmem_size_MB: 32 - 512

You are REQUIRED to estimate TTFT, TPOT, and peak_power for the config you propose, using only your knowledge of NPU architecture and LLM inference. Do NOT call a simulator — your estimate is the final answer for this iteration.

IMPORTANT:
- Configurations whose peak power exceeds the given budget are infeasible. Keep peak_power under the budget.
- Different configs should be proposed across iterations — diversity matters.
- Reason about whether the workload is compute-bound or memory-bound before proposing, since that determines which parameters matter.
"""


TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "propose_and_estimate",
        "description": "Propose an NPU configuration and estimate its TTFT, TPOT, and peak power.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Brief analysis (1-3 sentences) of bottleneck and parameter choice."},
                "num_sa": {"type": "integer", "minimum": 2, "maximum": 16},
                "num_vu": {"type": "integer", "minimum": 2, "maximum": 12},
                "sa_dim": {"type": "integer", "enum": SA_DIM_VALUES},
                "freq_GHz": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                "vmem_size_MB": {"type": "integer", "minimum": 32, "maximum": 512},
                "predicted_TTFT_sec": {"type": "number", "description": "Estimated time-to-first-token in seconds."},
                "predicted_TPOT_ms": {"type": "number", "description": "Estimated time-per-output-token in ms."},
                "predicted_peak_power_W": {"type": "number", "description": "Estimated peak power in Watts."},
            },
            "required": ["reasoning", "num_sa", "num_vu", "sa_dim", "freq_GHz", "vmem_size_MB",
                         "predicted_TTFT_sec", "predicted_TPOT_ms", "predicted_peak_power_W"],
        },
    },
}]


def _build_user_prompt(model_name: str, num_chips: int, tp: int, pp: int, bs: int,
                       power_budget_W: float, focus: str,
                       upper_bounds: dict[str, float],
                       history: list[dict[str, Any]]) -> str:
    parts = [
        f"Workload: {model_name} inference",
        f"Deployment: {num_chips} NPU chips, TP={tp}, PP={pp}, batch size={bs}",
        f"Power budget: {power_budget_W:.0f} W (peak power per chip must not exceed this)",
        f"Fixed (non-searched) parameters: HBM bandwidth = {upper_bounds['hbm_bw_GBps']:.0f} GB/s, HBM size = {upper_bounds['hbm_size_GB']:.0f} GB, ICI bandwidth = {upper_bounds['ici_bw_GBps']:.0f} GB/s, DCN bandwidth = {upper_bounds['dcn_bw_GBps']:.1f} GB/s",
        f"Optimization objective: {focus}",
    ]

    if history:
        parts.append("\nPrior proposals and your estimates (to promote diversity):")
        for i, h in enumerate(history[-10:], 1):
            cfg = h["config"]
            parts.append(
                f"  [{i}] num_sa={cfg['num_sa']}, num_vu={cfg['num_vu']}, sa_dim={cfg['sa_dim']}, "
                f"freq={cfg['freq_GHz']:.2f}GHz, vmem={cfg['vmem_size_MB']}MB "
                f"→ predicted TTFT={h['predicted_TTFT_sec']:.4f}s, "
                f"TPOT={h['predicted_TPOT_ms']:.2f}ms, peak_power={h['predicted_peak_power_W']:.0f}W"
            )
        parts.append("Propose a DIFFERENT configuration than the ones above.")
    else:
        parts.append("\nThis is the first proposal.")

    return "\n".join(parts)


def _call_llm(client: OpenAI, model: str, user_prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tools=TOOL_SCHEMA,
        tool_choice={"type": "function", "function": {"name": "propose_and_estimate"}},
    )
    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)


def _extract_config(args: dict[str, Any], upper_bounds: dict[str, float]) -> dict[str, Any]:
    return {
        "num_sa": int(args["num_sa"]),
        "num_vu": int(args["num_vu"]),
        "num_vu_ports": int(args["num_vu"]),
        "sa_dim": int(args["sa_dim"]),
        "freq_GHz": float(args["freq_GHz"]),
        "vmem_size_MB": int(args["vmem_size_MB"]),
        "hbm_bw_GBps": upper_bounds["hbm_bw_GBps"],
        "hbm_size_GB": upper_bounds["hbm_size_GB"],
        "ici_bw_GBps": upper_bounds["ici_bw_GBps"],
        "dcn_bw_GBps": upper_bounds["dcn_bw_GBps"],
    }


def _config_sig(config: dict[str, Any]) -> str:
    keys = ["num_sa", "num_vu", "sa_dim", "freq_GHz", "vmem_size_MB"]
    return json.dumps({k: config.get(k) for k in keys}, sort_keys=True)


def run_chatdse_search(
    model_name: str,
    chip_version: str = "4",
    num_chips: int = 8,
    tp: int = 4, pp: int = 2, dp: int = 1, ep: int = 1, bs: int = 16,
    pp_dcn: int = 1, dp_dcn: int = 1,
    power_budget_W: float = 300.0,
    output_dir: str = "design_results",
    upper_bounds: dict[str, float] | None = None,
    ttft_weight: float = 0.5,
    tpot_weight: float = 0.5,
    n_iterations: int = 60,
    openai_model: str = "gpt-5.4-mini",
) -> dict[str, Any]:
    if upper_bounds is None:
        upper_bounds = dict(DEFAULT_UPPER_BOUNDS)

    client = OpenAI()
    base_config = load_base_configs(model_name, chip_version)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "prefill" if tpot_weight == 0 else ("decode" if ttft_weight == 0 else "combined")
    run_dir = os.path.join(output_dir, f"{model_name}_chatdse_{label}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    focus_str = "Minimize TTFT (prefill)" if tpot_weight == 0 else (
        "Minimize TPOT (decode)" if ttft_weight == 0 else "Minimize a weighted sum of TTFT and TPOT"
    )

    print(f"\n{'='*60}")
    print(f"ChatDSE — {model_name} on {num_chips} chips (TP={tp}, PP={pp})")
    print(f"Optimization: {focus_str}")
    print(f"Power budget: {power_budget_W}W")
    print(f"LLM-proposal iterations: {n_iterations} (no simulator in the loop)")
    print(f"{'='*60}\n")

    history: list[dict[str, Any]] = []
    seen: set[str] = set()

    for i in range(1, n_iterations + 1):
        user_prompt = _build_user_prompt(
            model_name, num_chips, tp, pp, bs, power_budget_W,
            focus_str, upper_bounds, history,
        )
        try:
            args = _call_llm(client, openai_model, user_prompt)
        except Exception as e:
            print(f"  [iter {i:02d}] LLM ERROR: {e} — skipping")
            continue

        config = _extract_config(args, upper_bounds)
        sig = _config_sig(config)
        if sig in seen:
            print(f"  [iter {i:02d}] DUP — skipping")
            continue
        seen.add(sig)

        # Reject LLM-predicted infeasible configs
        if args["predicted_peak_power_W"] > power_budget_W * 1.05:
            print(f"  [iter {i:02d}] LLM-self-reported power {args['predicted_peak_power_W']:.0f}W > budget — skipping")
            continue

        entry = {
            "iter": i,
            "config": config,
            "reasoning": args.get("reasoning", ""),
            "predicted_TTFT_sec": float(args["predicted_TTFT_sec"]),
            "predicted_TPOT_ms": float(args["predicted_TPOT_ms"]),
            "predicted_peak_power_W": float(args["predicted_peak_power_W"]),
        }
        history.append(entry)
        print(
            f"  [iter {i:02d}] num_sa={config['num_sa']:2d} num_vu={config['num_vu']:2d} "
            f"sa={config['sa_dim']:3d} f={config['freq_GHz']:.2f} vmem={config['vmem_size_MB']:3d}  →  "
            f"pred TTFT={entry['predicted_TTFT_sec']:.4f}s TPOT={entry['predicted_TPOT_ms']:.2f}ms "
            f"Pwr={entry['predicted_peak_power_W']:.0f}W"
        )

    if not history:
        print("No valid proposals from LLM — aborting.")
        return {"best_result": None, "history": history}

    # Pick the best by LLM estimate
    def _score(e):
        return ttft_weight * e["predicted_TTFT_sec"] + tpot_weight * (e["predicted_TPOT_ms"] / 1000.0)

    best_entry = min(history, key=_score)
    print(f"\n>>> LLM's best-estimated config (iter {best_entry['iter']})")
    print(f"    Config: {json.dumps({k: best_entry['config'][k] for k in ['num_sa','num_vu','sa_dim','freq_GHz','vmem_size_MB']})}")
    print(f"    Predicted TTFT={best_entry['predicted_TTFT_sec']:.4f}s  TPOT={best_entry['predicted_TPOT_ms']:.2f}ms")

    # Verify once with real simulator — this is the ONLY sim call for ChatDSE
    print(f"\n>>> Verifying with NeuSim (single ground-truth simulation) <<<")
    try:
        results = run_simulation(
            model_name=model_name,
            chip_version=chip_version,
            chip_overrides=best_entry["config"],
            num_chips=num_chips,
            tp=tp, pp=pp, dp=dp, ep=ep, bs=bs,
            pp_dcn=pp_dcn, dp_dcn=dp_dcn,
            output_dir=os.path.join(run_dir, "verify"),
            upper_bounds=upper_bounds,
        )
    except Exception as e:
        print(f"  Verification FAILED: {e}")
        return {"best_result": None, "history": history}

    is_valid = not results["out_of_memory"] and results["total_power_peak_W"] <= power_budget_W
    status = "VALID" if is_valid else ("OOM" if results["out_of_memory"] else f"PWR={results['total_power_peak_W']:.0f}W")

    print(
        f"  Ground-truth: TTFT={results['TTFT_sec']:.4f}s  TPOT={results['TPOT_ms_request']:.4f}ms  "
        f"Pwr={results['total_power_peak_W']:.0f}W  {status}"
    )

    # run_simulation already prints "=== NeuSIM Simulation Results ===" once,
    # which generate_tables.py counts as sim=1.
    print(f"Combined: TTFT={results['TTFT_sec']:.4f}s, TPOT={results['TPOT_ms_request']:.4f}ms")

    best_result = {
        "iteration": best_entry["iter"],
        "chip_overrides": best_entry["config"],
        "results": {
            "TTFT_sec": results["TTFT_sec"],
            "TPOT_ms_request": results["TPOT_ms_request"],
            "total_power_peak_W": results["total_power_peak_W"],
        },
        "llm_predicted": {
            "TTFT_sec": best_entry["predicted_TTFT_sec"],
            "TPOT_ms": best_entry["predicted_TPOT_ms"],
            "peak_power_W": best_entry["predicted_peak_power_W"],
        },
        "is_valid": is_valid,
    }

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"best_result": best_result, "history": history}, f, indent=2)

    return {"best_result": best_result, "history": history}


def main():
    parser = argparse.ArgumentParser(description="ChatDSE baseline (LLM-as-zero-shot-surrogate)")
    parser.add_argument("model", type=str)
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
    parser.add_argument("--n-iterations", type=int, default=60)
    parser.add_argument("--openai-model", type=str, default="gpt-5.4-mini")

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
        n_iterations=args.n_iterations,
        openai_model=args.openai_model,
    )

    if args.disaggregated:
        print("\n>>> ChatDSE: PREFILL optimization <<<")
        prefill = run_chatdse_search(**common, ttft_weight=1.0, tpot_weight=0.0)
        print("\n>>> ChatDSE: DECODE optimization <<<")
        decode = run_chatdse_search(**common, ttft_weight=0.0, tpot_weight=1.0)

        print(f"\n{'='*60}\n  CHATDSE DISAGGREGATED SUMMARY\n{'='*60}")
        pb = prefill.get("best_result")
        db = decode.get("best_result")
        if pb:
            r = pb["results"]
            print(f"PREFILL (iter {pb['iteration']}): TTFT={r['TTFT_sec']:.4f}s, Pwr={r['total_power_peak_W']:.0f}W")
        if db:
            r = db["results"]
            print(f"DECODE (iter {db['iteration']}): TPOT={r['TPOT_ms_request']:.4f}ms, Pwr={r['total_power_peak_W']:.0f}W")
        if pb and db:
            print(f"\nCombined: TTFT={pb['results']['TTFT_sec']:.4f}s, TPOT={db['results']['TPOT_ms_request']:.4f}ms")
    else:
        run_chatdse_search(**common, ttft_weight=args.ttft_weight, tpot_weight=args.tpot_weight)


if __name__ == "__main__":
    main()
