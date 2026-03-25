"""NPU Design Agent (Agent B / Design Planner): designs NPU architecture using
a roofline-based world model + LLM reasoning + NeuSIM validation.

Given an LLM model and upper-bound constraints (hbm_bw, hbm_size, ici_bw, dcn_bw),
this agent uses a physics-based world model (roofline) to guide LLM design decisions,
runs NeuSIM for validation, and iterates until convergence.

Supports disaggregated serving: separately optimize a prefill chip and a decode chip.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

from neusim_runner import (
    apply_chip_overrides,
    format_results,
    load_base_configs,
    run_simulation,
)
from bottleneck_analysis import run_analysis
from world_model import LearnedRoofline
from design_memory import (
    DesignMemory,
    extract_workload_features,
    generate_reflection,
    _summarize_trajectory,
)
from neusim.configs.chips.ChipConfig import ChipConfig
from world_model import (
    RooflineModel,
    build_roofline,
    populate_from_ops,
    predict_cascade,
    predict_power,
    suggest_directions,
    format_for_llm,
)

# ── Tunable NPU parameters and their descriptions ──────────────────────────
TUNABLE_PARAMS = {
    "num_sa": "Number of systolic arrays for matrix multiplication (int, typical range: 2-16)",
    "num_vu": "Number of SIMD vector units for generic vector ops (int, typical range: 2-12)",
    "num_vu_ports": "Number of VU memory ports, usually equals num_vu (int)",
    "sa_dim": "Systolic array dimension NxN (int, typical: 64, 128, 256)",
    "freq_GHz": "Core clock frequency in GHz (float, range: 0.5-2.5)",
    "vmem_size_MB": "On-chip SRAM/vector memory size in MB (int, typical range: 32-512)",
    "TDP_W": "Thermal design power budget in Watts (float, typical range: 200-500)",
}

# Hard bounds to prevent exploiting simulator artifacts
MIN_FREQ_GHZ = 0.5

# Power params that scale with compute units — the agent can adjust these too
TUNABLE_POWER_PARAMS = {
    "static_power_W_per_sa": "Static power per systolic array (W)",
    "static_power_W_per_vu": "Static power per vector unit (W)",
    "static_power_vmem_W": "Static power of vector memory (W)",
    "dynamic_power_W_per_SA": "Dynamic power per systolic array when active (W)",
    "dynamic_power_W_per_VU": "Dynamic power per vector unit when active (W)",
    "dynamic_power_vmem_W": "Dynamic power of vector memory when active (W)",
}

# Upper-bound constraints: the agent can set these but must not exceed the max
CONSTRAINED_PARAMS = {
    "hbm_bw_GBps": "HBM bandwidth in GB/s (float, upper bound)",
    "hbm_size_GB": "HBM capacity in GB (int, upper bound)",
    "ici_bw_GBps": "Inter-chip interconnect bandwidth in GB/s (float, upper bound)",
    "dcn_bw_GBps": "Datacenter network bandwidth in GB/s (float, upper bound)",
}

DEFAULT_UPPER_BOUNDS = {
    "hbm_bw_GBps": 600.0,
    "hbm_size_GB": 16,
    "ici_bw_GBps": 125.0,
    "dcn_bw_GBps": 12.5,
}

# ── OpenAI tool definition ──────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "propose_npu_config",
            "description": (
                "Propose a new NPU architecture configuration. "
                "Only include parameters you want to change from the baseline. "
                "You may set hbm_bw_GBps, hbm_size_GB, ici_bw_GBps, dcn_bw_GBps "
                "but they will be clamped to their upper bounds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why you chose these parameters",
                    },
                    "num_sa": {"type": "integer", "description": TUNABLE_PARAMS["num_sa"]},
                    "num_vu": {"type": "integer", "description": TUNABLE_PARAMS["num_vu"]},
                    "num_vu_ports": {"type": "integer", "description": TUNABLE_PARAMS["num_vu_ports"]},
                    "sa_dim": {"type": "integer", "description": TUNABLE_PARAMS["sa_dim"]},
                    "freq_GHz": {"type": "number", "description": TUNABLE_PARAMS["freq_GHz"]},
                    "vmem_size_MB": {"type": "integer", "description": TUNABLE_PARAMS["vmem_size_MB"]},
                    "TDP_W": {"type": "number", "description": TUNABLE_PARAMS["TDP_W"]},
                    "hbm_bw_GBps": {"type": "number", "description": "HBM bandwidth in GB/s (max: see upper bounds)"},
                    "hbm_size_GB": {"type": "integer", "description": "HBM capacity in GB (max: see upper bounds)"},
                    "ici_bw_GBps": {"type": "number", "description": "ICI bandwidth in GB/s (max: see upper bounds)"},
                    "dcn_bw_GBps": {"type": "number", "description": "DCN bandwidth in GB/s (max: see upper bounds)"},
                    "static_power_W_per_sa": {"type": "number"},
                    "static_power_W_per_vu": {"type": "number"},
                    "static_power_vmem_W": {"type": "number"},
                    "dynamic_power_W_per_SA": {"type": "number"},
                    "dynamic_power_W_per_VU": {"type": "number"},
                    "dynamic_power_vmem_W": {"type": "number"},
                },
                "required": ["reasoning"],
            },
        },
    }
]


def _summarize_compute_demand(prefill_csv: str, decode_csv: str) -> str:
    """Read Agent 1's compute demand CSVs and produce a concise workload summary for the LLM."""
    sections = []
    for phase, csv_path in [("Prefill", prefill_csv), ("Decode", decode_csv)]:
        df = pd.read_csv(csv_path)
        total_flops = df["FLOP Count"].sum()
        compute_ops = df[df["FLOP Count"] > 0]
        comm_ops = df[df["FLOP Count"] == 0]

        # Top 5 compute-heavy operators with tensor shapes
        top = df.nlargest(5, "FLOP Count")
        top_lines = []
        for _, row in top.iterrows():
            pct = row["FLOP Count"] / total_flops * 100 if total_flops > 0 else 0
            top_lines.append(
                f"  {row['Description']}: {row['FLOP Count']:.2e} FLOPs ({pct:.1f}%), "
                f"count={row['Count']}, in={row['Input Tensor Shapes']}, out={row['Output Tensor Shapes']}"
            )

        sections.append(
            f"### {phase} Phase\n"
            f"- Total operators: {len(df)} ({len(compute_ops)} compute, {len(comm_ops)} communication/reshape)\n"
            f"- Total FLOPs: {total_flops:.3e}\n"
            f"- Top 5 operators by FLOPs:\n" + "\n".join(top_lines)
        )
    return "\n\n".join(sections)


def _build_system_prompt(
    model_name: str,
    base_chip_config: dict[str, Any],
    upper_bounds: dict[str, Any],
    power_budget_W: float,
    ttft_weight: float = 0.5,
    tpot_weight: float = 0.5,
) -> str:
    """Build the system prompt for the LLM agent."""
    # Describe the optimization focus based on weights
    if tpot_weight == 0:
        focus = (
            "Your PRIMARY objective: minimize TTFT (Time To First Token) — the prefill phase latency. "
            "This chip is specialized for PREFILL only. "
            "TPOT does not matter for this chip. "
            "Focus on maximizing compute throughput: more/bigger systolic arrays, higher frequency, larger SA dim."
        )
    elif ttft_weight == 0:
        focus = (
            "Your PRIMARY objective: minimize TPOT (Time Per Output Token) — the decode phase latency. "
            "This chip is specialized for DECODE only. "
            "TTFT does not matter for this chip. "
            "Focus on memory bandwidth and efficiency: HBM bandwidth matters most, "
            "fewer but efficient compute units, optimize for memory-bound workloads."
        )
    else:
        focus = (
            f"Your objective: minimize a weighted score = {ttft_weight:.1f} * (TTFT/baseline) + {tpot_weight:.1f} * (TPOT/baseline). "
            f"TTFT weight = {ttft_weight:.1f}, TPOT weight = {tpot_weight:.1f}."
        )

    return f"""You are an expert NPU (Neural Processing Unit) chip architect.

Your task: design an NPU chip architecture for the model "{model_name}", while keeping total power within budget.

## Optimization Goal
{focus}

## Upper-Bound Constraints
These parameters represent technology limits — you CAN set them, but they CANNOT exceed these maximums:
{json.dumps(upper_bounds, indent=2)}
You should typically set these to their max values for best performance, but you may lower them if, for example, reducing HBM size saves area/power and the model still fits.

## Power Budget (HARD CONSTRAINT)
- The chip's peak power (total_power_peak_W = static + dynamic peak) MUST stay under {power_budget_W:.0f} W
- total_power_peak_W = sum of all static power components + sum of all dynamic peak power components
- This scales with compute unit counts and their per-unit power values
- If your proposed config results in peak power > {power_budget_W:.0f} W, it is INVALID
- To reduce peak power: fewer/smaller compute units, or lower per-unit power values

## Baseline Chip Configuration
Current chip parameters (from which you start):
{json.dumps({k: v for k, v in base_chip_config.items() if k in TUNABLE_PARAMS or k in TUNABLE_POWER_PARAMS or k in CONSTRAINED_PARAMS}, indent=2)}

## Tunable Parameters — FULL SEARCH RANGE
Explore the ENTIRE range, including values BELOW the baseline defaults:
- num_sa: 2 to 16 (fewer SAs = less power, sometimes BETTER for memory-bound ops)
- num_vu: 2 to 12 (num_vu_ports should equal num_vu)
- sa_dim: 64, 128, or 256 (64 is better for small matmuls, 256 for large ones)
- freq_GHz: 0.5 to 2.0 (LOWER freq can IMPROVE latency for memory-bound ops by reducing power, leaving more headroom). MINIMUM is 0.5 GHz.
- vmem_size_MB: 32 to 512
- TDP_W: 100 to 500

Power parameters (scale proportionally with compute units):
{json.dumps(TUNABLE_POWER_PARAMS, indent=2)}

## Power Formula (USE THIS to predict valid configs!)
total_power_peak_W = static_power + dynamic_power_peak, where:
  static_power = (static_power_W_per_sa * num_sa) + (static_power_W_per_vu * num_vu) + static_power_vmem_W + static_power_hbm_W + static_power_ici_W
  dynamic_power_peak = (dynamic_power_W_per_SA * num_sa) + (dynamic_power_W_per_VU * num_vu) + dynamic_power_vmem_W + dynamic_power_hbm_W + dynamic_power_ici_W

The HBM and ICI power components are fixed (set by bandwidth). You control SA/VU/VMEM power.
To estimate peak power BEFORE simulation: total ≈ fixed_base + (static_power_W_per_sa + dynamic_power_W_per_SA) * num_sa + (static_power_W_per_vu + dynamic_power_W_per_VU) * num_vu + static_power_vmem_W + dynamic_power_vmem_W

## Key Trade-offs
- More SAs → faster matrix ops but higher power. USELESS if memory-bound (decode)
- Higher freq → faster clock but power scales quadratically. LOWER freq can be BETTER if it avoids power-limit bottleneck
- Larger SA dim → better for large matmuls (prefill), but WORSE for small decode matmuls (padding overhead)
- For DECODE: HBM bandwidth is almost always the bottleneck. More compute won't help. Minimize compute to save power.
- For PREFILL: compute is the bottleneck. More/bigger SAs help most.
- Larger VMEM → better data reuse, less HBM traffic
- CRITICAL: do NOT assume baseline defaults are good. Sometimes going BELOW defaults gives better results.

## Roofline World Model
You will receive a roofline analysis after each simulation that shows:
- **Compute ceiling** = num_sa × sa_dim² × 2 × freq_GHz (GFLOPS): max compute throughput
- **BW ceiling** = hbm_bw_GBps (GB/s): max memory bandwidth
- **Ridge point** = compute_ceiling / BW_ceiling (FLOP/byte): the crossover point
- Operators with arithmetic_intensity > ridge_point are **compute-bound** (limited by SAs)
- Operators with arithmetic_intensity < ridge_point are **memory-bound** (limited by HBM BW)

Use the roofline to reason about your design:
- If most time is in compute-bound ops → increase compute ceiling (more SAs, larger sa_dim, higher freq)
- If most time is in memory-bound ops → increase BW or reduce HBM traffic (larger VMEM); reducing compute saves power without hurting perf
- Watch **ridge crossings**: increasing compute may push ridge RIGHT, making some ops memory-bound
- The roofline predicts **cascade effects**: fixing one bottleneck reveals the next

When "Roofline-Suggested Directions" are provided, they include physics-based speedup estimates.
Use these as informed starting points, but you may deviate based on your own reasoning.

## IMPORTANT: Avoid Duplicate Configs
Do NOT propose a config you (or the seeds) already tried. Every iteration must explore a NEW configuration.
If you keep getting similar results, try a DRAMATICALLY different approach.

## Important Notes
- num_vu_ports should typically equal num_vu
- Power parameters should be physically consistent: if you double num_sa, the per-SA power stays the same but total SA power doubles
- Check memory footprint vs hbm_size_GB — OOM means the config is invalid

Call the `propose_npu_config` tool with your proposed configuration."""


def _build_iteration_prompt(
    iteration: int,
    results: dict[str, Any],
    history: list[dict[str, Any]],
    analysis_report: str = "",
    **_kwargs: Any,
) -> str:
    """Build the prompt for subsequent iterations with simulation feedback."""
    results_str = format_results(results)

    history_summary = ""
    if len(history) > 1:
        history_summary = "\n## Previous Iterations\n"
        for i, h in enumerate(history[:-1]):
            r = h["results"]
            history_summary += (
                f"  Iteration {i+1}: TTFT={r['TTFT_sec']:.4f}s, "
                f"TPOT={r['TPOT_ms_request']:.4f}ms, "
                f"PeakPwr={r.get('total_power_peak_W', 0):.1f}W, "
                f"Config: {json.dumps(h['chip_overrides'])}\n"
            )

    # Flag constraint violations
    violations = []
    if results.get("out_of_memory"):
        violations.append("OUT OF MEMORY — config is INVALID, reduce model footprint or increase HBM")

    violation_str = ""
    if violations:
        violation_str = "\n## CONSTRAINT VIOLATIONS\n" + "\n".join(f"- {v}" for v in violations)

    return f"""## Iteration {iteration} Results

{results_str}
{history_summary}{violation_str}

{analysis_report}

Based on the bottleneck analysis above, propose an improved NPU configuration.
Apply MULTIPLE parameter changes simultaneously — do NOT change just one parameter.
- If chip peak power (total_power_peak_W) is over budget, reduce compute units, lower per-unit power, or lower frequency. THIS IS A HARD CONSTRAINT.
- If power is under budget, you might have room to push harder.
- If OOM, the config is invalid.

Call `propose_npu_config` with your next design."""


def _extract_overrides(tool_args: dict[str, Any]) -> dict[str, Any]:
    """Extract chip config overrides from the tool call arguments."""
    overrides = {}
    valid_keys = set(TUNABLE_PARAMS.keys()) | set(TUNABLE_POWER_PARAMS.keys()) | set(CONSTRAINED_PARAMS.keys())
    for k, v in tool_args.items():
        if k in valid_keys and v is not None:
            overrides[k] = v
    # Clamp frequency to prevent simulator artifact exploitation
    if "freq_GHz" in overrides:
        overrides["freq_GHz"] = max(overrides["freq_GHz"], MIN_FREQ_GHZ)
    return overrides


def _config_signature(overrides: dict[str, Any]) -> str:
    """Create a hashable signature from config overrides for duplicate detection."""
    # Only use the architecture params that affect simulation results
    sig_keys = sorted(k for k in overrides if k in TUNABLE_PARAMS or k in CONSTRAINED_PARAMS)
    return json.dumps({k: overrides[k] for k in sig_keys}, sort_keys=True)


def _fast_power_check(
    base_config: dict[str, Any],
    overrides: dict[str, Any],
    upper_bounds: dict[str, float] | None,
) -> float:
    """Compute total_power_peak_W without running a full simulation.

    Delegates to the world model's predict_power for consistency.
    """
    return predict_power(base_config, overrides, upper_bounds)


def check_convergence(
    history: list[dict[str, Any]],
    threshold: float = 0.02,
    window: int = 10,
    ttft_weight: float = 0.5,
    tpot_weight: float = 0.5,
) -> bool:
    """Check if optimization has converged.

    Only checks the metric(s) that have non-zero weight.
    """
    if len(history) < window + 1:
        return False

    recent = history[-(window + 1):]

    for i in range(1, len(recent)):
        if tpot_weight > 0:
            prev = recent[i - 1]["results"]["TPOT_ms_request"]
            curr = recent[i]["results"]["TPOT_ms_request"]
            if prev > 0 and abs(curr - prev) / prev > threshold:
                return False
        if ttft_weight > 0:
            prev = recent[i - 1]["results"]["TTFT_sec"]
            curr = recent[i]["results"]["TTFT_sec"]
            if prev > 0 and abs(curr - prev) / prev > threshold:
                return False

    return True


# ── Local refinement perturbations ──────────────────────────────────────────
SA_DIM_VALUES = [64, 128, 256]
LOCAL_PERTURBATIONS = {
    "num_sa":       [-2, -1, 1, 2],
    "num_vu":       [-2, -1, 1, 2],
    "sa_dim":       [-1, 1],        # index shift in SA_DIM_VALUES
    "freq_GHz":     [-0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2],
    "vmem_size_MB": [-128, -64, -32, 32, 64, 128],
}
LOCAL_PARAM_BOUNDS = {
    "num_sa": (2, 16), "num_vu": (2, 12),
    "freq_GHz": (0.5, 2.0), "vmem_size_MB": (32, 512),
}


def _perturb_config(config: dict[str, Any]) -> dict[str, Any]:
    """Generate a neighbor by perturbing one random parameter."""
    import random
    new = dict(config)
    param = random.choice(list(LOCAL_PERTURBATIONS.keys()))
    delta = random.choice(LOCAL_PERTURBATIONS[param])

    if param == "sa_dim":
        cur_idx = SA_DIM_VALUES.index(config["sa_dim"]) if config["sa_dim"] in SA_DIM_VALUES else 1
        new_idx = max(0, min(2, cur_idx + delta))
        new["sa_dim"] = SA_DIM_VALUES[new_idx]
    elif param == "num_vu":
        lo, hi = LOCAL_PARAM_BOUNDS["num_vu"]
        new["num_vu"] = max(lo, min(hi, config["num_vu"] + delta))
        new["num_vu_ports"] = new["num_vu"]
    else:
        lo, hi = LOCAL_PARAM_BOUNDS[param]
        val = config[param] + delta
        if isinstance(config[param], int):
            new[param] = int(max(lo, min(hi, val)))
        else:
            new[param] = max(lo, min(hi, round(val, 2)))
    return new


def _get_top_k_configs(
    history: list[dict[str, Any]],
    power_budget_W: float,
    baseline_ttft: float,
    baseline_tpot: float,
    ttft_weight: float,
    tpot_weight: float,
    k: int = 3,
) -> list[tuple[dict[str, Any], float]]:
    """Extract top-k unique valid configs from history, sorted by score (ascending)."""
    scored: list[tuple[dict[str, Any], float, str]] = []
    for entry in history:
        if "error" in entry:
            continue
        r = entry["results"]
        if r.get("out_of_memory"):
            continue
        if r.get("total_power_peak_W", float("inf")) > power_budget_W:
            continue
        score = _compute_score(r, baseline_ttft, baseline_tpot, ttft_weight, tpot_weight)
        sig = _config_signature(entry["chip_overrides"])
        scored.append((entry["chip_overrides"], score, sig))

    # Deduplicate by signature, keep best score
    seen: dict[str, tuple[dict[str, Any], float]] = {}
    for cfg, sc, sig in scored:
        if sig not in seen or sc < seen[sig][1]:
            seen[sig] = (cfg, sc)

    unique = sorted(seen.values(), key=lambda x: x[1])
    return unique[:k]


def _run_local_refinement(
    best_config: dict[str, Any],
    best_score: float,
    best_result: dict[str, Any],
    *,
    model_name: str,
    chip_version: str,
    base_config: dict[str, Any],
    num_chips: int,
    tp: int, pp: int, dp: int, ep: int, bs: int,
    pp_dcn: int, dp_dcn: int,
    upper_bounds: dict[str, float],
    power_budget_W: float,
    ttft_weight: float,
    tpot_weight: float,
    run_dir: str,
    history: list[dict[str, Any]],
    seen_configs: set[str],
    baseline_ttft: float,
    baseline_tpot: float,
    max_refine_sims: int = 30,
    learned_roofline: LearnedRoofline | None = None,
) -> tuple[dict[str, Any] | None, float]:
    """Greedy local search from top-3 configs to push to boundaries."""

    # Extract top-3 starting points from full history
    top_configs = _get_top_k_configs(
        history, power_budget_W, baseline_ttft, baseline_tpot,
        ttft_weight, tpot_weight, k=3,
    )

    if not top_configs:
        print("\n  --- Local refinement: no valid configs to refine ---")
        return best_result, best_score

    n_starts = len(top_configs)
    trade_reserve = 5  # reserve sims for power-aware trades at the end
    coord_budget = max_refine_sims - trade_reserve
    sims_per_start = coord_budget // n_starts
    print(f"\n  --- Local refinement phase: {n_starts} starts × ~{sims_per_start} sims + {trade_reserve} trade ---")
    for i, (cfg, sc) in enumerate(top_configs):
        key_cfg = {k: cfg[k] for k in ("num_sa", "sa_dim", "freq_GHz", "vmem_size_MB") if k in cfg}
        print(f"  Start-{i}: score={sc:.4f}  {json.dumps(key_cfg)}")

    total_sim_count = 0

    def _try_config(neighbor, start_idx, local_label):
        """Evaluate a single config, return (score, entry) or None if invalid."""
        nonlocal total_sim_count, best_result, best_score

        sig = _config_signature(neighbor)
        if sig in seen_configs:
            return None
        seen_configs.add(sig)

        try:
            power = _fast_power_check(base_config, neighbor, upper_bounds)
            if power > power_budget_W:
                return None
        except Exception:
            return None

        # Surrogate gating: predict with world model, skip if confident it's worse
        if learned_roofline is not None and learned_roofline.confidence_score() >= 0.3:
            pred = learned_roofline.surrogate_predict(neighbor, base_config, upper_bounds)
            if pred is not None:
                # Predict the score using the target metric
                pred_score = _compute_score(
                    {"TTFT_sec": pred["predicted_TTFT_sec"], "TPOT_ms_request": pred["predicted_TPOT_ms"]},
                    baseline_ttft, baseline_tpot, ttft_weight, tpot_weight,
                )
                if not learned_roofline.should_simulate(pred_score, best_score, minimize=True):
                    return None  # surrogate says not worth simulating

        total_sim_count += 1
        try:
            results = run_simulation(
                model_name=model_name,
                chip_version=chip_version,
                chip_overrides=neighbor,
                num_chips=num_chips,
                tp=tp, pp=pp, dp=dp, ep=ep, bs=bs,
                pp_dcn=pp_dcn, dp_dcn=dp_dcn,
                output_dir=os.path.join(run_dir, f"refine_{total_sim_count:02d}"),
                upper_bounds=upper_bounds,
            )
        except Exception:
            return None

        is_valid = not results["out_of_memory"] and results["total_power_peak_W"] <= power_budget_W
        if not is_valid:
            return None

        score = _compute_score(results, baseline_ttft, baseline_tpot, ttft_weight, tpot_weight)

        # Track surrogate prediction accuracy for confidence calibration
        if learned_roofline is not None:
            pred = learned_roofline.surrogate_predict(neighbor, base_config, upper_bounds)
            if pred is not None:
                if ttft_weight > 0:
                    learned_roofline.record_prediction_error(pred["predicted_TTFT_sec"], results["TTFT_sec"])
                if tpot_weight > 0:
                    learned_roofline.record_prediction_error(pred["predicted_TPOT_ms"], results["TPOT_ms_request"])

        entry = {
            "iteration": f"refine-{local_label}",
            "chip_overrides": neighbor,
            "reasoning": f"Coordinate descent from start-{start_idx}",
            "results": {
                "TTFT_sec": results["TTFT_sec"],
                "TPOT_ms_request": results["TPOT_ms_request"],
                "throughput_tokens_per_sec_request": results["throughput_tokens_per_sec_request"],
                "avg_power_W_prefill": results["avg_power_W_prefill"],
                "avg_power_W_decode": results["avg_power_W_decode"],
                "max_power_W_prefill": results.get("max_power_W_prefill", 0),
                "max_power_W_decode": results.get("max_power_W_decode", 0),
                "total_power_peak_W": results["total_power_peak_W"],
                "mem_footprint_GB_prefill": results["mem_footprint_GB_prefill"],
                "mem_footprint_GB_decode": results["mem_footprint_GB_decode"],
                "out_of_memory": results["out_of_memory"],
            },
        }
        history.append(entry)

        key_cfg = {k: neighbor[k] for k in ("num_sa", "sa_dim", "freq_GHz", "vmem_size_MB") if k in neighbor}
        improved_global = score < best_score
        print(
            f"  [{local_label}] {json.dumps(key_cfg)}  →  "
            f"TTFT={results['TTFT_sec']:.4f}s  TPOT={results['TPOT_ms_request']:.4f}ms  "
            f"Pwr={results['total_power_peak_W']:.0f}W"
            f"{'  ★ GLOBAL best!' if improved_global else ''}"
        )

        if improved_global:
            best_result = entry
            best_score = score

        return score, entry

    # Directional coordinate descent: for each param, search up/down efficiently
    param_steps = {
        "num_sa":       1,
        "num_vu":       1,
        "sa_dim":       1,   # index step in SA_DIM_VALUES
        "freq_GHz":     0.1,
        "vmem_size_MB": 64,
    }
    sweep_order = ["freq_GHz", "num_sa", "sa_dim", "num_vu", "vmem_size_MB"]

    def _apply_step(cfg, param, direction):
        """Apply one step in the given direction. Returns new config or None if at bounds."""
        if param not in cfg:
            return None
        neighbor = dict(cfg)
        step = param_steps[param] * direction
        if param == "sa_dim":
            cur_idx = SA_DIM_VALUES.index(cfg["sa_dim"]) if cfg["sa_dim"] in SA_DIM_VALUES else 1
            new_idx = cur_idx + int(step)
            if new_idx < 0 or new_idx >= len(SA_DIM_VALUES):
                return None
            neighbor["sa_dim"] = SA_DIM_VALUES[new_idx]
        elif param == "num_vu":
            lo, hi = LOCAL_PARAM_BOUNDS["num_vu"]
            new_val = cfg["num_vu"] + int(step)
            if new_val < lo or new_val > hi:
                return None
            neighbor["num_vu"] = new_val
            neighbor["num_vu_ports"] = new_val
        else:
            lo, hi = LOCAL_PARAM_BOUNDS[param]
            new_val = cfg[param] + step
            if isinstance(cfg[param], int):
                new_val = int(new_val)
            else:
                new_val = round(new_val, 2)
            if new_val < lo or new_val > hi:
                return None
            neighbor[param] = new_val
        return neighbor

    for start_idx, (start_cfg, start_score) in enumerate(top_configs):
        if total_sim_count >= coord_budget:
            break

        current = dict(start_cfg)
        current_score = start_score

        print(f"\n  -- Coordinate descent from start-{start_idx} --")

        for sweep_round in range(3):
            improved_any = False
            for param in sweep_order:
                if total_sim_count >= coord_budget:
                    break

                # Try both directions, keep going in the one that improves
                for direction in [-1, 1]:
                    probe = _apply_step(current, param, direction)
                    if probe is None:
                        continue
                    label = f"s{start_idx}-{param}={probe.get(param)}"
                    result = _try_config(probe, start_idx, label)
                    if result is not None:
                        score, _ = result
                        if score < current_score:
                            print(f"    ↑ {param}: {current.get(param)} → {probe.get(param)}")
                            current = probe
                            current_score = score
                            improved_any = True
                            # Keep going in this direction
                            while total_sim_count < coord_budget:
                                next_probe = _apply_step(current, param, direction)
                                if next_probe is None:
                                    break
                                label2 = f"s{start_idx}-{param}={next_probe.get(param)}"
                                result2 = _try_config(next_probe, start_idx, label2)
                                if result2 is not None:
                                    s2, _ = result2
                                    if s2 < current_score:
                                        print(f"    ↑ {param}: {current.get(param)} → {next_probe.get(param)}")
                                        current = next_probe
                                        current_score = s2
                                    else:
                                        break  # no more improvement in this direction
                                else:
                                    break
                            break  # found improving direction, done with this param

            if not improved_any:
                break

    # Power-aware trade: try increasing compute (num_sa) by reducing other params (num_vu, freq)
    if total_sim_count < max_refine_sims:
        # Find the current overall best from history
        best_entry = _get_top_k_configs(
            history, power_budget_W, baseline_ttft, baseline_tpot,
            ttft_weight, tpot_weight, k=1,
        )
        if best_entry:
            trade_cfg, trade_score = best_entry[0]
            trade_cfg = dict(trade_cfg)
            print(f"\n  -- Power-aware trades from best config --")
            # Try: +1 SA with -1 VU
            for vu_delta, sa_delta in [(-1, 1), (-2, 1), (-1, 2), (-2, 2)]:
                if total_sim_count >= max_refine_sims:
                    break
                probe = dict(trade_cfg)
                new_vu = probe.get("num_vu", 4) + vu_delta
                new_sa = probe.get("num_sa", 4) + sa_delta
                if new_vu < 2 or new_sa < 2 or new_sa > 16:
                    continue
                probe["num_vu"] = new_vu
                probe["num_vu_ports"] = new_vu
                probe["num_sa"] = new_sa
                label = f"trade-sa{new_sa}-vu{new_vu}"
                result = _try_config(probe, "T", label)
                if result is not None:
                    score, _ = result
                    if score < trade_score:
                        trade_cfg = probe
                        trade_score = score
                        print(f"    ↑ trade improved!")

    print(f"\n  Local refinement total: {total_sim_count} simulations across {n_starts} starts")
    return best_result, best_score


## ── Seed configs for warm-start ──────────────────────────────────────────
# Diverse configs to probe the design space before the LLM takes over.
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


def _run_seed_phase(
    model_name: str,
    chip_version: str,
    num_chips: int,
    tp: int, pp: int, dp: int, ep: int, bs: int,
    pp_dcn: int, dp_dcn: int,
    upper_bounds: dict[str, float],
    power_budget_W: float,
    run_dir: str,
    ttft_weight: float,
    tpot_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, float, float | None, float | None]:
    """Run seed configs to explore the design space before the LLM loop.

    Returns (seed_history, best_result, best_score, baseline_ttft, baseline_tpot, best_full_results).
    """
    constrained = {k: upper_bounds[k] for k in ("hbm_bw_GBps", "hbm_size_GB", "ici_bw_GBps", "dcn_bw_GBps")}

    seed_history: list[dict[str, Any]] = []
    best_result = None
    best_score = float("inf")
    best_full_results = None
    baseline_tpot = None
    baseline_ttft = None

    print("  --- Seed phase: evaluating diverse configs ---")
    for seed in SEED_CONFIGS:
        name = seed.pop("name")
        overrides = {**seed, **constrained}
        seed["name"] = name  # restore

        print(f"  Seed [{name}]: {json.dumps({k: v for k, v in overrides.items() if k not in constrained}, sort_keys=True)}", end="  →  ", flush=True)

        try:
            results = run_simulation(
                model_name=model_name,
                chip_version=chip_version,
                chip_overrides=overrides,
                num_chips=num_chips,
                tp=tp, pp=pp, dp=dp, ep=ep, bs=bs,
                pp_dcn=pp_dcn, dp_dcn=dp_dcn,
                output_dir=os.path.join(run_dir, f"seed_{name}"),
                upper_bounds=upper_bounds,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        is_valid = not results["out_of_memory"] and results["total_power_peak_W"] <= power_budget_W
        print(
            f"TTFT={results['TTFT_sec']:.4f}s  TPOT={results['TPOT_ms_request']:.4f}ms  "
            f"Pwr={results['total_power_peak_W']:.0f}W  {'VALID' if is_valid else 'INVALID'}"
        )

        entry = {
            "iteration": f"seed-{name}",
            "chip_overrides": overrides,
            "reasoning": f"Seed config: {name}",
            "results": {
                "TTFT_sec": results["TTFT_sec"],
                "TPOT_ms_request": results["TPOT_ms_request"],
                "throughput_tokens_per_sec_request": results["throughput_tokens_per_sec_request"],
                "avg_power_W_prefill": results["avg_power_W_prefill"],
                "avg_power_W_decode": results["avg_power_W_decode"],
                "max_power_W_prefill": results.get("max_power_W_prefill", 0),
                "max_power_W_decode": results.get("max_power_W_decode", 0),
                "total_power_peak_W": results["total_power_peak_W"],
                "mem_footprint_GB_prefill": results["mem_footprint_GB_prefill"],
                "mem_footprint_GB_decode": results["mem_footprint_GB_decode"],
                "out_of_memory": results["out_of_memory"],
            },
        }
        seed_history.append(entry)

        if is_valid:
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
                best_full_results = results

    print(f"  --- Seed phase done: {len(seed_history)} configs evaluated ---")
    if best_result:
        r = best_result["results"]
        print(f"  Best seed: TTFT={r['TTFT_sec']:.4f}s, TPOT={r['TPOT_ms_request']:.4f}ms, Pwr={r['total_power_peak_W']:.0f}W")
    print()

    return seed_history, best_result, best_score, baseline_ttft, baseline_tpot, best_full_results


def _format_seed_results(seed_history: list[dict[str, Any]], power_budget_W: float) -> str:
    """Format seed results into a string for the LLM's initial context."""
    lines = ["## Seed Evaluation Results (pre-computed)",
             "These configs were already evaluated. Use them to understand the design space.\n"]
    for h in seed_history:
        r = h["results"]
        is_valid = not r["out_of_memory"] and r["total_power_peak_W"] <= power_budget_W
        cfg = {k: v for k, v in h["chip_overrides"].items()
               if k in ("num_sa", "num_vu", "sa_dim", "freq_GHz", "vmem_size_MB")}
        lines.append(
            f"  {h['iteration']}: {json.dumps(cfg)} → "
            f"TTFT={r['TTFT_sec']:.4f}s, TPOT={r['TPOT_ms_request']:.4f}ms, "
            f"Pwr={r['total_power_peak_W']:.0f}W {'✓' if is_valid else '✗ INVALID'}"
        )
    lines.append(
        "\nKey insight: LOWER frequency and FEWER compute units can IMPROVE latency for memory-bound ops "
        "(decode). Don't assume more compute = better. Explore the full range including values BELOW the baseline defaults."
    )
    return "\n".join(lines)


def _compute_score(
    results: dict[str, Any],
    baseline_ttft: float | None,
    baseline_tpot: float | None,
    ttft_weight: float,
    tpot_weight: float,
) -> float:
    """Compute weighted normalized score. Lower is better."""
    score = 0.0
    if ttft_weight > 0 and baseline_ttft and baseline_ttft > 0:
        score += ttft_weight * results["TTFT_sec"] / baseline_ttft
    if tpot_weight > 0 and baseline_tpot and baseline_tpot > 0:
        score += tpot_weight * results["TPOT_ms_request"] / baseline_tpot
    if score == 0:
        return float("inf")
    return score


def run_design_loop(
    model_name: str,
    chip_version: str,
    num_chips: int = 64,
    tp: int = 8,
    pp: int = 8,
    dp: int = 1,
    ep: int = 1,
    bs: int = 16,
    pp_dcn: int = 1,
    dp_dcn: int = 1,
    power_budget_W: float = 350.0,
    max_iterations: int = 50,
    convergence_threshold: float = 0.02,
    convergence_window: int = 10,
    output_dir: str = "design_results",
    openai_model: str = "gpt-4o",
    upper_bounds: dict[str, float] | None = None,
    ttft_weight: float = 0.5,
    tpot_weight: float = 0.5,
    label: str = "",
    compute_demand_dir: str | None = None,
    memory: DesignMemory | None = None,
) -> dict[str, Any]:
    """Main design loop: LLM proposes NPU configs, NeuSIM evaluates, iterate.

    Args:
        ttft_weight: Weight for TTFT in scoring (0.0 = ignore TTFT).
        tpot_weight: Weight for TPOT in scoring (0.0 = ignore TPOT).
        label: Optional label for output directory naming.
        memory: Cross-workload design memory for experience retrieval/storage.

    Returns the best configuration found.
    """
    client = OpenAI()

    if upper_bounds is None:
        upper_bounds = dict(DEFAULT_UPPER_BOUNDS)

    # Load base config to show the agent
    base_config = load_base_configs(model_name, chip_version)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{model_name}_{label}_{timestamp}" if label else f"{model_name}_{timestamp}"
    run_dir = os.path.join(output_dir, dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # Build compute demand summary from Agent 1 output
    compute_demand_summary = ""
    if compute_demand_dir:
        prefill_csv = os.path.join(compute_demand_dir, "inference_prefill.csv")
        decode_csv = os.path.join(compute_demand_dir, "inference_decode.csv")
        if os.path.exists(prefill_csv) and os.path.exists(decode_csv):
            compute_demand_summary = _summarize_compute_demand(prefill_csv, decode_csv)
            print(f"Loaded compute demand from: {compute_demand_dir}")
        else:
            print(f"Warning: compute demand CSVs not found in {compute_demand_dir}")

    # Extract workload features for memory retrieval
    workload_features = extract_workload_features(base_config, tp=tp, pp=pp, bs=bs)
    optimization_target = "prefill" if tpot_weight == 0 else ("decode" if ttft_weight == 0 else "combined")

    # Retrieve relevant design experiences from memory
    memory_context = ""
    if memory and memory.num_episodes() > 0:
        memory_context = memory.format_for_context(
            workload_features, optimization_target, top_k=3,
        )
        if memory_context:
            print(f"Retrieved {memory.num_episodes()} design experiences from memory")

    # Build system prompt (instructions only, no workload data)
    system_prompt = _build_system_prompt(
        model_name, base_config, upper_bounds, power_budget_W,
        ttft_weight=ttft_weight, tpot_weight=tpot_weight,
    )

    # Describe focus for print
    if tpot_weight == 0:
        focus_str = "PREFILL-only (minimize TTFT)"
    elif ttft_weight == 0:
        focus_str = "DECODE-only (minimize TPOT)"
    else:
        focus_str = f"weights: TTFT={ttft_weight:.1f}, TPOT={tpot_weight:.1f}"

    print(f"\n{'='*60}")
    print(f"NPU Design Agent — {model_name} on {num_chips} chips (TP={tp}, PP={pp})")
    print(f"Optimization: {focus_str}")
    print(f"Upper bounds: {json.dumps(upper_bounds)}")
    print(f"Power budget: {power_budget_W}W")
    print(f"{'='*60}\n")

    # ── Create online-learned roofline model ─────────────────────────────
    learned_roofline = LearnedRoofline(ema_alpha=0.3)

    # ── Seed phase: evaluate diverse configs to map the design space ──────
    seed_history, best_result, best_score, baseline_ttft, baseline_tpot, best_seed_full = _run_seed_phase(
        model_name=model_name, chip_version=chip_version,
        num_chips=num_chips, tp=tp, pp=pp, dp=dp, ep=ep, bs=bs,
        pp_dcn=pp_dcn, dp_dcn=dp_dcn,
        upper_bounds=upper_bounds, power_budget_W=power_budget_W,
        run_dir=run_dir,
        ttft_weight=ttft_weight, tpot_weight=tpot_weight,
    )
    history: list[dict[str, Any]] = list(seed_history)

    # ── Build initial conversation with seed knowledge ────────────────────
    seed_results_str = _format_seed_results(seed_history, power_budget_W)

    # Run bottleneck analysis on best seed to give initial design guidance
    seed_analysis_str = ""
    roofline_model = None
    if best_seed_full:
        analysis_str, roofline_model = run_analysis(
            best_seed_full, power_budget_W,
            ttft_weight=ttft_weight, tpot_weight=tpot_weight,
            base_config=base_config, upper_bounds=upper_bounds,
            learned=learned_roofline,
        )
        seed_analysis_str = "\n\n" + analysis_str

    best_seed_str = ""
    if best_result:
        r = best_result["results"]
        best_seed_str = (
            f"\n\nBest seed so far: TTFT={r['TTFT_sec']:.4f}s, TPOT={r['TPOT_ms_request']:.4f}ms, "
            f"Pwr={r['total_power_peak_W']:.0f}W, Config={json.dumps({k: v for k, v in best_result['chip_overrides'].items() if k in ('num_sa','num_vu','sa_dim','freq_GHz','vmem_size_MB')})}"
            f"{seed_analysis_str}"
            f"\n\nYour goal: BEAT this result. Apply MULTIPLE parameter changes simultaneously based on the bottleneck analysis above."
        )

    # Build first user message: compute demand (Input 1) + seed results
    user_msg_parts = [
        f"Design an NPU architecture for {model_name} inference. "
        f"We will use {num_chips} chips with TP={tp}, PP={pp}, BS={bs}.",
    ]
    if compute_demand_summary:
        user_msg_parts.append(
            f"\n\n## Workload Computing Demand (from Agent 1)\n"
            f"Use this to understand the workload characteristics — "
            f"which operators dominate, their tensor shapes, and compute vs memory patterns.\n\n"
            f"{compute_demand_summary}"
        )
    if memory_context:
        user_msg_parts.append(f"\n\n{memory_context}")
    user_msg_parts.append(
        f"\n\n{seed_results_str}{best_seed_str}\n\n"
        f"Now propose a config that IMPROVES on the best seed. Call the propose_npu_config tool."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "".join(user_msg_parts)},
    ]

    # Track seen configs to detect duplicates
    seen_configs: set[str] = set()
    for h in seed_history:
        seen_configs.add(_config_signature(h["chip_overrides"]))

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # 1. Trim conversation to avoid context length overflow.
        MAX_KEPT_PAIRS = 15
        if len(messages) > 2 + MAX_KEPT_PAIRS * 2:
            trimmed_count = (len(messages) - 2) // 2 - MAX_KEPT_PAIRS
            summary_lines = [f"[Context trimmed: {trimmed_count} earlier iterations removed]"]
            if best_result:
                br = best_result["results"]
                summary_lines.append(
                    f"Best so far (iter {best_result['iteration']}): "
                    f"TTFT={br['TTFT_sec']:.4f}s, TPOT={br['TPOT_ms_request']:.4f}ms, "
                    f"PeakPwr={br['total_power_peak_W']:.1f}W, "
                    f"Config={json.dumps(best_result['chip_overrides'])}"
                )
            summary_msg = {"role": "user", "content": "\n".join(summary_lines)}
            messages = messages[:2] + [summary_msg] + messages[-(MAX_KEPT_PAIRS * 2):]

        # 2. Ask LLM to propose config
        response = client.chat.completions.create(
            model=openai_model,
            messages=messages,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "propose_npu_config"}},
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # 3. Extract proposed config
        tool_call = msg.tool_calls[0]
        tool_args = json.loads(tool_call.function.arguments)
        reasoning = tool_args.get("reasoning", "")
        chip_overrides = _extract_overrides(tool_args)

        print(f"  Agent reasoning: {reasoning}")
        print(f"  Proposed config: {json.dumps(chip_overrides, indent=2)}")

        # 3b. Skip duplicate configs
        sig = _config_signature(chip_overrides)
        if sig in seen_configs:
            print(f"  ** DUPLICATE config — skipping simulation, asking for a new one")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": (
                    "DUPLICATE CONFIG: You already tried this exact configuration. "
                    "Propose something DIFFERENT. Try a parameter you haven't changed yet, "
                    "or try a dramatically different combination."
                ),
            })
            continue
        seen_configs.add(sig)

        # 3c. Fast power pre-check — reject configs that exceed power budget
        predicted_power = _fast_power_check(base_config, chip_overrides, upper_bounds)
        if predicted_power > power_budget_W:
            print(f"  ** POWER PRE-CHECK FAIL: {predicted_power:.1f}W > {power_budget_W:.0f}W budget — skipping simulation")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": (
                    f"REJECTED: Your config has peak power {predicted_power:.1f}W which exceeds the {power_budget_W:.0f}W budget. "
                    f"You need to reduce power by {predicted_power - power_budget_W:.1f}W. "
                    f"Options: fewer SAs, fewer VUs, lower per-unit power values, or lower vmem power. "
                    f"The current power breakdown: total = static + dynamic peak."
                ),
            })
            continue

        # 3d. Surrogate gating — predict with learned world model, skip if confident it's worse
        if learned_roofline.confidence_score() >= 0.3:
            pred = learned_roofline.surrogate_predict(chip_overrides, base_config, upper_bounds)
            if pred is not None:
                # Which metric to check depends on optimization focus
                skip = False
                if ttft_weight > 0 and tpot_weight == 0 and best_result:
                    best_ttft = best_result["results"]["TTFT_sec"]
                    skip = not learned_roofline.should_simulate(pred["predicted_TTFT_sec"], best_ttft)
                elif tpot_weight > 0 and ttft_weight == 0 and best_result:
                    best_tpot = best_result["results"]["TPOT_ms_request"]
                    skip = not learned_roofline.should_simulate(pred["predicted_TPOT_ms"], best_tpot)
                elif best_result:
                    pred_score = _compute_score(
                        {"TTFT_sec": pred["predicted_TTFT_sec"], "TPOT_ms_request": pred["predicted_TPOT_ms"]},
                        baseline_ttft, baseline_tpot, ttft_weight, tpot_weight,
                    )
                    skip = not learned_roofline.should_simulate(pred_score, best_score)

                if skip:
                    metric_str = f"TTFT={pred['predicted_TTFT_sec']:.4f}s" if ttft_weight > 0 else f"TPOT={pred['predicted_TPOT_ms']:.4f}ms"
                    print(f"  ** SURROGATE SKIP: predicted {metric_str} worse than best (conf={pred['confidence']:.0%})")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": (
                            f"SURROGATE PREDICTION: The world model predicts this config will achieve "
                            f"TTFT={pred['predicted_TTFT_sec']:.4f}s, TPOT={pred['predicted_TPOT_ms']:.4f}ms — "
                            f"worse than the current best. Try a MORE aggressive change."
                        ),
                    })
                    continue

        # 4. Run NeuSIM simulation
        iter_output_dir = os.path.join(run_dir, f"iter_{iteration:02d}")
        try:
            results = run_simulation(
                model_name=model_name,
                chip_version=chip_version,
                chip_overrides=chip_overrides,
                num_chips=num_chips,
                tp=tp,
                pp=pp,
                dp=dp,
                ep=ep,
                bs=bs,
                pp_dcn=pp_dcn,
                dp_dcn=dp_dcn,
                output_dir=iter_output_dir,
                upper_bounds=upper_bounds,
            )
        except Exception as e:
            error_msg = f"Simulation failed: {e}"
            print(f"  ERROR: {error_msg}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"Simulation failed with error: {error_msg}. Please try a different configuration.",
            })
            history.append({
                "iteration": iteration,
                "chip_overrides": chip_overrides,
                "reasoning": reasoning,
                "error": error_msg,
                "results": {"TPOT_ms_request": float("inf"), "TTFT_sec": float("inf")},
            })
            continue

        # 5. Print results
        print(format_results(results))

        # 5b. Update surrogate reference and track prediction accuracy
        learned_roofline.store_reference(results["TTFT_sec"], results["TPOT_ms_request"])
        pred = learned_roofline.surrogate_predict(chip_overrides, base_config, upper_bounds)
        if pred is not None:
            if ttft_weight > 0:
                learned_roofline.record_prediction_error(pred["predicted_TTFT_sec"], results["TTFT_sec"])
            if tpot_weight > 0:
                learned_roofline.record_prediction_error(pred["predicted_TPOT_ms"], results["TPOT_ms_request"])

        # 6. Record history
        entry = {
            "iteration": iteration,
            "chip_overrides": chip_overrides,
            "reasoning": reasoning,
            "results": {
                "TTFT_sec": results["TTFT_sec"],
                "TPOT_ms_request": results["TPOT_ms_request"],
                "throughput_tokens_per_sec_request": results["throughput_tokens_per_sec_request"],
                "avg_power_W_prefill": results["avg_power_W_prefill"],
                "avg_power_W_decode": results["avg_power_W_decode"],
                "max_power_W_prefill": results["max_power_W_prefill"],
                "max_power_W_decode": results["max_power_W_decode"],
                "total_power_peak_W": results["total_power_peak_W"],
                "mem_footprint_GB_prefill": results["mem_footprint_GB_prefill"],
                "mem_footprint_GB_decode": results["mem_footprint_GB_decode"],
                "out_of_memory": results["out_of_memory"],
            },
        }
        history.append(entry)

        # 7. Track best result using weighted normalized score (lower is better)
        is_valid = (
            not results["out_of_memory"]
            and results["total_power_peak_W"] <= power_budget_W
        )

        if is_valid and baseline_tpot is None:
            baseline_tpot = results["TPOT_ms_request"]
            baseline_ttft = results["TTFT_sec"]

        score = _compute_score(results, baseline_ttft, baseline_tpot, ttft_weight, tpot_weight)

        if is_valid and score < best_score:
            best_score = score
            best_result = entry
        elif not is_valid:
            violation = []
            if results["out_of_memory"]:
                violation.append("OOM")
            if results["total_power_peak_W"] > power_budget_W:
                violation.append(f"PeakPower {results['total_power_peak_W']:.1f}W > budget {power_budget_W:.0f}W")
            print(f"  ** Config INVALID: {', '.join(violation)} — skipped for best selection")

        # 8. Save iteration results
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)

        # 9. Check convergence
        valid_history = [h for h in history if "error" not in h]
        if check_convergence(
            valid_history, threshold=convergence_threshold, window=convergence_window,
            ttft_weight=ttft_weight, tpot_weight=tpot_weight,
        ):
            print(f"\n  Converged after {iteration} iterations!")
            break

        # 10. Build/update roofline world model and feed results back to LLM
        #     The learned_roofline is updated in-place by run_analysis,
        #     making each iteration's predictions more accurate.
        analysis_report, roofline_model = run_analysis(
            results, power_budget_W,
            ttft_weight=ttft_weight, tpot_weight=tpot_weight,
            base_config=base_config, upper_bounds=upper_bounds,
            learned=learned_roofline,
        )
        feedback = _build_iteration_prompt(
            iteration, results, history,
            ttft_weight=ttft_weight, tpot_weight=tpot_weight,
            analysis_report=analysis_report,
        )
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": feedback,
        })

    # ── Local refinement: greedy perturbation around LLM's best config ────
    if best_result and baseline_ttft and baseline_tpot:
        best_result, best_score = _run_local_refinement(
            best_config=best_result["chip_overrides"],
            best_score=best_score,
            best_result=best_result,
            model_name=model_name,
            chip_version=chip_version,
            base_config=base_config,
            num_chips=num_chips,
            tp=tp, pp=pp, dp=dp, ep=ep, bs=bs,
            pp_dcn=pp_dcn, dp_dcn=dp_dcn,
            upper_bounds=upper_bounds,
            power_budget_W=power_budget_W,
            ttft_weight=ttft_weight,
            tpot_weight=tpot_weight,
            run_dir=run_dir,
            history=history,
            seen_configs=seen_configs,
            baseline_ttft=baseline_ttft,
            baseline_tpot=baseline_tpot,
            learned_roofline=learned_roofline,
        )

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DESIGN COMPLETE — {focus_str}")
    print(f"{'='*60}")

    print(f"\n{'Iter':>4}  {'TTFT(s)':>8}  {'TPOT(ms)':>9}  {'PeakPwr(W)':>11}  {'AvgPwr_pre(W)':>14}  {'AvgPwr_dec(W)':>14}  {'OOM':>4}  {'Valid':>5}")
    print("-" * 85)
    for h in history:
        r = h["results"]
        if "error" in h:
            print(f"{h['iteration']:>4}  {'ERROR':>8}")
        else:
            is_v = not r["out_of_memory"] and r["total_power_peak_W"] <= power_budget_W
            print(
                f"{h['iteration']:>4}  "
                f"{r['TTFT_sec']:>8.4f}  "
                f"{r['TPOT_ms_request']:>9.4f}  "
                f"{r['total_power_peak_W']:>11.1f}  "
                f"{r['avg_power_W_prefill']:>14.2f}  "
                f"{r['avg_power_W_decode']:>14.2f}  "
                f"{'Y' if r['out_of_memory'] else 'N':>4}  "
                f"{'Y' if is_v else 'N':>5}"
            )

    if best_result:
        print(f"\nBest configuration (iteration {best_result['iteration']}):")
        print(f"  Config: {json.dumps(best_result['chip_overrides'], indent=2)}")
        r = best_result["results"]
        print(f"  TTFT:       {r['TTFT_sec']:.4f} sec")
        print(f"  TPOT:       {r['TPOT_ms_request']:.4f} ms/request")
        print(f"  Peak Power: {r['total_power_peak_W']:.1f} W")
        print(f"  Avg Power:  {r['avg_power_W_prefill']:.1f} W (prefill), {r['avg_power_W_decode']:.2f} W (decode)")
    else:
        print("\nNo valid configuration found within constraints.")

    # Save final summary
    summary = {
        "model_name": model_name,
        "chip_version": chip_version,
        "upper_bounds": upper_bounds,
        "power_budget_W": power_budget_W,
        "ttft_weight": ttft_weight,
        "tpot_weight": tpot_weight,
        "parallelism": {"num_chips": num_chips, "tp": tp, "pp": pp, "dp": dp, "ep": ep, "bs": bs},
        "best_result": best_result,
        "total_iterations": len(history),
        "history": history,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Store design experience in memory ──────────────────────────────────
    if memory and best_result:
        trajectory_str = _summarize_trajectory(history)
        reflection = generate_reflection(
            model_name=model_name,
            optimization_target=optimization_target,
            best_config=best_result["chip_overrides"],
            best_results=best_result["results"],
            trajectory_summary=trajectory_str,
            client=client,
        )
        print(f"  Design reflection: {reflection}")
        memory.store_episode(
            model_name=model_name,
            optimization_target=optimization_target,
            workload_features=workload_features,
            best_config=best_result["chip_overrides"],
            best_results=best_result["results"],
            parallelism={"num_chips": num_chips, "tp": tp, "pp": pp, "dp": dp, "ep": ep, "bs": bs},
            reflection=reflection,
            power_budget_W=power_budget_W,
        )
        # Consolidate semantic principles if enough episodes exist
        if memory.num_episodes() >= 3:
            principles = memory.consolidate(client=client)
            print(f"  Consolidated {len(principles)} design principles")

    print(f"\nResults saved to: {run_dir}")
    return summary


def run_disaggregated(args, upper_bounds: dict[str, float], memory: DesignMemory | None = None) -> None:
    """Run disaggregated serving: optimize prefill chip and decode chip separately."""
    common = dict(
        model_name=args.model,
        chip_version=args.chip_version,
        num_chips=args.num_chips,
        tp=args.tp,
        pp=args.pp,
        dp=args.dp,
        ep=args.ep,
        bs=args.bs,
        pp_dcn=args.pp_dcn,
        dp_dcn=args.dp_dcn,
        power_budget_W=args.power_budget,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        convergence_window=args.convergence_window,
        output_dir=args.output_dir,
        openai_model=args.openai_model,
        upper_bounds=upper_bounds,
        compute_demand_dir=getattr(args, "compute_demand_dir", None),
        memory=memory,
    )

    print("\n" + "=" * 60)
    print("  DISAGGREGATED SERVING: Prefill Chip + Decode Chip")
    print("=" * 60)

    # Phase 1: Prefill chip
    print("\n>>> Phase 1: Optimizing PREFILL chip (TTFT only) <<<\n")
    prefill_summary = run_design_loop(
        **common,
        ttft_weight=1.0,
        tpot_weight=0.0,
        label="prefill",
    )

    # Phase 2: Decode chip
    print("\n>>> Phase 2: Optimizing DECODE chip (TPOT only) <<<\n")
    decode_summary = run_design_loop(
        **common,
        ttft_weight=0.0,
        tpot_weight=1.0,
        label="decode",
    )

    # Combined report
    print("\n" + "=" * 60)
    print("  DISAGGREGATED RESULTS SUMMARY")
    print("=" * 60)

    prefill_best = prefill_summary.get("best_result")
    decode_best = decode_summary.get("best_result")

    if prefill_best:
        r = prefill_best["results"]
        print(f"\nPREFILL Chip (iter {prefill_best['iteration']}):")
        print(f"  Config: {json.dumps(prefill_best['chip_overrides'], indent=2)}")
        print(f"  TTFT:       {r['TTFT_sec']:.4f} sec")
        print(f"  Peak Power: {r['total_power_peak_W']:.1f} W")
    else:
        print("\nPREFILL Chip: No valid configuration found.")

    if decode_best:
        r = decode_best["results"]
        print(f"\nDECODE Chip (iter {decode_best['iteration']}):")
        print(f"  Config: {json.dumps(decode_best['chip_overrides'], indent=2)}")
        print(f"  TPOT:       {r['TPOT_ms_request']:.4f} ms/request")
        print(f"  Peak Power: {r['total_power_peak_W']:.1f} W")
    else:
        print("\nDECODE Chip: No valid configuration found.")

    if prefill_best and decode_best:
        pr = prefill_best["results"]
        dr = decode_best["results"]
        print(f"\nCombined: TTFT={pr['TTFT_sec']:.4f}s (prefill chip), TPOT={dr['TPOT_ms_request']:.4f}ms (decode chip)")


def main():
    parser = argparse.ArgumentParser(
        description="NPU Design Agent: iteratively design NPU architecture using LLM + NeuSIM"
    )
    parser.add_argument(
        "model", type=str,
        help="Model name (e.g., llama3-8b, llama3_1-405b, deepseekv2-236b)",
    )
    parser.add_argument("--chip-version", type=str, default="4",
                        help="Base chip version to start from (default: 4)")
    parser.add_argument("--num-chips", type=int, default=64,
                        help="Number of NPU chips (default: 64)")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism degree")
    parser.add_argument("--pp", type=int, default=8, help="Pipeline parallelism degree")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism degree")
    parser.add_argument("--bs", type=int, default=16, help="Global batch size")
    parser.add_argument("--pp-dcn", type=int, default=1, help="Pipeline parallel degree DCN")
    parser.add_argument("--dp-dcn", type=int, default=1, help="Data parallel degree DCN")
    parser.add_argument("--power-budget", type=float, default=350.0,
                        help="Power budget in Watts (default: 350)")
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="Maximum design iterations (default: 50)")
    parser.add_argument("--convergence-threshold", type=float, default=0.02,
                        help="Convergence threshold as relative change (default: 0.02 = 2%%)")
    parser.add_argument("--convergence-window", type=int, default=10,
                        help="Consecutive iterations with <threshold change to converge (default: 10)")
    parser.add_argument("--output-dir", type=str, default="design_results",
                        help="Output directory for results")
    parser.add_argument("--openai-model", type=str, default="gpt-4o",
                        help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--max-hbm-bw", type=float, default=DEFAULT_UPPER_BOUNDS["hbm_bw_GBps"],
                        help=f"Upper bound for hbm_bw_GBps (default: {DEFAULT_UPPER_BOUNDS['hbm_bw_GBps']})")
    parser.add_argument("--max-hbm-size", type=float, default=DEFAULT_UPPER_BOUNDS["hbm_size_GB"],
                        help=f"Upper bound for hbm_size_GB (default: {DEFAULT_UPPER_BOUNDS['hbm_size_GB']})")
    parser.add_argument("--max-ici-bw", type=float, default=DEFAULT_UPPER_BOUNDS["ici_bw_GBps"],
                        help=f"Upper bound for ici_bw_GBps (default: {DEFAULT_UPPER_BOUNDS['ici_bw_GBps']})")
    parser.add_argument("--max-dcn-bw", type=float, default=DEFAULT_UPPER_BOUNDS["dcn_bw_GBps"],
                        help=f"Upper bound for dcn_bw_GBps (default: {DEFAULT_UPPER_BOUNDS['dcn_bw_GBps']})")

    # Optimization weights
    parser.add_argument("--ttft-weight", type=float, default=0.5,
                        help="Weight for TTFT in scoring (default: 0.5)")
    parser.add_argument("--tpot-weight", type=float, default=0.5,
                        help="Weight for TPOT in scoring (default: 0.5)")
    parser.add_argument("--disaggregated", action="store_true",
                        help="Disaggregated serving: optimize separate prefill and decode chips")
    parser.add_argument("--compute-demand-dir", type=str, default=None,
                        help="Path to Agent 1 compute demand output directory (contains inference_prefill.csv and inference_decode.csv)")
    parser.add_argument("--memory-dir", type=str, default="design_memory",
                        help="Directory for cross-workload design memory (default: design_memory)")
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable cross-workload design memory")

    args = parser.parse_args()

    upper_bounds = {
        "hbm_bw_GBps": args.max_hbm_bw,
        "hbm_size_GB": args.max_hbm_size,
        "ici_bw_GBps": args.max_ici_bw,
        "dcn_bw_GBps": args.max_dcn_bw,
    }

    memory = None if args.no_memory else DesignMemory(args.memory_dir)

    if args.disaggregated:
        run_disaggregated(args, upper_bounds, memory=memory)
    else:
        run_design_loop(
            model_name=args.model,
            chip_version=args.chip_version,
            num_chips=args.num_chips,
            tp=args.tp,
            pp=args.pp,
            dp=args.dp,
            ep=args.ep,
            bs=args.bs,
            pp_dcn=args.pp_dcn,
            dp_dcn=args.dp_dcn,
            power_budget_W=args.power_budget,
            max_iterations=args.max_iterations,
            convergence_threshold=args.convergence_threshold,
            convergence_window=args.convergence_window,
            output_dir=args.output_dir,
            openai_model=args.openai_model,
            upper_bounds=upper_bounds,
            ttft_weight=args.ttft_weight,
            tpot_weight=args.tpot_weight,
            compute_demand_dir=args.compute_demand_dir,
            memory=memory,
        )


if __name__ == "__main__":
    main()
