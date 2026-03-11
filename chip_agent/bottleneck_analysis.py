"""Bottleneck Analysis Agent: uses LLM to analyze per-operator NeuSIM results
and produce multi-parameter design recommendations for Agent 2.

Sits between NeuSIM and Agent 2 in the design loop:
  Agent 2 → NPU Config → NeuSIM → **Analysis Agent (LLM)** → feedback → Agent 2
"""

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

ANALYSIS_SYSTEM_PROMPT = """You are an NPU performance analyst. Given NeuSIM bottleneck data, state the core bottleneck in ONE sentence. No recommendations, no parameter suggestions, no trade-off discussion. Just the fact.

Example outputs:
- "Prefill is 78% compute-bound with 42% SA utilization — SAs are under-utilized due to tile padding."
- "Decode is 93% memory-bound, HBM BW at 88% utilization — pure bandwidth bottleneck."
- "ICI accounts for 35% of prefill time — communication overhead dominates."

Rules:
- ONE sentence only
- State bottleneck type, percentage, and the key utilization number
- Do NOT suggest any parameter changes
- Do NOT say "consider" or "recommend" or "should" """


def analyze_phase(ops: list, phase: str) -> dict[str, Any]:
    """Extract bottleneck statistics from per-operator simulation results."""
    if not ops:
        return {}

    time_by_category: dict[str, float] = {"Compute": 0, "Memory": 0, "ICI/NVLink": 0}
    total_time_ns = 0
    total_flops = 0
    total_mem_bytes = 0

    op_details = []
    for op in ops:
        s = op.stats
        weighted_time = s.execution_time_ns * s.count
        total_time_ns += weighted_time
        total_flops += s.flop_count * s.count
        total_mem_bytes += s.memory_traffic_bytes * s.count

        cat = s.bounded_by if s.bounded_by in time_by_category else "Compute"
        time_by_category[cat] += weighted_time

        op_details.append({
            "name": op.description or op.name,
            "bounded_by": s.bounded_by,
            "time_ns": weighted_time,
            "count": s.count,
            "flop_count": s.flop_count * s.count,
            "mem_bytes": s.memory_traffic_bytes * s.count,
            "sa_time_ns": s.sa_time_ns * s.count,
            "memory_time_ns": s.memory_time_ns * s.count,
            "ici_time_ns": s.ici_time_ns * s.count,
            "flops_util": s.flops_util,
            "hbm_bw_util": s.hbm_bw_util,
        })

    op_details.sort(key=lambda x: x["time_ns"], reverse=True)

    cat_pct = {}
    for cat, t in time_by_category.items():
        cat_pct[cat] = t / total_time_ns * 100 if total_time_ns > 0 else 0

    avg_flops_util = 0
    avg_hbm_util = 0
    for d in op_details:
        w = d["time_ns"] / total_time_ns if total_time_ns > 0 else 0
        avg_flops_util += d["flops_util"] * w
        avg_hbm_util += d["hbm_bw_util"] * w

    return {
        "phase": phase,
        "total_time_ns": total_time_ns,
        "total_flops": total_flops,
        "total_mem_bytes": total_mem_bytes,
        "cat_pct": cat_pct,
        "avg_flops_util": avg_flops_util,
        "avg_hbm_util": avg_hbm_util,
        "top_ops": op_details[:5],
        "num_ops": len(ops),
    }


def _format_analysis_data(
    prefill: dict[str, Any],
    decode: dict[str, Any],
    chip_config: dict[str, Any],
    power_budget_W: float,
    ttft_weight: float,
    tpot_weight: float,
) -> str:
    """Format bottleneck data into a structured prompt for the Analysis LLM."""
    lines = []

    # Chip config
    current_power = chip_config.get("total_power_peak_W", 0)
    lines.append(f"## Current Chip Config")
    cfg_keys = ["num_sa", "num_vu", "sa_dim", "freq_GHz", "vmem_size_MB",
                "hbm_bw_GBps", "hbm_size_GB", "ici_bw_GBps"]
    cfg_str = {k: chip_config[k] for k in cfg_keys if k in chip_config}
    lines.append(f"Parameters: {json.dumps(cfg_str)}")
    lines.append(f"Peak power: {current_power:.1f}W / {power_budget_W:.0f}W budget "
                 f"(headroom: {power_budget_W - current_power:.1f}W)")

    # Optimization target
    if tpot_weight == 0:
        lines.append(f"Optimization: PREFILL only (minimize TTFT)")
    elif ttft_weight == 0:
        lines.append(f"Optimization: DECODE only (minimize TPOT)")
    else:
        lines.append(f"Optimization: TTFT weight={ttft_weight}, TPOT weight={tpot_weight}")

    # Per-phase data
    for analysis in [prefill, decode]:
        if not analysis:
            continue
        phase = analysis["phase"]
        cat = analysis["cat_pct"]
        lines.append(f"\n## {phase} Phase Bottleneck Data")
        lines.append(
            f"- Time breakdown: Compute {cat.get('Compute', 0):.1f}% | "
            f"Memory {cat.get('Memory', 0):.1f}% | "
            f"ICI {cat.get('ICI/NVLink', 0):.1f}%"
        )
        lines.append(f"- Avg SA utilization: {analysis['avg_flops_util']:.1%}")
        lines.append(f"- Avg HBM BW utilization: {analysis['avg_hbm_util']:.1%}")
        lines.append(f"- Total FLOPs: {analysis['total_flops']:.2e}")
        lines.append(f"- Total memory traffic: {analysis['total_mem_bytes'] / 1e9:.2f} GB")

        lines.append("- Top 5 time-consuming operators:")
        for op in analysis["top_ops"]:
            pct = op["time_ns"] / analysis["total_time_ns"] * 100 if analysis["total_time_ns"] > 0 else 0
            lines.append(
                f"  - {op['name']}: {pct:.1f}% of time, {op['bounded_by']}-bound, "
                f"SA util {op['flops_util']:.0%}, HBM util {op['hbm_bw_util']:.0%}, "
                f"FLOPs={op['flop_count']:.2e}, mem={op['mem_bytes'] / 1e9:.2f}GB"
            )

    return "\n".join(lines)


def run_analysis(
    results: dict[str, Any],
    power_budget_W: float,
    ttft_weight: float = 0.5,
    tpot_weight: float = 0.5,
) -> str:
    """Main entry point: analyze NeuSIM results using LLM and return recommendations.

    Args:
        results: dict returned by neusim_runner.run_simulation()
        power_budget_W: power budget constraint
        ttft_weight: optimization weight for TTFT
        tpot_weight: optimization weight for TPOT

    Returns:
        LLM-generated analysis report with multi-parameter recommendations.
    """
    prefill_ops = results.get("prefill_ops", [])
    decode_ops = results.get("decode_ops", [])

    prefill_analysis = analyze_phase(prefill_ops, "Prefill") if prefill_ops else {}
    decode_analysis = analyze_phase(decode_ops, "Decode") if decode_ops else {}

    # Build chip config dict
    chip_config = dict(results.get("chip_overrides", {}))
    chip_config["total_power_peak_W"] = results.get("total_power_peak_W", 0)

    # Format data for LLM
    data_prompt = _format_analysis_data(
        prefill_analysis, decode_analysis,
        chip_config, power_budget_W,
        ttft_weight, tpot_weight,
    )

    # Call LLM for analysis
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"{data_prompt}\n\n"
                f"State the #1 bottleneck in ONE sentence. No suggestions."
            )},
        ],
        temperature=0.1,
        max_tokens=80,
    )

    llm_analysis = response.choices[0].message.content

    # Structured data + LLM one-liner diagnosis
    lines = ["## Bottleneck Analysis"]

    for analysis in [prefill_analysis, decode_analysis]:
        if not analysis:
            continue
        cat = analysis["cat_pct"]
        lines.append(
            f"- {analysis['phase']}: Compute {cat.get('Compute', 0):.0f}% | "
            f"Memory {cat.get('Memory', 0):.0f}% | "
            f"ICI {cat.get('ICI/NVLink', 0):.0f}% "
            f"(SA util {analysis['avg_flops_util']:.0%}, HBM util {analysis['avg_hbm_util']:.0%})"
        )

    lines.append(f"- Diagnosis: {llm_analysis}")

    return "\n".join(lines)
