"""Bottleneck Analysis (Agent A / World Model Constructor).

Builds a physics-based roofline world model from NeuSIM simulation results.
Replaces the previous LLM-based one-sentence diagnosis with deterministic
roofline analysis — the world model provides structured, interpretable
bottleneck information that Agent B (Design Planner) can reason with.

Sits between NeuSIM and Agent B in the design loop:
  Agent B → NPU Config → NeuSIM → **World Model Constructor** → roofline → Agent B
"""

import json
from typing import Any

from world_model import (
    LearnedRoofline,
    RooflineModel,
    build_roofline,
    populate_from_ops,
    format_for_llm,
    suggest_directions,
)


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


def run_analysis(
    results: dict[str, Any],
    power_budget_W: float,
    ttft_weight: float = 0.5,
    tpot_weight: float = 0.5,
    base_config: dict[str, Any] | None = None,
    upper_bounds: dict[str, float] | None = None,
    learned: LearnedRoofline | None = None,
) -> tuple[str, RooflineModel | None]:
    """Main entry point: build roofline world model from NeuSIM results.

    Args:
        results: dict returned by neusim_runner.run_simulation()
        power_budget_W: power budget constraint
        ttft_weight: optimization weight for TTFT
        tpot_weight: optimization weight for TPOT
        base_config: base chip config dict (for roofline construction)
        upper_bounds: upper-bound constraints
        learned: optional LearnedRoofline to use for corrected predictions
                 and to update with this simulation's data

    Returns:
        (formatted_analysis_string, RooflineModel or None)
    """
    prefill_ops = results.get("prefill_ops", [])
    decode_ops = results.get("decode_ops", [])

    prefill_analysis = analyze_phase(prefill_ops, "Prefill") if prefill_ops else {}
    decode_analysis = analyze_phase(decode_ops, "Decode") if decode_ops else {}

    # Build structured stats summary
    stats_lines = ["## Bottleneck Analysis"]
    for analysis in [prefill_analysis, decode_analysis]:
        if not analysis:
            continue
        cat = analysis["cat_pct"]
        stats_lines.append(
            f"- {analysis['phase']}: Compute {cat.get('Compute', 0):.0f}% | "
            f"Memory {cat.get('Memory', 0):.0f}% | "
            f"ICI {cat.get('ICI/NVLink', 0):.0f}% "
            f"(SA util {analysis['avg_flops_util']:.0%}, HBM util {analysis['avg_hbm_util']:.0%})"
        )

    # Build roofline world model if base_config is available
    roofline = None
    if base_config is not None:
        chip_overrides = results.get("chip_overrides", {})
        roofline = build_roofline(chip_overrides, base_config, power_budget_W, upper_bounds)
        roofline = populate_from_ops(roofline, prefill_ops, decode_ops, learned=learned)

        # Update learned model from this simulation's actual data
        if learned is not None:
            learned.update(roofline)

        # Generate suggestions using learned corrections
        suggestions = suggest_directions(
            roofline, base_config, upper_bounds,
            ttft_weight=ttft_weight, tpot_weight=tpot_weight,
            learned=learned,
        )

        # Format roofline analysis with learned corrections
        roofline_str = format_for_llm(roofline, suggestions=suggestions, learned=learned)
        stats_lines.append("")
        stats_lines.append(roofline_str)

    return "\n".join(stats_lines), roofline
