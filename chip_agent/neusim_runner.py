"""NeuSIM runner: programmatic interface to run NeuSIM simulations and extract metrics.

Wraps the NeuSIM Python API so that Agent 2 can call a single function to:
  1. Build a chip config + model config
  2. Run the ops generator (prefill + decode)
  3. Run power analysis
  4. Return TPOT_ms_request, TTFT_sec, and total_power_W
"""

import contextlib
import csv
import io
import json
import os
import sys
import tempfile
from math import ceil
from pathlib import Path
from typing import Any

# Add NeuSim to path
_NEUSIM_ROOT = Path(__file__).resolve().parent.parent / "NeuSim"
sys.path.insert(0, str(_NEUSIM_ROOT))

from neusim.npusim.frontend.llm_ops_generator import (
    DeepSeekOpsGenerator,
    LLMOpsGeneratorInference,
)
from neusim.npusim.frontend import power_analysis_lib as power_lib
from neusim.npusim.frontend import run_sim_lib

CONFIGS_PATH = _NEUSIM_ROOT / "configs"


def load_base_configs(model_name: str, chip_version: str) -> dict[str, Any]:
    """Load and merge model, chip, and system JSON configs."""
    model_cfg = json.load(open(CONFIGS_PATH / "models" / f"{model_name}.json"))
    chip_cfg = json.load(open(CONFIGS_PATH / "chips" / f"tpuv{chip_version}.json"))
    sys_cfg = json.load(open(CONFIGS_PATH / "systems" / "system_config.json"))
    return {**sys_cfg, **chip_cfg, **model_cfg}


def apply_chip_overrides(
    config: dict[str, Any],
    overrides: dict[str, Any],
    upper_bounds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply NPU architecture parameter overrides to the config.

    The overrides dict may contain any ChipConfig field, e.g.:
      num_sa, num_vu, num_vu_ports, sa_dim, freq_GHz, vmem_size_MB,
      TDP_W, hbm_bw_GBps, hbm_size_GB, ici_bw_GBps, dcn_bw_GBps, etc.

    If upper_bounds is provided, constrained parameters are clamped to their max values.
    """
    for k, v in overrides.items():
        config[k] = v
    # Enforce upper bounds on constrained parameters
    if upper_bounds:
        for k, max_val in upper_bounds.items():
            if k in config:
                config[k] = min(config[k], max_val)
    return config


def apply_parallelism(
    config: dict[str, Any],
    num_chips: int,
    tp: int,
    pp: int,
    dp: int = 1,
    ep: int = 1,
    bs: int = 16,
    pp_dcn: int = 1,
    dp_dcn: int = 1,
) -> dict[str, Any]:
    """Set parallelism degrees, batch size, and derived fields."""
    config["num_chips"] = num_chips
    config["tensor_parallelism_degree"] = tp
    config["pipeline_parallelism_degree"] = pp
    config["data_parallelism_degree"] = dp
    config["data_parallel_degree_dcn"] = dp_dcn
    config["tensor_parallel_degree_dcn"] = 1
    config["pipeline_parallel_degree_dcn"] = pp_dcn
    config["global_batch_size"] = bs
    config["microbatch_size_ici"] = ceil(bs / dp_dcn / pp / pp_dcn)
    config["microbatch_size_dcn"] = ceil(bs / pp_dcn)

    if ep > 1:
        config["expert_parallelism_degree"] = ep
        config["expert_parallel_degree_dcn"] = 1

    # Map parallelism to ICI axes — simple heuristic
    # For a single pod, axes = log2(degree) if degree > 1, else 0
    import math
    config["num_tensor_parallel_axes"] = max(0, int(math.log2(tp))) if tp > 1 else 0
    config["num_pipeline_parallel_axes"] = max(0, int(math.log2(pp))) if pp > 1 else 0
    config["num_data_parallel_axes"] = max(0, int(math.log2(dp))) if dp > 1 else 0
    if ep > 1:
        config["num_expert_parallel_axes"] = max(0, int(math.log2(ep)))

    return config


def run_simulation(
    model_name: str,
    chip_version: str,
    chip_overrides: dict[str, Any],
    num_chips: int = 64,
    tp: int = 8,
    pp: int = 8,
    dp: int = 1,
    ep: int = 1,
    bs: int = 16,
    pp_dcn: int = 1,
    dp_dcn: int = 1,
    output_dir: str | None = None,
    pg_config: str = "NoPG",
    upper_bounds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run a full NeuSIM simulation and return key metrics.

    Returns:
        dict with keys:
            TTFT_sec, TPOT_ms_request, throughput_tokens_per_sec_request,
            total_power_W_prefill, total_power_W_decode,
            avg_power_W_prefill, avg_power_W_decode,
            mem_footprint_GB_prefill, mem_footprint_GB_decode,
            out_of_memory, chip_config (the overrides used)
    """
    # 1. Build config
    config = load_base_configs(model_name, chip_version)
    config = apply_chip_overrides(config, chip_overrides, upper_bounds=upper_bounds)
    config = apply_parallelism(config, num_chips, tp, pp, dp, ep, bs, pp_dcn, dp_dcn)

    # 2. Set output path
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="neusim_agent_")
    os.makedirs(output_dir, exist_ok=True)
    config["output_file_path"] = os.path.join(output_dir, f"inference-v{chip_version}.csv")

    # 3. Create ops generator
    is_deepseek = "deepseek" in model_name.lower()
    if is_deepseek:
        ops_gen = DeepSeekOpsGenerator(config)
    else:
        ops_gen = LLMOpsGeneratorInference(config)

    # 4. Run simulation (generate operators with performance stats)
    # Suppress verbose per-operator prints from NeuSIM
    with contextlib.redirect_stdout(io.StringIO()):
        ops, prefill_ops, decode_ops = ops_gen.generate(
            dump_to_file=True,
            separate_prefill_decode=True,
            analyze_energy=False,  # we run power analysis separately
        )

        # 5. Run power analysis on all operators
        from neusim.configs.chips.ChipConfig import ChipConfig
        chip_config_obj = ChipConfig.model_validate(config)

        for op in ops:
            power_lib.analyze_operator_energy(op, chip_config_obj, pg_config=pg_config)

    # 6. Get statistics from trace
    prefill_stats = run_sim_lib.get_statistics_from_trace_file(prefill_ops)
    decode_stats = run_sim_lib.get_statistics_from_trace_file(decode_ops)

    # 7. Compute TTFT and TPOT
    pp_val = config["pipeline_parallelism_degree"]
    pp_dcn_val = config["pipeline_parallel_degree_dcn"]
    input_seqlen = config["input_seqlen"]
    output_seqlen = config["output_seqlen"]

    prefill_pod_latency_ns = (
        prefill_stats["total_execution_time_non_pp_ns"]
        + prefill_stats["total_pp_ici_time_ns"]
    ) * pp_val
    prefill_tot_latency_ns = (
        prefill_pod_latency_ns + prefill_stats["total_pp_dcn_time_ns"]
    ) * pp_dcn_val
    TTFT_sec = prefill_tot_latency_ns / 1e9

    decode_pod_latency_ns = (
        decode_stats["total_execution_time_non_pp_ns"]
        + decode_stats["total_pp_ici_time_ns"]
    ) * pp_val / output_seqlen
    decode_tot_latency_ns = (
        decode_pod_latency_ns
        + decode_stats["total_pp_dcn_time_ns"] / output_seqlen
    ) * pp_dcn_val
    TPOT_ms_request = decode_tot_latency_ns / 1e6
    throughput_tokens_per_sec_request = 1e3 / TPOT_ms_request if TPOT_ms_request > 0 else 0

    # 8. Compute power metrics
    prefill_total_energy_J = sum(
        op.stats.total_energy_J for op in prefill_ops if op.stats.total_energy_J > 0
    )
    prefill_total_time_ns = sum(
        op.stats.execution_time_ns * op.stats.count for op in prefill_ops
    )
    decode_total_energy_J = sum(
        op.stats.total_energy_J for op in decode_ops if op.stats.total_energy_J > 0
    )
    decode_total_time_ns = sum(
        op.stats.execution_time_ns * op.stats.count for op in decode_ops
    )

    avg_power_W_prefill = (
        prefill_total_energy_J / (prefill_total_time_ns / 1e9)
        if prefill_total_time_ns > 0 else 0
    )
    avg_power_W_decode = (
        decode_total_energy_J / (decode_total_time_ns / 1e9)
        if decode_total_time_ns > 0 else 0
    )

    # Peak per-operator power
    max_power_W_prefill = max(
        (op.stats.total_power_W for op in prefill_ops if op.stats.total_power_W > 0),
        default=0,
    )
    max_power_W_decode = max(
        (op.stats.total_power_W for op in decode_ops if op.stats.total_power_W > 0),
        default=0,
    )

    # 9. Memory footprint
    mem_footprint_prefill_GB = ops_gen.compute_memory_footprint_bytes("prefill") / (1024**3)
    mem_footprint_decode_GB = ops_gen.compute_memory_footprint_bytes("decode") / (1024**3)
    hbm_size_GB = config["hbm_size_GB"]
    out_of_memory = (hbm_size_GB < mem_footprint_prefill_GB) or (hbm_size_GB < mem_footprint_decode_GB)

    # 10. Chip peak power from config
    total_power_peak_W = chip_config_obj.total_power_peak_W

    return {
        "TTFT_sec": TTFT_sec,
        "TPOT_ms_request": TPOT_ms_request,
        "throughput_tokens_per_sec_request": throughput_tokens_per_sec_request,
        "avg_power_W_prefill": avg_power_W_prefill,
        "avg_power_W_decode": avg_power_W_decode,
        "max_power_W_prefill": max_power_W_prefill,
        "max_power_W_decode": max_power_W_decode,
        "total_power_peak_W": total_power_peak_W,
        "mem_footprint_GB_prefill": mem_footprint_prefill_GB,
        "mem_footprint_GB_decode": mem_footprint_decode_GB,
        "hbm_size_GB": hbm_size_GB,
        "out_of_memory": out_of_memory,
        "chip_overrides": chip_overrides,
        "output_dir": output_dir,
        # raw stats for debugging
        "prefill_stats": prefill_stats,
        "decode_stats": decode_stats,
        # per-operator data for bottleneck analysis
        "prefill_ops": prefill_ops,
        "decode_ops": decode_ops,
    }


def format_results(results: dict[str, Any]) -> str:
    """Format simulation results into a readable string."""
    lines = [
        "=== NeuSIM Simulation Results ===",
        f"  TTFT (Time To First Token):     {results['TTFT_sec']:.4f} sec",
        f"  TPOT (Time Per Output Token):   {results['TPOT_ms_request']:.4f} ms/request",
        f"  Throughput:                      {results['throughput_tokens_per_sec_request']:.2f} tokens/sec/request",
        f"  Avg Power (prefill):             {results['avg_power_W_prefill']:.2f} W",
        f"  Avg Power (decode):              {results['avg_power_W_decode']:.2f} W",
        f"  Max Op Power (prefill):          {results['max_power_W_prefill']:.2f} W",
        f"  Max Op Power (decode):           {results['max_power_W_decode']:.2f} W",
        f"  Chip Peak Power (config):        {results['total_power_peak_W']:.2f} W",
        f"  Memory Footprint (prefill):      {results['mem_footprint_GB_prefill']:.2f} GB",
        f"  Memory Footprint (decode):       {results['mem_footprint_GB_decode']:.2f} GB",
        f"  HBM Size:                        {results['hbm_size_GB']} GB",
        f"  Out of Memory:                   {results['out_of_memory']}",
    ]
    return "\n".join(lines)
