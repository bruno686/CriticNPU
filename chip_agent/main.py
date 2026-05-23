"""Multi-Agent NPU Design Pipeline.

Runs the full pipeline:
  Agent 1 (compute demand extraction) → Agent 2 (NPU architecture design)

Agent 1 extracts per-operator compute demand from the model config,
then Agent 2 uses that workload profile to design the optimal NPU chip.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

NEUSIM_CONFIGS = Path(__file__).resolve().parent.parent / "NeuSim" / "configs" / "models"


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent NPU Design: Agent 1 (compute demand) → Agent 2 (NPU design)"
    )
    parser.add_argument(
        "model", type=str,
        help="Model name (e.g., llama3-8b, llama2-13b, deepseekv2-236b)",
    )
    # Shared parallelism args
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallelism degree")
    parser.add_argument("--pp", type=int, default=2, help="Pipeline parallelism degree")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism degree")
    parser.add_argument("--bs", type=int, default=16, help="Global batch size")
    parser.add_argument("--pp-dcn", type=int, default=1, help="Pipeline parallel degree DCN")
    parser.add_argument("--dp-dcn", type=int, default=1, help="Data parallel degree DCN")
    # Agent 2 specific args
    parser.add_argument("--chip-version", type=str, default="4", help="Base chip version")
    parser.add_argument("--num-chips", type=int, default=8, help="Number of NPU chips")
    parser.add_argument("--power-budget", type=float, default=300.0, help="Power budget in Watts")
    parser.add_argument("--max-iterations", type=int, default=50, help="Max design iterations")
    parser.add_argument("--convergence-threshold", type=float, default=0.02)
    parser.add_argument("--convergence-window", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="design_results")
    parser.add_argument("--openai-model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--max-hbm-bw", type=float, default=600.0)
    parser.add_argument("--max-hbm-size", type=float, default=16)
    parser.add_argument("--max-ici-bw", type=float, default=125.0)
    parser.add_argument("--max-dcn-bw", type=float, default=12.5)
    parser.add_argument("--ttft-weight", type=float, default=0.5)
    parser.add_argument("--tpot-weight", type=float, default=0.5)
    parser.add_argument("--disaggregated", action="store_true",
                        help="Optimize separate prefill and decode chips")
    # Skip Agent 1 if compute demand already exists
    parser.add_argument("--skip-agent1", action="store_true",
                        help="Skip Agent 1 and reuse existing compute demand CSVs")
    # Memory
    parser.add_argument("--memory-dir", type=str, default="design_memory",
                        help="Directory for cross-workload design memory (default: design_memory)")
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable cross-workload design memory")

    args = parser.parse_args()

    model_config = NEUSIM_CONFIGS / f"{args.model}.json"
    if not model_config.exists():
        print(f"Error: Model config not found: {model_config}")
        available = [f.stem for f in NEUSIM_CONFIGS.glob("*.json")]
        print(f"Available models: {', '.join(sorted(available))}")
        sys.exit(1)

    # ── Step 1: Agent 1 — Compute Demand Extraction ──────────────────────
    print("=" * 60)
    print("  Step 1: Agent 1 — Computing Demand Extraction")
    print("=" * 60)

    from computedemandagent import ComputeDemandAgent

    agent1 = ComputeDemandAgent(
        model_config_path=str(model_config),
        tp=args.tp, pp=args.pp, dp=args.dp, ep=args.ep, bs=args.bs,
    )
    compute_demand_dir = str(agent1.output_dir)

    if args.skip_agent1:
        prefill_csv = agent1.output_dir / "inference_prefill.csv"
        decode_csv = agent1.output_dir / "inference_decode.csv"
        if prefill_csv.exists() and decode_csv.exists():
            print(f"Reusing existing compute demand: {compute_demand_dir}")
        else:
            print("Existing CSVs not found, running Agent 1...")
            agent1.run()
    else:
        agent1.run()

    # ── Step 2: Agent 2 — NPU Architecture Design ───────────────────────
    print()
    print("=" * 60)
    print("  Step 2: Agent 2 — NPU Architecture Design")
    print(f"  (using compute demand from: {compute_demand_dir})")
    print("=" * 60)

    from npudesignagent import run_design_loop, run_disaggregated
    from design_memory import DesignMemory

    upper_bounds = {
        "hbm_bw_GBps": args.max_hbm_bw,
        "hbm_size_GB": args.max_hbm_size,
        "ici_bw_GBps": args.max_ici_bw,
        "dcn_bw_GBps": args.max_dcn_bw,
    }

    memory = None if args.no_memory else DesignMemory(args.memory_dir)

    if args.disaggregated:
        # Attach compute_demand_dir to args for run_disaggregated
        args.compute_demand_dir = compute_demand_dir
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
            compute_demand_dir=compute_demand_dir,
            memory=memory,
        )


if __name__ == "__main__":
    main()
