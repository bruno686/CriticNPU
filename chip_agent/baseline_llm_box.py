"""Baseline: LLM-Box — two-agent LLM coordination for NPU DSE (no physics prior).

Two LLM agents collaborate each iteration:

  Proposer agent — given history + constraint status, proposes the next config.
  Critic   agent — given the Proposer's proposal + history, either accepts
                   (sends to simulator) or returns a one-line objection
                   asking the Proposer to revise.

The goal of the Critic is to push the Proposer away from rehashed or risky
proposals. No calibrated roofline or bottleneck digest is used — Critic and
Proposer reason only over raw simulator numbers from past iterations.

This baseline tests whether multi-agent LLM coordination (without physics
priors) is already enough for competitive NPU DSE.
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

warnings.filterwarnings("ignore")

DEFAULT_UPPER_BOUNDS = {
    "hbm_bw_GBps": 600.0,
    "hbm_size_GB": 16,
    "ici_bw_GBps": 125.0,
    "dcn_bw_GBps": 12.5,
}

SA_DIM_VALUES = [64, 128, 256]

PROPOSER_PROMPT = """You are a PROPOSER agent in a multi-agent NPU design system.

Your job: given the objective, deployment setting, power budget, and full history of simulated configurations with their TTFT/TPOT/power, propose ONE next configuration.

DESIGN SPACE:
- num_sa (systolic arrays): 2-16
- num_vu (vector units): 2-12
- sa_dim (SA side length): one of {64, 128, 256}
- freq_GHz: 0.5 - 2.0
- vmem_size_MB: 32 - 512

A CRITIC agent will review your proposal. If the critic accepts, your config is simulated; if not, you will see the critic's objection and must revise.

Propose diverse, well-reasoned configurations. Do not repeat prior configs. Do not propose configurations that will clearly exceed the power budget.
"""

CRITIC_PROMPT = """You are a CRITIC agent in a multi-agent NPU design system.

Your job: given the proposer's latest configuration together with the history of already-simulated configs, decide whether to ACCEPT the proposal for simulation or REJECT it with a one-line reason.

Reject when:
- The config duplicates (or is trivially close to) a config already in history.
- The config is very likely infeasible (e.g., far above the power budget based on the number of SAs, VUs, and clock frequency).
- The proposal makes no use of the lessons from previous simulated iterations.

Otherwise, ACCEPT. Be strict about duplication, but tolerant of exploration in under-sampled regions of the design space.
"""


PROPOSER_TOOL = [{
    "type": "function",
    "function": {
        "name": "propose_config",
        "description": "Propose the next NPU configuration.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "num_sa": {"type": "integer", "minimum": 2, "maximum": 16},
                "num_vu": {"type": "integer", "minimum": 2, "maximum": 12},
                "sa_dim": {"type": "integer", "enum": SA_DIM_VALUES},
                "freq_GHz": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                "vmem_size_MB": {"type": "integer", "minimum": 32, "maximum": 512},
            },
            "required": ["reasoning", "num_sa", "num_vu", "sa_dim", "freq_GHz", "vmem_size_MB"],
        },
    },
}]

CRITIC_TOOL = [{
    "type": "function",
    "function": {
        "name": "review_proposal",
        "description": "Accept or reject the proposer's configuration.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["accept", "reject"]},
                "reason": {"type": "string", "description": "One sentence explaining the decision."},
            },
            "required": ["decision", "reason"],
        },
    },
}]


def _history_block(history: list[dict[str, Any]]) -> str:
    if not history:
        return "History: (empty — first iteration)"
    lines = ["History (simulated so far):"]
    for i, h in enumerate(history, 1):
        cfg = h["config"]
        r = h["results"]
        status = "VALID" if h["is_valid"] else ("OOM" if r.get("oom") else f"over-budget @ {r['total_power_peak_W']:.0f}W")
        lines.append(
            f"  [{i}] num_sa={cfg['num_sa']}, num_vu={cfg['num_vu']}, sa_dim={cfg['sa_dim']}, "
            f"freq={cfg['freq_GHz']:.2f}GHz, vmem={cfg['vmem_size_MB']}MB  →  "
            f"TTFT={r['TTFT_sec']:.4f}s, TPOT={r['TPOT_ms_request']:.4f}ms, pwr={r['total_power_peak_W']:.0f}W [{status}]"
        )
    return "\n".join(lines)


def _context_block(model_name: str, num_chips: int, tp: int, pp: int, bs: int,
                   power_budget_W: float, focus: str,
                   upper_bounds: dict[str, float]) -> str:
    return "\n".join([
        f"Workload: {model_name} inference",
        f"Deployment: {num_chips} chips, TP={tp}, PP={pp}, batch size={bs}",
        f"Power budget: {power_budget_W:.0f} W peak per chip",
        f"Fixed: HBM BW={upper_bounds['hbm_bw_GBps']:.0f}, HBM size={upper_bounds['hbm_size_GB']:.0f}GB, ICI={upper_bounds['ici_bw_GBps']:.0f}, DCN={upper_bounds['dcn_bw_GBps']:.1f}",
        f"Objective: {focus}",
    ])


def _proposer(client: OpenAI, model: str, context: str, history: str,
              revision_note: str | None = None) -> dict[str, Any]:
    user = context + "\n\n" + history
    if revision_note:
        user += f"\n\nCRITIC objection on your last proposal: {revision_note}\nRevise your proposal."
    else:
        user += "\n\nPropose the next configuration."

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": PROPOSER_PROMPT},
                  {"role": "user", "content": user}],
        tools=PROPOSER_TOOL,
        tool_choice={"type": "function", "function": {"name": "propose_config"}},
    )
    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)


def _critic(client: OpenAI, model: str, context: str, history: str, proposal: dict[str, Any]) -> dict[str, Any]:
    proposal_str = (
        f"Proposer's proposal: num_sa={proposal['num_sa']}, num_vu={proposal['num_vu']}, "
        f"sa_dim={proposal['sa_dim']}, freq={proposal['freq_GHz']:.2f}GHz, vmem={proposal['vmem_size_MB']}MB.\n"
        f"Proposer's reasoning: {proposal.get('reasoning', '(none)')}"
    )
    user = context + "\n\n" + history + "\n\n" + proposal_str + "\n\nReview and decide."
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": CRITIC_PROMPT},
                  {"role": "user", "content": user}],
        tools=CRITIC_TOOL,
        tool_choice={"type": "function", "function": {"name": "review_proposal"}},
    )
    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)


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


def run_llm_box_search(
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
    max_critic_rejections: int = 3,
    openai_model: str = "gpt-5.4-mini",
) -> dict[str, Any]:
    if upper_bounds is None:
        upper_bounds = dict(DEFAULT_UPPER_BOUNDS)

    client = OpenAI()
    base_config = load_base_configs(model_name, chip_version)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "prefill" if tpot_weight == 0 else ("decode" if ttft_weight == 0 else "combined")
    run_dir = os.path.join(output_dir, f"{model_name}_llm_box_{label}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    focus_str = "Minimize TTFT (prefill)" if tpot_weight == 0 else (
        "Minimize TPOT (decode)" if ttft_weight == 0 else "Minimize weighted TTFT + TPOT"
    )

    print(f"\n{'='*60}")
    print(f"LLM-Box — {model_name} on {num_chips} chips (TP={tp}, PP={pp})")
    print(f"Optimization: {focus_str}")
    print(f"Power budget: {power_budget_W}W  |  Simulation budget: {n_iterations}")
    print(f"{'='*60}\n")

    context = _context_block(model_name, num_chips, tp, pp, bs, power_budget_W, focus_str, upper_bounds)
    history: list[dict[str, Any]] = []
    seen: set[str] = set()
    best_result: dict[str, Any] | None = None
    best_score = float("inf")
    sim_count = 0

    for i in range(1, n_iterations + 1):
        hist_block = _history_block(history)
        revision_note = None
        proposal = None

        # Proposer / Critic rounds
        for r_idx in range(max_critic_rejections):
            try:
                proposal = _proposer(client, openai_model, context, hist_block, revision_note)
            except Exception as e:
                print(f"  [iter {i:02d}] proposer ERROR: {e}")
                proposal = None
                break

            try:
                verdict = _critic(client, openai_model, context, hist_block, proposal)
            except Exception as e:
                print(f"  [iter {i:02d}] critic ERROR: {e} — accepting by default")
                verdict = {"decision": "accept", "reason": "(critic errored)"}

            if verdict["decision"] == "accept":
                break
            revision_note = verdict.get("reason", "(none)")
            print(f"  [iter {i:02d}] critic REJECTED (round {r_idx+1}): {revision_note}")

        if proposal is None:
            continue

        config = _extract_config(proposal, upper_bounds)
        sig = _config_sig(config)
        if sig in seen:
            print(f"  [iter {i:02d}] DUP after critic — skipping")
            continue
        seen.add(sig)

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
            print(f"  [iter {i:02d}] sim ERROR: {e}")
            continue

        is_valid = not results["out_of_memory"] and results["total_power_peak_W"] <= power_budget_W
        status = "VALID" if is_valid else ("OOM" if results["out_of_memory"] else f"PWR={results['total_power_peak_W']:.0f}W")

        entry = {
            "iter": i,
            "config": config,
            "proposer_reasoning": proposal.get("reasoning", ""),
            "results": {
                "TTFT_sec": results["TTFT_sec"],
                "TPOT_ms_request": results["TPOT_ms_request"],
                "total_power_peak_W": results["total_power_peak_W"],
                "oom": results["out_of_memory"],
            },
            "is_valid": is_valid,
        }
        history.append(entry)

        print(
            f"  [iter {i:02d}] num_sa={config['num_sa']:2d} num_vu={config['num_vu']:2d} "
            f"sa={config['sa_dim']:3d} f={config['freq_GHz']:.2f} vmem={config['vmem_size_MB']:3d}  →  "
            f"TTFT={results['TTFT_sec']:.4f}s TPOT={results['TPOT_ms_request']:.4f}ms "
            f"Pwr={results['total_power_peak_W']:.0f}W  {status}"
        )

        if is_valid:
            score = ttft_weight * results["TTFT_sec"] + tpot_weight * (results["TPOT_ms_request"] / 1000.0)
            if score < best_score:
                best_score = score
                best_result = entry

    if best_result is None:
        print("\nNo valid configurations found.")
        return {"best_result": None, "history": history, "sim_count": sim_count}

    r = best_result["results"]
    print(f"\n{'='*60}\n  LLM-BOX BEST RESULT\n{'='*60}")
    print(f"Best iteration: {best_result['iter']}")
    print(f"Config: {json.dumps({k: best_result['config'][k] for k in ['num_sa','num_vu','sa_dim','freq_GHz','vmem_size_MB']})}")
    print(f"TTFT={r['TTFT_sec']:.4f}s  TPOT={r['TPOT_ms_request']:.4f}ms  Pwr={r['total_power_peak_W']:.0f}W")
    print(f"Combined: TTFT={r['TTFT_sec']:.4f}s, TPOT={r['TPOT_ms_request']:.4f}ms")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"best_result": best_result, "history": history, "sim_count": sim_count}, f, indent=2)

    return {"best_result": best_result, "history": history, "sim_count": sim_count}


def main():
    parser = argparse.ArgumentParser(description="LLM-Box baseline (two-agent LLM, no physics prior)")
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
    parser.add_argument("--max-critic-rejections", type=int, default=3)
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
        max_critic_rejections=args.max_critic_rejections,
        openai_model=args.openai_model,
    )

    if args.disaggregated:
        print("\n>>> LLM-Box: PREFILL optimization <<<")
        prefill = run_llm_box_search(**common, ttft_weight=1.0, tpot_weight=0.0)
        print("\n>>> LLM-Box: DECODE optimization <<<")
        decode = run_llm_box_search(**common, ttft_weight=0.0, tpot_weight=1.0)

        print(f"\n{'='*60}\n  LLM-BOX DISAGGREGATED SUMMARY\n{'='*60}")
        pb = prefill.get("best_result")
        db = decode.get("best_result")
        if pb:
            r = pb["results"]
            print(f"PREFILL (iter {pb['iter']}): TTFT={r['TTFT_sec']:.4f}s, Pwr={r['total_power_peak_W']:.0f}W")
        if db:
            r = db["results"]
            print(f"DECODE (iter {db['iter']}): TPOT={r['TPOT_ms_request']:.4f}ms, Pwr={r['total_power_peak_W']:.0f}W")
        if pb and db:
            print(f"\nCombined: TTFT={pb['results']['TTFT_sec']:.4f}s, TPOT={db['results']['TPOT_ms_request']:.4f}ms")
    else:
        run_llm_box_search(**common, ttft_weight=args.ttft_weight, tpot_weight=args.tpot_weight)


if __name__ == "__main__":
    main()
