"""Launch all 480 experiments: 6 models × 16 configs × 5 methods.

Baselines run in parallel (no API). Agent uses gpt-5.4-mini.
Results are saved to experiment_480/results/ with structured naming.
After all runs complete, call collect_results() to generate tables.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUTDIR = Path("experiment_480")
LOGDIR = OUTDIR / "logs"
RESDIR = OUTDIR / "results"
HBM_SIZE = 80

MODELS = [
    {"name": "llama3-8b",       "chips": 8,  "group": "small"},
    {"name": "llama2-13b",      "chips": 8,  "group": "small"},
    {"name": "llama3-70b",      "chips": 64, "group": "large"},
    {"name": "llama3_1-405b",   "chips": 64, "group": "large"},
    {"name": "deepseekv2-236b", "chips": 16, "group": "dsv2"},
    {"name": "deepseekv3-671b", "chips": 64, "group": "large"},
]

CONFIGS_SMALL = [
    {"id": "C1",  "tp":2, "pp":4, "bs":16, "pw":250},
    {"id": "C2",  "tp":2, "pp":4, "bs":16, "pw":300},
    {"id": "C3",  "tp":2, "pp":4, "bs":16, "pw":400},
    {"id": "C4",  "tp":4, "pp":2, "bs":16, "pw":250},
    {"id": "C5",  "tp":4, "pp":2, "bs":16, "pw":300},
    {"id": "C6",  "tp":4, "pp":2, "bs":16, "pw":400},
    {"id": "C7",  "tp":8, "pp":1, "bs":16, "pw":250},
    {"id": "C8",  "tp":8, "pp":1, "bs":16, "pw":300},
    {"id": "C9",  "tp":8, "pp":1, "bs":16, "pw":400},
    {"id": "C10", "tp":4, "pp":2, "bs":1,  "pw":300},
    {"id": "C11", "tp":4, "pp":2, "bs":32, "pw":300},
    {"id": "C12", "tp":4, "pp":2, "bs":48, "pw":300},
    {"id": "C13", "tp":2, "pp":4, "bs":1,  "pw":300},
    {"id": "C14", "tp":8, "pp":1, "bs":1,  "pw":300},
    {"id": "C15", "tp":4, "pp":2, "bs":16, "pw":350},
    {"id": "C16", "tp":2, "pp":4, "bs":16, "pw":350},
]

CONFIGS_LARGE = [
    {"id": "C1",  "tp":4,  "pp":16, "bs":16, "pw":250},
    {"id": "C2",  "tp":4,  "pp":16, "bs":16, "pw":300},
    {"id": "C3",  "tp":4,  "pp":16, "bs":16, "pw":400},
    {"id": "C4",  "tp":8,  "pp":8,  "bs":16, "pw":250},
    {"id": "C5",  "tp":8,  "pp":8,  "bs":16, "pw":300},
    {"id": "C6",  "tp":8,  "pp":8,  "bs":16, "pw":400},
    {"id": "C7",  "tp":16, "pp":4,  "bs":16, "pw":250},
    {"id": "C8",  "tp":16, "pp":4,  "bs":16, "pw":300},
    {"id": "C9",  "tp":16, "pp":4,  "bs":16, "pw":400},
    {"id": "C10", "tp":8,  "pp":8,  "bs":1,  "pw":300},
    {"id": "C11", "tp":8,  "pp":8,  "bs":32, "pw":300},
    {"id": "C12", "tp":8,  "pp":8,  "bs":48, "pw":300},
    {"id": "C13", "tp":4,  "pp":16, "bs":1,  "pw":300},
    {"id": "C14", "tp":16, "pp":4,  "bs":1,  "pw":300},
    {"id": "C15", "tp":8,  "pp":8,  "bs":16, "pw":350},
    {"id": "C16", "tp":4,  "pp":16, "bs":16, "pw":350},
]

CONFIGS_DSV2 = [
    {"id": "C1",  "tp":2, "pp":8, "bs":16, "pw":250},
    {"id": "C2",  "tp":2, "pp":8, "bs":16, "pw":300},
    {"id": "C3",  "tp":2, "pp":8, "bs":16, "pw":400},
    {"id": "C4",  "tp":4, "pp":4, "bs":16, "pw":250},
    {"id": "C5",  "tp":4, "pp":4, "bs":16, "pw":300},
    {"id": "C6",  "tp":4, "pp":4, "bs":16, "pw":400},
    {"id": "C7",  "tp":8, "pp":2, "bs":16, "pw":250},
    {"id": "C8",  "tp":8, "pp":2, "bs":16, "pw":300},
    {"id": "C9",  "tp":8, "pp":2, "bs":16, "pw":400},
    {"id": "C10", "tp":4, "pp":4, "bs":1,  "pw":300},
    {"id": "C11", "tp":4, "pp":4, "bs":32, "pw":300},
    {"id": "C12", "tp":4, "pp":4, "bs":48, "pw":300},
    {"id": "C13", "tp":2, "pp":8, "bs":1,  "pw":300},
    {"id": "C14", "tp":8, "pp":2, "bs":1,  "pw":300},
    {"id": "C15", "tp":4, "pp":4, "bs":16, "pw":350},
    {"id": "C16", "tp":2, "pp":8, "bs":16, "pw":350},
]

BASELINES = ["baseline_bo", "baseline_es", "baseline_greedy", "baseline_grid"]


def get_configs(model):
    group = next(m["group"] for m in MODELS if m["name"] == model)
    if group == "small":
        return CONFIGS_SMALL
    elif group == "dsv2":
        return CONFIGS_DSV2
    else:
        return CONFIGS_LARGE


def get_chips(model):
    return next(m["chips"] for m in MODELS if m["name"] == model)


def run_baseline(model, chips, cfg, method):
    c = cfg
    tag = f"{model}_{method}_{c['id']}"
    logfile = LOGDIR / f"{tag}.log"

    cmd = [
        sys.executable, f"{method}.py", model,
        "--chip-version", "4",
        "--num-chips", str(chips),
        "--tp", str(c["tp"]),
        "--pp", str(c["pp"]),
        "--bs", str(c["bs"]),
        "--power-budget", str(c["pw"]),
        "--max-hbm-size", str(HBM_SIZE),
        "--disaggregated",
        "--output-dir", str(RESDIR),
    ]

    with open(logfile, "w") as lf:
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)


def run_agent(model, chips, cfg):
    c = cfg
    tag = f"{model}_agent_{c['id']}"
    logfile = LOGDIR / f"{tag}.log"

    cmd = [
        sys.executable, "main.py", model,
        "--tp", str(c["tp"]),
        "--pp", str(c["pp"]),
        "--bs", str(c["bs"]),
        "--num-chips", str(chips),
        "--power-budget", str(c["pw"]),
        "--max-hbm-size", str(HBM_SIZE),
        "--disaggregated",
        "--skip-agent1",
        "--no-memory",
        "--max-iterations", "8",
        "--openai-model", "gpt-5.4-mini",
        "--output-dir", str(RESDIR),
    ]

    with open(logfile, "w") as lf:
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)


def launch_all():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    RESDIR.mkdir(parents=True, exist_ok=True)

    procs = []

    # Launch all baselines first (no API, can fully parallelize)
    print("=== Launching baselines ===")
    for model_info in MODELS:
        model = model_info["name"]
        chips = model_info["chips"]
        configs = get_configs(model)

        for cfg in configs:
            for method in BASELINES:
                p = run_baseline(model, chips, cfg, method)
                procs.append((f"{model}_{method}_{cfg['id']}", p))

    print(f"Launched {len(procs)} baseline jobs")

    # Launch agents (need API, gpt-5.4-mini)
    print("=== Launching agents ===")
    agent_procs = []
    for model_info in MODELS:
        model = model_info["name"]
        chips = model_info["chips"]
        configs = get_configs(model)

        for cfg in configs:
            p = run_agent(model, chips, cfg)
            agent_procs.append((f"{model}_agent_{cfg['id']}", p))

    procs.extend(agent_procs)
    print(f"Launched {len(agent_procs)} agent jobs")
    print(f"Total: {len(procs)} jobs")
    print(f"Monitor: tail -f {LOGDIR}/*.log")

    # Wait for all
    print("\nWaiting for all jobs to complete...")
    start = time.time()
    completed = 0
    for tag, p in procs:
        p.wait()
        completed += 1
        if completed % 50 == 0:
            elapsed = time.time() - start
            print(f"  {completed}/{len(procs)} complete ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - start
    print(f"\n=== All {len(procs)} experiments complete in {elapsed:.0f}s ===")


def collect_results():
    """Parse all log files and generate result tables."""
    results = {}  # results[config_id][model][method] = {"ttft": ..., "tpot": ..., "sims": ...}

    for logfile in sorted(LOGDIR.glob("*.log")):
        name = logfile.stem  # e.g., llama3-8b_baseline_bo_C5
        parts = name.split("_")

        # Parse model, method, config
        # Find config ID (starts with C)
        config_id = None
        for p in parts:
            if p.startswith("C") and p[1:].isdigit():
                config_id = p
                break

        if not config_id:
            continue

        # Method
        if "agent" in name:
            method = "NPUAgent"
        elif "baseline_bo" in name:
            method = "BO"
        elif "baseline_es" in name:
            method = "CMA-ES"
        elif "baseline_greedy" in name:
            method = "Greedy"
        elif "baseline_grid" in name:
            method = "Grid"
        else:
            continue

        # Model (everything before the method)
        method_idx = name.find("_baseline_") if "_baseline_" in name else name.find("_agent_")
        model = name[:method_idx]

        # Parse results from log
        text = logfile.read_text()
        ttft = tpot = sims_prefill = sims_decode = None

        # Look for Combined: line
        import re
        m = re.search(r"TTFT=([0-9.]+)s.*TPOT=([0-9.]+)ms", text)
        if m:
            ttft = float(m.group(1))
            tpot = float(m.group(2))

        # Or separate Prefill/Decode lines
        if ttft is None:
            m_p = re.search(r"Prefill.*TTFT=([0-9.]+)", text)
            m_d = re.search(r"Decode.*TPOT=([0-9.]+)", text)
            if m_p:
                ttft = float(m_p.group(1))
            if m_d:
                tpot = float(m_d.group(1))

        # Count sims
        sims = text.count("Simulation Results") or text.count("→")

        if config_id not in results:
            results[config_id] = {}
        if model not in results[config_id]:
            results[config_id][model] = {}

        results[config_id][model][method] = {
            "ttft": ttft,
            "tpot": tpot,
            "sims": sims,
        }

    # Save raw results
    with open(OUTDIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Collected results for {len(results)} configs")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        collect_results()
    else:
        launch_all()
