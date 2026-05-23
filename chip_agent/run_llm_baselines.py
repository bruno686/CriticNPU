"""Launch LLM-based baselines across 6 models × 16 configs.

Covers three new methods: ChatDSE, LLM-DSE, LLM-Box.
Also optionally re-runs NPUAgent (main.py) under the unified gpt-5.4-mini
setting. All jobs write logs into experiment_480/logs/ with names that
generate_tables.py can parse out of the box.

Concurrency is throttled (default: 12 simultaneous API-using jobs) to stay
under provider rate limits. Each method gets its own worker pool per model
slice so failures in one method do not block others.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
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

# Mapping method-name (as used in log filenames) → script path.
#   The log pattern determines how generate_tables.py classifies the run:
#   "_chatdse_" / "_llm_dse_" / "_llm_box_" / "_agent_".
METHOD_SCRIPTS = {
    "chatdse":  "baseline_chatdse.py",
    "llm_dse":  "baseline_llm_dse.py",
    "llm_box":  "baseline_llm_box.py",
    "agent":    "main.py",
}


def get_configs(model):
    group = next(m["group"] for m in MODELS if m["name"] == model)
    if group == "small":
        return CONFIGS_SMALL
    if group == "dsv2":
        return CONFIGS_DSV2
    return CONFIGS_LARGE


def get_chips(model):
    return next(m["chips"] for m in MODELS if m["name"] == model)


def build_cmd(method: str, model: str, chips: int, cfg: dict, openai_model: str) -> list[str]:
    script = METHOD_SCRIPTS[method]
    c = cfg

    base = [
        sys.executable, script, model,
        "--num-chips", str(chips),
        "--tp", str(c["tp"]),
        "--pp", str(c["pp"]),
        "--bs", str(c["bs"]),
        "--power-budget", str(c["pw"]),
        "--max-hbm-size", str(HBM_SIZE),
        "--disaggregated",
        "--output-dir", str(RESDIR),
        "--openai-model", openai_model,
    ]

    if method == "agent":
        base += [
            "--skip-agent1",
            "--no-memory",
            "--max-iterations", "8",
        ]
    else:
        base += ["--n-iterations", "60", "--chip-version", "4"]

    return base


def job_logfile(method: str, model: str, cfg: dict) -> Path:
    # Name pattern must match generate_tables.py both the method classifier
    # (substrings like "_chatdse_") AND the config-id regex
    # "tp(\d+)_pp(\d+)_bs(\d+)_(\d+)W". Keep tp/pp/bs/pw suffix for parity
    # with existing baseline_bo / baseline_es log names.
    c = cfg
    suffix = f"tp{c['tp']}_pp{c['pp']}_bs{c['bs']}_{c['pw']}W"
    if method == "agent":
        tag = f"{model}_agent_{c['id']}_{suffix}"
    elif method == "chatdse":
        tag = f"{model}_chatdse_{c['id']}_{suffix}"
    elif method == "llm_dse":
        tag = f"{model}_llm_dse_{c['id']}_{suffix}"
    elif method == "llm_box":
        tag = f"{model}_llm_box_{c['id']}_{suffix}"
    else:
        tag = f"{model}_{method}_{c['id']}_{suffix}"
    return LOGDIR / f"{tag}.log"


def run_concurrent(jobs: list[tuple[str, list[str], Path]], max_workers: int) -> None:
    """Run jobs with a bounded worker pool. jobs = list of (tag, cmd, logfile)."""
    sem = threading.Semaphore(max_workers)
    threads: list[threading.Thread] = []
    start = time.time()
    done = {"n": 0}
    lock = threading.Lock()

    def _run(tag, cmd, logfile):
        with sem:
            with open(logfile, "w") as lf:
                p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
                p.wait()
            with lock:
                done["n"] += 1
                if done["n"] % 10 == 0 or done["n"] == len(jobs):
                    elapsed = time.time() - start
                    print(f"  {done['n']}/{len(jobs)} done  ({elapsed:.0f}s elapsed)")

    for tag, cmd, logfile in jobs:
        t = threading.Thread(target=_run, args=(tag, cmd, logfile), daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def launch(methods: list[str], models_filter: list[str] | None,
           configs_filter: list[str] | None, openai_model: str,
           max_workers: int, skip_existing: bool) -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    RESDIR.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, list[str], Path]] = []
    for model_info in MODELS:
        model = model_info["name"]
        if models_filter and model not in models_filter:
            continue
        chips = model_info["chips"]
        for cfg in get_configs(model):
            if configs_filter and cfg["id"] not in configs_filter:
                continue
            for method in methods:
                logfile = job_logfile(method, model, cfg)
                if skip_existing and logfile.exists() and logfile.stat().st_size > 0:
                    continue
                cmd = build_cmd(method, model, chips, cfg, openai_model)
                jobs.append((logfile.stem, cmd, logfile))

    if not jobs:
        print("No jobs to run (filters excluded everything or all logs already exist).")
        return

    print(f"=== Launching {len(jobs)} LLM jobs across {len(methods)} methods ===")
    print(f"Methods: {', '.join(methods)}")
    print(f"Model: {openai_model}")
    print(f"Concurrency: {max_workers}")
    print(f"Log dir: {LOGDIR}")
    print()

    run_concurrent(jobs, max_workers)
    print(f"\n=== Done: {len(jobs)} runs complete ===")


def main():
    parser = argparse.ArgumentParser(description="Run LLM-based baselines on the 16×6 grid")
    parser.add_argument("--methods", type=str,
                        default="chatdse,llm_dse,llm_box,agent",
                        help="Comma-separated methods to run (subset of: chatdse,llm_dse,llm_box,agent)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated models to restrict to (default: all 6)")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config IDs to restrict to (default: all 16)")
    parser.add_argument("--openai-model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--max-workers", type=int, default=12,
                        help="Max concurrent API-using jobs (default: 12)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a run if its log already exists and is non-empty")

    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_SCRIPTS]
    if unknown:
        sys.exit(f"Unknown methods: {unknown}. Valid: {list(METHOD_SCRIPTS)}")

    models_filter = [s.strip() for s in args.models.split(",")] if args.models else None
    configs_filter = [s.strip() for s in args.configs.split(",")] if args.configs else None

    launch(methods, models_filter, configs_filter,
           args.openai_model, args.max_workers, args.skip_existing)


if __name__ == "__main__":
    main()
