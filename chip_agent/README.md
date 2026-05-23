# CriticNPU: Roofline-Guided LLM Agent for NPU Architecture Design

CriticNPU is a multi-agent LLM system that automatically designs NPU
architectures for LLM inference. It uses a calibrated roofline world model
to guide GPT-driven design decisions, then validates with NeuSIM.

This repository contains the **agent code** and **all baseline implementations**
used in the paper. Experiment outputs (`experiment_480/`, design caches, etc.)
are excluded from the repo to keep it lightweight — re-running the scripts
will regenerate them.

## Repository layout

```
chip_agent/
├── main.py                          Orchestrates the full 3-agent pipeline
│
├── Core agents
├── computedemandagent.py            Agent 1: extract per-operator compute demand
├── npudesignagent.py                Agent 3: roofline-guided NPU designer (CriticNPU)
├── bottleneck_analysis.py           Agent 2: one-sentence bottleneck diagnosis
├── world_model.py                   Calibrated roofline world model
├── design_memory.py                 Episodic + semantic cross-workload memory
├── neusim_runner.py                 Programmatic NeuSIM bridge
│
├── Baselines (7)
├── baseline_bo.py                   Bayesian Optimization (GP + EI)
├── baseline_es.py                   CMA-ES
├── baseline_greedy.py               Greedy coordinate descent
├── baseline_grid.py                 Exhaustive grid search
├── baseline_chatdse.py              ChatDSE — LLM as zero-shot surrogate
├── baseline_llm_dse.py              LLM-DSE — single-agent LLM in the loop
├── baseline_llm_box.py              LLM-Box — two-agent LLM (proposer + critic)
│
├── Runners & table generation
├── run_all_experiments.py           Launches 6 models × 16 configs × classical baselines
├── run_llm_baselines.py             Launches LLM-based baselines with rate limiting
└── generate_tables.py               Parses logs → main LaTeX win-rate table + 16 subtables
```

## Setup

```bash
# 1. Install NeuSIM in editable mode (from the NeuSim/ directory, sibling repo)
cd ../NeuSim && pip install -e ".[dev]"

# 2. Install agent dependencies
pip install pandas openai python-dotenv matplotlib scipy scikit-learn numpy

# 3. Configure OpenAI access
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

Python 3.12 required.

## Run the full agent pipeline (one workload, one deployment)

```bash
python main.py llama3-8b \
    --tp 4 --pp 2 --bs 16 \
    --num-chips 8 \
    --power-budget 300 \
    --max-hbm-size 80 \
    --disaggregated
```

Important flags:
- `--disaggregated`: design a separate prefill chip and a separate decode chip
- `--skip-agent1`: reuse cached compute-demand CSVs if available
- `--no-memory`: disable cross-run memory (useful for fair benchmark runs)
- `--max-iterations 8`: cap LLM iterations (default for paper experiments)

## Run a single baseline

All baselines share the same CLI:

```bash
python baseline_bo.py llama3-8b \
    --chip-version 4 --num-chips 8 \
    --tp 4 --pp 2 --bs 16 \
    --power-budget 300 --max-hbm-size 80 \
    --disaggregated
```

Replace `baseline_bo.py` with any of `baseline_es.py`, `baseline_greedy.py`,
`baseline_grid.py`, `baseline_chatdse.py`, `baseline_llm_dse.py`, `baseline_llm_box.py`.

## Reproduce the paper's main experiment (6 models × 16 deployments × 8 methods)

```bash
# Classical baselines (no API calls) — fully parallel
python run_all_experiments.py

# LLM-based baselines (gpt-5.4-mini) — throttled to 12 concurrent jobs
python run_llm_baselines.py \
    --methods chatdse,llm_dse,llm_box,agent \
    --openai-model gpt-5.4-mini \
    --max-workers 12

# Aggregate logs into the main win-rate LaTeX table + 16 subtables
python generate_tables.py
# Output: ../current_result.tex
```

Total budget: ~768 runs, each running `--disaggregated` (prefill + decode).
Expect several hours of wall time and substantial OpenAI API spend.

## Search-space constraints (paper Table 1)

Every method optimizes a 5-parameter NPU configuration
(`num_sa`, `num_vu`, `sa_dim`, `freq_GHz`, `vmem_size_MB`) under five constraints:

| Constraint | Value | Type |
|---|---|---|
| Power budget | 250 / 300 / 350 / 400 W (per deployment) | hard |
| HBM capacity | ≤ 80 GB | hard (OOM) |
| HBM bandwidth | ≤ 600 GB/s | upper bound |
| ICI bandwidth | ≤ 125 GB/s | upper bound |
| DCN bandwidth | ≤ 12.5 GB/s | upper bound |

## Citation

```
@inproceedings{criticnpu2026,
  title  = {Learning a Roofline World Model for NPU Architecture Design},
  author = {Anonymous},
  year   = {2026},
  booktitle = {Submitted to NeurIPS 2026}
}
```

## License

[Add your license here — e.g. MIT, Apache-2.0]
