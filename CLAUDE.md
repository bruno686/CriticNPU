# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains two projects:

- **NeuSim** (`NeuSim/`): An open-source simulator for modeling performance, power, and carbon behavior of Neural Processing Units (NPUs) running ML workloads. Supports TPUv2-v6p hardware configs and models including LLMs (Llama, DeepSeek), DLRM, DiT, and GLIGEN.
- **chip_agent** (`chip_agent/`): A multi-agent system (3 GPT-4o agents) that automatically designs optimal NPU chip configurations for LLM inference, using NeuSim as its evaluation engine. Matches or beats Bayesian Optimization, CMA-ES, and Greedy baselines with ~50% fewer simulations.

## Build & Development Commands

### NeuSim

```bash
# Install (from NeuSim/ directory)
pip install -e .          # runtime only
pip install -e ".[dev]"   # with dev dependencies (pytest, ruff, mypy)

# Run all tests
cd NeuSim && pytest

# Run a single test file
cd NeuSim && pytest neusim/configs/tests/test_llm_config.py

# Run a single test
cd NeuSim && pytest neusim/configs/tests/test_llm_config.py::test_function_name

# Lint (from NeuSim/ directory)
./lint.sh

# Lint manually
ruff check neusim/        # lint checks
ruff format --check neusim/  # format check
mypy neusim/              # type checking (non-blocking)
```

Python version: 3.12.2

### chip_agent

```bash
pip install pandas openai python-dotenv matplotlib

# Full pipeline (recommended) — from chip_agent/ directory
python main.py llama3-8b --tp 4 --pp 2 --bs 16 --num-chips 8 --power-budget 300 --disaggregated

# Skip Agent 1 (reuse existing compute demand CSVs)
python main.py llama3-8b --tp 4 --pp 2 --bs 16 --num-chips 8 --power-budget 300 --skip-agent1

# Run Agent 1 standalone
python computedemandagent.py ../NeuSim/configs/models/llama3-8b.json --tp 4 --pp 2 --bs 16

# Run baselines (same interface: baseline_bo.py, baseline_es.py, baseline_greedy.py)
python baseline_grid.py llama3-8b --chip-version 4 --num-chips 8 --tp 4 --pp 2 --bs 16 --power-budget 300 --disaggregated
```

Requires `chip_agent/.env` file with `OPENAI_API_KEY`.

## Architecture

### NeuSim Simulation Pipeline

```
Model Config (JSON) → Ops Generator → Operator List (FLOPS/memory traffic)
    → Backend Simulator → Performance Results
    → Power Analysis → Energy + Carbon Footprint
    → SLO Analysis (find optimal config)
```

Entry points in `neusim/run_scripts/`:
1. `run_sim.py` — Performance simulation (sweeps chips, batch sizes, parallelism configs; uses Ray)
2. `energy_operator_analysis_main.py` — Power/energy analysis (reads perf results)
3. `carbon_analysis_main.py` — Carbon footprint analysis (reads power results)
4. `slo_analysis_main.py` — Finds optimal configs meeting SLO targets
5. `run_single_op_main.py` — Single operator simulation
6. `run_agent_pipeline.py` — Runs the chip_agent design pipeline from within NeuSim

### NeuSim Package Structure (`neusim/`)

- **`configs/`** — Pydantic v2 config classes for chips (`ChipConfig`), models (`LLMConfig`, `DiTConfig`, `DLRMConfig`, `GLIGENConfig`), and systems (`SystemConfig`). All validation uses Pydantic v2.
- **`npusim/backend/`** — Core performance simulation engine. `npusim_lib.py` models hardware components (systolic arrays, vector units, SRAM, HBM, ICI) and calculates per-operator execution time and bottlenecks.
- **`npusim/frontend/`** — Ops generators and analysis:
  - `Operator.py` — Universal operator representation with stats fields (execution time, FLOPS, memory traffic, per-device timings for MXU/VPU/Transpose/Permute)
  - `llm_ops_generator.py` / `llm_ops_lib.py` — LLM model graph generation with parallelism support (TP, PP, DP, EP). Base class `LLMOpsGeneratorBase` with `Training` and `Inference` variants.
  - `dlrm_ops_generator.py`, `dit_ops_generator.py`, `gligen_ops_generator.py` — Other model generators
  - `power_analysis_lib.py` / `dvfs_power_getter.py` — Power gating and DVFS modeling
  - `energy_carbon_analysis_lib.py` — Carbon footprint calculations
  - `run_sim_lib.py` — Simulation orchestration; generates valid parallelism configs via `generate_parallelism_configs_llm()`
  - `memory_footprint_analysis_lib.py` — Memory consumption tracking
  - `query_results_helper_lib.py` — Results parsing and analysis
- **`xla_hlo_parser/`** — XLA HLO trace parser framework
- **Tests colocated with modules**: Each subpackage has a `tests/` directory (configs/tests, backend/tests, frontend/tests, xla_hlo_parser/tests)

### chip_agent Architecture

Three GPT-4o agents collaborate in a pipeline:

```
Agent 1 (computedemandagent.py): Extract per-operator compute demand from model config
    → Agent 3 (npudesignagent.py): Iteratively propose NPU configs via LLM + NeuSIM loop
        ↔ Agent 2 (bottleneck_analysis.py): One-sentence bottleneck diagnosis after each sim
        ↔ NeuSIM (neusim_runner.py): Programmatic bridge to NeuSim simulation
        ↔ Design Memory (design_memory.py): Cross-workload experience (episodic + semantic)
```

- `main.py` — Unified entry point orchestrating the full pipeline
- `neusim_runner.py` — Programmatic NeuSim interface; `load_base_configs()` merges JSON configs, `apply_chip_overrides()` modifies NPU params
- `design_memory.py` — Two-tier memory: episodic (per-run reflections) and semantic (consolidated principles). Uses per-chip workload similarity (cosine on feature vectors) for retrieval.
- Agent 3's design loop: seed warm-start (9 configs) → LLM iterations (max 8 rounds) → coordinate descent refinement → reflect & store
- `--disaggregated` mode designs separate prefill and decode chips
- Baselines: `baseline_grid.py`, `baseline_bo.py`, `baseline_es.py`, `baseline_greedy.py` — all share the same design space, constraints, and scoring

### Configuration Files (`configs/`)

- `chips/` — Hardware configs (tpuv2-v6p): SA/VU counts, memory bandwidth, power characteristics
- `models/` — DNN architecture params: dimensions, parallelism degrees, batch sizes
- `systems/` — Datacenter params: PUE, carbon intensity

### Key Design Patterns

- **Distributed via Ray**: Design space exploration uses Ray for multi-machine parallelization
- **Extensible model support**: New models require implementing an ops generator class in `npusim/frontend/` and adding a JSON config in `configs/models/`
- **Bottleneck analysis**: Backend tracks SA-bound, VU-bound, HBM-bound, and ICI-bound bottlenecks per operator
- **Multi-chip parallelism**: Supports tensor, pipeline, data, and expert parallelism with automatic valid combination generation
- **Agent 2 "facts only" design**: Bottleneck agent outputs one sentence with no recommendations — more detail hurts Agent 3's exploration

## CI/CD

GitHub Actions workflows (`NeuSim/.github/workflows/`):
- `test.yml` — Runs `pytest` on push/PR
- `lint.yml` — Runs `./lint.sh` on push/PR

## Lint Rules

The `lint.sh` script treats these as **blocking errors**: E9 (syntax), F821 (undefined names), F822 (undefined exports), F823 (undefined locals). Other ruff/mypy issues are warnings only.

Ruff config in `pyproject.toml`: selects E, F, I, B, UP rules; ignores E501 (line length). Target Python 3.12.
