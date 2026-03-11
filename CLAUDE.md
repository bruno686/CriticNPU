# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains two projects:

- **NeuSim** (`NeuSim/`): An open-source simulator for modeling performance, power, and carbon behavior of Neural Processing Units (NPUs) running ML workloads. Supports TPUv2-v6p hardware configs and models including LLMs (Llama, DeepSeek), DLRM, DiT, and GLIGEN.
- **chip_agent** (`chip_agent/`): A lightweight companion tool that uses an OpenAI LLM agent to extract per-operator compute demand from ML model configs, producing CSV files for use with NeuSim.

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
pip install pandas openai python-dotenv
python computedemandagent.py <model_config.json> [--tp N --pp N --dp N --ep N --bs N]
```

Requires `.env` file with `OPENAI_API_KEY`.

## Architecture

### Simulation Pipeline

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

### NeuSim Package Structure (`neusim/`)

- **`configs/`** — Pydantic v2 config classes for chips, models (LLM, DLRM, DiT, GLIGEN), and systems
- **`npusim/backend/`** — Core performance simulation engine. `npusim_lib.py` models hardware components (systolic arrays, vector units, SRAM, HBM, ICI) and calculates per-operator execution time and bottlenecks
- **`npusim/frontend/`** — Ops generators and analysis:
  - `Operator.py` — Universal operator representation with stats fields (execution time, FLOPS, memory traffic)
  - `llm_ops_generator.py` / `llm_ops_lib.py` — LLM model graph generation with parallelism support (TP, PP, DP, EP)
  - `dlrm_ops_generator.py`, `dit_ops_generator.py`, `gligen_ops_generator.py` — Other model generators
  - `power_analysis_lib.py` / `dvfs_power_getter.py` — Power gating and DVFS modeling
  - `energy_carbon_analysis_lib.py` — Carbon footprint calculations
  - `run_sim_lib.py` — Simulation orchestration with Ray for distributed execution
  - `query_results_helper_lib.py` — Results parsing and analysis
- **`xla_hlo_parser/`** — XLA HLO trace parser framework

### Configuration Files (`configs/`)

- `chips/` — Hardware configs (tpuv2-v6p): SA/VU counts, memory bandwidth, power characteristics
- `models/` — DNN architecture params: dimensions, parallelism degrees, batch sizes
- `systems/` — Datacenter params: PUE, carbon intensity

### Key Design Patterns

- **Distributed via Ray**: Design space exploration uses Ray for multi-machine parallelization
- **Extensible model support**: New models require implementing an ops generator class in `npusim/frontend/` and adding a JSON config in `configs/models/`
- **Bottleneck analysis**: Backend tracks SA-bound, VU-bound, HBM-bound, and ICI-bound bottlenecks per operator
- **Multi-chip parallelism**: Supports tensor, pipeline, data, and expert parallelism with automatic valid combination generation
- **Tests colocated with modules**: Each subpackage has a `tests/` directory (configs/tests, backend/tests, frontend/tests, xla_hlo_parser/tests)

## CI/CD

GitHub Actions workflows (`NeuSim/.github/workflows/`):
- `test.yml` — Runs `pytest` on push/PR
- `lint.yml` — Runs `./lint.sh` on push/PR

## Lint Rules

The `lint.sh` script treats these as **blocking errors**: E9 (syntax), F821 (undefined names), F822 (undefined exports), F823 (undefined locals). Other ruff/mypy issues are warnings only.

Ruff config in `pyproject.toml`: selects E, F, I, B, UP rules; ignores E501 (line length). Target Python 3.12.
