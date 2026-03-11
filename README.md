# Multi-Agent System for NPU Architecture Design

An LLM-driven NPU architecture design system. Given an LLM model structure and hardware constraints, it automatically designs optimal NPU chip configurations using a multi-agent system powered by GPT-4o and the [NeuSim](NeuSim/) performance simulator.

<img src="chip_agent/assets/flow.png" width="600">

## System Overview

The system consists of **three LLM-powered agents** and a **cross-workload design memory**:

- **Agent 1** (`computedemandagent.py`): Extracts per-operator compute demand from model configs
- **Agent 2** (`bottleneck_analysis.py`): Diagnoses performance bottlenecks from NeuSIM results
- **Agent 3** (`npudesignagent.py`): Iteratively designs NPU architecture using LLM + NeuSIM

```
                    ┌─────────────────────────┐
                    │  LLM Model Config JSON  │
                    │ (llama3-8b, llama2-13b)  │
                    └───────────┬──────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  Agent 1: Compute Demand       │
                │  (GPT-4o)                      │
                │  Model config → per-operator   │
                │  FLOPs, tensor shapes, counts  │
                └───────────────┬────────────────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                   │
              ▼                 ▼                   ▼
    ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
    │  Compute     │  │  Constraints    │  │  Design Memory   │
    │  Demand CSV  │  │  (power, HBM,   │  │  (cross-workload │
    │              │  │   ICI, area)    │  │   experiences)   │
    └──────┬───────┘  └───────┬─────────┘  └────────┬─────────┘
           │                  │                      │
           └──────────────────┼──────────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  Agent 3: NPU Design         │
               │  (GPT-4o)                    │
               │  Proposes NPU config         │◄──────────────┐
               └──────────────┬───────────────┘               │
                              │                               │
                              ▼                               │
               ┌──────────────────────────────┐               │
               │  NeuSIM Simulation           │               │
               │  → TTFT, TPOT, Power, OOM    │               │
               └──────────────┬───────────────┘               │
                              │                               │
                              ▼                               │
               ┌──────────────────────────────┐               │
               │  Agent 2: Bottleneck Analysis│               │
               │  (GPT-4o)                    │               │
               │  One-sentence bottleneck     │───────────────┘
               │  diagnosis (facts only)      │  feedback loop
               └──────────────────────────────┘  (max 50 iters)
```

| Agent | Role | LLM | Key Design Choice |
|-------|------|-----|-------------------|
| **Agent 1** | Extract per-operator compute demand | GPT-4o | Sharding-aware FLOP calculation with verification |
| **Agent 2** | Diagnose bottleneck from NeuSIM results | GPT-4o | Facts only, no recommendations (prevents misleading Agent 3) |
| **Agent 3** | Iteratively design NPU architecture | GPT-4o | Seed warm-start + LLM-guided search with power pre-check |
| **Design Memory** | Store and retrieve cross-workload experiences | GPT-4o | Per-chip workload similarity + hierarchical memory (episodic + semantic) |

---

## Method

### Problem Formulation

**Input:**
- LLM model structure (e.g., llama3-8b: 32 layers, d_model=4096)
- Hardware constraints: HBM bandwidth/size, ICI bandwidth, DCN bandwidth (upper bounds)
- Power budget (e.g., 300W hard constraint)
- Parallelism config: num_chips, TP, PP, DP, batch size

**Output:**
- Optimal NPU chip configuration (num_sa, num_vu, sa_dim, freq_GHz, vmem_size_MB, ...)
- Minimizing TTFT (Time To First Token) and/or TPOT (Time Per Output Token)

**Tunable Parameters:**

| Parameter | Description | Range |
|-----------|-------------|-------|
| `num_sa` | Number of systolic arrays | 2 - 16 |
| `num_vu` | Number of vector units | 2 - 12 |
| `sa_dim` | Systolic array dimension (N×N) | 64, 128, 256 |
| `freq_GHz` | Core clock frequency | 0.5 - 2.0 GHz |
| `vmem_size_MB` | On-chip SRAM size | 32 - 512 MB |
| Power params | Per-unit static/dynamic power | Varies |

**Constraints:**
- `total_power_peak_W ≤ power_budget` (hard constraint)
- `memory_footprint ≤ hbm_size_GB` (no OOM)
- `hbm_bw_GBps`, `hbm_size_GB`, `ici_bw_GBps`, `dcn_bw_GBps` ≤ upper bounds

### Design Loop

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Step 1: SEED WARM-START                                         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  9 diverse pre-defined configs spanning the design space   │  │
│  │                                                            │  │
│  │  min-compute ──► low-compute ──► balanced ──► high-compute │  │
│  │  (2 SA, 0.5G)   (4 SA, 0.8G)   (8 SA, 1.0G)  (12 SA,1.2)│  │
│  │                                                            │  │
│  │  Each seed: run NeuSIM → record (TTFT, TPOT, Power, OOM)  │  │
│  └────────────────────────────────┬───────────────────────────┘  │
│                                   │                              │
│                                   ▼                              │
│  Step 2: LLM ITERATION LOOP (max 50 iterations)                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                                                            │  │
│  │  Agent 3 (GPT-4o) proposes config                         │  │
│  │       │                                                    │  │
│  │       ▼                                                    │  │
│  │  Duplicate detection → YES → reject, ask for new one      │  │
│  │       │ NO                                                 │  │
│  │       ▼                                                    │  │
│  │  Fast power pre-check → FAIL → reject (no simulation)     │  │
│  │       │ PASS                                               │  │
│  │       ▼                                                    │  │
│  │  NeuSIM simulation → TTFT, TPOT, Power, Memory            │  │
│  │       │                                                    │  │
│  │       ▼                                                    │  │
│  │  Agent 2 → one-sentence bottleneck diagnosis               │  │
│  │       │                                                    │  │
│  │       ▼                                                    │  │
│  │  Feedback to Agent 3 → next iteration                     │  │
│  │       │                                                    │  │
│  │  Converged? (10 iters < 2% change) → STOP                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                   │                              │
│                                   ▼                              │
│  Step 3: REFLECT & STORE                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  LLM reflects on trajectory → design lesson                │  │
│  │  Store episode in cross-workload memory                    │  │
│  │  Consolidate semantic principles (if ≥3 episodes)          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Agent 2: Bottleneck Analysis Design

Agent 2 evolved through three iterations. A key finding: **more analysis hurts performance** because LLM architects are easily misled by detailed recommendations.

| Version | Output Style | Prefill TTFT | vs Baseline |
|---------|-------------|-------------|-------------|
| v1: Detailed recommendations | Multi-paragraph with parameter suggestions | 0.1859 s | +20.2% worse |
| v2: Concise recommendations | 2-3 sentences with parameter values | 0.1637 s | +5.8% worse |
| **v3: Facts only** | **One sentence, no recommendations** | **0.1495 s** | **-3.4% better** |
| (no analysis) | No Agent 2 | 0.1547 s | baseline |

The final design: Agent 2 reports **only bottleneck facts** (e.g., "Prefill is 78% compute-bound with 42% SA utilization — SAs are under-utilized due to tile padding.") and lets Agent 3 decide what to do. This prevents Agent 2 from constraining Agent 3's exploration space.

### Cross-Workload Design Memory

The memory module enables **experience transfer across workloads** — the key advantage of LLM-based design over Bayesian Optimization.

**Architecture:**
```
DesignMemory
├── Episodic Memory    — specific design runs with LLM-generated reflections
│   e.g., "For llama3-8b prefill, low freq (0.6) + max SA count was optimal
│          because power savings from low freq funded additional SAs"
├── Semantic Memory    — general principles consolidated from multiple episodes
│   e.g., "Lowering core frequency below 0.7 GHz frees enough power budget
│          for 2-3 additional SAs, improving compute-bound workloads"
└── Retrieval          — per-chip workload similarity (cosine on feature vectors)
```

**Workload similarity** is computed on per-chip features after parallelism sharding:
- `d_ff / TP`, `num_heads / TP`, `num_layers / PP` (per-chip dimensions)
- Arithmetic intensity (FLOP/byte ratio — compute vs memory bound indicator)
- Architecture type (dense vs MoE)

This correctly identifies that llama3-8b with TP=4 and llama3-70b with TP=8 have **identical per-chip workloads** (similarity=0.986) and should use the same NPU config — which matches our experimental results.

### Disaggregated Serving Mode

With `--disaggregated`, the system designs **two separate chips**:

1. **Prefill chip** — optimized purely for TTFT (compute-bound)
2. **Decode chip** — optimized purely for TPOT (memory-bound)

Each chip runs the full design loop independently with appropriate optimization weights.

### Key Design Decisions

| Mechanism | Why |
|-----------|-----|
| **Seed warm-start** | LLM tends to stay near default configs. Seeds force exploration of the full design space, especially low-frequency regions that are often optimal. |
| **Fast power pre-check** | `ChipConfig.total_power_peak_W` is deterministic from config params (~0.001s vs ~2s for full simulation). Saves ~30% of wasted iterations. |
| **Duplicate detection** | GPT-4o often re-proposes the same config in long conversations. Hash-based detection catches this instantly. |
| **Facts-only analysis** | Detailed LLM recommendations consistently hurt Agent 3's search quality by constraining its exploration space. |
| **Context trimming** | Keep system prompt + last 15 iteration pairs + best-so-far summary to stay within token limits. |

### Scoring Function

For combined optimization:
```
score = ttft_weight × (TTFT / baseline_TTFT) + tpot_weight × (TPOT / baseline_TPOT)
```
Where `baseline_TTFT` and `baseline_TPOT` come from the first valid configuration found. Lower score = better.

---

## Experimental Results

All experiments: power budget = 300W, disaggregated mode (separate prefill & decode chips).

### llama3-8b (8 chips, TP=4, PP=2, BS=16)

#### Agent Variants Comparison (Prefill TTFT)

| Method | TTFT | vs Grid Search | Simulations |
|--------|------|----------------|-------------|
| Grid Search | 0.1859 s | baseline | 1440 |
| Agent (no analysis) | 0.1547 s | **-16.8%** | ~82 |
| Agent + detailed analysis (v1) | 0.1859 s | +0.0% | ~82 |
| Agent + concise analysis (v2) | 0.1637 s | -11.9% | ~82 |
| **Agent + facts-only analysis (v3)** | **0.1495 s** | **-19.6%** | **~82** |

#### Best Results: Agent (v3) vs Grid Search

| | Grid Search | Agent (v3) | Improvement |
|---|---|---|---|
| **Prefill TTFT** | 0.1859 s | **0.1495 s** | **-19.6%** |
| **Decode TPOT** | 7.2406 ms | **4.5256 ms** | **-37.5%** |
| **Simulations run** | 1440 | **~82** | **17.6x fewer** |

Best prefill chip config:

| Parameter | Grid Search | Agent (v3) |
|-----------|-------------|------------|
| num_sa | 8 | **9** |
| sa_dim | 256 | 256 |
| freq_GHz | 0.8 | **0.6** |
| num_vu | 6 | **6** |
| vmem_size_MB | 128 | **256** |
| Peak Power | 277 W | **295 W** |

### llama2-13b (8 chips, TP=4, PP=2, BS=1)

| | Grid Search | Agent | Improvement |
|---|---|---|---|
| **Prefill TTFT** | 0.0373 s | **0.0286 s** | **-23.3%** |
| **Decode TPOT** | 6.3353 ms | **3.9597 ms** | **-37.5%** |
| **Simulations run** | 1440 | **55** | **26.2x fewer** |

### llama3-70b (16 chips, TP=8, PP=2, BS=16)

| | Agent (v3) |
|---|---|
| **Prefill TTFT** | **0.6086 s** |
| **Decode TPOT** | **17.3315 ms** |

Best prefill config: num_sa=9, sa_dim=256, freq_GHz=0.59, vmem_size_MB=420, Peak Power=293W

**Cross-workload observation:** llama3-8b (TP=4) and llama3-70b (TP=8) converge to nearly identical optimal configs because their per-chip workloads are the same after parallelism sharding (d_ff/TP=3584, heads/TP=8). This validates the per-chip workload similarity metric used in the design memory.

### Summary

The multi-agent system consistently:

1. **Finds better configs** — explores parameter values outside the grid (e.g., `num_sa=9`, `freq_GHz=0.54`) that grid search cannot reach
2. **Uses 17-26x fewer simulations** — seed warm-start + LLM-guided search + power pre-check avoids wasted evaluations
3. **Maximizes power budget utilization** — Agent configs land at ~295-299W (just under 300W budget), while grid search best configs only reach 277W
4. **Facts-only analysis outperforms detailed analysis** — Agent 2 v3 (one-sentence diagnosis) achieves the best results, while detailed recommendations (v1, v2) hurt performance

> **Note:** Decode TPOT improvements below `freq_GHz=0.8` are partly influenced by a NeuSIM tile-shape selection artifact where execution time scales linearly with frequency for memory-bound operations.

---

## Baseline: Grid Search

`baseline_grid.py` provides a brute-force grid search baseline for comparison:

```
Search grid (720 combinations):
  num_sa:       [4, 6, 8, 10, 12]
  num_vu:       [4, 6, 8, 10]
  sa_dim:       [64, 128, 256]
  freq_GHz:     [0.8, 1.0, 1.2, 1.4]
  vmem_size_MB: [128, 256, 384]
```

Same constraints, same scoring, same power budget. Fixed HBM/ICI/DCN at max values.

---

## Setup

```bash
pip install pandas openai python-dotenv matplotlib
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## Usage

All commands should be run from the `chip_agent/` directory:

```bash
cd chip_agent
```

### Full Pipeline (Recommended)

`main.py` runs the full multi-agent pipeline: Agent 1 → Agent 2 → Agent 3, with cross-workload memory enabled by default:

```bash
# Basic run
python main.py llama3-8b --tp 4 --pp 2 --bs 16 --num-chips 8 --power-budget 300

# Disaggregated mode (separate prefill & decode chips)
python main.py llama3-8b --tp 4 --pp 2 --bs 16 --num-chips 8 --power-budget 300 --disaggregated

# Reuse existing Agent 1 results (skip re-extraction)
python main.py llama3-8b --tp 4 --pp 2 --bs 16 --num-chips 8 --power-budget 300 --skip-agent1

# Disable cross-workload memory (for ablation)
python main.py llama3-8b --tp 4 --pp 2 --bs 16 --num-chips 8 --power-budget 300 --no-memory
```

Or via shell script: `./main.sh llama3-8b --tp 4 --pp 2 --bs 16 --num-chips 8 --power-budget 300`

### Run Agents Individually

```bash
# Agent 1: Compute Demand Extraction
python computedemandagent.py ../NeuSim/configs/models/llama3-8b.json --tp 4 --pp 2 --bs 16

# Agent 3: NPU Design (with Agent 1 output as input)
python npudesignagent.py llama3-8b --chip-version 4 --num-chips 8 \
  --tp 4 --pp 2 --bs 16 --power-budget 300 --disaggregated \
  --compute-demand-dir results/llama3-8b/dp1-tp4-pp2-dpdcn1-tpdcn1-ppdcn1-bs16
```

### Baseline: Grid Search

```bash
python baseline_grid.py llama3-8b --chip-version 4 --num-chips 8 \
  --tp 4 --pp 2 --bs 16 --power-budget 300 --disaggregated
```

### Plot Results

```bash
python plot_iterations.py design_results/<run_dir>/history.json [output.png]
```

## Project Structure

```
chip_agent/
├── main.py                # Unified entry point: Agent 1 → Agent 2 → Agent 3
├── main.sh                # Shell wrapper for main.py
├── computedemandagent.py  # Agent 1: compute demand extraction (GPT-4o)
├── bottleneck_analysis.py # Agent 2: bottleneck analysis (GPT-4o)
├── npudesignagent.py      # Agent 3: NPU architecture design (GPT-4o + NeuSIM)
├── design_memory.py       # Cross-workload design memory (episodic + semantic)
├── neusim_runner.py       # NeuSIM simulation wrapper (Python API)
├── baseline_grid.py       # Grid search baseline for comparison
├── plot_iterations.py     # Visualization of design iterations
├── design_results/        # Output directory for Agent 3 runs
├── design_memory/         # Persistent cross-workload memory store
├── results/               # Pre-generated compute demand CSVs
└── .env                   # OpenAI API key (not committed)
```
