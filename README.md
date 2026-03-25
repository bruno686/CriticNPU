# Physics-Grounded Multi-Agent System for NPU Architecture Design

A multi-agent system that automatically designs optimal NPU chip configurations for LLM inference. The system equips LLM agents with a **physics-based world model (roofline model)** that transforms chip design from black-box optimization into **physics-informed reasoning** — matching or beating Bayesian Optimization, CMA-ES, and Greedy Local Search baselines across 6 LLM models while using **~35% fewer simulations**.

---

## Table of Contents

- [Motivation & Design Philosophy](#motivation--design-philosophy)
- [Problem Formulation](#problem-formulation)
- [System Architecture](#system-architecture)
- [The Roofline World Model (Core Contribution)](#the-roofline-world-model-core-contribution)
- [Agent Details](#agent-details)
- [Design Loop](#design-loop)
- [Cross-Workload Design Memory](#cross-workload-design-memory)
- [Experimental Results](#experimental-results)
- [Key Findings](#key-findings)
- [Baselines](#baselines)
- [Setup & Usage](#setup--usage)
- [Project Structure](#project-structure)

---

## Motivation & Design Philosophy

### Why not just use Bayesian Optimization?

Traditional optimization methods (BO, CMA-ES, evolutionary search) treat the NPU simulator as a **black box**: they propose parameters, observe results, and fit a statistical surrogate model. This ignores the rich physical structure of the problem:

- NPU performance is governed by the **roofline model** — a well-known analytical framework that decomposes performance into compute-bound and memory-bound regimes
- The relationship between chip parameters and performance is **causal and interpretable** — adding systolic arrays increases the compute ceiling, not just "a number that correlates with TTFT"
- Fixing one bottleneck **predictably reveals the next** — this cascade structure enables multi-step planning

BO's GP surrogate can fit curves but cannot reason about *why* a configuration is good. It doesn't know that "adding SAs is useless when the workload is memory-bound." It has to discover this empirically, wasting simulation budget.

### Why not just use an LLM as optimizer?

Existing LLM-for-optimization approaches (OPRO, EvoPrompting) use LLMs as **reactive pattern matchers**: they see numbers, guess the next set of numbers based on trends. The LLM has no internal model of *how* the system works — it cannot answer counterfactual questions like "if I add 2 SAs, will TTFT improve?" without actually running a simulation.

This leads to two problems:
1. **Every step of cognition requires a simulation call** — the LLM cannot reason ahead
2. **LLMs have cognitive biases** — e.g., the frequency anchoring bias (higher frequency = faster is "common sense," but under power constraints, lower frequency is often optimal because it frees power budget for more compute units)

### Our approach: Physics-Grounded World Model

We equip the LLM design agent with a **domain-specific world model** based on the roofline model — a first-principles analytical framework from computer architecture. This world model:

1. **Tells the LLM *why* performance is what it is** — "these 5 MatMul operators are compute-bound because their arithmetic intensity (200 FLOP/byte) exceeds the ridge point (150 FLOP/byte)"
2. **Predicts *how* parameter changes will affect performance** — "adding 2 SAs shifts the ridge point right by 33%, making 3 operators cross from compute-bound to memory-bound"
3. **Enables multi-step cascade planning** — "lower freq → save power → add more SAs → compute ceiling increases → prefill speedup"

The simulator (NeuSim) is reduced from "every-round evaluator" to "key-point validator." Most reasoning happens analytically inside the world model.

**Analogy:** A human chip architect doesn't design by running 65 simulations. They use mental models (roofline, power budgeting, Amdahl's law) to reason about the design space, and only simulate to validate. Our LLM agent works the same way.

---

## Problem Formulation

### Input

**LLM Model Structure** (from JSON config):
- Architecture: `d_model`, `d_ff`, `num_heads`, `num_kv_heads`, `d_head`, `num_layers`
- Sequence lengths: `input_seqlen`, `output_seqlen`
- FFN type: `dense`, `llama` (gated), `deepseek_moe`
- MoE parameters (if applicable): `num_routed_experts`, etc.

**Deployment Configuration**:
- Number of chips, tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP), expert parallelism (EP), batch size (BS)

**Constraints**:
- **Hard constraint**: `total_power_peak_W ≤ power_budget` (e.g., 300W)
- **Hard constraint**: no OOM — `memory_footprint ≤ hbm_size_GB`
- **Upper bounds**: `hbm_bw_GBps ≤ 600`, `hbm_size_GB ≤ 16-32`, `ici_bw_GBps ≤ 125`, `dcn_bw_GBps ≤ 12.5`

### Output

**Optimal NPU Chip Configuration** — the following parameters are searched:

| Parameter | Description | Search Range |
|-----------|-------------|--------------|
| `num_sa` | Number of systolic arrays (matrix multiply units) | 2 – 16 |
| `num_vu` | Number of SIMD vector units | 2 – 12 |
| `sa_dim` | Systolic array dimension (N×N) | {64, 128, 256} |
| `freq_GHz` | Core clock frequency | 0.5 – 2.0 GHz |
| `vmem_size_MB` | On-chip SRAM / vector memory size | 32 – 512 MB |
| Power params | Per-unit static/dynamic power (6 parameters) | Varies |
| Bandwidth params | HBM/ICI/DCN bandwidth (4 parameters, capped at upper bounds) | ≤ upper bounds |

### Objective

```
minimize  score = ttft_weight × (TTFT / baseline_TTFT) + tpot_weight × (TPOT / baseline_TPOT)
subject to  total_power_peak_W ≤ power_budget
            memory_footprint ≤ hbm_size_GB
```

In **disaggregated serving** mode, two separate chips are designed:
- **Prefill chip**: `ttft_weight=1.0, tpot_weight=0.0` — minimize TTFT only
- **Decode chip**: `ttft_weight=0.0, tpot_weight=1.0` — minimize TPOT only

---

## System Architecture

```
  LLM Model Config (JSON)
        │
        ▼
  ┌─────────────────────────────┐
  │  Agent 1: Compute Demand    │   Runs once. GPT-4o extracts per-operator
  │  (computedemandagent.py)    │   FLOPs, tensor shapes, execution counts.
  └──────────────┬──────────────┘
                 │  CSV (per-op workload profile)
                 ▼
  ┌─────────────────────────────┐     ┌──────────────────────────┐
  │  Agent B: Design Planner    │◄────│  Roofline World Model    │
  │  (npudesignagent.py)        │     │  (world_model.py)        │
  │  GPT-4o reasons WITH the    │     │                          │
  │  world model to propose     │     │  • Compute/BW ceilings   │
  │  chip configurations        │     │  • Ridge point           │
  │                             │     │  • Operator mapping      │
  └──────────────┬──────────────┘     │  • Cascade prediction    │
                 │  chip config       │  • Power model           │
                 ▼                    └──────────┬───────────────┘
  ┌─────────────────────────────┐                │
  │  NeuSIM Simulation          │                │
  │  → TTFT, TPOT, Power, OOM  │                │
  │  → Per-operator breakdown   │                │
  └──────────────┬──────────────┘                │
                 │                               │
                 ▼                               │
  ┌─────────────────────────────┐                │
  │  Agent A: World Model       │────────────────┘
  │  Constructor                │   Builds roofline model from
  │  (bottleneck_analysis.py)   │   NeuSIM results. No LLM call —
  │                             │   pure physics computation.
  └──────────────┬──────────────┘
                 │  Structured roofline analysis
                 └──► Fed back to Agent B for next iteration
```

| Component | File | Role |
|-----------|------|------|
| **Agent 1** | `computedemandagent.py` | Extract per-operator compute demand from model config (GPT-4o) |
| **Agent A** | `bottleneck_analysis.py` | Build roofline world model from NeuSIM results (deterministic, no LLM) |
| **Agent B** | `npudesignagent.py` | Design Planner — proposes chip configs using roofline-informed reasoning (GPT-4o) |
| **World Model** | `world_model.py` | Roofline construction, cascade prediction, direction suggestion (pure physics) |
| **Simulator** | `neusim_runner.py` | Programmatic bridge to NeuSIM |
| **Memory** | `design_memory.py` | Cross-workload experience store (episodic + semantic) |

---

## The Roofline World Model (Core Contribution)

### What is a World Model in this context?

A world model is the agent's internal representation of how the environment works — it allows the agent to **predict the consequences of actions without actually taking them**.

The NeuSIM simulator is NOT a world model from the agent's perspective. It's a black box that answers "what is the TTFT of config X?" but not "**why** is it that value?" or "**how** will it change if I modify Y?"

The roofline model IS a world model because:
- It has a **simple, interpretable structure** that the LLM can reason with (`perf = min(compute_ceiling, intensity × BW_ceiling)`)
- It provides **causal explanations** ("this op is slow because it's compute-bound and the SA ceiling is too low")
- It enables **counterfactual prediction** ("adding 2 SAs would increase the ceiling by 33%, giving ~1.2x speedup on compute-bound ops")
- It supports **multi-step planning** ("lower freq → save power → add SAs → shift ridge → predict new bottleneck")

### Three-Layer Architecture

#### Layer 1: Roofline Model

The roofline model defines two performance ceilings:

```
Compute ceiling = num_sa × sa_dim² × 2 × freq_GHz    (GFLOPS)
Memory BW ceiling = hbm_bw_GBps                        (GB/s)
Ridge point = Compute ceiling / BW ceiling              (FLOP/byte)
```

Every operator has an **arithmetic intensity** = FLOPs / memory_traffic_bytes (FLOP/byte).

- If `intensity > ridge_point` → **compute-bound**: performance limited by SAs, adding BW won't help
- If `intensity < ridge_point` → **memory-bound**: performance limited by HBM BW, adding SAs won't help

This single framework tells the agent exactly which parameters matter for which operators.

#### Layer 2: Power Model

Power is an analytical function of chip parameters (no simulation needed):

```
total_power_peak_W = static_power + dynamic_power_peak
  static_power  = (per_sa_static × num_sa) + (per_vu_static × num_vu) + vmem_static + hbm_static + ici_static
  dynamic_power = (per_sa_dynamic × num_sa) + (per_vu_dynamic × num_vu) + vmem_dynamic + hbm_dynamic + ici_dynamic
```

Given a power budget, the agent can analytically compute the **feasible region** — e.g., "at freq=0.5 GHz, the maximum num_sa that fits within 300W is 9."

#### Layer 3: Bottleneck Cascade Model

When you change a parameter, the roofline shifts, and operators may change regime:

```
Example cascade:
  Current: num_sa=6, ridge=655 → prefill 53% compute-bound

  Action: +2 SAs → new ridge=874
  Effect: Q/K/V projections cross from compute-bound to memory-bound
  Net:    prefill speedup ~1.12x on compute-bound ops, but memory-bound ops unchanged

  Next bottleneck: now memory-bound ops dominate → increase SRAM to reduce HBM traffic
```

The agent can plan this **entire chain without running any simulation**. It only calls NeuSIM to validate the final proposed configuration.

### How It Changes the Agent's Reasoning

**Without world model (old approach):**
```
LLM sees: "TTFT=0.08s, SA util=42%, HBM util=88%"
LLM thinks: "SA utilization is low, maybe reduce SA? Last time that helped..."
→ Pattern matching on numbers. No understanding of WHY.
```

**With world model (new approach):**
```
LLM sees roofline analysis:
  "Ridge point = 655 FLOP/byte
   Top operators: FFN-gate at intensity=1806 (compute-bound, 1.5x above ridge)
                  FlashAttention at intensity=144 (memory-bound, 2.2x below ridge)
   Prefill: 53% compute, 25% memory, 21% ICI
   Power headroom: 65W"

LLM reasons: "The FFN-gate ops are compute-bound and dominate time.
  Adding 2 SAs would increase compute ceiling by 33%.
  But that costs ~36W, which is within the 65W headroom.
  The memory-bound FlashAttention ops won't benefit, but they're only 25% of time.
  Predicted prefill speedup: ~1.12x."
→ Physics-informed causal reasoning. Every step has a physical justification.
```

---

## Agent Details

### Agent 1: Compute Demand Extraction

**Purpose:** Analyze the LLM model structure and produce a per-operator workload profile.

**How it works:**
1. Reads model config JSON (layer count, hidden dims, attention heads, parallelism degrees)
2. GPT-4o applies sharding formulas to compute per-operator demands under the given parallelism
3. Python-side verification pass recomputes FLOP counts and tensor shapes
4. Outputs two CSVs: `inference_prefill.csv` and `inference_decode.csv`

**Output format** (each row = one operator):

| Column | Example |
|--------|---------|
| Description | `-Q-5`, `FlashAttention-6` |
| Count | 32 |
| FLOP Count | 2.15e+09 |
| Input Tensor Shapes | `[16, 4096, 1024] bf16` |
| Output Tensor Shapes | `[16, 4096, 1024] bf16` |

### Agent A: World Model Constructor (bottleneck_analysis.py)

**Purpose:** Build the roofline world model from NeuSIM simulation results. **No LLM call** — this is pure deterministic physics computation.

**How it works:**
1. Receives NeuSIM per-operator results (execution time, FLOPs, memory traffic, bottleneck classification)
2. Computes chip ceilings from config (compute ceiling, BW ceiling, ridge point)
3. Maps each operator onto the roofline plane (arithmetic intensity, regime classification, distance to ridge)
4. Aggregates phase-level bottleneck distribution (% compute-bound / memory-bound / ICI-bound)
5. Generates direction suggestions with cascade predictions

**Design decision:** Agent A was previously an LLM (GPT-4o) that produced a one-sentence bottleneck diagnosis. We found that **more LLM analysis hurts performance** because it anchors the design agent's exploration. Replacing the LLM with a deterministic roofline model provides richer, more structured information without the anchoring effect.

### Agent B: Design Planner (npudesignagent.py)

**Purpose:** The main decision-maker. Uses the roofline world model to propose chip configurations via GPT-4o.

**How it works:**
1. Receives the roofline world model (compute/BW ceilings, ridge point, per-operator mapping, cascade predictions, direction suggestions)
2. Reasons about which parameters to change based on the dominant bottleneck regime
3. Proposes a chip configuration via the `propose_npu_config` OpenAI function call
4. Must provide a `reasoning` field explaining the design rationale

**System prompt includes roofline concepts** — the LLM is explicitly taught to reason with compute ceilings, ridge points, and regime crossings.

---

## Design Loop

### Step 1: Seed Warm-Start

Evaluate 9 diverse pre-defined configs spanning the design space:

| Seed | num_sa | sa_dim | freq | num_vu | vmem |
|------|--------|--------|------|--------|------|
| min-compute | 2 | 64 | 0.8 | 2 | 128 |
| low-compute | 4 | 64 | 0.8 | 4 | 128 |
| balanced | 8 | 128 | 1.0 | 6 | 256 |
| high-compute | 12 | 256 | 1.2 | 8 | 384 |
| max-compute | 16 | 256 | 1.5 | 10 | 384 |
| low-freq-big | 8 | 256 | 0.8 | 6 | 384 |
| hi-freq-small | 4 | 128 | 1.4 | 4 | 128 |
| vlow-freq-tiny | 2 | 64 | 0.5 | 2 | 128 |
| med-freq-mid | 6 | 128 | 0.65 | 4 | 256 |

Purpose: map the design landscape and give the agent a starting point. Without seeds, GPT-4o stays near defaults and misses counterintuitive regions.

After seed phase, the **initial roofline world model** is built from the best seed's simulation results.

### Step 2: LLM Iteration Loop (max 8 rounds)

Each round:
1. Agent B (GPT-4o) receives roofline analysis + history → proposes config
2. **Duplicate detection** — reject configs already tried
3. **Fast power pre-check** — analytical power computation (~0.001s), reject over-budget configs without simulation
4. **NeuSIM simulation** — full simulation (~2s), get TTFT, TPOT, power, per-operator breakdown
5. **Agent A builds updated world model** — roofline mapping of all operators, direction suggestions with cascade predictions
6. **Feedback to Agent B** — roofline analysis + suggestions injected into conversation

### Step 3: Local Refinement (Coordinate Descent)

After the LLM loop, deterministic post-hoc refinement from top-3 configs:

1. For each starting config, run **directional coordinate descent** on each parameter axis
2. For each axis, probe both directions; if one improves, keep walking that direction until bounds or no further gain
3. **Power-aware trade**: if adding SAs exceeds power, try simultaneously reducing VUs to free headroom

This step corrects the LLM's **frequency anchoring bias** — the LLM stops at freq=0.8-0.9 GHz ("higher frequency = faster"), but the actual optimum is often freq=0.5 GHz (lower frequency → less per-unit power → more SAs within budget → more compute).

### Step 4: Reflect & Store

After the loop completes, GPT-4o generates a design lesson from the trajectory, stored in cross-workload design memory.

---

## Cross-Workload Design Memory

### Architecture

```
DesignMemory
├── Episodic Memory    Specific design runs with LLM-generated reflections
│     "For llama3-8b prefill, low freq (0.5 GHz) + max SA count was optimal
│      because power savings from low freq funded additional SAs."
│
├── Semantic Memory    General principles consolidated from multiple episodes
│     "Lowering core frequency below 0.7 GHz frees enough power budget for
│      2-3 additional SAs, improving compute-bound workloads."
│
└── Retrieval          Per-chip workload similarity (cosine on feature vectors)
```

### Per-Chip Workload Similarity

The key insight: **optimal NPU config depends on per-chip workload, not model size**. After parallelism sharding:
- llama3-8b with TP=4: per-chip d_ff = 14336/4 = **3584**, heads = 32/4 = **8**
- llama3-70b with TP=8: per-chip d_ff = 28672/8 = **3584**, heads = 64/8 = **8**

Same per-chip workload → same optimal config.

Similarity features: `d_ff/TP`, `num_heads/TP`, `num_layers/PP`, `d_head`, `arithmetic_intensity`, `batch_size`, `is_moe`.

---

## Experimental Results

### Setup

- **Simulator:** NeuSIM (analytical NPU performance model)
- **LLM:** GPT-4o (for Agent 1, Agent B, and design memory reflection)
- **Mode:** Disaggregated serving (separate prefill and decode chips)
- **Power budget:** 300W per chip
- **All methods use identical** constraints, design space, scoring function, and simulator

### Prefill Chip (minimize TTFT) — 6 Models × 4 Methods

| Model | Agent (WM) | BO | CMA-ES | Greedy |
|-------|-----------|-----|--------|--------|
| llama3-8b (8 chips, TP=4, PP=2) | **0.1351s / 42 sim** | 0.1366s / 57 sim | 0.1974s / 65 sim | 0.1430s / 65 sim |
| llama2-13b (8 chips, TP=4, PP=2) | **0.2112s / 42 sim** | 0.2112s / 57 sim | 0.2112s / 65 sim | 0.2279s / 65 sim |
| llama3-70b (64 chips, TP=8, PP=8) | **0.1408s / 41 sim** | 0.1408s / 57 sim | 0.1835s / 65 sim | 0.1408s / 65 sim |
| llama3.1-405b (64 chips, TP=8, PP=8) | **0.6678s / 40 sim** | 0.6722s / 57 sim | 0.9266s / 65 sim | 0.6678s / 65 sim |
| DeepSeek-v2-236b (16 chips, TP=8, PP=2) | **2.7418s / 42 sim** | 2.7668s / 57 sim | 2.8257s / 65 sim | 2.7418s / 64 sim |
| DeepSeek-v3-671b (64 chips, TP=8, PP=8) | **3.2220s / 42 sim** | 3.2371s / 57 sim | 3.3326s / 65 sim | 3.2220s / 65 sim |

**Bold** = best or tied-for-best TTFT for that model.

### Decode Chip (minimize TPOT) — 6 Models × 4 Methods

| Model | Agent (WM) | BO | CMA-ES | Greedy |
|-------|-----------|-----|--------|--------|
| llama3-8b | 4.5256ms / 30 sim | 4.5256ms / 57 sim | 4.5256ms / 65 sim | 4.5256ms / 63 sim |
| llama2-13b | 7.0333ms / 31 sim | 7.0333ms / 57 sim | 7.0333ms / 65 sim | 7.0333ms / 65 sim |
| llama3-70b | 13.1139ms / 31 sim | 13.1139ms / 57 sim | 13.1139ms / 65 sim | 13.1139ms / 65 sim |
| llama3.1-405b | 62.9332ms / 32 sim | 62.9332ms / 57 sim | 62.9332ms / 65 sim | 62.9332ms / 65 sim |
| DeepSeek-v2-236b | 97.8701ms / 41 sim | 97.8701ms / 57 sim | 97.8701ms / 65 sim | 97.8701ms / 65 sim |
| DeepSeek-v3-671b | 80.3154ms / 42 sim | 80.3154ms / 57 sim | 80.3154ms / 65 sim | 80.3154ms / 65 sim |

All methods converge to the **same TPOT** for every model — decode is purely HBM bandwidth-bound, compute resources are irrelevant. The agent consistently uses **~35-50% fewer simulations**.

### Why Decode TPOT is Identical Across All Methods

Decode-phase operators (matrix-vector multiplies for single-token generation) have extremely low arithmetic intensity — far below the ridge point of ANY chip configuration in the design space. This means:

```
TPOT ≈ model_weight_bytes / hbm_bw_GBps
```

Since all methods share the same HBM bandwidth upper bound (600 GB/s) and model weights are fixed, TPOT is identical. The optimal decode chip minimizes compute to save power (num_sa=2, num_vu=2, freq=0.5 GHz).

### Summary: Agent vs Best Baseline

| Model | Phase | Agent | Best Baseline | Δ | Agent Sims | BL Sims |
|-------|-------|-------|---------------|---|------------|---------|
| llama3-8b | prefill | 0.1351s | 0.1366s (BO) | **-1.1%** | 42 | 57 |
| llama2-13b | prefill | 0.2112s | 0.2112s (BO) | tie | 42 | 57 |
| llama3-70b | prefill | 0.1408s | 0.1408s (BO/Greedy) | tie | 41 | 57 |
| llama3.1-405b | prefill | 0.6678s | 0.6678s (Greedy) | tie | 40 | 65 |
| DeepSeek-v2-236b | prefill | 2.7418s | 2.7418s (Greedy) | tie | 42 | 64 |
| DeepSeek-v3-671b | prefill | 3.2220s | 3.2220s (Greedy) | tie | 42 | 65 |

**Across all 6 models: the agent never loses to any baseline.** It matches or beats the best baseline while using 35-40% fewer simulations.

---

## Key Findings

### 1. Physics-Based World Model Enables Efficient Design

The roofline world model lets the agent reason about design consequences without running the simulator. This reduces simulation calls from ~57-65 (baselines) to ~40-42 (agent) while achieving the same or better results.

### 2. LLM Anchoring Bias is Real and Correctable

The LLM consistently stops at freq=0.8-0.9 GHz because "higher frequency = faster" is common sense. The actual optimum is freq=0.5 GHz for compute-bound prefill workloads under power constraints — lower frequency reduces per-unit power, leaving more thermal budget for additional SAs. The roofline model makes this counterintuitive trade-off visible, and the coordinate descent step corrects remaining bias.

### 3. Decode is Pure HBM-Bound (Compute is Irrelevant)

All methods converge to the same TPOT for every model. The roofline model explains this: decode operators sit far below the ridge point (memory-bound). The optimal decode chip minimizes compute (num_sa=2, freq=0.5) to save power — adding SAs has zero effect on TPOT. BO and CMA-ES waste simulations discovering this; the roofline model knows it immediately.

### 4. CMA-ES Struggles with Discrete Parameters

CMA-ES is designed for continuous optimization but the design space has discrete parameters (sa_dim ∈ {64, 128, 256}). It consistently finds worse prefill configs than the agent, BO, and Greedy across all models.

### 5. Cross-Workload Transfer via Roofline

Models with the same per-chip workload after sharding converge to the same optimal config — the roofline distribution (not the model name) determines the design. This enables experience transfer: design knowledge from llama3-8b (TP=4) directly applies to llama3-70b (TP=8) because both have identical per-chip arithmetic intensity profiles.

---

## Contributions

1. **Core contribution: Physics-grounded world model for LLM design agents.** We equip LLM agents with a domain-specific world model (roofline) that transforms hardware design from black-box optimization into physics-informed reasoning. The world model provides causal explanations, counterfactual predictions, and multi-step cascade planning — capabilities that neither BO's statistical surrogate nor vanilla LLM agents possess.

2. **Bottleneck cascade planning.** Based on the world model, the agent plans multi-step bottleneck transition paths (e.g., "lower freq → free power → add SAs → shift ridge → new bottleneck is memory → add SRAM") instead of greedy single-step parameter tuning.

3. **Comprehensive experimental validation.** Across 6 LLM models (llama3-8b to DeepSeek-v3-671b), 4 optimization baselines (BO, CMA-ES, Greedy, Grid), and disaggregated prefill/decode chip design: the agent matches or beats every baseline on every model while using ~35% fewer simulations.

---

## Baselines

| Baseline | File | Method | Sims |
|----------|------|--------|------|
| **Bayesian Optimization** | `baseline_bo.py` | GP surrogate + Expected Improvement (15 init + 50 BO iters) | 57 |
| **CMA-ES** | `baseline_es.py` | Covariance Matrix Adaptation Evolution Strategy (pop=10, budget=65) | 65 |
| **Greedy Local Search** | `baseline_greedy.py` | Single-parameter perturbation from seed configs + random restart | 63-65 |
| **Grid Search** | `baseline_grid.py` | Exhaustive 720-point grid | 720 |

All baselines use the same design space, constraints, power budget, and scoring as the agent.

---

## Setup & Usage

### Prerequisites

```bash
# NeuSIM (from NeuSim/ directory)
cd NeuSim && pip install -e .

# chip_agent dependencies
pip install pandas openai python-dotenv matplotlib
```

Create `chip_agent/.env`:
```
OPENAI_API_KEY=sk-...
```

### Running the Agent

All commands from `chip_agent/`:

```bash
cd chip_agent

# Full pipeline: Agent 1 (compute demand) → Agent B (design with world model)
python main.py llama3-8b --tp 4 --pp 2 --bs 16 --num-chips 8 --power-budget 300 --disaggregated

# Skip Agent 1 (reuse existing compute demand CSVs)
python main.py llama3-8b --tp 4 --pp 2 --bs 16 --num-chips 8 --power-budget 300 --disaggregated --skip-agent1

# Disable cross-workload memory (for ablation)
python main.py llama3-8b --tp 4 --pp 2 --bs 16 --num-chips 8 --power-budget 300 --no-memory

# Larger models
python main.py llama3-70b --tp 8 --pp 8 --bs 16 --num-chips 64 --power-budget 300 --disaggregated
python main.py deepseekv2-236b --tp 8 --pp 2 --bs 16 --num-chips 16 --power-budget 300 --max-hbm-size 32 --disaggregated
```

### Running Baselines

```bash
python baseline_bo.py llama3-8b --chip-version 4 --num-chips 8 --tp 4 --pp 2 --bs 16 --power-budget 300 --disaggregated
python baseline_es.py llama3-8b --chip-version 4 --num-chips 8 --tp 4 --pp 2 --bs 16 --power-budget 300 --disaggregated
python baseline_greedy.py llama3-8b --chip-version 4 --num-chips 8 --tp 4 --pp 2 --bs 16 --power-budget 300 --disaggregated
```

### Summarizing Results

```bash
python summarize_results.py
```

---

## Project Structure

```
chip_agent/
├── main.py                   # Unified entry point: Agent 1 → Agent B pipeline
├── world_model.py            # ★ Roofline world model (core contribution)
│                             #   - build_roofline(): compute/BW ceilings, ridge point
│                             #   - populate_from_ops(): map operators to roofline
│                             #   - predict_cascade(): predict bottleneck shifts
│                             #   - suggest_directions(): physics-based design suggestions
│                             #   - format_for_llm(): structured output for Agent B
├── bottleneck_analysis.py    # Agent A: world model constructor (deterministic, no LLM)
├── npudesignagent.py         # Agent B: design planner (GPT-4o + world model reasoning)
├── computedemandagent.py     # Agent 1: compute demand extraction (GPT-4o)
├── design_memory.py          # Cross-workload design memory (episodic + semantic)
├── neusim_runner.py          # NeuSIM simulation wrapper (Python API)
├── baseline_bo.py            # Bayesian Optimization baseline
├── baseline_es.py            # CMA-ES (Evolutionary Search) baseline
├── baseline_greedy.py        # Greedy Local Search baseline
├── baseline_grid.py          # Grid Search baseline
├── summarize_results.py      # Results comparison table generator
├── run_all_experiments.sh    # Batch runner for all models × all methods
├── plot_iterations.py        # Visualization of design iterations
├── design_results/           # Output directory for design runs
├── design_results_all/       # Batch experiment results
├── results/                  # Pre-generated compute demand CSVs
└── .env                      # OpenAI API key (not committed)

NeuSim/                       # NPU performance simulator (separate project)
├── neusim/
│   ├── configs/              # Pydantic v2 configs for chips, models, systems
│   ├── npusim/backend/       # Core simulation engine (npusim_lib.py)
│   ├── npusim/frontend/      # Ops generators, power analysis, simulation orchestration
│   └── xla_hlo_parser/       # XLA HLO trace parser
├── configs/
│   ├── chips/                # Hardware configs (tpuv2-v6p)
│   ├── models/               # LLM model configs (llama, deepseek, etc.)
│   └── systems/              # Datacenter configs
└── run_scripts/              # Entry points for batch simulation
```
