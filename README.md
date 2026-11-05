<div align="center">

<h1>CriticNPU</h1>
<h3>Mechanistic Critics for Sample-Efficient<br>NPU Design-Space Exploration</h3>

<p>
  <img src="https://img.shields.io/badge/NeurIPS_2026-Accepted-7c3aed?style=flat-square&amp;labelColor=18181b" alt="Accepted at NeurIPS 2026">
  <img src="https://img.shields.io/badge/Python-3.12-0d9488?style=flat-square&amp;labelColor=18181b" alt="Python 3.12">
</p>

<p>Yicheng He · Yuqi Xue · Jerry Wu · Jian Huang · Bin Hu<br>
<sub>University of Illinois Urbana-Champaign</sub></p>

<p>
  <a href="#quick-start">Quick start</a> &nbsp;·&nbsp;
  <a href="#method">Method</a> &nbsp;·&nbsp;
  <a href="chip_agent/README.md#reproduce">Reproduction</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

</div>

---

CriticNPU searches for NPU architectures tailored to LLM inference. An **online-calibrated mechanistic critic** screens configurations before simulation and turns operator-level feedback into guidance for the next design.

## Results

| Prefill win rate | Decode win rate | Simulator calls |
| :---: | :---: | :---: |
| **74%** | **87.5%** | **Roughly half** |

Across **6 LLMs × 16 deployment settings**, win rate measures best-or-tied-best latency. Simulator-call savings are relative to competitive DSE baselines.

## Method

| Component | Role |
| --- | --- |
| **Compute Demand Agent** | Extracts operator-level compute and memory demand from the model and deployment setting. |
| **Mechanistic Critic** | Calibrates Roofline limits, filters weak candidates, diagnoses compute/HBM/communication bottlenecks, and predicts parameter-change effects. |
| **Architecture Design Agent** | Proposes NPU configurations using the workload, constraints, search history, and critic feedback. |

The critic uses deterministic calculations; the agents use an LLM. **NeuSIM** evaluates candidate latency, power, and memory usage. Prefill and decode architectures are optimized separately for **TTFT** and **TPOT**.

## Quick start

### 1. Install

Requires Conda and Python 3.12.

```bash
git clone https://github.com/bruno686/CriticNPU.git
cd CriticNPU

conda create -n neusim python=3.12 -y
conda activate neusim
python -m pip install -e "./NeuSim[dev]"
python -m pip install pandas openai python-dotenv matplotlib scipy scikit-learn numpy
```

### 2. Configure

```bash
cd chip_agent
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

### 3. Design

```bash
python main.py llama3-8b \
    --tp 4 --pp 2 --bs 16 --num-chips 8 \
    --power-budget 300 --max-hbm-size 80 \
    --disaggregated
```

→ [Experiment guide](chip_agent/README.md) · [Full reproduction](chip_agent/README.md#reproduce)

## Repository

| Directory | Contents |
| --- | --- |
| [**chip_agent/**](chip_agent/README.md) | CriticNPU, seven DSE baselines, experiment runners, and result tables. |
| [**NeuSim/**](NeuSim/README.md) | Analytical NPU simulator with performance, power, and memory modeling. |

**Baselines:** BO · CMA-ES · Greedy · Grid · ChatDSE · LLM-DSE · LLM-Box

## Citation

```bibtex
@inproceedings{criticnpu2026,
  title     = {Mechanistic Critics for Sample-Efficient {NPU} Design-Space Exploration},
  author    = {He, Yicheng and Xue, Yuqi and Wu, Jerry and Huang, Jian and Hu, Bin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
