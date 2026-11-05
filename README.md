# CriticNPU: Mechanistic Critics for Sample-Efficient NPU Design-Space Exploration

Official implementation of **"Mechanistic Critics for Sample-Efficient NPU
Design-Space Exploration"**, accepted at **NeurIPS 2026**.

**Authors:** Yicheng He, Yuqi Xue, Jerry Wu, Jian Huang, and Bin Hu

**Affiliation:** University of Illinois Urbana-Champaign

The system designs a custom NPU configuration for a given LLM inference workload
(model, parallelism, batch size, power budget). An online-calibrated mechanistic
critic screens candidates before simulation and provides bottleneck-guided
feedback afterward. Across six LLM workloads and 16 deployment settings per
model, CriticNPU achieves best-or-tied-best latency in 74% of prefill and 87.5%
of decode settings with roughly half the simulator calls of competitive baselines.

## Repository structure

```
.
├── chip_agent/           # The agent — proposes NPU designs (this paper)
│   ├── main.py           # End-to-end pipeline
│   ├── npudesignagent.py # Design agent + roofline world model
│   ├── computedemandagent.py / bottleneck_analysis.py
│   ├── world_model.py / design_memory.py / neusim_runner.py
│   ├── baseline_{bo,es,greedy,grid,chatdse,llm_dse,llm_box}.py
│   ├── run_all_experiments.py / run_llm_baselines.py / generate_tables.py
│   └── README.md         # ← detailed method + reproduction guide
│
├── NeuSim/               # The simulator the agent calls
│   ├── neusim/           # Performance, power, carbon modeling
│   ├── configs/          # Chip + model + system JSON configs
│   ├── run_scripts/      # Standalone NeuSim launchers
│   └── README.md         # ← NeuSim documentation
│
└── README.md             # this file
```

## What's in each subdirectory

### `chip_agent/` — CriticNPU and baselines

A critic-guided search framework with three components:

1. **Compute Demand Agent** — extracts operator-level compute and memory demand from the LLM specification and deployment setting.
2. **Online-Calibrated Roofline Critic** — filters weak candidates, summarizes compute, memory, and communication bottlenecks, and predicts the effect of parameter changes. The critic is deterministic; it does not make LLM calls.
3. **Architecture Design Agent** — proposes the next NPU configuration using the workload, constraints, search history, and critic feedback.

Comes with seven baselines: BO, CMA-ES, Greedy, Grid, ChatDSE, LLM-DSE, LLM-Box.

See [chip_agent/README.md](chip_agent/README.md) for installation,
reproduction commands, and the search space specification.

### `NeuSim/` — NPU simulator

NeuSim is an open-source analytical simulator for NPU performance, power,
and carbon. The agent calls it via `chip_agent/neusim_runner.py`. NeuSim
is a separate, independently usable project; CriticNPU uses it as the
evaluation engine but does not modify its internals.

See [NeuSim/README.md](NeuSim/README.md) for NeuSim documentation.

## Quick start

```bash
# 1. Set up NeuSim (the simulator)
cd NeuSim
conda create -n neusim python=3.12 && conda activate neusim
pip install -e ".[dev]"

# 2. Set up the agent
cd ../chip_agent
pip install pandas openai python-dotenv matplotlib scipy scikit-learn
cp .env.example .env       # then add your OPENAI_API_KEY

# 3. Run the agent on one workload
python main.py llama3-8b \
    --tp 4 --pp 2 --bs 16 --num-chips 8 \
    --power-budget 300 --max-hbm-size 80 \
    --disaggregated
```

For the full 6-model × 16-deployment × 8-method experiment used in the paper,
see the reproduction section of [chip_agent/README.md](chip_agent/README.md).

## Citation

```
@inproceedings{criticnpu2026,
  title     = {Mechanistic Critics for Sample-Efficient {NPU} Design-Space Exploration},
  author    = {He, Yicheng and Xue, Yuqi and Wu, Jerry and Huang, Jian and Hu, Bin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

## License

[Add license here]
