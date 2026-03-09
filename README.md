# chip_agent

A lightweight tool that uses an LLM agent to extract per-operator compute demand from LLM model configs. It generates CSV files containing FLOP counts, tensor shapes, and execution counts for each operator — ready to feed into [NeuSim](../NeuSim/) for hardware performance simulation.

## How It Works

1. Reads a model config JSON (architecture params, parallelism degrees, batch size, etc.)
2. Sends the config to an OpenAI LLM agent, which applies architectural knowledge and sharding formulas to derive per-operator compute demand
3. The agent produces operator lists for both **prefill** and **decode** phases
4. A Python-side verification pass recomputes FLOP counts and tensor shapes to ensure correctness
5. Outputs two CSV files (`inference_prefill.csv`, `inference_decode.csv`)

## Output Format

Each CSV row represents one operator with the following columns:

| Column | Description |
|---|---|
| `Description` | Operator name (e.g., `-Q-5`, `FlashAttention-6`) |
| `Count` | Number of times the operator executes per forward pass |
| `FLOP Count` | Total floating-point operations |
| `Input Tensor Shapes` | Input tensor dimensions with dtype |
| `Output Tensor Shapes` | Output tensor dimensions with dtype |

## Supported Models

Pre-generated results are included for:

- Llama 2 13B
- Llama 3 8B
- Llama 3 70B
- Llama 3.1 405B
- DeepSeek-V2 236B (MoE)

## Setup

```bash
pip install pandas openai python-dotenv
```

Create a `.env` file in this directory with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## Usage

```bash
python computedemandagent.py <model_config.json> [options]
```

### Options

| Flag | Description |
|---|---|
| `--tp` | Tensor parallelism degree |
| `--pp` | Pipeline parallelism degree |
| `--dp` | Data parallelism degree |
| `--ep` | Expert parallelism degree |
| `--bs` | Global batch size |
| `--dpdcn` | Data parallel degree (DCN) |
| `--tpdcn` | Tensor parallel degree (DCN) |
| `--ppdcn` | Pipeline parallel degree (DCN) |
| `--epdcn` | Expert parallel degree (DCN) |
| `--output-dir` | Output directory (default: `results/<model>/<parallelism>/`) |

### Example

```bash
python computedemandagent.py ../NeuSim/configs/models/llama3-8b.json --tp 4 --pp 2 --bs 16
```

This generates:
```
results/llama3-8b/dp1-tp4-pp2-dpdcn1-tpdcn1-ppdcn1-bs16/inference_prefill.csv
results/llama3-8b/dp1-tp4-pp2-dpdcn1-tpdcn1-ppdcn1-bs16/inference_decode.csv
```

## Project Structure

```
chip_agent/
├── computedemandagent.py   # Main agent script
├── results/                # Pre-generated compute demand CSVs
│   ├── deepseekv2-236b/
│   ├── llama2-13b/
│   ├── llama3-8b/
│   ├── llama3-70b/
│   └── llama3_1-405b/
└── .env                    # OpenAI API key (not committed)
```
