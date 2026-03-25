#!/bin/bash
# Run all LLM models × all methods (agent + 4 baselines), disaggregated mode
# Results saved to design_results_all/

set -e
OUTPUT="design_results_all"
mkdir -p $OUTPUT

# Model configs: model_name  num_chips  tp  pp  bs  power  max_hbm_size  extra_args
declare -a MODELS=(
    "llama3-8b       8  4 2 16 300  16"
    "llama2-13b      8  4 2 16 300  32"
    "llama3-70b     64  8 8 16 300  16"
    "llama3_1-405b  64  8 8 16 300  32"
    "deepseekv2-236b 16 8 2 16 300  32"
    "deepseekv3-671b 64 8 8 16 300  16"
)

run_agent() {
    local model=$1 nc=$2 tp=$3 pp=$4 bs=$5 pwr=$6 hbm=$7
    echo "[AGENT] $model: nc=$nc tp=$tp pp=$pp bs=$bs pwr=$pwr hbm=$hbm"
    python main.py $model --tp $tp --pp $pp --bs $bs --num-chips $nc \
        --power-budget $pwr --max-hbm-size $hbm --disaggregated \
        --max-iterations 8 --output-dir $OUTPUT --no-memory \
        --skip-agent1 2>&1 | tail -5
    echo "[AGENT] $model done"
}

run_baseline() {
    local baseline=$1 model=$2 nc=$3 tp=$4 pp=$5 bs=$6 pwr=$7 hbm=$8
    echo "[${baseline}] $model: nc=$nc tp=$tp pp=$pp"
    python baseline_${baseline}.py $model --chip-version 4 --num-chips $nc \
        --tp $tp --pp $pp --bs $bs --power-budget $pwr --max-hbm-size $hbm \
        --disaggregated --output-dir $OUTPUT 2>&1 | tail -3
    echo "[${baseline}] $model done"
}

# First generate compute demand CSVs for all models (agent1)
echo "=== Step 1: Generate compute demand for all models ==="
for entry in "${MODELS[@]}"; do
    read -r model nc tp pp bs pwr hbm <<< "$entry"
    CSV_DIR="results/${model}/dp1-tp${tp}-pp${pp}-dpdcn1-tpdcn1-ppdcn1-bs${bs}"
    if [ -f "${CSV_DIR}/inference_prefill.csv" ] && [ -f "${CSV_DIR}/inference_decode.csv" ]; then
        echo "  $model: compute demand CSVs exist, skipping"
    else
        echo "  $model: generating compute demand..."
        python computedemandagent.py ../NeuSim/configs/models/${model}.json --tp $tp --pp $pp --bs $bs 2>&1 | tail -3
    fi
done

# Run baselines in parallel (no API needed)
echo ""
echo "=== Step 2: Run baselines (parallel per model) ==="
for entry in "${MODELS[@]}"; do
    read -r model nc tp pp bs pwr hbm <<< "$entry"
    run_baseline bo     $model $nc $tp $pp $bs $pwr $hbm &
    run_baseline es     $model $nc $tp $pp $bs $pwr $hbm &
    run_baseline greedy $model $nc $tp $pp $bs $pwr $hbm &
done
echo "Waiting for all baselines..."
wait
echo "All baselines done."

# Run agent sequentially (API rate limits)
echo ""
echo "=== Step 3: Run agent with world model ==="
for entry in "${MODELS[@]}"; do
    read -r model nc tp pp bs pwr hbm <<< "$entry"
    run_agent $model $nc $tp $pp $bs $pwr $hbm
done

echo ""
echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "Results in: $OUTPUT/"
