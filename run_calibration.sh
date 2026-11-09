#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
output=${1:?Usage: bash run_calibration.sh FRESH_OUTPUT_DIRECTORY}
mkdir -p "$(dirname "$output")"
mkdir "$output"
local_cores=$(seq -s, 0 47)
remote_cores=$(seq -s, 48 95)
for bucket in 8 32 128; do
    for depth in 1 2 4; do
        bash scripts/run_npu.sh 10 env OMP_NUM_THREADS=8 .venv/bin/python train.py \
            --model .cache/models/Qwen3-0.6B --mode pipeline --cpu-optimizer fused \
            --bucket-mb "$bucket" --depth "$depth" --threads 8 --cpu-list "$local_cores" \
            --sequence 128 --accumulation 2 --warmup 2 --steps 5 --seed 123 \
            --data .cache/data/training_tokens/calibration.json \
            --output "$output/local_b${bucket}_d${depth}.json" \
            > "$output/local_b${bucket}_d${depth}.log" 2>&1
    done
done
for location in local remote; do
    if [[ "$location" == local ]]; then cores=$local_cores; else cores=$remote_cores; fi
    for threads in 4 8 16 32; do
        if [[ "$location" == local && "$threads" == 8 ]]; then continue; fi
        bash scripts/run_npu.sh 10 env OMP_NUM_THREADS="$threads" .venv/bin/python train.py \
            --model .cache/models/Qwen3-0.6B --mode pipeline --cpu-optimizer fused \
            --bucket-mb 32 --depth 2 --threads "$threads" --cpu-list "$cores" \
            --sequence 128 --accumulation 2 --warmup 2 --steps 5 --seed 123 \
            --data .cache/data/training_tokens/calibration.json \
            --output "$output/${location}_threads${threads}.json" \
            > "$output/${location}_threads${threads}.log" 2>&1
    done
done
for mode in sync zero; do
    for setting in '8 8' '32 8' '128 8' '32 4' '32 16' '32 32'; do
        read -r bucket threads <<< "$setting"
        bash scripts/run_npu.sh 10 env OMP_NUM_THREADS="$threads" .venv/bin/python train.py \
            --model .cache/models/Qwen3-0.6B --mode "$mode" --bucket-mb "$bucket" \
            --threads "$threads" --cpu-list "$local_cores" --sequence 128 \
            --accumulation 2 --warmup 2 --steps 5 --seed 123 \
            --data .cache/data/training_tokens/calibration.json \
            --output "$output/${mode}_b${bucket}_t${threads}.json" \
            > "$output/${mode}_b${bucket}_t${threads}.log" 2>&1
    done
done
.venv/bin/python select_plan.py "$output"
