# NUMA-Aware Optimizer Offload for LLM Training on Ascend NPUs

A bounded CPU–NPU optimizer-offload pipeline for full-parameter training of
Qwen3 models on an Ascend 910B, together with an upstream DeepSpeed patch that
releases the Python GIL during native CPU Adam updates.

## Overview

FP32 AdamW master weights and both moments live on the CPU; BF16 model weights
and accumulated gradients stay on the NPU. After the complete accumulated
backward pass, parameters are divided into fixed-capacity buckets (large
tensors are split at element boundaries) and three operations are pipelined
across buckets: gradient download to pinned host memory, the CPU AdamW update,
and upload of the updated BF16 weights. Global gradient clipping is applied
once per step before any update, and the optimizer call returns only after
every updated weight is back on the NPU, so parameter semantics are unchanged.
Depth one of the same implementation is the synchronous-offload baseline.

- **Bounded gradient transfer**: for bucket capacity B elements and in-flight
  depth d, pinned staging is bounded by 4dB bytes (one BF16 download and one
  BF16 upload buffer per slot); a staging slot is reused only after its weight
  upload completes.
- **Measured speed on real Qwen training**: across 33 independent evaluation
  processes (198 timed updates) on held-out public SQuAD 2.0 documents, the
  pipeline reaches **1.430× / 1.516× / 1.590×** complete-step speedups over
  independently tuned synchronous offload on Qwen3-0.6B / 1.7B / 4B. Against
  actual DeepSpeed ZeRO-Offload: **1.247× on 0.6B**, statistically overlapping
  on 1.7B (0.63% median gap), and **8.52% slower on 4B** — with higher NPU and
  CPU memory peaks than ZeRO in all cases.
- **NUMA, bucket and concurrency tuning**: 28 calibration runs over bucket
  sizes 8/32/128 MiB, depths 1/2/4, 4–32 CPU threads and local/remote NUMA
  affinity, selected on complete-step time (8 MiB buckets, depth 4, 8 local
  threads won); more threads is not monotonically better, and remote affinity
  adds visible cost.
- **Measured capacity bounds**: all three offload methods complete 2,048-token
  Qwen3-4B updates where the tested resident fused optimizer (20 bytes of
  state per parameter) OOMs at 256 tokens; all methods share the measured
  2,048-token grid boundary on 0.6B/1.7B/4B.
- **DeepSpeed CPUAdam GIL patch**: releasing the GIL around the native update
  (including the shared CPU FusedAdam configuration) yields a median
  **1.151×** complete-training speedup (range 1.145–1.156×) across three
  paired Qwen3-0.6B runs; 53 focused upstream CPU tests pass and submitted as
  [PR #9035](https://github.com/deepspeedai/DeepSpeed/pull/9035).

## Hardware

Measurements use one 64 GiB Ascend 910B2C NPU (PCIe device `0000:48:00.0`,
active 32.0 GT/s ×8 link) attached to NUMA node 0 of a host with two Intel
Xeon Platinum 8468 sockets (48 physical cores each, 192 logical CPUs). Runtime:
Python 3.11.14, PyTorch 2.9.0, torch-npu 2.9.0.post1, CANN 8.5.1,
Transformers 4.57.6, DeepSpeed 0.19.9 — full inventory in
[`environment.json`](environment.json).

## Links

- Technical report (full methods, memory tables, ranges, limitations):
  [`reports/technical_report.md`](reports/technical_report.md)
  (Chinese walkthrough: [`reports/中文讲解.md`](reports/中文讲解.md))
- DeepSpeed draft PR #9035 — release the GIL during native CPU Adam:
  https://github.com/deepspeedai/DeepSpeed/pull/9035
- Method lineage — SuperOffload:
  https://supercomputing-system-ai-lab.github.io/projects/superoffload/
- Upstream reference: [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)

## Repository contents

- `offload.py` — `PipelinedAdamW`: bounded-bucket offload pipeline (download →
  CPU AdamW → weight write-back) with `torch`, `fused` and `deepspeed` CPU
  optimizer backends and checkpoint save/restore.
- `cpu_adam.cpp` / `cpu_adam.py` — fused FP32 CPU AdamW kernel (OpenMP + SIMD)
  that releases the Python GIL during the native update.
- `train.py` — full-parameter Qwen3 training driver with `resident`, `sync`,
  `pipeline` and actual DeepSpeed `zero` (stage 2, CPU optimizer offload)
  modes, CPU affinity control and whole-step/optimizer timing.
- `test_offload.py` — independent CPU-reference numerical tests (accumulated
  gradients, global clipping, depths 1/3, checkpoint resume, FP32/BF16).
- `run_calibration.sh`, `select_plan.py`, `plot_calibration.py` — joint
  bucket/depth/thread/NUMA calibration on complete training steps.
- `run_study.py`, `summarize.py` — three-process, per-model evaluation driver
  and aggregation; results in `results/evaluation_public/`.
- `run_capacity.py`, `plot_capacity.py` — measured sequence-capacity grid
  (256–8192 tokens); results in `results/capacity_public/`.
- `analyze_pipeline.py`, `plot_overlap.py` — CPU-worker/PCIe-DMA interval
  overlap analysis (51.71% of recorded DMA time overlapped CPU updates in the
  profiled capture); records in `results/qwen06_pipeline_overlap/`.
- `validate_npu.py`, `validate_zero.py`, `validate_resident.py`,
  `run_deepspeed_patch.py`, `build_cpu_adam_pair.py`, `check_cpu_adam_patch.py`,
  `run_cpu_adam_comparison.py`, `plot_cpu_adam_comparison.py` — correctness
  and paired-throughput harnesses for the baselines and the GIL patch.
- `contributions/` — DeepSpeed PR material: `cpu_adam_gil.patch`, the PR body
  and submission context.
- `reports/` — technical report (EN/中文), CV summary, teaching/interview
  notes; `environment.json` — pinned software/hardware/model-revision record;
  `results/` — raw per-run JSONs, summary tables and figures.
- `scripts/` — CANN environment + card health guard (`run_npu.sh`), pinned
  ModelScope Qwen downloader, SQuAD tokenizer.

## Reproduction

Requires an Ascend 910B host with CANN 8.5.1 and a Python 3.11 environment.

```bash
python -m venv --system-site-packages .venv
DS_BUILD_OPS=0 .venv/bin/python -m pip install -r requirements.txt

# Public model weights (Qwen ModelScope revisions pinned in environment.json)
.venv/bin/python scripts/download_qwen.py --sources environment.json

# Public SQuAD 2.0 dev documents → calibration/evaluation token partitions
mkdir -p .cache/data
curl -L --fail https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json \
  -o .cache/data/squad-v2-dev.json
.venv/bin/python scripts/tokenize_training_data.py \
  --model .cache/models/Qwen3-0.6B --data .cache/data/squad-v2-dev.json \
  --output .cache/data/training_tokens

# Numerical tests against the independent CPU reference
.venv/bin/python -m unittest test_offload.py

# Correctness on the NPU (fused CPU AdamW and actual ZeRO-Offload);
# run_npu.sh sources CANN and refuses to start unless the card is healthy/idle
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
bash scripts/run_npu.sh 10 .venv/bin/python validate_npu.py --cpu-optimizer fused
bash scripts/run_npu.sh 10 .venv/bin/python validate_zero.py

# Smallest end-to-end training run
bash scripts/run_npu.sh 10 .venv/bin/python train.py \
  --model .cache/models/Qwen3-0.6B --mode pipeline --bucket-mb 32 --depth 2 \
  --sequence 128 --accumulation 2 --steps 6 --warmup 2 \
  --output results/qwen06_pipeline.json

# Full studies (each requires an exclusive performance window)
bash run_calibration.sh results/calibration_public
.venv/bin/python run_study.py --selection results/calibration_public/selection.json \
  --output results/evaluation_public
.venv/bin/python summarize.py results/evaluation_public
.venv/bin/python run_capacity.py --selection results/calibration_public/selection.json \
  --evaluation results/evaluation_public --output results/capacity_public
.venv/bin/python plot_capacity.py results/capacity_public
```

Card index, core lists and NUMA placement in the scripts reflect the measured
host (card 10 on NUMA node 0, local cores 0–47); adapt them to your machine.
The DeepSpeed GIL patch reproduction (clone upstream, apply
`contributions/cpu_adam_gil.patch`, run 53 focused CPU tests and the paired
Qwen3-0.6B training comparison) is documented in
[`contributions/README.md`](contributions/README.md).

## Limitations

Results apply to the single measured host, one Ascend 910B2C, dense pretrained
Qwen3 models with eager attention, and BF16 forward/backward with FP32 AdamW
state. Short training runs establish update semantics and execution cost, not
downstream-task convergence. See the technical report for the full list.
