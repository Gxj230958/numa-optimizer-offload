# Optimizer offload on Ascend

- Built a NUMA-aware CPU–NPU optimizer pipeline for full-parameter Qwen3
  training, preserving accumulated-gradient clipping and checkpoint resume
  with bounded pinned buffers and FP32 AdamW master weights and moments.
- Evaluated 33 independent 0.6B/1.7B/4B training processes on held-out public
  documents; achieved 1.43–1.59× complete-step speedups over tuned synchronous
  offload, while characterizing the model-dependent tradeoff against actual
  ZeRO-Offload, including 8.52% slower 4B steps and higher memory use.
- Measured fixed-grid training capacity: all three offload methods completed
  2,048-token Qwen3-4B updates while the tested resident optimizer ran out of
  memory at 256 tokens; smaller models shared the same measured sequence boundary.
- Implemented fused CPU AdamW and a DeepSpeed CPUAdam GIL contribution;
  measured a median 1.151× complete-training speedup across three controlled
  pairs, with 51 focused upstream CPU tests and actual Ascend FP32/BF16 and
  ZeRO-Offload numerical validation.

[Technical report](technical_report.md) ·
[Implementation](../offload.py) ·
[DeepSpeed contribution](../contributions/README.md)
