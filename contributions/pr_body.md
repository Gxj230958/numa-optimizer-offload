Release the Python GIL during native CPU Adam updates so independent CPU workers
can update concurrently and the training thread can submit gradient downloads
and updated-weight uploads. The optimizer registry lookup and step-state update
retain the GIL; each kernel invocation uses its own copy of the scalar optimizer
configuration.

CPU `FusedAdam` shares that configuration across parameter groups, so the change
covers its caller as well as `DeepSpeedCPUAdam`. The added concurrent-update test
runs two independent optimizers with different hyperparameters and compares
parameters and both moments against PyTorch AdamW over four steps. The separate
ZenFlow update path retains its existing behavior.

The open CPUAdam/GIL pull-request and issue searches on November 8, 2026 found
no existing change implementing this native-update GIL release.

## Complete training measurements

Public pretrained Qwen3-0.6B trains with BF16 parameters and FP32 CPU AdamW
master weights and moments. A bounded four-worker pipeline uses 8 MiB buckets,
eight OpenMP threads per worker and affinity to 48 physical cores on the NPU's
local NUMA node. Both variants use the same native compiler flags, sequence
length 256, microbatch one, two-way accumulation and global clipping at 1.0.
The public document inputs are separate from calibration inputs.

Each entry is the median of six complete training steps after two warmup steps
in a fresh process. Variant order alternates across three paired seeds. Loading,
optimizer construction and compilation precede timing.

| Seed | Upstream step | GIL-release step | Speedup | NPU peak, both |
| --- | ---: | ---: | ---: | ---: |
| 700 | 729.22 ms | 636.85 ms | 1.145× | 3.764 GiB |
| 701 | 720.79 ms | 626.29 ms | 1.151× | 3.764 GiB |
| 702 | 723.13 ms | 625.30 ms | 1.156× | 3.764 GiB |

The median paired speedup is **1.151×**. Peak process CPU RSS is at most
8.577 GiB for either variant. All paired gradient norms match exactly, and the
maximum same-input loss difference across the 24 warmup/measured step pairs is
`4.77e-7`. These throughput measurements exercise concurrent CPUAdam through
the bucket pipeline; the ZeRO engine comparison below validates updates.

## Validation

Hardware: two Intel Xeon Platinum 8468 sockets and one Ascend 910B2C, with
driver 25.0.rc1.1, CANN 8.5.1, PyTorch 2.9.0 and torch-npu
2.9.0.post1+gitdc51c2d. Submission validation uses DeepSpeed
`0.19.10+02f7978`, based on master revision
`02f7978481cb3d3297f111c02b16d768b890b256`. The paired training measurements
use revision `1a956270f42fbe7ffa485fdff45499eb77318469`; the native CPU Adam
sources are unchanged between these revisions.

- 53 focused CPU tests pass: strict Adam/AdamW state updates, concurrent
  CPUAdam/CPU FusedAdam updates, unsupported AMSGrad rejection and preservation
  of optimizer-state dtypes across checkpoint restoration.
- Changed-file pre-commit checks pass, including clang-format 18.1.3.
- Ascend FP32/BF16 reference comparisons pass four consecutive steps with
  three-way gradient accumulation, global clipping, pipeline depths one and
  three, and checkpoint restore. Maximum tested FP32 weight error: `4.47e-8`.
- Actual single-rank ZeRO stage 2 with BF16 parameters and CPU optimizer offload
  passes four accumulated updates. Maximum reference weight difference:
  `0.00048828125`.

The focused tests can be selected in a CPU DeepSpeed installation with:

```bash
PYTHONPATH="$PWD/tests" DS_ACCELERATOR=cpu OMP_NUM_THREADS=2 MAX_JOBS=1 pytest \
  --confcutdir=tests/unit/ops/adam \
  tests/unit/ops/adam/test_cpu_adam.py \
  -k 'test_cpu_adam_strict_state_updates or test_cpu_adam_concurrent_updates or test_cpu_adam_rejects_amsgrad or (test_cpu_optimizer_load_state_dict_keeps_state_dtype and not lion and not adagrad)' -q
pre-commit run --files csrc/adam/cpu_adam_impl.cpp \
  tests/unit/ops/adam/test_cpu_adam.py
```

The CI selector requests the complete `tests/unit/v1` scope for the C++ change.
The CPU workflow also runs `tests/unit`, which includes the added regression.

## Limitations

Device integration covers single-rank Ascend BF16 CPU offload and the FP32/BF16
optimizer reference cases above. ZeRO stages 1/3, multi-rank execution, FP16,
CUDA, XPU and other accelerators were not measured. Short training comparisons
establish update agreement and execution cost; they do not establish downstream
convergence.
