"""Single-NPU DeepSpeed ZeRO-Offload baseline with FP32 CPU AdamW state."""

import os
from pathlib import Path
import tempfile


def create_engine(model, *, batch, accumulation, bucket_mb, max_grad_norm=1.,
                  lr=1e-5, weight_decay=.01):
    import deepspeed

    os.environ['LOCAL_RANK'] = '0'
    rendezvous = tempfile.TemporaryDirectory(prefix='qwen-zero-')
    deepspeed.init_distributed(
        dist_backend='hccl', auto_mpi_discovery=False,
        init_method=(Path(rendezvous.name)/'rendezvous').as_uri(), rank=0, world_size=1)
    config = {
        'train_micro_batch_size_per_gpu': batch,
        'gradient_accumulation_steps': accumulation,
        'gradient_clipping': max_grad_norm,
        'bf16': {'enabled': True},
        'optimizer': {'type': 'Adam', 'params': {
            'lr': lr, 'betas': [.9, .999], 'eps': 1e-8,
            'weight_decay': weight_decay, 'adam_w_mode': True}},
        'zero_optimization': {
            'stage': 2,
            'offload_optimizer': {'device': 'cpu', 'pin_memory': True},
            'contiguous_gradients': True,
            'overlap_comm': True,
            'reduce_bucket_size': int(bucket_mb*1024**2)//2,
        },
        'steps_per_print': 1000000,
    }
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(),
                                          config=config, dist_init_required=False)
    return engine, rendezvous
