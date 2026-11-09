"""Full-parameter Qwen3 training with resident and offloaded AdamW."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import resource
import random
import time

import torch
import torch_npu
from transformers import AutoConfig, AutoModelForCausalLM

from offload import PipelinedAdamW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--mode', choices=['resident', 'sync', 'pipeline', 'zero'], required=True)
    parser.add_argument('--cpu-optimizer', choices=['torch', 'deepspeed', 'fused'], default='fused')
    parser.add_argument('--resident-optimizer', choices=['fused','torch'], default='fused')
    parser.add_argument('--bucket-mb', type=float, default=32)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--sequence', type=int, default=128)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--cpu-list')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--random-init', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.cpu_list:
        os.sched_setaffinity(0, [int(x) for x in args.cpu_list.split(',')])
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.npu.set_device(0)
    if args.random_init:
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        config._attn_implementation = 'eager'
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='eager')
    model = model.to('npu').train()
    model.config.use_cache = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    engine = rendezvous = None
    if args.mode == 'zero':
        from zero import create_engine
        engine, rendezvous = create_engine(model, batch=args.batch,
                                          accumulation=args.accumulation,
                                          bucket_mb=args.bucket_mb)
        optimizer = engine.optimizer
    elif args.mode == 'resident':
        # FP32 master weights and moments match the offload numerical contract.
        masters = [p.detach().float().requires_grad_() for p in model.parameters()]
        optimizer = (torch_npu.optim.NpuFusedAdamW(masters,lr=1e-5)
                     if args.resident_optimizer=='fused' else
                     torch.optim.AdamW(masters, lr=1e-5, foreach=False))
    else:
        optimizer = PipelinedAdamW(model.parameters(), bucket_mb=args.bucket_mb,
                                  depth=1 if args.mode == 'sync' else args.depth,
                                  lr=1e-5, max_grad_norm=1.,
                                  cpu_optimizer=args.cpu_optimizer,
                                  record_cpu_trace=args.profile)
    generator = torch.Generator().manual_seed(args.seed+1)
    if args.data:
        dataset = json.loads(args.data.read_text())
        documents = [d['token_ids'] for d in dataset['documents']
                     if len(d['token_ids'])>=args.sequence]
        rng = random.Random(args.seed+1)
        batches = []
        for _ in range(args.accumulation):
            rows = []
            for _ in range(args.batch):
                document = rng.choice(documents)
                start = rng.randrange(len(document)-args.sequence+1)
                rows.append(document[start:start+args.sequence])
            batches.append(torch.tensor(rows,device='npu'))
        training_inputs = {'source':dataset['source'],'partition':dataset['partition']}
    else:
        batches = [torch.randint(0, model.config.vocab_size,
                             (args.batch, args.sequence), generator=generator).to('npu')
               for _ in range(args.accumulation)]
        training_inputs = 'seeded synthetic token IDs'
    trace_dir = str(args.output.with_suffix('')) + '_trace'
    profiler = (torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir),
        schedule=torch_npu.profiler.schedule(wait=0, warmup=args.warmup,
                                             active=args.steps, repeat=1),
        record_shapes=True) if args.profile else contextlib.nullcontext())
    rows = []
    with profiler:
        for step in range(args.warmup+args.steps):
            if engine is None:
                model.zero_grad(set_to_none=True)
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            losses = []
            if engine is not None:
                for batch in batches:
                    loss = engine(input_ids=batch, labels=batch).loss
                    losses.append(loss.detach())
                    engine.backward(loss)
                    if engine.is_gradient_accumulation_boundary():
                        torch.npu.synchronize()
                        backward_end = time.perf_counter()
                    engine.step()
                norm = float(engine.get_global_grad_norm())
            else:
                for batch in batches:
                    loss = model(input_ids=batch, labels=batch).loss
                    losses.append(loss.detach())
                    (loss/args.accumulation).backward()
                torch.npu.synchronize()
                backward_end = time.perf_counter()
            if args.mode == 'resident':
                for p, master in zip(model.parameters(), masters):
                    if master.grad is None:
                        master.grad = p.grad.float()
                    else:
                        master.grad.copy_(p.grad)
                norm = torch.nn.utils.clip_grad_norm_(masters, 1.).item()
                optimizer.step()
                with torch.no_grad():
                    for p, master in zip(model.parameters(), masters):
                        p.copy_(master)
                optimizer.zero_grad(set_to_none=args.resident_optimizer!='fused')
            elif engine is None:
                norm = optimizer.step()
            torch.npu.synchronize()
            end = time.perf_counter()
            row = {'step': step, 'warmup': step < args.warmup,
                   'step_ms': (end-start)*1000,
                   'forward_backward_ms': (backward_end-start)*1000,
                   'optimizer_ms': (end-backward_end)*1000,
                   'peak_npu_bytes': torch.npu.max_memory_allocated(),
                   'peak_npu_reserved_bytes':torch.npu.max_memory_reserved(),
                   'cpu_max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'loss': float(torch.stack(losses).mean().cpu()),
                   'grad_norm': norm}
            rows.append(row)
            print(json.dumps(row), flush=True)
            if args.profile:
                profiler.step()
    if isinstance(optimizer, PipelinedAdamW):
        optimizer.close()
        if args.profile:
            args.output.with_suffix('.cpu_trace.json').write_text(
                json.dumps({'traceEvents':optimizer.cpu_trace},indent=2)+'\n')
    result = {'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'training_inputs': training_inputs, 'steps': rows}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if engine is not None:
        torch.distributed.destroy_process_group()
        rendezvous.cleanup()


if __name__ == '__main__':
    main()
