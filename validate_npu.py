"""Compare offloaded NPU updates with independent FP32 CPU AdamW updates."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401 -- register the Ascend backend

from offload import PipelinedAdamW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-optimizer', choices=['torch', 'deepspeed', 'fused'], default='torch')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        for depth in (1, 3):
            torch.manual_seed(20)
            model = torch.nn.Sequential(torch.nn.Linear(67, 129), torch.nn.GELU(), torch.nn.Linear(129, 31)).to(device='npu', dtype=dtype)
            masters = [p.detach().cpu().float().requires_grad_() for p in model.parameters()]
            reference = torch.optim.AdamW(masters, lr=0.003, weight_decay=0.1)
            pipeline = PipelinedAdamW(model.parameters(), bucket_mb=0.01, depth=depth,
                                     lr=0.003, weight_decay=0.1, max_grad_norm=0.1,
                                     cpu_optimizer=args.cpu_optimizer)
            errors = []
            for step in range(4):
                pipeline.zero_grad()
                for _ in range(3):
                    x = torch.randn(7, 67, device='npu', dtype=dtype)
                    y = torch.randn(7, 31, device='npu', dtype=dtype)
                    ((model(x).float()-y.float()).square().mean()/3).backward()
                for master, p in zip(masters, model.parameters()):
                    master.grad = p.grad.detach().cpu().float()
                reference_norm = torch.nn.utils.clip_grad_norm_(masters, 0.1)
                reference.step()
                pipeline_norm = pipeline.step()
                for master, p in zip(masters, model.parameters()):
                    expected = master.detach().to(dtype).float()
                    actual = p.detach().cpu().float()
                    errors.append(float((expected-actual).abs().max()))
                    try:
                        torch.testing.assert_close(actual, expected, rtol=3e-6,
                            atol=2e-7 if dtype == torch.float32 else 0.0005,
                            msg=lambda msg:f'{dtype}, depth={depth}, step={step}: {msg}')
                    except AssertionError:
                        cpu_master = torch.empty(p.numel(), dtype=torch.float32)
                        for bucket in pipeline.buckets:
                            for part in bucket.slices:
                                if part.parameter is p:
                                    n = part.end-part.start
                                    cpu_master[part.start:part.end].copy_(
                                        bucket.master.detach()[part.offset:part.offset+n])
                        uploaded = cpu_master.reshape(p.shape).to(dtype).float()
                        print(json.dumps({
                            'reference_norm':float(reference_norm),
                            'pipeline_norm':pipeline_norm,
                            'cpu_master_error':float((uploaded-expected).abs().max()),
                            'weight_transfer_error':float((actual-uploaded).abs().max()),
                        }),flush=True)
                        raise
                if step == 1:
                    state = copy.deepcopy(pipeline.state_dict())
                    pipeline.close()
                    pipeline = PipelinedAdamW(model.parameters(), bucket_mb=0.01, depth=depth,
                                             cpu_optimizer=args.cpu_optimizer)
                    pipeline.load_state_dict(state)
            pipeline.close()
            rows.append({'dtype': str(dtype), 'depth': depth, 'steps': 4,
                         'cpu_optimizer': args.cpu_optimizer,
                         'accumulation_steps': 3, 'max_weight_error': max(errors),
                         'checkpoint_resume': True, 'passed': True})
    output = args.output or Path(f'results/npu_correctness_{args.cpu_optimizer}.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
