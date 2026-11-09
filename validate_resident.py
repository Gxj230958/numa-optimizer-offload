"""Validate the fused NPU resident baseline's persistent FP32 gradient buffers."""

import json
from pathlib import Path

import torch
import torch_npu


def main():
    torch.npu.set_device(0)
    torch.manual_seed(82)
    parameters = [torch.randn(67,129,device='npu',requires_grad=True),
                  torch.randn(31,device='npu',requires_grad=True)]
    reference = [p.detach().cpu().clone().requires_grad_() for p in parameters]
    npu = torch_npu.optim.NpuFusedAdamW(parameters,lr=.003,weight_decay=.1)
    cpu = torch.optim.AdamW(reference,lr=.003,weight_decay=.1,foreach=False)
    errors = []
    for step in range(4):
        for p,r in zip(parameters,reference):
            gradient = torch.randn_like(r)
            r.grad = gradient
            if p.grad is None:
                p.grad = gradient.to('npu')
            else:
                p.grad.copy_(gradient)
        torch.nn.utils.clip_grad_norm_(parameters,.1)
        torch.nn.utils.clip_grad_norm_(reference,.1)
        npu.step();cpu.step()
        for p,r in zip(parameters,reference):
            torch.testing.assert_close(p.detach().cpu(),r.detach(),atol=5e-7,rtol=3e-6)
            errors.append(float((p.detach().cpu()-r.detach()).abs().max()))
        npu.zero_grad(set_to_none=False)
    result = {'steps':4,'max_weight_error':max(errors),'passed':True}
    Path('results/resident_correctness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
