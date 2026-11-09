"""Check ZeRO-Offload against the same accumulated BF16 model gradients."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401 -- register the Ascend backend

from offload import PipelinedAdamW
from zero import create_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/zero_correctness.json'))
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(57)
    model = torch.nn.Sequential(torch.nn.Linear(67,129),torch.nn.GELU(),
                                torch.nn.Linear(129,31)).to('npu',dtype=torch.bfloat16)
    reference = copy.deepcopy(model)
    pipeline = PipelinedAdamW(reference.parameters(),lr=.003,weight_decay=.1,
                             max_grad_norm=.1,bucket_mb=.01,depth=2)
    engine, rendezvous = create_engine(model,batch=7,accumulation=3,bucket_mb=.01,
                                       max_grad_norm=.1,lr=.003,weight_decay=.1)
    errors = []
    for step in range(4):
        pipeline.zero_grad()
        for _ in range(3):
            x = torch.randn(7,67,device='npu',dtype=torch.bfloat16)
            y = torch.randn(7,31,device='npu',dtype=torch.bfloat16)
            loss = (engine(x).float()-y.float()).square().mean()
            engine.backward(loss)
            engine.step()
            ((reference(x).float()-y.float()).square().mean()/3).backward()
        pipeline.step()
        error = max(float((p.detach().float()-r.detach().float()).abs().max())
                    for p,r in zip(model.parameters(),reference.parameters()))
        errors.append(error)
        for p,r in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p.float(),r.float(),atol=.001,rtol=.01)
    pipeline.close()
    torch.distributed.destroy_process_group()
    rendezvous.cleanup()
    output = {'steps':4,'accumulation':3,'dtype':'bfloat16',
              'max_weight_errors':errors,'passed':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
