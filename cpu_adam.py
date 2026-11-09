"""Fused FP32 CPU AdamW that releases Python execution during native updates."""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


class FusedCPUAdam(torch.optim.Optimizer):
    def __init__(self, parameters, *, lr, betas, eps, weight_decay):
        super().__init__(parameters, dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
        self.native = load('qwen_cpu_adam', [str(Path(__file__).with_suffix('.cpp'))],
                           extra_cflags=['-O3','-march=native','-fopenmp','-fno-math-errno'],
                           extra_ldflags=['-fopenmp'])
        self.threads = torch.get_num_threads()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group['params']:
                state = self.state[parameter]
                if not state:
                    state.update(step=0,exp_avg=torch.zeros_like(parameter),
                                 exp_avg_sq=torch.zeros_like(parameter))
                state['step'] += 1
                self.native.update(parameter,parameter.grad,state['exp_avg'],state['exp_avg_sq'],
                                   group['lr'],*group['betas'],group['eps'],
                                   group['weight_decay'],state['step'],self.threads)
