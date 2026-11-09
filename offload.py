"""Bounded gradient/AdamW/weight pipeline for dense CPU and Ascend training."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import threading
import time
from typing import Iterable

import torch


@dataclass
class Slice:
    parameter: torch.Tensor
    start: int
    end: int
    offset: int


@dataclass
class Bucket:
    slices: list[Slice]
    master: torch.Tensor
    optimizer: torch.optim.Optimizer


class PipelinedAdamW:
    """FP32 CPU AdamW with at most ``depth`` transfers/updates in flight.

    Call after the complete accumulated backward pass. All trainable parameters
    must have dense gradients. Global clipping precedes every parameter update.
    ``depth=1`` is the synchronous offload baseline with identical numerics.
    """

    def __init__(self, parameters: Iterable[torch.Tensor], *, bucket_mb=32,
                 depth=2, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, max_grad_norm=1.0, cpu_optimizer="torch",
                 record_cpu_trace=False):
        self.parameters = [p for p in parameters if p.requires_grad]
        if not self.parameters:
            raise ValueError("No trainable parameters")
        self.device = self.parameters[0].device
        self.dtype = self.parameters[0].dtype
        if self.device.type not in ("cpu", "npu"):
            raise ValueError("Supported devices are CPU and Ascend NPU")
        if any(p.device != self.device or p.dtype != self.dtype or not p.is_contiguous()
               for p in self.parameters):
            raise ValueError("Parameters must be contiguous with one device and dtype")
        if depth < 1 or bucket_mb <= 0:
            raise ValueError("depth and bucket_mb must be positive")
        self.depth = depth
        self.cpu_optimizer = cpu_optimizer
        self.record_cpu_trace = record_cpu_trace
        self.cpu_trace = []
        self.max_grad_norm = max_grad_norm
        self.npu = self.device.type == "npu"
        self.bucket_elements = max(1, int(bucket_mb * 1024**2) // self.parameters[0].element_size())
        self.buckets = []
        self.pool = ThreadPoolExecutor(max_workers=depth)
        self.download_stream = torch.npu.Stream(device=self.device) if self.npu else None
        self.upload_stream = torch.npu.Stream(device=self.device) if self.npu else None
        self.slots = []
        pending, used = [], 0
        options = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        if cpu_optimizer == "deepspeed":
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer_class = DeepSpeedCPUAdam
            options.update(adamw_mode=True, fp32_optimizer_states=True)
        elif cpu_optimizer == "fused":
            from cpu_adam import FusedCPUAdam
            optimizer_class = FusedCPUAdam
        elif cpu_optimizer == "torch":
            optimizer_class = torch.optim.AdamW
            options["foreach"] = False
        else:
            raise ValueError("cpu_optimizer must be torch, fused or deepspeed")

        def append_bucket():
            master = torch.empty(used, dtype=torch.float32)
            for part in pending:
                master[part.offset:part.offset + part.end - part.start].copy_(
                    part.parameter.detach().view(-1)[part.start:part.end])
            master.requires_grad_(True)
            self.buckets.append(Bucket(list(pending), master, optimizer_class([master], **options)))

        for parameter in self.parameters:
            start = 0
            while start < parameter.numel():
                size = min(parameter.numel() - start, self.bucket_elements - used)
                pending.append(Slice(parameter, start, start + size, used))
                start += size
                used += size
                if used == self.bucket_elements:
                    append_bucket()
                    pending, used = [], 0
        if pending:
            append_bucket()
        capacity = max(b.master.numel() for b in self.buckets)
        for _ in range(min(depth, len(self.buckets))):
            self.slots.append((
                torch.empty(capacity, dtype=self.dtype, pin_memory=self.npu),
                torch.empty(capacity, dtype=self.dtype, pin_memory=self.npu),
            ))

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def _norm_and_scale(self):
        if any(p.grad is None or p.grad.is_sparse for p in self.parameters):
            raise ValueError("Every trainable parameter needs a dense accumulated gradient")
        norms = [torch.linalg.vector_norm(p.grad.float()) for p in self.parameters]
        norm = torch.linalg.vector_norm(torch.stack(norms))
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Non-finite global gradient norm; no parameters updated")
        scale = 1.0 if self.max_grad_norm is None else min(1.0, self.max_grad_norm / (norm.item() + 1e-6))
        return norm.item(), scale

    def _cpu_update(self, bucket, slot, ready, scale):
        if ready is not None:
            ready.synchronize()
        if self.record_cpu_trace:
            start = time.time_ns()/1000
        with torch.autograd.profiler.record_function("offload.cpu_adamw"):
            # Each update owns its FP32 gradient while staging slots are reused.
            bucket.master.grad = slot[0][:bucket.master.numel()].float().clone()
            bucket.master.grad.mul_(scale)
            bucket.optimizer.step()
            bucket.master.grad = None
            slot[1][:bucket.master.numel()].copy_(bucket.master.detach())
        if self.record_cpu_trace:
            self.cpu_trace.append({'name':'CPU AdamW bucket','ph':'X','cat':'optimizer',
                                   'pid':os.getpid(),'tid':threading.get_native_id(),
                                   'ts':start,'dur':time.time_ns()/1000-start,
                                   'args':{'elements':bucket.master.numel()}})

    @torch.no_grad()
    def _download(self, bucket, slot):
        def copy():
            for part in bucket.slices:
                n = part.end - part.start
                slot[0][part.offset:part.offset+n].copy_(
                    part.parameter.grad.view(-1)[part.start:part.end], non_blocking=self.npu)
        if self.npu:
            with torch.npu.stream(self.download_stream):
                copy()
                ready = torch.npu.Event()
                ready.record(self.download_stream)
                return ready
        copy()
        return None

    @torch.no_grad()
    def _upload(self, bucket, slot):
        def copy():
            for part in bucket.slices:
                n = part.end - part.start
                part.parameter.view(-1)[part.start:part.end].copy_(
                    slot[1][part.offset:part.offset+n], non_blocking=self.npu)
        if self.npu:
            with torch.npu.stream(self.upload_stream):
                copy()
                done = torch.npu.Event()
                done.record(self.upload_stream)
                return done
        copy()
        return None

    def step(self):
        norm, scale = self._norm_and_scale()
        if self.npu:
            self.download_stream.wait_stream(torch.npu.current_stream(self.device))
        active = deque()
        available = deque((slot, None) for slot in self.slots)

        def finish():
            bucket, slot, future = active.popleft()
            future.result()
            available.append((slot, self._upload(bucket, slot)))

        for bucket in self.buckets:
            if not available:
                finish()
            slot, uploaded = available.popleft()
            if uploaded is not None:
                uploaded.synchronize()
            with torch.autograd.profiler.record_function("offload.gradient_d2h"):
                ready = self._download(bucket, slot)
            future = self.pool.submit(self._cpu_update, bucket, slot, ready, scale)
            active.append((bucket, slot, future))
        while active:
            finish()
        for _, uploaded in available:
            if uploaded is not None:
                uploaded.synchronize()
        return norm

    def state_dict(self):
        return {"masters": [b.master.detach().clone() for b in self.buckets],
                "optimizers": [b.optimizer.state_dict() for b in self.buckets],
                "parameter_shapes": [tuple(p.shape) for p in self.parameters],
                "bucket_elements": self.bucket_elements,
                "cpu_optimizer": self.cpu_optimizer,
                "max_grad_norm": self.max_grad_norm}

    @torch.no_grad()
    def load_state_dict(self, state):
        if (state["parameter_shapes"] != [tuple(p.shape) for p in self.parameters]
                or state["bucket_elements"] != self.bucket_elements
                or state["cpu_optimizer"] != self.cpu_optimizer):
            raise ValueError("Checkpoint requires the same parameter shapes, bucket size and CPU optimizer")
        self.max_grad_norm = state["max_grad_norm"]
        for bucket, master, optimizer in zip(self.buckets, state["masters"], state["optimizers"], strict=True):
            bucket.master.copy_(master)
            bucket.optimizer.load_state_dict(optimizer)
            for part in bucket.slices:
                n = part.end-part.start
                part.parameter.view(-1)[part.start:part.end].copy_(master[part.offset:part.offset+n])

    def close(self):
        self.pool.shutdown(wait=True)
