"""Compile the upstream CPUAdam change and run its focused CPU tests."""

from pathlib import Path
import sys

import pytest
import torch
from torch.utils.cpp_extension import load

from deepspeed.ops.op_builder import CPUAdamBuilder, FusedAdamBuilder


def build_extensions():
    root = Path('upstream/DeepSpeed').resolve()
    modules = {}
    for kind in (CPUAdamBuilder,FusedAdamBuilder):
        builder = kind()
        modules[builder.NAME] = load(
            name=f'{builder.NAME}_gil',
            sources=[str(root/path) for path in builder.sources()],
            extra_include_paths=[str(root/path) for path in builder.include_paths()],
            extra_cflags=builder.cxx_args(),
            extra_ldflags=builder.extra_ldflags()+['-fopenmp'],
        )
    return modules


def main():
    torch.set_num_threads(2)
    modules = build_extensions()
    for kind in (CPUAdamBuilder,FusedAdamBuilder):
        kind.load = lambda self,verbose=True:modules[self.NAME]
    root = Path('upstream/DeepSpeed').resolve()
    sys.path.insert(0,str(root/'tests'))
    tests = root/'tests/unit/ops/adam/test_cpu_adam.py'
    raise SystemExit(pytest.main([
        str(tests),'-q',f'--confcutdir={tests.parent}',
        '-k','test_cpu_adam_concurrent_updates or test_cpu_adam_strict_state_updates or test_cpu_adam_rejects_amsgrad',
    ]))


if __name__ == '__main__':
    main()
