"""Run an existing training or validation entry point with the compiled CPUAdam patch."""

import argparse
import importlib
import json
from pathlib import Path
import runpy
import sys

import torch
from deepspeed.ops.op_builder import CPUAdamBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['base','gil'],default='gil')
    parser.add_argument('entry')
    args, remaining = parser.parse_known_args()
    name = f'cpu_adam_{args.variant}'
    sys.path.insert(0,str((Path('.torch_extensions')/name).resolve()))
    extension = importlib.import_module(name)

    CPUAdamBuilder.load = lambda self,verbose=True:extension
    sys.argv = [args.entry,*remaining]
    print(json.dumps({'cpu_adam_implementation':args.variant,
                      'source':'upstream/DeepSpeed/csrc/adam/cpu_adam_impl.cpp',
                      'extension':extension.__file__,'torch':torch.__version__}),flush=True)
    runpy.run_path(args.entry,run_name='__main__')


if __name__ == '__main__':
    main()
