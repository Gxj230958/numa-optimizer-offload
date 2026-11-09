"""Measure the CPUAdam GIL patch in paired complete Qwen training runs."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess

from build_cpu_adam_pair import BASE_REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--card',default='10')
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    records = {}
    for repeat in range(3):
        variants = ('base','gil') if repeat%2==0 else ('gil','base')
        for variant in variants:
            name = f'{variant}_{repeat}'
            destination = args.output/f'{name}.json'
            command = ['bash','scripts/run_npu.sh',args.card,'env',
                       'OMP_NUM_THREADS=8',
                       f'TORCH_EXTENSIONS_DIR={Path(".torch_extensions").resolve()}',
                       '.venv/bin/python','run_deepspeed_patch.py','--variant',variant,
                       'train.py','--model','.cache/models/Qwen3-0.6B',
                       '--mode','pipeline','--cpu-optimizer','deepspeed',
                       '--bucket-mb','8','--depth','4','--threads','8',
                       '--cpu-list',','.join(map(str,range(48))),
                       '--sequence','256','--accumulation','2','--warmup','2',
                       '--steps','6','--seed',str(700+repeat),
                       '--data','.cache/data/training_tokens/evaluation.json',
                       '--output',str(destination)]
            print(name,flush=True)
            with (args.output/f'{name}.log').open('w') as handle:
                subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
            records[variant,repeat] = json.loads(destination.read_text())
    pairs = []
    for repeat in range(3):
        base,gil = (records[variant,repeat] for variant in ('base','gil'))
        pair = {'seed':700+repeat}
        for variant,result in [('base',base),('gil',gil)]:
            measured = [row for row in result['steps'] if not row['warmup']]
            pair[variant] = {
                'step_ms':statistics.median(row['step_ms'] for row in measured),
                'optimizer_ms':statistics.median(row['optimizer_ms'] for row in measured),
                'peak_npu_gib':max(row['peak_npu_bytes'] for row in measured)/1024**3,
                'peak_cpu_gib':max(row['cpu_max_rss_kib'] for row in measured)/1024**2}
        pair['step_speedup'] = pair['base']['step_ms']/pair['gil']['step_ms']
        pair['max_loss_difference'] = max(abs(a['loss']-b['loss'])
            for a,b in zip(base['steps'],gil['steps']))
        pair['max_gradient_norm_difference'] = max(abs(a['grad_norm']-b['grad_norm'])
            for a,b in zip(base['steps'],gil['steps']))
        pairs.append(pair)
    summary = {'source_revision':BASE_REVISION,
        'pairs':pairs,'median_step_speedup':statistics.median(p['step_speedup'] for p in pairs),
        'min_step_speedup':min(p['step_speedup'] for p in pairs),
        'max_step_speedup':max(p['step_speedup'] for p in pairs)}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
