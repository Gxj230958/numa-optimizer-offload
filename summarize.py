"""Summarize three independent full-training repetitions and export figures."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    selection = json.loads((args.directory/'selection.json').read_text())
    expected = {(f'Qwen3-{model}',mode,seed)
                for model in ('0.6B','1.7B','4B')
                for mode in ('resident','sync','pipeline','zero')
                if (model,mode)!=('4B','resident')
                for seed in range(600,603)}
    observed = set()
    groups = {}
    for path in sorted(args.directory.glob('qwen*.json')):
        result = json.loads(path.read_text())
        measured = [r for r in result['steps'] if not r['warmup']]
        config = result['config']
        key = (Path(config['model']).name,config['mode'])
        identity = (*key,config['seed'])
        if identity not in expected or identity in observed:
            raise ValueError(f'Unexpected or duplicate run: {path}')
        observed.add(identity)
        fixed = {'sequence':256,'batch':1,'accumulation':2,'warmup':2,
                 'steps':6,'random_init':False,'profile':False,
                 'cpu_optimizer':'fused','resident_optimizer':'fused'}
        if key[1]!='resident':
            plan = selection['plans'][key[1]]['config']
            fixed.update({name:plan[name] for name in
                          ('bucket_mb','depth','threads','cpu_list')})
        else:
            fixed.update({'bucket_mb':32,'depth':1,'threads':8,
                          'cpu_list':','.join(map(str,range(48)))})
        if any(config[name]!=value for name,value in fixed.items()):
            raise ValueError(f'Run differs from the evaluation protocol: {path}')
        if (result['training_inputs']['partition']!='evaluation'
                or [r['warmup'] for r in result['steps']]!=[True]*2+[False]*6):
            raise ValueError(f'Incorrect data partition or step count: {path}')
        for step in result['steps']:
            for name in ('step_ms','forward_backward_ms','optimizer_ms','loss','grad_norm'):
                if not math.isfinite(step[name]):
                    raise ValueError(f'Nonfinite {name} in {path}, step {step["step"]}')
        groups.setdefault(key,[]).append({
            'source':str(path),'seed':config['seed'],
            'step_ms':statistics.median(r['step_ms'] for r in measured),
            'optimizer_ms':statistics.median(r['optimizer_ms'] for r in measured),
            'peak_npu_gib':max(r['peak_npu_bytes'] for r in result['steps'])/1024**3,
            'peak_reserved_gib':max(r['peak_npu_reserved_bytes'] for r in result['steps'])/1024**3,
            'peak_cpu_gib':max(r['cpu_max_rss_kib'] for r in result['steps'])/1024**2,
            'loss_start':result['steps'][0]['loss'],'loss_end':result['steps'][-1]['loss']})
    if observed!=expected:
        raise ValueError(f'Missing evaluation runs: {sorted(expected-observed)}')
    rows = []
    for (model,mode),runs in groups.items():
        if len(runs)!=3 or len({r['seed'] for r in runs})!=3:
            raise ValueError(f'{model}/{mode} requires three independent seeds')
        times = [r['step_ms'] for r in runs]
        rows.append({'model':model,'mode':mode,'step_ms':statistics.median(times),
                     'min_ms':min(times),'max_ms':max(times),
                     'optimizer_ms':statistics.median(r['optimizer_ms'] for r in runs),
                     'peak_npu_gib':max(r['peak_npu_gib'] for r in runs),
                     'peak_reserved_gib':max(r['peak_reserved_gib'] for r in runs),
                     'peak_cpu_gib':max(r['peak_cpu_gib'] for r in runs),'runs':runs})
    comparisons = {}
    for model in sorted({r['model'] for r in rows}):
        modes = {r['mode']:r for r in rows if r['model']==model}
        pipeline = modes['pipeline']['step_ms']
        comparisons[model] = {f'pipeline_speedup_over_{mode}':r['step_ms']/pipeline
                              for mode,r in modes.items() if mode!='pipeline'}
    (args.directory/'summary.json').write_text(json.dumps(
        {'protocol':{'independent_runs':len(observed),'measured_steps':len(observed)*6,
                     'partition':'evaluation','frozen_configurations':True,
                     'memory_scope':'maximum across warmup and measured steps',
                     'finite_losses_gradients_and_times':True},
         'rows':rows,'comparisons':comparisons},indent=2)+'\n')
    fields = [key for key in rows[0] if key!='runs']
    with (args.directory/'summary.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:row[key] for key in fields} for row in rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(r['model'] for r in rows))
    modes = ['resident','sync','pipeline','zero']
    fig,axes = plt.subplots(1,2,figsize=(10,3.6))
    for index,mode in enumerate(modes):
        selected = [next((r for r in rows if r['model']==model and r['mode']==mode),None)
                    for model in models]
        positions = [i+(index-1.5)*.19 for i,r in enumerate(selected) if r]
        records = [r for r in selected if r]
        values = [r['step_ms']/1000 for r in records]
        axes[0].bar(positions,values,.18,label=mode,yerr=[
            [(r['step_ms']-r['min_ms'])/1000 for r in records],
            [(r['max_ms']-r['step_ms'])/1000 for r in records]],capsize=2)
        axes[1].bar(positions,[r['peak_npu_gib'] for r in records],.18,label=mode)
    for axis in axes:
        axis.set_xticks(range(len(models)),[m.replace('Qwen3-','') for m in models])
        axis.set_xlabel('Qwen3 model')
        axis.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Complete training step (s)')
    axes[1].set_ylabel('Peak NPU allocated memory (GiB)')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.directory/'training.pdf')
    fig.savefig(args.directory/'training.png',dpi=180)
    print(json.dumps(comparisons,indent=2))


if __name__ == '__main__':
    main()
