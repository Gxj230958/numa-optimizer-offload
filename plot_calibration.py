"""Plot full-step calibration measurements without using evaluation results."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('selection',type=Path)
    args = parser.parse_args()
    data = json.loads(args.selection.read_text())
    candidates = {Path(r['source']).stem:r for r in data['candidates']}
    output = args.selection.parent
    fields = ['configuration','mode','bucket_mb','depth','threads','numa',
              'median_step_ms','min_step_ms','max_step_ms']
    with (output/'calibration.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for name,row in candidates.items():
            config = row['config']
            writer.writerow({'configuration':name,'mode':config['mode'],
                'bucket_mb':config['bucket_mb'],
                'depth':1 if config['mode']=='sync' else config['depth'],
                'threads':config['threads'],
                'numa':'remote' if name.startswith('remote') else 'local',
                **{k:row[k] for k in fields[-3:]}})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(12,3.5))
    for depth in (1,2,4):
        rows = [candidates[f'local_b{bucket}_d{depth}'] for bucket in (8,32,128)]
        axes[0].plot(range(3),[r['median_step_ms'] for r in rows],
                     marker='o',label=f'depth {depth}')
    axes[0].set_xticks(range(3),['8','32','128'])
    axes[0].set_xlabel('Gradient bucket (MiB), 8 threads')
    axes[0].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.15),
                   ncol=3,fontsize=8)
    for locality in ('local','remote'):
        rows = [candidates['local_b32_d2' if (locality,threads)==('local',8)
                           else f'{locality}_threads{threads}']
                for threads in (4,8,16,32)]
        axes[1].plot(range(4),[r['median_step_ms'] for r in rows],
                     marker='o',label=locality)
    axes[1].set_xticks(range(4),['4','8','16','32'])
    axes[1].set_xlabel('CPU threads, 32 MiB / depth 2')
    axes[1].legend(frameon=False)
    modes = ['sync','zero','pipeline']
    chosen = [data['plans'][mode] for mode in modes]
    axes[2].bar(modes,[r['median_step_ms'] for r in chosen],
                color=['#999999','#c59135','#3575a8'])
    axes[2].set_xlabel('Independently selected configuration')
    for axis in axes:
        axis.set_ylabel('Complete training step (ms)')
        axis.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(output/'calibration.pdf')
    fig.savefig(output/'calibration.png',dpi=180)


if __name__ == '__main__':
    main()
