"""Plot paired complete-training measurements of native CPUAdam kernels."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary',type=Path)
    args = parser.parse_args()
    pairs = json.loads(args.summary.read_text())['pairs']
    output = args.summary.parent
    fields = ['seed','variant','step_ms','optimizer_ms','peak_npu_gib','peak_cpu_gib']
    with (output/'comparison.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            for variant in ('base','gil'):
                writer.writerow({'seed':pair['seed'],'variant':variant,**pair[variant]})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(7.5,3.5))
    for pair in pairs:
        for axis,metric in zip(axes,('step_ms','optimizer_ms')):
            axis.plot((0,1),[pair[v][metric] for v in ('base','gil')],
                      marker='o',label=f'Seed {pair["seed"]}')
    for axis in axes:
        axis.set_xticks((0,1),('Upstream','Release GIL'))
        axis.set_xlim(-.2,1.2)
        axis.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Complete training step (ms)')
    axes[1].set_ylabel('Exposed optimizer interval (ms)')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output/'comparison.pdf')
    fig.savefig(output/'comparison.png',dpi=180)


if __name__ == '__main__':
    main()
