"""Export measured training-capacity bounds from a fixed sequence-length grid."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    summary = json.loads((args.directory/'summary.json').read_text())
    rows = []
    for record in summary['rows']:
        lower = record['largest_completed_sequence']
        upper = record['first_out_of_memory_sequence']
        if lower is None:
            bound = f'<{upper}'
        elif upper is None:
            bound = f'≥{lower}'
        else:
            bound = f'[{lower}, {upper})'
        rows.append({'model':record['model'],'mode':record['mode'],
                     'largest_completed_sequence':lower,
                     'first_out_of_memory_sequence':upper,'capacity_bound':bound,
                     'batch':summary['batch'],'accumulation':summary['accumulation']})
    with (args.directory/'capacity.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    models = list(dict.fromkeys(row['model'] for row in rows))
    modes = ['resident','sync','pipeline','zero']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    grid = summary['grid']
    fig,axes = plt.subplots(1,len(models),figsize=(4*len(models),3.8),
                           sharey=True,squeeze=False)
    for axis,model in zip(axes[0],models):
        for index,mode in enumerate(modes):
            record = next((r for r in rows if r['model']==model and r['mode']==mode),None)
            if record is None:
                continue
            lower = record['largest_completed_sequence']
            upper = record['first_out_of_memory_sequence']
            color = colors[index]
            if lower is not None and upper is not None:
                axis.plot([lower,upper],[index,index],color=color,alpha=.5,linewidth=2)
            if lower is not None:
                axis.plot(lower,index,'o',color=color,markersize=7)
            if upper is not None:
                axis.plot(upper,index,'x',color=color,markersize=8,markeredgewidth=1.7)
            if lower is None:
                axis.annotate(record['capacity_bound'],(upper,index),xytext=(-7,0),
                              textcoords='offset points',ha='right',va='center',fontsize=9)
            elif upper is None:
                axis.annotate(record['capacity_bound'],(lower,index),xytext=(7,0),
                              textcoords='offset points',ha='left',va='center',fontsize=9)
        axis.set_xscale('log',base=2)
        axis.set_xlim(grid[0]/2,grid[-1]*2)
        axis.set_xticks(grid,[str(length) for length in grid],fontsize=8)
        axis.set_yticks(range(len(modes)),modes)
        axis.set_ylim(len(modes)-.4,-.6)
        axis.set_xlabel('Sequence length (tokens)')
        axis.set_title(model)
        axis.grid(axis='x',alpha=.2)
        axis.spines[['top','right']].set_visible(False)
    fig.legend(handles=[
        Line2D([],[],color='black',marker='o',linestyle='None',label='Largest completed'),
        Line2D([],[],color='black',marker='x',linestyle='None',label='First out of memory')],
        loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.suptitle(f"Training capacity: batch {summary['batch']}, accumulation {summary['accumulation']}")
    fig.tight_layout(rect=(0,0,1,.84))
    fig.savefig(args.directory/'capacity.pdf')
    fig.savefig(args.directory/'capacity.png',dpi=180)
    plt.close(fig)
    print(args.directory/'capacity.csv')


if __name__ == '__main__':
    main()
