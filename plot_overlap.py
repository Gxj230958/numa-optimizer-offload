"""Export the recorded CPU-update and PCIe-DMA intervals as a figure."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('events',type=Path)
    args = parser.parse_args()
    data = json.loads(args.events.read_text())
    cpu,dma = data['cpu'],data['dma']
    origin = min(float(event['ts']) for event in cpu+dma)
    workers = sorted({event['tid'] for event in cpu})
    lanes = [('PCIe DMA',dma,'#c59135')]+[
        (f'CPU worker {index+1}',[event for event in cpu if event['tid']==worker],'#3575a8')
        for index,worker in enumerate(workers)]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,1,figsize=(10,4.5))
    for axis in axes:
        for index,(label,events,color) in enumerate(lanes):
            intervals = [((float(event['ts'])-origin)/1000,float(event['dur'])/1000)
                         for event in events]
            axis.broken_barh(intervals,(index-.35,.7),facecolors=color)
        axis.set_yticks(range(len(lanes)),[label for label,_,_ in lanes])
        axis.set_xlabel('Time from first recorded transfer (ms)')
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='x',alpha=.2)
    first_update = (min(float(event['ts']) for event in cpu)-origin)/1000
    detail_start = max(0,first_update-5)
    axes[1].set_xlim(detail_start,detail_start+100)
    axes[0].set_title('Measured CPU update and PCIe transfer overlap')
    axes[1].set_title('First CPU update: 100 ms detail')
    fig.tight_layout()
    fig.savefig(args.events.parent/'overlap.pdf')
    fig.savefig(args.events.parent/'overlap.png',dpi=180)


if __name__ == '__main__':
    main()
