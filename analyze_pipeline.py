"""Join measured CPU worker spans with Ascend PCIe DMA device events."""

import argparse
import json
from pathlib import Path


def merge(intervals):
    result = []
    for start,end in sorted(intervals):
        if result and start<=result[-1][1]:
            result[-1][1] = max(result[-1][1],end)
        else:
            result.append([start,end])
    return result


def total(intervals):
    return sum(end-start for start,end in merge(intervals))


def overlap(left,right):
    left,right = merge(left),merge(right)
    i = j = 0
    result = 0.
    while i<len(left) and j<len(right):
        result += max(0.,min(left[i][1],right[j][1])-max(left[i][0],right[j][0]))
        if left[i][1]<right[j][1]:
            i += 1
        else:
            j += 1
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-trace',type=Path,required=True)
    parser.add_argument('--cpu-trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    raw = json.loads(args.device_trace.read_text())
    events = raw if isinstance(raw,list) else raw['traceEvents']
    copies = [event for event in events if event.get('ph')=='X' and
              event.get('args',{}).get('Task Type')=='PCIE_DMA_SQE']
    steps = [event for event in events if event.get('ph')=='X' and
             event.get('name','').startswith('ProfilerStep#')]
    cpu = json.loads(args.cpu_trace.read_text())['traceEvents']
    def intervals(rows):
        return [(float(row['ts']),float(row['ts'])+float(row['dur'])) for row in rows]
    recorded = intervals([event for event in events if event.get('ph')=='X'])
    start,end = min(a for a,b in recorded),max(b for a,b in recorded)
    cpu = [event for event in cpu if start<=event['ts'] and event['ts']+event['dur']<=end]
    cpu_intervals,dma_intervals = intervals(cpu),intervals(copies)
    shared = overlap(cpu_intervals,dma_intervals)
    summary = {'cpu_busy_ms':total(cpu_intervals)/1000,
               'pcie_dma_busy_ms':total(dma_intervals)/1000,
               'cpu_dma_overlap_ms':shared/1000,
               'fraction_dma_overlapped_with_cpu':shared/total(dma_intervals),
               'cpu_buckets':len(cpu),'pcie_dma_events':len(copies),
               'device_source':str(args.device_trace),'cpu_source':str(args.cpu_trace)}
    if steps:
        step_intervals = intervals(steps)
        step_time = total(step_intervals)
        dma_in_steps = overlap(dma_intervals,step_intervals)
        summary.update(profiled_steps=len(steps),profiled_step_ms=step_time/1000,
                       pcie_dma_within_step_ms=dma_in_steps/1000,
                       fraction_step_with_dma=dma_in_steps/step_time)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'overlap_events.json').write_text(json.dumps({
        'cpu':cpu,'dma':copies,'steps':steps,'device_source':str(args.device_trace),
        'cpu_source':str(args.cpu_trace)},indent=2)+'\n')
    metadata = [{'ph':'M','name':'thread_name','pid':event['pid'],'tid':event['tid'],
                 'args':{'name':f"CPU AdamW worker {event['tid']}"}} for event in cpu]
    (args.output/'timeline.json').write_text(json.dumps({'traceEvents':events+cpu+metadata})+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
