"""Find the largest runnable public-document context in a fixed length grid."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--card',default='10')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    args.output.mkdir(parents=True)
    rows = []
    for model in ('0.6B','1.7B','4B'):
        for mode in ('resident','sync','pipeline','zero'):
            config = (selection['plans'][mode]['config'] if mode!='resident' else
                      {'threads':8,'cpu_list':','.join(map(str,range(48))),
                       'bucket_mb':32,'depth':1})
            row = {'model':f'Qwen3-{model}','mode':mode,'runs':[]}
            if (model,mode)!=('4B','resident'):
                baseline = args.evaluation/f'qwen{model}_{mode}_0.json'
                result = json.loads(baseline.read_text())
                if result['config']['sequence']!=256:
                    raise ValueError(f'Expected 256-token baseline: {baseline}')
                row['runs'].append({'sequence':256,'status':'completed',
                                    'source':str(baseline)})
                lengths = (512,1024,2048,4096,8192)
            else:
                lengths = (256,512,1024,2048,4096,8192)
            for length in lengths:
                name = f'qwen{model}_{mode}_s{length}'
                destination = args.output/f'{name}.json'
                log_path = args.output/f'{name}.log'
                command = ['bash','scripts/run_npu.sh',args.card,'env',
                    f'OMP_NUM_THREADS={config["threads"]}',
                    f'TORCH_EXTENSIONS_DIR={Path(".torch_extensions").resolve()}',
                    '.venv/bin/python','train.py','--model',f'.cache/models/Qwen3-{model}',
                    '--mode',mode,'--sequence',str(length),'--accumulation','2',
                    '--warmup','1','--steps','3','--seed','800',
                    '--data','.cache/data/training_tokens/evaluation.json',
                    '--bucket-mb',str(config['bucket_mb']),'--depth',str(config['depth']),
                    '--threads',str(config['threads']),'--cpu-list',config['cpu_list'],
                    '--output',str(destination)]
                print(name,flush=True)
                with log_path.open('w') as log:
                    process = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if process.returncode:
                    if 'out of memory' not in log_path.read_text().lower():
                        raise RuntimeError(f'Training failed outside the capacity limit: {log_path}')
                    row['runs'].append({'sequence':length,'status':'out_of_memory',
                                        'source':str(log_path)})
                    break
                row['runs'].append({'sequence':length,'status':'completed',
                                    'source':str(destination)})
            completed = [r['sequence'] for r in row['runs'] if r['status']=='completed']
            failed = [r['sequence'] for r in row['runs'] if r['status']=='out_of_memory']
            row['largest_completed_sequence'] = max(completed,default=None)
            row['first_out_of_memory_sequence'] = min(failed,default=None)
            rows.append(row)
            (args.output/'summary.json').write_text(json.dumps({
                'grid':[256,512,1024,2048,4096,8192],'batch':1,'accumulation':2,
                'precision':'BF16 model; FP32 AdamW master weights and moments',
                'rows':rows},indent=2)+'\n')


if __name__ == '__main__':
    main()
