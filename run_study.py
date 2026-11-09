"""Evaluate frozen offload configurations on held-out public documents."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--card',default='10')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    args.output.mkdir(parents=True,exist_ok=args.resume)
    saved_selection = args.output/'selection.json'
    if saved_selection.exists() and json.loads(saved_selection.read_text())!=selection:
        raise ValueError('The output directory uses a different calibration selection')
    saved_selection.write_text(json.dumps(selection,indent=2)+'\n')
    for model in ('0.6B','1.7B','4B'):
        modes = ('resident','sync','pipeline','zero') if model!='4B' else ('sync','pipeline','zero')
        for repeat in range(3):
            for mode in modes[repeat:]+modes[:repeat]:
                config = (selection['plans'][mode]['config'] if mode!='resident' else
                          {'threads':8,'cpu_list':','.join(map(str,range(48))),
                           'bucket_mb':32,'depth':1})
                name = f'qwen{model}_{mode}_{repeat}'
                result_path = args.output/f'{name}.json'
                if args.resume and result_path.exists():
                    saved = json.loads(result_path.read_text())['config']
                    if (saved['seed']!=600+repeat or saved['mode']!=mode
                            or Path(saved['model']).name!=f'Qwen3-{model}'
                            or saved['sequence']!=256 or saved['accumulation']!=2):
                        raise ValueError(f'Completed run has different settings: {result_path}')
                    print(f'Complete: {name}',flush=True)
                    continue
                command = ['bash','scripts/run_npu.sh',args.card,'env',
                    f'OMP_NUM_THREADS={config["threads"]}',
                    f'TORCH_EXTENSIONS_DIR={Path(".torch_extensions").resolve()}',
                    '.venv/bin/python','train.py','--model',f'.cache/models/Qwen3-{model}',
                    '--mode',mode,'--sequence','256','--accumulation','2','--warmup','2',
                    '--steps','6','--seed',str(600+repeat),
                    '--data','.cache/data/training_tokens/evaluation.json',
                    '--bucket-mb',str(config['bucket_mb']),'--depth',str(config['depth']),
                    '--threads',str(config['threads']),'--cpu-list',config['cpu_list'],
                    '--output',str(args.output/f'{name}.json')]
                print(name,flush=True)
                with (args.output/f'{name}.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__ == '__main__':
    main()
