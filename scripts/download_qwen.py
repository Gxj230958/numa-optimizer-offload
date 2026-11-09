"""Download the official Qwen ModelScope releases into the project input cache."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import time

import requests


ROOT = Path(__file__).resolve().parents[1]/'.cache/models'


def download(model, item):
    path = ROOT/model/item['Path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == item['Size']:
        return
    session = requests.Session()
    session.trust_env = False
    url = f"https://modelscope.cn/models/Qwen/{model}/resolve/{item['Revision']}/{item['Path']}"
    temporary = path.with_name(path.name+'.partial')
    if item['Size'] > 64*1024**2:
        prefix = temporary.stat().st_size if temporary.exists() else 0
        part_size = 64*1024**2
        offsets = list(range(prefix, item['Size'], part_size))

        def fetch_part(offset):
            end = min(offset+part_size, item['Size'])-1
            part = path.with_name(path.name+f'.part-{offset}')
            if part.exists() and part.stat().st_size == end-offset+1:
                return part
            transfer = requests.Session()
            transfer.trust_env = False
            for attempt in range(3):
                try:
                    with transfer.get(url, headers={'Range': f'bytes={offset}-{end}'},
                                      stream=True, timeout=(30,120)) as response:
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise IOError('Server did not honor the requested byte range')
                        with part.open('wb') as handle:
                            for chunk in response.iter_content(4*1024**2):
                                handle.write(chunk)
                    if part.stat().st_size != end-offset+1:
                        raise IOError('Incomplete byte range')
                    return part
                except (requests.RequestException, IOError):
                    if attempt == 2:
                        raise

        with ThreadPoolExecutor(max_workers=8) as pool:
            parts = list(pool.map(fetch_part, offsets))
        assembled = path.with_name(path.name+'.assembling')
        with assembled.open('wb') as handle:
            for part in ([temporary] if prefix else []) + parts:
                with part.open('rb') as source:
                    while chunk := source.read(4*1024**2):
                        handle.write(chunk)
        if assembled.stat().st_size != item['Size']:
            raise IOError('Incomplete assembled model file')
        assembled.replace(path)
        for part in ([temporary] if prefix else []) + parts:
            part.unlink()
        print(f'{model}/{item["Path"]}: {item["Size"]} bytes', flush=True)
        return
    for attempt in range(3):
        offset = temporary.stat().st_size if temporary.exists() else 0
        try:
            with session.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                             stream=True, timeout=(20,60)) as response:
                response.raise_for_status()
                mode = 'ab' if offset and response.status_code == 206 else 'wb'
                with temporary.open(mode) as handle:
                    for chunk in response.iter_content(1024**2):
                        handle.write(chunk)
            if temporary.stat().st_size != item['Size']:
                raise IOError(f'Incomplete transfer: {temporary.stat().st_size} of {item["Size"]} bytes')
            temporary.rename(path)
            print(f'{model}/{item["Path"]}: {item["Size"]} bytes', flush=True)
            return
        except (requests.RequestException, IOError) as error:
            print(f'{model}/{item["Path"]}: {type(error).__name__}, attempt {attempt+1}',flush=True)
            if attempt == 2:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='*', default=['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-1.7B'])
    parser.add_argument('--sources',type=Path,
                        help='Experiment environment JSON containing pinned model source records')
    args = parser.parse_args()
    pinned = ({entry['repository'].rsplit('/',1)[-1]:entry
               for entry in json.loads(args.sources.read_text())['models'].values()}
              if args.sources else None)
    session = requests.Session()
    session.trust_env = False
    for model in args.models:
        start = time.monotonic()
        if pinned is not None:
            files = list(pinned[model]['files'])
        else:
            response = session.get(f'https://modelscope.cn/api/v1/models/Qwen/{model}/repo/files',
                                   params={'Revision': 'master', 'Recursive': 'true'}, timeout=30)
            response.raise_for_status()
            files = [item for item in response.json()['Data']['Files']
                     if item['Type'] == 'blob' and
                     (Path(item['Path']).suffix in ('.json','.safetensors','.txt','.model')
                      or item['Path'] == 'LICENSE')]
        files.sort(key=lambda item: (item['Size'] > 64*1024**2, item['Path']))
        for item in files:
            download(model, item)
        provenance = {'repository': f'https://modelscope.cn/models/Qwen/{model}',
                      'files': [{key: item[key] for key in ('Path','Revision','Size')} for item in files]}
        (ROOT/model/'source.json').write_text(json.dumps(provenance,indent=2)+'\n')
        print(f'{model} complete in {time.monotonic()-start:.1f}s',flush=True)


if __name__ == '__main__':
    main()
