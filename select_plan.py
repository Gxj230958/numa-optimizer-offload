"""Select fixed execution settings from independent calibration documents."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    candidates = []
    for path in sorted(args.directory.glob('*.json')):
        data = json.loads(path.read_text())
        if 'steps' not in data:
            continue
        times = [s['step_ms'] for s in data['steps'] if not s['warmup']]
        candidates.append({'source':str(path),'config':data['config'],
                           'median_step_ms':statistics.median(times),
                           'min_step_ms':min(times),'max_step_ms':max(times)})
    plans = {mode:min((c for c in candidates if c['config']['mode']==mode),
                      key=lambda c:c['median_step_ms'])
             for mode in ('pipeline','sync','zero')}
    result = {'selection_metric':'median complete training step time',
              'plans':plans,'candidates':candidates}
    (args.directory/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(plans,indent=2))


if __name__ == '__main__':
    main()
