"""Prepare disjoint public-document token sets for full-parameter training."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    articles = json.loads(args.data.read_text())['data']
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,partition in [('calibration',articles[:16]),('evaluation',articles[16:])]:
        documents = []
        for article in partition:
            text = '\n\n'.join(p['context'] for p in article['paragraphs'])
            documents.append({'title':article['title'],
                              'token_ids':tokenizer.encode(text,add_special_tokens=False)})
        result = {'source':'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
                  'partition':name,'tokenizer':args.model,'documents':documents}
        (args.output/f'{name}.json').write_text(json.dumps(result)+'\n')
        print(name,len(documents),sum(len(d['token_ids']) for d in documents))


if __name__ == '__main__':
    main()
