#!/usr/bin/env python3
"""Replay the frozen source-cluster analysis with Python's standard library."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from protocol import ROOT, MODELS

def reproduce(split, output):
    status = json.loads((ROOT / 'provenance/status.json').read_text())
    if not status['results_complete']:
        raise ValueError('Prepared artifact has no completed response projections yet')
    target = output.resolve()
    if target.is_relative_to(ROOT.parent.resolve()) and not target.is_relative_to((ROOT / '.local').resolve()):
        raise ValueError('Replay inside the release must stay in the ignored extension .local directory')
    for protected in ('results', 'reference', 'data', 'prompts', 'provenance', 'scripts', 'tests'):
        if target.is_relative_to((ROOT / protected).resolve()):
            raise ValueError('Replay output must be outside frozen artifact directories')
    for model in MODELS:
        destination = output / split / model / 'responses.jsonl'
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / 'results' / split / model / 'responses.jsonl'
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError('Refusing to overwrite different response records in the replay output')
        if not destination.exists():
            shutil.copyfile(source, destination)
    subprocess.run([sys.executable, '-S', str(ROOT / 'reference/analyze.py'), '--data-dir', str(ROOT / 'data'),
                    '--results-dir', str(output), '--split', split], check=True)
    actual = json.loads((output / split / 'analysis/summary.json').read_text())
    actual.pop('provenance')
    expected = json.loads((ROOT / 'results/expected_analysis' / (split + '.json')).read_text())
    if actual != expected:
        raise ValueError('Replay differs from preserved original numerical analysis')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('dev', 'test', 'all'), default='all')
    p.add_argument('--output', type=Path, default=ROOT / '.local/replay')
    args = p.parse_args()
    for split in ('dev', 'test') if args.split == 'all' else (args.split,):
        reproduce(split, args.output)
    print(json.dumps({'status': 'PASS', 'exact_original_analysis': True, 'output': str(args.output)}))

if __name__ == '__main__':
    main()
