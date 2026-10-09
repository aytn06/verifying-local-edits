#!/usr/bin/env python3
"""Optional fresh local replication using pinned data and native tokenizers.

Never used by CPU replay. A fresh output directory and explicit inference flag
are mandatory. No retry/resume and no overwrite of the historical study.
"""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from protocol import ROOT, MODELS, canonical, file_sha, rows, verify_request, parse_with_error

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def append(path, obj):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(canonical(obj) + '\n')
        stream.flush()
        os.fsync(stream.fileno())

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-inference', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT / '.local/data')
    p.add_argument('--model-dir', type=Path, default=ROOT / '.local/models')
    p.add_argument('--split', choices=('dev', 'test', 'all'), default='all')
    p.add_argument('--model', choices=(*MODELS, 'all'), default='all')
    p.add_argument('--development-results', type=Path, help='Completed independent development run when executing test separately')
    args = p.parse_args()
    if not args.run_inference:
        p.error('Explicit --run-inference required')
    destination = args.output.resolve()
    if destination.exists() or (destination.is_relative_to(ROOT.parent.resolve()) and not destination.is_relative_to((ROOT / '.local').resolve())):
        p.error('Choose a new output directory outside frozen release contents')
    deps = json.loads((ROOT / 'provenance/dependencies.json').read_text())
    for package, version in deps['versions'].items():
        assert importlib.metadata.version(package) == version, 'Environment version differs: ' + package
    models = MODELS if args.model == 'all' else (args.model,)
    splits = ('dev', 'test') if args.split == 'all' else (args.split,)
    if args.split == 'all' and models != MODELS:
        p.error('A full replication requires both models; use an explicit split for an individual block')
    if 'test' in splits and (args.split != 'all' or models != MODELS):
        if args.development_results is None:
            p.error('A separate test block requires --development-results from both completed model blocks and their analysis')
        development = json.loads((args.development_results / 'replication.json').read_text())
        assert development['kind'] == 'new_independent_replication'
        assert development['historical_results_replaced'] is False
        assert development['frozen_original_sha256'] == deps['freeze_original_sha256']
        assert set(development['models']) == set(MODELS) and 'dev' in development['splits']
        for name in MODELS:
            completed = json.loads((args.development_results / 'dev' / name / 'completed.json').read_text())
            assert completed['calls'] == 60
            assert file_sha(args.development_results / 'dev' / name / 'responses.jsonl') == completed['responses_sha256']
            assert all(r['independent_replication'] is True for r in rows(args.development_results / 'dev' / name / 'responses.jsonl'))
        assert (args.development_results / 'dev/analysis/summary.json').is_file()
        analysis = json.loads((args.development_results / 'dev/analysis/summary.json').read_text())
        assert analysis['provenance']['analyzer_sha256'] == file_sha(ROOT / 'reference/analyze.py')
        assert analysis['provenance']['data_sha256'] == deps['prepared_files_sha256']['dev.jsonl']
        for name in MODELS:
            assert analysis['provenance']['responses_sha256']['dev/' + name + '/responses.jsonl'] == file_sha(args.development_results / 'dev' / name / 'responses.jsonl')
    for split in splits:
        assert file_sha(args.data_dir / (split + '.jsonl')) == deps['prepared_files_sha256'][split + '.jsonl']
    for name in models:
        for filename, expected in deps['models'][name]['files'].items():
            assert file_sha(args.model_dir / name / filename) == expected
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'replication.json').write_text(json.dumps({'kind': 'new_independent_replication', 'started_utc': now(),
        'historical_results_replaced': False, 'retries': 0, 'splits': splits, 'models': models,
        'frozen_original_sha256': deps['freeze_original_sha256']}, indent=2) + '\n')
    for split in splits:
        data = {r['case_id']: r for r in rows(args.data_dir / (split + '.jsonl'))}
        for name in models:
            model, tokenizer = load(str(args.model_dir / name), trust_remote_code=False)
            eos = set(tokenizer.eos_token_ids)
            assert eos == set(deps['models'][name]['eos_token_ids'])
            requests = rows(ROOT / 'provenance/requests' / (split + '_' + name + '.jsonl'))
            # Reconstruct all request IDs before starting this block.
            prepared = [verify_request(data[r['case_id']], r, tokenizer) for r in requests]
            root = destination / split / name
            root.mkdir(parents=True, exist_ok=False)
            sampler = make_sampler(temp=0.0)
            for req, (_, rendered, ids) in zip(requests, prepared):
                assert len(ids) + 64 <= deps['models'][name]['context_limit']
                append(root / 'receipts.jsonl', {'call_id': req['call_id'], 'index': req['index'], 'started_utc': now()})
                pieces, generated, last = [], [], None
                started = time.monotonic()
                for chunk in stream_generate(model, tokenizer, prompt=ids, max_tokens=64, sampler=sampler):
                    pieces.append(chunk.text)
                    generated.append(int(chunk.token))
                    last = chunk
                elapsed, raw = time.monotonic() - started, ''.join(pieces)
                raw_record = {'call_id': req['call_id'], 'raw': raw, 'generated_token_ids': generated,
                              'seconds_unrounded': elapsed, 'finish_reason': getattr(last, 'finish_reason', None), 'finished_utc': now()}
                append(root / 'raw_responses.jsonl', raw_record)
                assert last is not None and last.finish_reason in ('stop', 'length')
                assert last.prompt_tokens == len(ids) and last.generation_tokens == len(generated)
                eos_output = [t for t in generated if t in eos]
                assert (last.finish_reason == 'stop' and len(eos_output) == 1 and generated[-1] in eos) or (last.finish_reason == 'length' and not eos_output)
                output = [t for t in generated if t not in eos]
                decoded = tokenizer.decode(output)
                verdict, error = parse_with_error(raw)
                result = {k: req[k] for k in ('call_id', 'index', 'case_id', 'source_group', 'label', 'split', 'model', 'condition', 'data_hash', 'rendered_hash')}
                result.update(raw_record)
                result.update(verdict=verdict, parse_error=error, input_tokens=len(ids), output_tokens=len(output),
                              output_token_ids=output, eos_output_token_ids=eos_output, generated_tokens_including_eos=len(generated),
                              decoded_output=decoded, decode_matches_raw=decoded == raw, timing_scope='stream_generation_only',
                              freeze_sha256=deps['freeze_original_sha256'], is_mock=False, independent_replication=True)
                append(root / 'responses.jsonl', result)
                mx.clear_cache()
            (root / 'completed.json').write_text(json.dumps({'calls': len(requests), 'time': now(),
                'responses_sha256': file_sha(root / 'responses.jsonl'), 'historical_results_replaced': False}) + '\n')
            del model, tokenizer
            mx.clear_cache()
        if split == 'dev' and models == MODELS:
            # Same execution/format gate as the original run, with no accuracy
            # threshold and no opportunity to select or tune a condition.
            subprocess.run([sys.executable, '-S', str(ROOT / 'reference/analyze.py'), '--data-dir', str(args.data_dir),
                            '--results-dir', str(destination), '--split', 'dev'], check=True)
    print(json.dumps({'status': 'COMPLETE', 'kind': 'new_independent_replication', 'output': str(destination)}))

if __name__ == '__main__':
    main()
