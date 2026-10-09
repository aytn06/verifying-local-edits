#!/usr/bin/env python3
"""Validate public commitments, paired fixtures and cached result preservation."""
import argparse
from collections import Counter
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from protocol import ROOT, MODELS, CONDITIONS, file_sha, rows, load_reference, parse_verdict, digest
from reproduce import reproduce

def validate(prepared_only=False, run_tests=True):
    sums = json.loads((ROOT / 'provenance/checksums.json').read_text())
    actual_files = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
                    and not any(part in ('.local', '__pycache__') for part in p.relative_to(ROOT).parts)
                    and p != ROOT / 'provenance/checksums.json'}
    assert actual_files == set(sums), 'Untracked or missing extension files'
    for name, expected in sums.items():
        assert file_sha(ROOT / name) == expected, 'Checksum mismatch: ' + name
    freeze = json.loads((ROOT / 'provenance/freeze.public.json').read_text())
    deps = json.loads((ROOT / 'provenance/dependencies.json').read_text())
    assert file_sha(ROOT.parent / 'scripts/external_protocol.py') == deps['shared_protocol_sha256']
    for name in ('analyze.py', 'test_analysis.py', 'prepare.py'):
        assert file_sha(ROOT / 'reference' / name) == freeze['script_hashes']['scripts/replication/' + name]
    excluded = set(json.loads((ROOT / 'data/prior_source_exclusions.json').read_text())['excluded_source_ids'])
    seen, requests = set(excluded), 0
    for split, n in (('dev', 10), ('test', 50)):
        data = rows(ROOT / 'data' / (split + '.jsonl'))
        assert len(data) == 2*n
        by_id = {r['case_id']: r for r in data}
        groups = {r['group_id'] for r in data}
        assert len(groups) == n and not groups & seen
        seen.update(groups)
        for gid in groups:
            pair = [r for r in data if r['group_id'] == gid]
            assert Counter(r['label'] for r in pair) == {0: 1, 1: 1}
            assert len({r['original_source_group'] for r in pair}) == len({r['original_seed_hash'] for r in pair}) == 1
        for model in MODELS:
            reqs = rows(ROOT / 'provenance/requests' / (split + '_' + model + '.jsonl'))
            assert len(reqs) == 6*n and len({r['call_id'] for r in reqs}) == 6*n
            assert {(r['case_id'], r['condition']) for r in reqs} == {(cid, c) for cid in by_id for c in CONDITIONS}
            assert [r['index'] for r in reqs] == list(range(6*n))
            for req in reqs:
                assert req['model'] == model and req['split'] == split
                assert req['label'] == by_id[req['case_id']]['label']
                assert req['source_group'] == by_id[req['case_id']]['group_id']
                assert req['max_output_tokens'] == 64 and req['temperature'] == 0
                assert req['template_settings'] == json.loads((ROOT / 'prompts/spec.json').read_text())['native_template_settings'][model]
                assert req['input_tokens'] + 64 <= freeze['models'][model]['context_limit']
                assert not {'messages', 'rendered_prompt', 'input_token_ids'} & req.keys()
            requests += len(reqs)
    for name in sums:
        path = ROOT / name
        assert path.suffix not in ('.safetensors', '.arrow', '.bin', '.pt', '.zip')
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'/(?:Users|home)/[A-Za-z0-9_.-]+/', text), 'Private home path: ' + name
        assert not re.search(r'(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,})', text), 'Credential-like string'
    status = json.loads((ROOT / 'provenance/status.json').read_text())
    for mapped in json.loads((ROOT / 'provenance/source_mapping.json').read_text()):
        assert file_sha(ROOT / mapped['public']) == mapped['public_sha256']
        if mapped['transformation'] == 'byte-identical':
            assert mapped['source_sha256'] == mapped['public_sha256']
    if not prepared_only:
        assert status['results_complete'], 'Responses not yet projected; use --prepared-only for construction checks'
        analyzer = load_reference('analyze')
        for split in ('dev', 'test'):
            response = []
            for model in MODELS:
                rr = rows(ROOT / 'results' / split / model / 'responses.jsonl')
                expected = rows(ROOT / 'provenance/requests' / (split + '_' + model + '.jsonl'))
                assert [r['call_id'] for r in rr] == [r['call_id'] for r in expected]
                for row in rr:
                    assert row['raw'] == '' or parse_verdict(row['raw']) is not None
                    assert not {'decoded_output', 'generated_token_ids', 'output_token_ids'} & row.keys()
                    if not row['raw_redacted']: assert digest(row['raw']) == row['raw_sha256']
                response.extend(rr)
            analyzer.validate_matrix(rows(ROOT / 'data' / (split + '.jsonl')), response, split, analyzer.EXPECTED_SOURCES[split])
        with tempfile.TemporaryDirectory(prefix='checklist-cpu-replay-') as temporary:
            for split in ('dev', 'test'):
                reproduce(split, Path(temporary))
    if run_tests:
        for folder in ('reference', 'tests'):
            subprocess.run([sys.executable, '-S', '-m', 'unittest', 'discover', '-s', str(ROOT / folder), '-p', 'test_*.py', '-v'], check=True)
    return {'status': 'PASS', 'prepared_only': prepared_only, 'committed_files': len(sums), 'request_matrix': requests,
            'result_calls': status['completed_projected_calls'], 'frozen_analyzer_exact': True}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-only', action='store_true')
    args = p.parse_args()
    print(json.dumps(validate(args.prepared_only), indent=2))
