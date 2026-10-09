#!/usr/bin/env python3
"""Independently obtain pinned data/models, or hydrate from an existing raw file.

No inference. Third-party text/assets are written only to the chosen local area.
"""
import argparse
import json
from pathlib import Path
from protocol import ROOT, MODELS, file_sha, rows, load_reference, verify_request, digest, canonical

def write_once(path, body, expected=None):
    if expected and digest(body) != expected:
        raise ValueError('Frozen file commitment mismatch: ' + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf-8') != body:
            raise ValueError('Refusing to replace existing material')
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(body)

def hydrate(raw, output):
    prepare = load_reference('prepare')
    deps = json.loads((ROOT / 'provenance/dependencies.json').read_text())
    exclusions = json.loads((ROOT / 'data/prior_source_exclusions.json').read_text())['excluded_source_ids']
    normalized, rejects = prepare.normalize_rows(raw)
    assert len(normalized) == 6348 and not rejects
    pairs, conflicts, ineligible = prepare.make_pairs(normalized, set(exclusions), 20261009)
    selected, strata = prepare.choose_splits(pairs, 10, 50, 20261009)
    prepare.verify_splits(selected, exclusions)
    assert len(pairs) == 71 and len(ineligible) == 5 and not conflicts
    checked = 0
    for split in ('dev', 'test'):
        body = prepare.jsonl_text(selected[split])
        assert digest(body) == deps['prepared_files_sha256'][split + '.jsonl']
        metadata = [{k: v for k, v in r.items() if k not in {'source', 'original', 'edited', 'edit_types'}} for r in selected[split]]
        assert metadata == rows(ROOT / 'data' / (split + '.jsonl'))
        index = {r['case_id']: r for r in selected[split]}
        for model in MODELS:
            for req in rows(ROOT / 'provenance/requests' / (split + '_' + model + '.jsonl')):
                verify_request(index[req['case_id']], req)
                checked += 1
        write_once(output / (split + '.jsonl'), body, deps['prepared_files_sha256'][split + '.jsonl'])
    return {'status': 'PASS', 'cases': 120, 'source_groups': 60, 'exact_original_files': 2,
            'request_messages_and_renderings': checked, 'selection_reexecuted': True}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--source-json', type=Path)
    source.add_argument('--source-arrow', type=Path, help='Existing Arrow stream; requires optional pyarrow')
    source.add_argument('--download', action='store_true', help='Explicit pinned dataset network download')
    p.add_argument('--models', nargs='+', choices=MODELS, help='Explicit pinned model downloads; never inference')
    p.add_argument('--output', type=Path, default=ROOT / '.local/data')
    args = p.parse_args()
    if args.output.resolve().is_relative_to((ROOT / 'data').resolve()):
        p.error('Hydrated text must not replace distributed metadata')
    deps = json.loads((ROOT / 'provenance/dependencies.json').read_text())
    path = args.source_json
    if args.download or args.models:
        from huggingface_hub import hf_hub_download, snapshot_download
    if args.download:
        d = deps['dataset']
        path = Path(hf_hub_download(repo_id=d['repo'], repo_type='dataset', revision=d['revision'], filename=d['file'], local_dir=ROOT / '.local/downloads'))
        if d.get('sha256'):
            assert file_sha(path) == d['sha256']
    if path:
        raw = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(raw, dict):
            raw = raw.get('data', raw.get('train', raw))
        assert isinstance(raw, list)
        print(json.dumps(hydrate(raw, args.output), indent=2))
    if args.source_arrow:
        import pyarrow.ipc as ipc
        assert file_sha(args.source_arrow) == deps['dataset']['original_arrow_sha256']
        with args.source_arrow.open('rb') as stream:
            raw = ipc.open_stream(stream).read_all().to_pylist()
        print(json.dumps(hydrate(raw, args.output), indent=2))
    for model in args.models or []:
        d = deps['models'][model]
        target = ROOT / '.local/models' / model
        snapshot_download(repo_id=d['repo'], revision=d['revision'], local_dir=target, allow_patterns=[*d['files'], 'LICENSE', 'LICENSE.txt', 'USE_POLICY.md'])
        for filename, sha in d['files'].items():
            assert file_sha(target / filename) == sha
        print(json.dumps({'status': 'PASS', 'model': model, 'pinned_files_verified': len(d['files'])}))
    if not (path or args.source_arrow or args.models):
        p.error('Choose a source file or an explicit download option')

if __name__ == '__main__':
    main()
