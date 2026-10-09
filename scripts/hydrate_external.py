#!/usr/bin/env python3
"""Opt-in independent dependency download and exact frozen-sample hydration.

No inference. Downloaded data/model/tokenizer material stays in ignored .local/.
Read THIRD_PARTY.md before obtaining or redistributing upstream materials.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
from external_protocol import ROOT, canonical, digest, make_request, normalize, render_without_tokenizer

def read_rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def write_exact(path,content,expected=None):
    if expected and hashlib.sha256(content).hexdigest()!=expected:raise ValueError(f'Frozen hash mismatch: {path.name}')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content:raise ValueError(f'Refusing overwrite: {path}')
    else:path.write_bytes(content)
def hydrate(raw_path,output):
    dependencies=json.loads((ROOT/'provenance/dependencies.json').read_text())
    try:raw=json.loads(raw_path.read_text())
    except json.JSONDecodeError:raw=read_rows(raw_path)
    if isinstance(raw,dict):
        for key in ('data','train'):
            if key in raw and isinstance(raw[key],list):raw=raw[key];break
    if not isinstance(raw,list):raise ValueError('Expected list of SummEdits records (JSON or JSONL)')
    index={str(r['id']):r for r in raw}
    if len(index)!=len(raw):raise ValueError('Duplicate upstream case IDs')
    restored={};checks=0
    for split in ('dev','test'):
        metas=read_rows(ROOT/f'data/external/{split}.metadata.jsonl');selected=[]
        for meta in metas:
            row=normalize(index[meta['case_id']])
            for key in ('case_id','changed','domain','edit_types','group_id','label'):assert row[key]==meta[key],(split,key)
            assert digest(canonical(row))==meta['data_hash']
            for key,expected in meta['text_sha256'].items():assert digest(row[key])==expected
            selected.append(row)
        body=''.join(canonical(r)+'\n' for r in selected).encode()
        write_exact(output/f'{split}.jsonl',body,dependencies['prepared_files_original_sha256'][f'{split}.jsonl'])
        restored[split]={r['case_id']:r for r in selected}
        for req in read_rows(ROOT/f'provenance/{split}_requests.metadata.jsonl'):
            messages=make_request(restored[split][req['case_id']],req['condition'])
            assert digest(canonical(messages))==req['messages_sha256']
            assert digest(render_without_tokenizer(messages))==req['rendered_hash']
            checks+=1
    write_exact(output/'sample_manifest.json',(ROOT/'data/external/sample_manifest.json').read_bytes(),dependencies['prepared_files_original_sha256']['sample_manifest.json'])
    return {'prepared_cases':140,'exact_original_file_hashes_verified':3,'request_message_and_render_hashes_verified':checks}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-json',type=Path,help='Already independently obtained pinned SummEdits JSON; no network')
    p.add_argument('--download',action='store_true',help='Explicitly download pinned SummEdits and hydrate the frozen sample')
    p.add_argument('--models',nargs='+',choices=['qwen3_8b','qwen3_17b'],help='Explicitly download pinned model/tokenizer dependencies; never infer')
    p.add_argument('--output',type=Path,default=ROOT/'.local/external')
    args=p.parse_args();deps=json.loads((ROOT/'provenance/dependencies.json').read_text())
    if args.download or args.models:
        try:from huggingface_hub import hf_hub_download,snapshot_download
        except ImportError:raise SystemExit('Install requirements-download.txt for explicit network downloads.')
    source=args.source_json
    if args.download:
        d=deps['dataset'];source=Path(hf_hub_download(repo_id=d['id'],repo_type='dataset',revision=d['revision'],filename=d['file'],local_dir=ROOT/'.local/downloads/summedits'))
        assert source.stat().st_size==d['bytes']
        assert file_sha(source)==d['sha256'], 'Pinned raw dataset checksum mismatch'
    if source:print(json.dumps(hydrate(source,args.output),indent=2))
    for key in args.models or []:
        d=deps['models'][key];dest=ROOT/'.local/models'/key
        snapshot_download(repo_id=d['model_id'],revision=d['revision'],local_dir=dest,allow_patterns=[*d['files'],'LICENSE','README.md'])
        for name,expected in d['files'].items():assert file_sha(dest/name)==expected,(key,name)
        print(f'Pinned snapshot hashes verified: {key}')
    if not (source or args.models):p.error('Choose --source-json, --download, or --models; nothing is downloaded by default.')
if __name__=='__main__':main()
