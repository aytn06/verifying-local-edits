#!/usr/bin/env python3
"""Validate substantive public-data preservation and offline table reproduction.

Original hashes attest private provenance; redacted public projections cannot
validate removed third-party prose or reproduce unavailable private file bytes.
All shipped hashes, labels, IDs, configuration, measured costs, exact constructed
fixtures, cached controlled replies and frozen prompt fixtures are checked here.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, re, sys
from external_protocol import ROOT, canonical, digest, make_request, render_without_tokenizer
from reproduce_tables import read, rows, reproduce, controlled
import controlled_editing_data as construction
sys.path.insert(0,str(ROOT/'reference_code'))
import run_controlled_verification as verifier
import run_controlled_rule_ablation as diagnostic

def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def release_files():
    return [p for p in sorted(ROOT.rglob('*')) if p.is_file() and not any(part in ('.git','.local','__pycache__','generated','replications') for part in p.relative_to(ROOT).parts) and p.name!='release_checksums.json' and p.suffix!='.pyc']
def write_checksums():
    result={str(p.relative_to(ROOT)):file_sha(p) for p in release_files()}
    (ROOT/'provenance/release_checksums.json').write_text(json.dumps(result,indent=2)+'\n')
    return len(result)
def validate():
    checksums=read('provenance/release_checksums.json')
    for name,expected in checksums.items():assert file_sha(ROOT/name)==expected,('public file hash changed',name)
    actual={str(p.relative_to(ROOT)) for p in release_files()}
    assert actual==set(checksums),('unlisted or missing public files',sorted(actual^set(checksums)))
    mappings=read('provenance/source_mapping.json')['entries']
    for m in mappings:
        assert file_sha(ROOT/m['public_path'])==m['public_sha256'],m['public_path']
        if m['transformation']=='byte-identical':assert m['original_sha256']==m['public_sha256']
    built=construction.build_records()
    for objects,name in zip(built,['worlds','tasks','gold','stress_spans']):assert objects==rows(f'data/controlled_editing/{name}.jsonl'),name
    construction.validate_records(*built)
    cases={r['case_id']:r for r in built[1]};controlled_count=0;paired_tokens={}
    prompt=read('prompts/controlled.json');assert prompt['system']==verifier.SYSTEM and prompt['instructions']==verifier.INSTRUCTIONS and prompt['configuration']==verifier.CONFIG
    for model in ('qwen3_8b','qwen3_17b'):
        seen=set()
        for stage in ('smoke','development','evaluation'):
            for r in rows(f'results/controlled_verification/{model}/{stage}_judgments.jsonl'):
                key=r['case_id'],r['condition_id'];assert key not in seen;seen.add(key)
                assert verifier.object_sha(verifier.make_messages(cases[r['case_id']],r['condition_id']))==r['messages_sha256']
                assert r['config']==prompt['configuration'] and r['model_revision']==prompt['models'][model]['revision']
                assert len(r['input_token_ids'])==r['input_tokens'] and len(r['output_token_ids'])==r['output_tokens']
                if model=='qwen3_8b':paired_tokens[key]=r['input_token_ids']
                else:assert r['input_token_ids']==paired_tokens[key]
                controlled_count+=1
        assert seen=={(c,k) for c in cases for k in verifier.CONDITIONS}
    assert controlled_count==864
    diagnostic_rows=rows('results/controlled_rule_ablation/qwen3_8b/judgments.jsonl')
    assert len(diagnostic_rows)==len({r['case_id'] for r in diagnostic_rows})==48
    for r in diagnostic_rows:
        assert r['split']=='development' and cases[r['case_id']]['split']=='development'
        assert verifier.object_sha(diagnostic.ablated_messages(cases[r['case_id']]))==r['messages_sha256']
        parsed=controlled.parse_primary_response(r['raw_response'])
        assert all(parsed[k]==r[k] for k in parsed)
        assert len(r['input_token_ids'])==r['input_tokens'] and len(r['output_token_ids'])==r['output_tokens']
    for f in read('tests/prompt_fixtures.json'):
        messages=make_request(f['row'],f['condition'])
        assert digest(canonical(messages))==f['expected_messages_sha256']
        assert digest(render_without_tokenizer(messages))==f['expected_rendered_sha256']
    assert file_sha(ROOT/'scripts/external_protocol.py')==read('provenance/prompt_equivalence.json')['public_adapter_sha256']
    meta={split:rows(f'data/external/{split}.metadata.jsonl') for split in ('dev','test')}
    assert not {r['group_id'] for r in meta['dev']} & {r['group_id'] for r in meta['test']}
    external_calls=0
    for split in ('dev','test'):
        n=40 if split=='dev' else 100;indexed={r['case_id']:r for r in meta[split]}
        assert len(indexed)==len(meta[split])==len({r['group_id'] for r in meta[split]})==n
        assert Counter(r['label'] for r in meta[split])=={0:n//2,1:n//2}
        predictions=rows(f'results/external/{split}_predictions.metadata.jsonl')
        if split=='dev':predictions+=rows('results/external/delta_claims_v1.metadata.jsonl')
        seen=set()
        for r in predictions:
            key=r['case_id'],r['condition'];assert key not in seen;seen.add(key)
            m=indexed[r['case_id']];assert r['label']==m['label'] and r['source_group']==m['group_id'] and r['data_hash']==m['data_hash'] and r['split']==split
            assert not r['is_mock'] and r['parse_error'] is None
            for s in r['intermediate']:
                assert s['seconds']>=0 and s['input_tokens']>0 and s['output_tokens']>0
                assert not {'messages','rendered_prompt','input_token_ids','output_token_ids','raw','decoded_output'}&set(s)
                external_calls+=1
        conditions={'whole','matched','delta_prompt'}|({'delta_claims_v1'} if split=='dev' else set())
        assert seen=={(cid,c) for cid in indexed for c in conditions}
    assert external_calls==500
    test_domains=Counter(r['domain'] for r in meta['test']);assert test_domains['sales_call']+test_domains['sales_email']==18
    # Check all text files for machine paths and common credential forms. Construct
    # patterns from parts so validation's own source is not a false positive.
    patterns=[re.compile('/'+'Users/'),re.compile('/'+'home/'),re.compile('/'+'private/'),re.compile(r'\b'+'sk-'+r'[A-Za-z0-9]{24,}'),re.compile('hf_'+r'[A-Za-z0-9]{25,}')]
    for p in release_files():
        try:s=p.read_text()
        except UnicodeDecodeError:continue
        assert not any(pattern.search(s) for pattern in patterns),('private path/credential pattern',str(p.relative_to(ROOT)))
    report=reproduce();assert len(report['tables'])==10
    return {'status':'PASS','public_files':len(checksums),'source_mappings':len(mappings),'controlled_fixture_rows':[len(x) for x in built],'controlled_primary_judgments':controlled_count,'controlled_diagnostic_judgments':len(diagnostic_rows),'external_metadata_calls':external_calls,'owned_prompt_fixtures':18,'numeric_tables':10,'network_calls':0,'model_calls':0}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write-checksums',action='store_true',help='Maintainer only: inventory final release after intentional edits; original source mapping still immutable')
    args=p.parse_args()
    if args.write_checksums:print(f'Wrote {write_checksums()} release file hashes.')
    print(json.dumps(validate(),indent=2))
if __name__=='__main__':main()
