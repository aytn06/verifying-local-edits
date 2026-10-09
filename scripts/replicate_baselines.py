#!/usr/bin/env python3
"""Optional fresh baseline replication on Apple Silicon; never part of CPU replay.

Requires independently obtained data/weights, explicit --run-inference, and a new
output directory. Retains frozen messages, greedy decoding, 64-token cap, zero
retries, original labels, and prepared case/condition order. The historical test
was executed once; any new run is a separate replication, never a replacement.
"""
import argparse, datetime, importlib.metadata, json, os, time
from pathlib import Path
from external_protocol import ROOT, canonical, digest, make_request, parse_verdict, render_without_tokenizer
from hydrate_external import file_sha, read_rows

def persist(path,obj):
    with path.open('a') as f:
        f.write(canonical(obj)+'\n');f.flush();os.fsync(f.fileno())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--run-inference',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'.local/models/qwen3_8b');p.add_argument('--data',type=Path,default=ROOT/'.local/external')
    args=p.parse_args()
    if not args.run_inference:p.error('Inference requires explicit --run-inference.')
    protected=[ROOT/'results',ROOT/'data',ROOT/'provenance',ROOT/'prompts']
    if args.output.exists() or any(args.output.resolve().is_relative_to(q.resolve()) for q in protected):p.error('Choose a fresh output directory outside frozen release records.')
    dep=json.loads((ROOT/'provenance/dependencies.json').read_text())
    for package in ('mlx','mlx-lm','transformers','tokenizers'):
        assert importlib.metadata.version(package)==dep['versions'][package],package
    for name,expected in dep['models']['qwen3_8b']['files'].items():assert file_sha(args.model/name)==expected,name
    assert file_sha(args.data/f'{args.split}.jsonl')==dep['prepared_files_original_sha256'][f'{args.split}.jsonl']
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler
    model,tokenizer=load(str(args.model),trust_remote_code=False);sampler=make_sampler(temp=0.0)
    records=read_rows(args.data/f'{args.split}.jsonl');known={(r['case_id'],r['condition']):r for r in read_rows(ROOT/f'provenance/{args.split}_requests.metadata.jsonl')}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'replication.json').write_text(json.dumps({'kind':'new_independent_replication','historical_results_replaced':False,'split':args.split,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'settings':dep['baseline_decoding']},indent=2))
    for row in records:
        for condition in ('whole','matched','delta_prompt'):
            messages=make_request(row,condition);k=known[row['case_id'],condition]
            rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            assert rendered==render_without_tokenizer(messages)
            ids=tokenizer.encode(rendered,add_special_tokens=False)
            assert digest(canonical(ids))==k['input_token_ids_sha256']
            assert digest(rendered)==k['rendered_hash'] and len(ids)+64<=40960
            call={'case_id':row['case_id'],'condition':condition,'input_tokens':len(ids),'rendered_hash':digest(rendered),'input_token_ids_sha256':digest(canonical(ids))}
            persist(args.output/'receipts.jsonl',call)
            started=time.monotonic();pieces=[];tokens=[];last=None
            for chunk in stream_generate(model,tokenizer,prompt=ids,max_tokens=64,sampler=sampler):
                pieces.append(chunk.text);tokens.append(int(chunk.token));last=chunk
            elapsed=time.monotonic()-started
            if last is None or last.finish_reason not in ('stop','length'):raise RuntimeError('Unknown/partial call: stop; do not silently retry.')
            assert last.prompt_tokens==len(ids) and last.generation_tokens==len(tokens)
            eos=set(tokenizer.eos_token_ids);end=[t for t in tokens if t in eos]
            assert (last.finish_reason=='stop' and len(end)==1 and tokens[-1] in eos) or (last.finish_reason=='length' and not end)
            output=[t for t in tokens if t not in eos];raw=''.join(pieces);assert tokenizer.decode(output)==raw
            persist(args.output/'predictions.jsonl',{**call,'split':args.split,'source_group':row['group_id'],'label':row['label'],'verdict':parse_verdict(raw),'raw':raw,'output_tokens':len(output),'eos_output_tokens':len(end),'seconds':round(elapsed,4),'seconds_unrounded':elapsed,'finish_reason':last.finish_reason})
    (args.output/'complete.json').write_text(json.dumps({'complete':True,'calls':len(records)*3})+'\n')
if __name__=='__main__':main()
