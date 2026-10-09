"""Independent portable adapter for the frozen experimental specification.

No third-party research-kit implementation is distributed. Prompt strings are
loaded unchanged from the recorded scientific specification. No inference here.
"""
from pathlib import Path
import difflib
import hashlib
import json
import re
ROOT = Path(__file__).resolve().parents[1]
def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()
def token_differences(original, edited):
    before, after = original.split(), edited.split()
    changes=[]
    matcher=difflib.SequenceMatcher(None, before, after, autojunk=False)
    for op,i,j,p,q in matcher.get_opcodes():
        if op != 'equal':
            changes.append(f'{op} at original words [{i}:{j}]: {before[i:j]!r} -> {after[p:q]!r}')
    return changes[:12]+([f'... {len(changes)-12} further changes'] if len(changes)>12 else [])
def make_request(row, condition):
    spec=json.loads((ROOT/'prompts/external_baselines.json').read_text())
    payload={'source_document':row['source'],'edited_summary':row['edited']}
    if condition in ('matched','delta_prompt'):
        payload['original_summary']=row['original']
        payload['token_differences']=token_differences(row['original'], row['edited'])
    instruction=spec['instructions'][condition]
    return [{'role':'system','content':spec['system']},
            {'role':'user','content':instruction+'\n'+json.dumps(payload,ensure_ascii=False)}]
def render_without_tokenizer(messages):
    """Exact executed 8B baseline rendering for these system/user-only messages.

    This fixture renderer does not ship or replace the upstream tokenizer. Fresh
    inference must use that pinned tokenizer and compare this rendered hash.
    """
    if [m['role'] for m in messages] != ['system','user']:
        raise ValueError('Frozen baseline requires one system and one user message')
    return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n<think>\n\n</think>\n\n'
def parse_verdict(raw):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ValueError('duplicate key')
            out[k]=v
        return out
    try: value=json.loads(raw.strip(),object_pairs_hook=unique)
    except (ValueError,TypeError):return None
    if isinstance(value,dict) and set(value)=={'verdict'} and isinstance(value['verdict'],str) and value['verdict'] in ('yes','no'):
        return value['verdict']
    return None

def normalize(raw):
    source,original,edited=(str(raw[k]).strip() for k in ('doc','seed_summary','summary'))
    label=int(raw['label'])
    if label not in (0,1) or not all((source,original,edited)):
        raise ValueError('Invalid frozen sample row')
    return {'case_id':str(raw['id']),'group_id':digest(source),'source':source,
            'original':original,'edited':edited,'label':label,'domain':str(raw.get('domain','')),
            'edit_types':raw.get('edit_types') or [],'changed':original!=edited}
