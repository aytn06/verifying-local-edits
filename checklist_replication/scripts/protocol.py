"""Portable, independently implemented adapter for the frozen specification.

Only filesystem glue and loading differ from the private execution driver.
No upstream research-kit implementation or tokenizer assets are distributed.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'scripts'))
from external_protocol import canonical, digest, token_differences, parse_verdict

MODELS = ('qwen3_8b', 'llama32_3b')
CONDITIONS = ('matched', 'matched_checklist', 'delta_prompt')

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def load_reference(name):
    spec = importlib.util.spec_from_file_location('extension_' + name, ROOT / 'reference' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def spec():
    return json.loads((ROOT / 'prompts/spec.json').read_text(encoding='utf-8'))

def make_request(row, condition):
    specification = spec()
    if condition not in CONDITIONS:
        raise ValueError('Unknown condition')
    payload = {'source_document': row['source'], 'edited_summary': row['edited'],
               'original_summary': row['original'],
               'token_differences': token_differences(row['original'], row['edited'])}
    return [{'role': 'system', 'content': specification['system']},
            {'role': 'user', 'content': specification['instructions'][condition] + '\n' + json.dumps(payload, ensure_ascii=False)}]

def render_fixture(messages, model):
    """Exact system/user fixture rendering; native tokenizer remains authoritative."""
    if [m['role'] for m in messages] != ['system', 'user']:
        raise ValueError('Expected exactly one system and one user message')
    if model == 'qwen3_8b':
        return ''.join('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' for m in messages) + '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    if model != 'llama32_3b':
        raise ValueError('Unknown model')
    return ('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
            'Cutting Knowledge Date: December 2023\nToday Date: 09 Oct 2026\n\n'
            + messages[0]['content'].strip() + '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
            + messages[1]['content'].strip() + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')

def verify_request(row, req, tokenizer=None):
    assert digest(canonical(row)) == req['data_hash'], 'Data record commitment mismatch'
    messages = make_request(row, req['condition'])
    assert digest(canonical(messages)) == req['messages_sha256'], 'Message commitment mismatch'
    rendered = render_fixture(messages, req['model'])
    assert digest(rendered) == req['rendered_hash'], 'Rendered prompt commitment mismatch'
    if tokenizer is None:
        return messages, rendered, None
    native = tokenizer.apply_chat_template(messages, **req['template_settings'])
    assert native == rendered, 'Native chat template differs from frozen rendering'
    ids = tokenizer.encode(native, add_special_tokens=False)
    assert len(ids) == req['input_tokens'] and digest(canonical(ids)) == req['input_token_ids_sha256']
    original = {k: v for k, v in req.items() if k not in ('call_id', 'messages_sha256', 'input_token_ids_sha256')}
    original.update(messages=messages, rendered_prompt=rendered, input_token_ids=ids)
    assert digest(canonical(original)) == req['call_id'], 'Full original request commitment mismatch'
    return messages, rendered, ids

def project_response(row):
    """Retain scoring/cost fields; remove potentially copyrighted generated prose."""
    fields = ('call_id', 'index', 'case_id', 'source_group', 'label', 'split', 'model', 'condition',
              'data_hash', 'rendered_hash', 'verdict', 'parse_error', 'input_tokens', 'output_tokens',
              'seconds_unrounded', 'finish_reason', 'generated_tokens_including_eos', 'decode_matches_raw',
              'timing_scope', 'freeze_sha256', 'is_mock', 'finished_utc')
    out = {k: row[k] for k in fields if k in row}
    raw = row['raw']
    out['raw_sha256'] = digest(raw)
    # Even schema-valid text is reduced to the original strict JSON string only.
    out['raw'] = raw if parse_verdict(raw) is not None else ''
    out['raw_redacted'] = parse_verdict(raw) is None
    for field in ('generated_token_ids', 'output_token_ids', 'eos_output_token_ids'):
        if field in row:
            out[field + '_sha256'] = digest(canonical(row[field]))
            out[field + '_count'] = len(row[field])
    if 'decoded_output' in row:
        out['decoded_output_sha256'] = digest(row['decoded_output'])
    return out

def parse_with_error(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw.strip(), object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None, 'not_strict_json'
    verdict = parse_verdict(raw)
    return (verdict, None) if verdict is not None else (None, 'wrong_schema')
