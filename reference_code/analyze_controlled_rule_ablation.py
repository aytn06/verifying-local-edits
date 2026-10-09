#!/usr/bin/env python3
"""CPU audit of the post-smoke, development-only E6-removal diagnostic.

No held-out test or confirmatory inference is performed. The primary verifier
results remain unchanged. One source statement is removed; this does not reveal
the model's rationale or establish a general causal mechanism.
"""
from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path

import analyze_controlled_verification as audit
import run_controlled_verification as primary

ROOT = audit.ROOT
MODEL = 'qwen3_8b'
CONDITION = 'whole_factual_without_e6'


def ablated_case(case):
    if [e['evidence_id'] for e in case['evidence']] != [f'E{i}' for i in range(7)]:
        raise ValueError('Expected exact E0-E6 source evidence')
    if not case['evidence'][-1]['text'].startswith('Explicit editorial prominence rule:'):
        raise ValueError('E6 is not the expected explicit ordering rule')
    result = copy.deepcopy(case)
    result['evidence'] = result['evidence'][:-1]
    return result


def validate_protocol(root, cases):
    folder = root / 'results/controlled_rule_ablation'
    protocol = audit.read_json(folder / 'protocol.json')
    selected = [c for c in cases if c['split'] == 'development']
    audit.equal(len(selected), 48, 'Ablation must use all 48 development cases')
    expected = {
        'experiment': 'controlled_rule_ablation_v1', 'status': 'post_smoke_exploratory_development_diagnostic',
        'model_key': MODEL, 'model_spec': primary.MODELS[MODEL], 'condition_id': CONDITION,
        'comparison_condition': 'whole_factual', 'config': dict(primary.CONFIG, base_seed=primary.CONFIG['base_seed'] + 30000),
        'primary_run_protocol_sha256': audit.sha(root / 'results/controlled_verification/run_protocol.json'),
        'primary_model_manifest_sha256': audit.sha(root / 'results/controlled_verification' / MODEL / 'model_manifest.json'),
        'primary_runner_sha256': audit.sha(root / 'scripts/run_controlled_verification.py'),
        'ablation_runner_sha256': audit.sha(root / 'scripts/run_controlled_rule_ablation.py'),
        'case_ids': [c['case_id'] for c in selected],
        'original_case_sha256': {c['case_id']: audit.object_sha(c) for c in selected},
        'removed_evidence': {c['case_id']: c['evidence'][6] for c in selected},
        'retained_evidence_sha256': {c['case_id']: audit.object_sha(c['evidence'][:6]) for c in selected},
        'messages_sha256': {c['case_id']: audit.object_sha(primary.make_messages(ablated_case(c), 'whole_factual')) for c in selected},
        'api_spend_usd': 0, 'human_labels': 0,
    }
    for field, value in expected.items():
        audit.equal(protocol.get(field), value, f'Ablation protocol mismatch: {field}')
    return protocol, selected


def validate_rows(rows, cases, protocol, rendered, tokenizer, folder):
    """Exact lattice, metadata, parser and all-call token audit; no missing outputs."""
    audit.equal([r.get('case_id') for r in rows], [c['case_id'] for c in cases], 'Ablation must contain a unique ordered complete case lattice')
    config = protocol['config']
    for index, (row, case) in enumerate(zip(rows, cases)):
        cid = case['case_id']
        batch = index // config['batch_size']
        seed = config['base_seed'] + batch
        entry = rendered[cid]
        precall = folder / f'batch_{batch:03d}_inputs.json'
        group = cases[batch * config['batch_size']:(batch + 1) * config['batch_size']]
        expected_pre = {'case_ids': [c['case_id'] for c in group], 'batch_seed': seed, 'batch_index': batch,
                        'inputs': [rendered[c['case_id']] for c in group],
                        'protocol_sha256': audit.sha(folder.parent / 'protocol.json')}
        audit.equal(audit.read_json(precall), expected_pre, 'Ablation pre-call manifest mismatch')
        timing = audit.read_json(folder / f'batch_{batch:03d}_timing.json')
        audit.equal(timing['case_ids'], expected_pre['case_ids'], 'Ablation timing IDs mismatch')
        audit.equal(timing['scope'], 'batch_generation_only_not_serving_latency', 'Ablation timing scope mismatch')
        if type(timing['wall_seconds']) not in (int, float) or not math.isfinite(timing['wall_seconds']) or timing['wall_seconds'] < 0:
            raise ValueError('Invalid ablation timing')
        expected = {**{k: case[k] for k in ('case_id', 'pair_id', 'base_scenario_id', 'template_family')},
            'judgment_id': f'{MODEL}:{cid}:{CONDITION}:0', 'split': 'development',
            'run_stage': 'post_smoke_exploratory', 'condition_id': CONDITION, 'comparison_condition_id': 'whole_factual',
            'model_key': MODEL, 'model_id': protocol['model_spec']['model_id'], 'model_revision': protocol['model_spec']['revision'],
            'config': config, 'case_sha256': audit.object_sha(case), 'batch_index': batch, 'batch_seed': seed,
            'attempt_index': 0, 'precall_manifest_sha256': audit.sha(precall),
            'messages_sha256': entry['messages_sha256'], 'render_metadata': entry['render_metadata'],
            'input_token_ids': entry['input_token_ids'], 'input_tokens': len(entry['input_token_ids']),
            'prediction_source': 'local_model', 'human_label': False, 'api_spend_usd': 0, 'semantic_certification': False}
        for field, value in expected.items():
            audit.equal(row.get(field), value, f'Ablation row mismatch: {cid}:{field}')
        for field in ('input_tokens', 'output_tokens', 'batch_index', 'batch_seed', 'attempt_index'):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError(f'Invalid ablation integer: {field}')
        for field in ('format_valid', 'operational_accept', 'normalized_format_valid', 'hit_token_limit', 'human_label', 'semantic_certification'):
            if type(row.get(field)) is not bool:
                raise ValueError(f'Invalid ablation Boolean: {field}')
        ids = row.get('output_token_ids')
        if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids + row['input_token_ids']):
            raise ValueError('Invalid ablation token IDs')
        audit.equal(row['output_tokens'], len(ids), 'Ablation output cost differs from token IDs')
        if len(ids) > config['max_tokens']:
            raise ValueError('Ablation output exceeds frozen token budget')
        audit.equal(row['hit_token_limit'], len(ids) >= config['max_tokens'], 'Ablation token-limit flag mismatch')
        audit.equal(tokenizer.decode(ids), row['raw_response'], 'Ablation output does not decode from saved token IDs')
        for field, value in primary.inspect_response(row['raw_response']).items():
            audit.equal(row.get(field), value, f'Ablation parser replay mismatch: {field}')
        for field, value in audit.parse_primary_response(row['raw_response']).items():
            audit.equal(row[field], value, f'Independent ablation parser mismatch: {field}')
        if any(row.get(field) is not None for field in ('human_ratings', 'usefulness_gain', 'expectation_risk', 'gold')):
            raise ValueError('Unsupported human/outcome/gold field in ablation output')
    return {'generation_calls': len(rows), 'input_tokens': sum(r['input_tokens'] for r in rows),
            'output_tokens': sum(r['output_tokens'] for r in rows)}


def paired_diagnostic(before, after):
    """All 48 pairs and class-specific observed differences; no population CI."""
    a, b = {r['case_id']: r for r in before}, {r['case_id']: r for r in after}
    if len(a) != len(before) or len(b) != len(after) or set(a) != set(b) or not a:
        raise ValueError('Exploratory comparison needs one-to-one unique paired cases')
    transitions, details = Counter(), []
    for cid in a:
        left, right = a[cid], b[cid]
        for key in ('model_key', 'base_scenario_id', 'family_id', 'split', 'factual_valid', 'prominence_valid'):
            audit.equal(left[key], right[key], f'Ablation paired metadata mismatch: {key}')
        audit.equal(left['split'], 'development', 'Exploratory ablation cannot include evaluation cases')
        for row in (left, right):
            audit.equal(row['condition_id'], 'whole_factual', 'Exploratory pairing requires the factual-only declared target')
            audit.equal(row['declared_target_gold'], row['factual_valid'], 'Exploratory pairing has wrong target gold')
        lv = left['verdict'] if left['format_valid'] else 'malformed'
        rv = right['verdict'] if right['format_valid'] else 'malformed'
        transitions[f'{lv}->{rv}'] += 1
        details.append({'case_id': cid, 'construction_class': left['construction_class'],
            'factual_valid': left['factual_valid'], 'before_verdict': lv, 'after_verdict': rv,
            'before_factual_error': left['declared_target_error'], 'after_factual_error': right['declared_target_error'],
            'changed_raw_response': left['raw_response'] != right['raw_response'], 'changed_verdict': lv != rv,
            'input_token_change': right['input_tokens'] - left['input_tokens'],
            'output_token_change': right['output_tokens'] - left['output_tokens']})
    summaries = []
    for kind in ('all', 'acceptable', 'factual_invalid', 'prominence_invalid'):
        lrows = [r for r in before if kind == 'all' or r['construction_class'] == kind]
        rrows = [r for r in after if kind == 'all' or r['construction_class'] == kind]
        for metric in ('declared_target_error', 'factual_valid_false_reject_operational', 'factual_valid_valid_no',
                       'assembled_factual_error_admission', 'acceptable_false_reject_operational', 'malformed'):
            l, r = audit.rate_record(lrows, metric), audit.rate_record(rrows, metric)
            audit.equal(l['denominator'], r['denominator'], 'Exploratory denominator differs across same cases')
            summaries.append({'construction_class': kind, 'metric': metric, 'before_numerator': l['numerator'],
                'after_numerator': r['numerator'], 'eligible_cases': l['denominator'], 'before_rate': l['rate'],
                'after_rate': r['rate'], 'difference_after_minus_before': r['rate'] - l['rate'] if l['rate'] is not None else None})
    return {'cases': len(a), 'verdict_transitions': dict(transitions),
            'changed_raw_responses': sum(r['changed_raw_response'] for r in details),
            'changed_verdicts': sum(r['changed_verdict'] for r in details), 'summaries': summaries, 'cases_detail': details,
            'scope': 'post-smoke exploratory development-only diagnostic; no confirmatory or held-out interval'}


def analyze(root=ROOT):
    root = Path(root)
    out = root / 'results/controlled_rule_ablation'
    folder = out / MODEL
    cases, labels, _ = audit.validate_dataset(root)
    primary_protocol, _ = audit.validate_protocol(root, cases)
    protocol, selected = validate_protocol(root, cases)
    primary_audit_path = root / 'results/controlled_verification/qwen3_8b_audit.json'
    primary_audit = audit.read_json(primary_audit_path)
    audit.equal(primary_audit['status'], 'complete', 'Completed independently audited primary 8B experiment required')
    audit.assert_unchanged(primary_audit['current_file_sha256'])
    frozen = audit.snapshot([out / 'protocol.json', out / 'rendered_inputs.json', primary_audit_path,
                            root / 'scripts/run_controlled_rule_ablation.py'] + list(folder.glob('*.json')) + list(folder.glob('*.jsonl')))
    tokenizer, _, manifest = audit.validate_model(root, MODEL, primary_protocol, cases)
    rendered = audit.read_json(out / 'rendered_inputs.json')
    expected_render = {c['case_id']: audit.expected_entry(ablated_case(c), 'whole_factual', tokenizer) for c in selected}
    audit.equal(rendered, expected_render, 'Ablation rendering differs from independent E6-only removal and CPU tokenizer')
    rows = audit.strict_jsonl(folder / 'judgments.jsonl')
    totals = validate_rows(rows, selected, protocol, rendered, tokenizer, folder)
    completion = {'model_key': MODEL, 'expected': 48, 'completed': len(rows),
        'format_valid': sum(r['format_valid'] for r in rows), **totals,
        'cache_sha256': audit.sha(folder / 'judgments.jsonl'), 'protocol_sha256': audit.sha(out / 'protocol.json'),
        'scope': 'post_smoke_exploratory_development_only', 'human_labels': 0, 'api_spend_usd': 0}
    audit.equal(audit.read_json(folder / 'completion.json'), completion, 'Ablation completion/cost/cache mismatch')
    # The altered prompt is still declared full-candidate factual verification.
    normalized_condition = [dict(r, condition_id='whole_factual') for r in rows]
    after = audit.join_predictions(normalized_condition, cases, labels)
    primary_folder = root / 'results/controlled_verification' / MODEL
    before_raw = audit.strict_jsonl(primary_folder / 'smoke_judgments.jsonl') + audit.strict_jsonl(primary_folder / 'development_judgments.jsonl')
    before = audit.join_predictions([r for r in before_raw if r['condition_id'] == 'whole_factual'], cases, labels)
    report = {'status': 'complete_post_smoke_exploratory_development_diagnostic', 'experiment': protocol['experiment'],
        'model': MODEL, 'model_manifest': manifest, 'human_labels': 0, 'api_spend_usd': 0, **totals,
        'primary_audit_sha256': audit.sha(primary_audit_path), 'ablation_protocol_sha256': audit.sha(out / 'protocol.json'),
        'analysis_script_sha256': audit.sha(Path(__file__)), 'current_file_sha256': frozen,
        'paired_diagnostic': paired_diagnostic(before, after),
        'limitations': ['Chosen after the development smoke; not a held-out or confirmatory test.',
            'Only two development lexical families sharing the same sentence grammar.',
            'Binary verdicts do not reveal internal reasoning or prove a general mechanism.',
            'Model inputs omit E6 only, but batching order/companions differ; no cross-batch bitwise guarantee.',
            'No human ratings, audience expectations, personalization usefulness, or real-error prevalence measured.']}
    audit.assert_unchanged(frozen)
    audit.assert_unchanged(primary_audit['current_file_sha256'])
    (out / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    audit.write_csv(out / 'paired_summary.csv', report['paired_diagnostic']['summaries'])
    audit.write_csv(out / 'paired_cases.csv', report['paired_diagnostic']['cases_detail'])
    text = ['# Exploratory source-rule removal diagnostic', '',
        'This was selected after the development smoke. It uses all 48 development cases and no evaluation cases; the primary results remain unchanged.', '',
        f"Observed verdict transitions: {report['paired_diagnostic']['verdict_transitions']}. "
        f"New generation calls: {len(rows)}; returned output tokens: {totals['output_tokens']}; human ratings: 0; API spend: $0.", '',
        '| Class | Factual-task errors with E6 | Factual-task errors without E6 | Cases |', '|---|---:|---:|---:|']
    for r in report['paired_diagnostic']['summaries']:
        if r['metric'] == 'declared_target_error':
            text.append(f"| {r['construction_class']} | {r['before_numerator']} | {r['after_numerator']} | {r['eligible_cases']} |")
    text += ['', *['- ' + item for item in report['limitations']]]
    (out / 'analysis_note.md').write_text('\n'.join(text) + '\n')
    return report


if __name__ == '__main__':
    result = analyze()
    print(json.dumps({k: result[k] for k in ('status', 'generation_calls', 'input_tokens', 'output_tokens')}))
