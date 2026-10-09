#!/usr/bin/env python3
"""Explicit Qwen chat prefill and structural-only response contract.

No model weights or MLX runtime are imported. The original collection runner
and its recorded outcomes remain unchanged. This helper belongs to a new,
separately versioned development experiment, not a repair of old result files.
"""
from __future__ import annotations

import hashlib
import json
import re

from run_model_study import canonical, inspect_output, span


CONTRACT_VERSION = "qwen-explicit-prefill-structural-v1"
ASSISTANT_HEADER = "<|im_start|>assistant\n"
EMPTY_THOUGHT = "<think>\n\n</think>\n\n"
EMPTY_SUFFIX = re.compile(r"<\|im_start\|>assistant\n<think>\s*</think>\s*\Z")
EMPTY_RESPONSE_PREFIX = re.compile(r"\A\s*<think>\s*</think>\s*")
COMPLETE_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\Z", re.DOTALL)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_verified_nonthinking(tokenizer, messages):
    """Return ``(token_ids, metadata)`` with an inspected empty-thought prefill.

    Qwen tokenizer revisions differ: a documented keyword can be silently
    ignored by an older Jinja template. Inspect the rendered string instead of
    inferring effective behavior from the argument. This is explicit prompting,
    not a guarantee that a model cannot subsequently emit reasoning text.
    """
    render_args = {"tokenize": False, "add_generation_prompt": True}
    requested = tokenizer.apply_chat_template(messages, enable_thinking=False, **render_args)
    enabled = tokenizer.apply_chat_template(messages, enable_thinking=True, **render_args)
    if not isinstance(requested, str) or not isinstance(enabled, str):
        raise ValueError("Tokenizer must render chat text when tokenize=False")
    adapted = requested
    added = False
    if EMPTY_SUFFIX.search(requested) is None:
        if not requested.endswith(ASSISTANT_HEADER):
            raise ValueError("Unknown assistant template suffix; explicit Qwen adaptation is unsafe")
        adapted += EMPTY_THOUGHT
        added = True
    if EMPTY_SUFFIX.search(adapted) is None:
        raise ValueError("Explicit nonthinking prefill verification failed")
    # Chat-template strings already contain control tokens; never add BOS/EOS.
    tokens = tokenizer.encode(adapted, add_special_tokens=False)
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, int) for t in tokens):
        raise ValueError("Tokenizer must return a nonempty list of integer token IDs")
    template = getattr(tokenizer, "chat_template", None)
    return tokens, {
        "contract_version": CONTRACT_VERSION,
        "requested_enable_thinking": False,
        "template_flag_changes_render": requested != enabled,
        "template_supplied_empty_thought_prefill": bool(EMPTY_SUFFIX.search(requested)),
        "manual_empty_thought_prefill_added": added,
        "effective_prompt_has_empty_thought_prefill": True,
        "control_scope": "explicit_assistant_prefill_not_provider_hard_disable_or_behavior_guarantee",
        "rendered_prompt_sha256": _sha(adapted),
        "unadapted_prompt_sha256": _sha(requested),
        "enabled_prompt_sha256": _sha(enabled),
        "chat_template_sha256": _sha(template) if isinstance(template, str) else None,
        "rendered_suffix": adapted[-len(ASSISTANT_HEADER + EMPTY_THOUGHT):],
        "input_token_count": len(tokens),
    }


def normalize_response(response):
    """Remove one empty thought prefix and one complete outer code fence.

    No substantive reasoning, partial fence, arbitrary prose, JSON repair,
    citation synthesis, or length truncation is permitted.
    """
    if not isinstance(response, str):
        raise ValueError("Response must be text")
    text, count = EMPTY_RESPONSE_PREFIX.subn("", response.strip(), count=1)
    match = COMPLETE_FENCE.fullmatch(text)
    fence = match is not None
    if match:
        text = match.group(1).strip()
    return text, {
        "empty_reasoning_prefix_removed": bool(count),
        "complete_markdown_wrapper_removed": fence,
        "response_normalization": "at_most_one_empty_thought_prefix_then_one_complete_code_fence",
    }


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate_json_key")
        obj[key] = value
    return obj


def _no_constant(value):
    raise ValueError("nonstandard_json_constant")


def inspect_contract(req, action, response, *, max_evidence_ids=3):
    """Check a new response with the frozen structural gate plus declared guards.

    Invalid input spans raise instead of becoming apparent model failures.
    Output rejection retains the original canonical and does not erase proposal
    text or grant semantic approval. Evidence IDs certify only ID membership.
    """
    if action not in ("local", "rewrite"):
        raise ValueError("Unknown generation action")
    original = canonical(req)
    if not isinstance(original, str) or not original:
        raise ValueError("Missing canonical source")
    if action == "local":
        lo, hi = span(req)
        if not 0 <= lo <= hi <= len(original):
            raise ValueError("Local span falls outside canonical source")
        declared = req["local_span"].get("text")
        if declared is not None and declared != original[lo:hi]:
            raise ValueError("Local span source text/version mismatch")
    declared_hash = req.get("canonical_sha256")
    if declared_hash is not None and declared_hash != _sha(original):
        raise ValueError("Canonical source hash mismatch")
    if max_evidence_ids is not None and (not isinstance(max_evidence_ids, int) or max_evidence_ids < 0):
        raise ValueError("Evidence count cap must be a nonnegative integer or None")
    normalized, normalization = normalize_response(response)
    extra_errors = []
    parsed = None
    try:
        parsed = json.loads(normalized, object_pairs_hook=_unique_object, parse_constant=_no_constant)
    except ValueError as exc:
        reason = str(exc)
        extra_errors.append(reason if reason in ("duplicate_json_key", "nonstandard_json_constant") else "invalid_normalized_json")
    outcome = inspect_output(req, action, normalized)
    if isinstance(parsed, dict) and isinstance(parsed.get("evidence_ids"), list):
        if max_evidence_ids is not None and len(parsed["evidence_ids"]) > max_evidence_ids:
            extra_errors.append("evidence_id_count")
    if extra_errors:
        outcome["structural_errors"] = list(dict.fromkeys(outcome["structural_errors"] + extra_errors))
        outcome.update(structural_accepted=False, served_text=original,
                       edit_distance=0.0, retained_canonical=True)
    outcome.update(normalization)
    outcome.update(contract_version=CONTRACT_VERSION, normalized_response=normalized,
                   max_evidence_ids=max_evidence_ids,
                   evidence_check_scope="identifier_membership_only_not_entailment_or_disclosure_approval")
    # The legacy field indicates a wrapper normalized anywhere in this helper.
    outcome["markdown_wrapper_removed"] = bool(outcome["markdown_wrapper_removed"] or normalization["complete_markdown_wrapper_removed"])
    assert outcome["approved_for_serving"] is False
    return outcome
