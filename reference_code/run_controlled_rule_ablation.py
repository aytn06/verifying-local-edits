#!/usr/bin/env python3
"""Post-smoke exploratory E6 removal on the 48 development candidates only.

Uses the frozen primary whole-factual instruction and cached Qwen3-8B. This is
an input ablation, not a correction of primary results or a held-out test.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time

import run_controlled_verification as primary

ROOT = primary.ROOT
OUT = ROOT / "results/controlled_rule_ablation"
MODEL_KEY = "qwen3_8b"
CONDITION = "whole_factual_without_e6"
CONFIG = dict(primary.CONFIG, base_seed=primary.CONFIG["base_seed"] + 30000)


def ablated_messages(case):
    """Remove only E6; preserve all factual source sentences and candidate bytes."""
    primary.validate_case(case)
    removed = [e for e in case["evidence"] if e["evidence_id"] == "E6"]
    if len(removed) != 1 or not removed[0]["text"].startswith("Explicit editorial prominence rule:"):
        raise ValueError("Expected exactly the declared E6 editorial-rule sentence")
    if [e["evidence_id"] for e in case["evidence"]] != [f"E{i}" for i in range(7)]:
        raise ValueError("Unexpected source layout; do not silently alter the ablation")
    changed = copy.deepcopy(case)
    changed["evidence"] = changed["evidence"][:6]
    return primary.make_messages(changed, "whole_factual")


def prepare(*, freeze):
    cases, _, _, tokenizer = primary.prepare(MODEL_KEY, freeze=False)
    selected = [c for c in cases if c["split"] == "development"]
    if len(selected) != 48:
        raise ValueError("Expected all 48 development cases")
    rendered = {}
    for case in selected:
        messages = ablated_messages(case)
        tokens, meta = primary.render_verified_nonthinking(tokenizer, messages)
        rendered[case["case_id"]] = {"messages": messages, "messages_sha256": primary.object_sha(messages),
                                    "input_token_ids": tokens, "input_token_ids_sha256": primary.object_sha(tokens),
                                    "render_metadata": meta}
    protocol = {
        "experiment": "controlled_rule_ablation_v1", "status": "post_smoke_exploratory_development_diagnostic",
        "decision_basis": "six-case primary development smoke showed factual-whole rejection of factually true ordering violations; no primary evaluation outputs informed this design",
        "hypothesis_scope": "paired sensitivity of factual-only verdicts to the irrelevant explicit E6 ordering instruction; rationale and causal mechanism inside the model remain unobserved",
        "model_key": MODEL_KEY, "model_spec": primary.MODELS[MODEL_KEY], "condition_id": CONDITION,
        "comparison_condition": "whole_factual", "config": CONFIG,
        "primary_run_protocol_sha256": primary.sha_file(primary.OUT / "run_protocol.json"),
        "primary_model_manifest_sha256": primary.sha_file(primary.OUT / MODEL_KEY / "model_manifest.json"),
        "primary_runner_sha256": primary.sha_file(primary.__file__), "ablation_runner_sha256": primary.sha_file(__file__),
        "case_ids": [c["case_id"] for c in selected],
        "original_case_sha256": {c["case_id"]: primary.object_sha(c) for c in selected},
        "removed_evidence": {c["case_id"]: c["evidence"][6] for c in selected},
        "retained_evidence_sha256": {c["case_id"]: primary.object_sha(c["evidence"][:6]) for c in selected},
        "messages_sha256": {k: v["messages_sha256"] for k, v in rendered.items()},
        "invariants": "same complete candidate, E0-E5, system/instruction, tokenizer, model, greedy128 budget and parser; only E6 removed; no retries",
        "labels": "unchanged construction-based factual labels; no human judgments or audience-expectation labels",
        "api_spend_usd": 0, "human_labels": 0,
    }
    for filename, payload in (("protocol.json", protocol), ("rendered_inputs.json", rendered)):
        path = OUT / filename
        if freeze:
            primary.write_frozen(path, payload)
        elif not path.exists() or json.loads(path.read_text()) != payload:
            raise ValueError("Missing/changed ablation freeze; do not alter a running experiment")
    return selected, rendered, tokenizer


def assert_primary_complete():
    directory = primary.OUT / MODEL_KEY
    for stage, expected in (("smoke", 18), ("development", 126), ("evaluation", 288)):
        path = directory / f"{stage}_completion.json"
        if not path.exists():
            raise ValueError("Primary 8B collection must finish before this GPU diagnostic")
        completed = json.loads(path.read_text())
        if completed["completed"] != expected or completed["cache_sha256"] != primary.sha_file(directory / f"{stage}_judgments.jsonl"):
            raise ValueError("Primary completion count/hash does not match its cache")


def collect():
    assert_primary_complete()
    cases, rendered, cpu_tokenizer = prepare(freeze=False)
    directory = OUT / MODEL_KEY
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "judgments.jsonl"
    rows = primary.read_jsonl(cache) if cache.exists() else []
    saved = {}
    by_id = {c["case_id"]: c for c in cases}
    for row in rows:
        cid = row["case_id"]
        if cid in saved or cid not in by_id:
            raise ValueError("Duplicate or unknown ablation case")
        expected = {"judgment_id": f"{MODEL_KEY}:{cid}:{CONDITION}:0", "condition_id": CONDITION,
                    "model_key": MODEL_KEY, "model_id": primary.MODELS[MODEL_KEY]["model_id"],
                    "model_revision": primary.MODELS[MODEL_KEY]["revision"], "config": CONFIG,
                    "case_sha256": primary.object_sha(by_id[cid]),
                    "messages_sha256": rendered[cid]["messages_sha256"],
                    "input_token_ids": rendered[cid]["input_token_ids"],
                    **primary.inspect_response(row["raw_response"])}
        if any(row.get(k) != value for k, value in expected.items()):
            raise ValueError("Ablation cache no longer matches the frozen input/parser")
        if (row["output_tokens"] != len(row["output_token_ids"])
                or row["input_tokens"] != len(row["input_token_ids"])
                or cpu_tokenizer.decode(row["output_token_ids"]) != row["raw_response"]):
            raise ValueError("Cached ablation text/tokens do not replay")
        saved[cid] = row
    if len(saved) < len(cases):
        import mlx.core as mx
        from mlx_lm import load, batch_generate
        from mlx_lm.sample_utils import make_sampler
        tick = time.perf_counter()
        model, tokenizer = load(str(primary.model_directory(MODEL_KEY)), trust_remote_code=False)
        print(json.dumps({"event": "loaded_ablation", "seconds": time.perf_counter()-tick}), flush=True)
        for entry in rendered.values():
            ids, meta = primary.render_verified_nonthinking(tokenizer, entry["messages"])
            if ids != entry["input_token_ids"] or meta != entry["render_metadata"]:
                raise ValueError("Ablation runtime tokenizer disagrees with frozen rendering")
        for batch_index, start in enumerate(range(0, len(cases), CONFIG["batch_size"])):
            group = cases[start:start+CONFIG["batch_size"]]
            present = [c["case_id"] in saved for c in group]
            if all(present):
                continue
            if any(present):
                raise ValueError("Partial ablation batch; no silent resampling")
            entries = [rendered[c["case_id"]] for c in group]
            seed = CONFIG["base_seed"] + batch_index
            prefile = directory / f"batch_{batch_index:03d}_inputs.json"
            primary.write_frozen(prefile, {"case_ids": [c["case_id"] for c in group], "batch_seed": seed,
                                         "batch_index": batch_index, "inputs": entries,
                                         "protocol_sha256": primary.sha_file(OUT / "protocol.json")})
            mx.random.seed(seed)
            tick = time.perf_counter()
            result = batch_generate(model, tokenizer, [e["input_token_ids"] for e in entries], max_tokens=CONFIG["max_tokens"],
                                    sampler=make_sampler(temp=0.0), return_token_ids=True,
                                    prefill_batch_size=4, completion_batch_size=4)
            elapsed = time.perf_counter()-tick
            if len(result.texts) != len(group) or len(result.token_ids) != len(group):
                raise ValueError("Ablation result cardinality differs from the input")
            batch_rows = []
            for case, entry, text, ids in zip(group, entries, result.texts, result.token_ids):
                if tokenizer.decode(ids) != text:
                    raise ValueError("Ablation response cannot be decoded from its returned IDs")
                cid = case["case_id"]
                row = {"judgment_id": f"{MODEL_KEY}:{cid}:{CONDITION}:0", "case_id": cid,
                       "pair_id": case["pair_id"], "base_scenario_id": case["base_scenario_id"],
                       "template_family": case["template_family"], "split": "development",
                       "run_stage": "post_smoke_exploratory", "condition_id": CONDITION,
                       "comparison_condition_id": "whole_factual", "model_key": MODEL_KEY,
                       "model_id": primary.MODELS[MODEL_KEY]["model_id"], "model_revision": primary.MODELS[MODEL_KEY]["revision"],
                       "config": CONFIG, "case_sha256": primary.object_sha(case), "batch_index": batch_index,
                       "batch_seed": seed, "attempt_index": 0, "precall_manifest_sha256": primary.sha_file(prefile),
                       "messages_sha256": entry["messages_sha256"], "render_metadata": entry["render_metadata"],
                       "input_token_ids": entry["input_token_ids"], "output_token_ids": ids,
                       "input_tokens": len(entry["input_token_ids"]), "output_tokens": len(ids),
                       "hit_token_limit": len(ids) >= CONFIG["max_tokens"], "raw_response": text,
                       "prediction_source": "local_model", "human_label": False, "api_spend_usd": 0,
                       "semantic_certification": False, **primary.inspect_response(text)}
                batch_rows.append(row)
            primary.append_rows(cache, batch_rows)
            saved.update((r["case_id"], r) for r in batch_rows)
            primary.write_frozen(directory / f"batch_{batch_index:03d}_timing.json",
                                 {"case_ids": [r["case_id"] for r in batch_rows], "wall_seconds": elapsed,
                                  "scope": "batch_generation_only_not_serving_latency", "stats": dataclasses.asdict(result.stats)})
            mx.clear_cache()
            print(json.dumps({"event": "ablation_batch", "completed": len(saved), "expected": 48,
                              "format_valid_in_batch": sum(r["format_valid"] for r in batch_rows)}), flush=True)
    if set(saved) != set(by_id):
        raise ValueError("Incomplete ablation cache")
    rows = list(saved.values())
    summary = {"model_key": MODEL_KEY, "expected": 48, "completed": len(rows),
               "format_valid": sum(r["format_valid"] for r in rows), "generation_calls": len(rows),
               "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
               "cache_sha256": primary.sha_file(cache), "protocol_sha256": primary.sha_file(OUT / "protocol.json"),
               "scope": "post_smoke_exploratory_development_only", "human_labels": 0, "api_spend_usd": 0}
    primary.write_frozen(directory / "completion.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "run"), required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        cases, rendered, _ = prepare(freeze=True)
        print(json.dumps({"event": "ablation_frozen_cpu_only", "cases": len(cases), "prompts": len(rendered)}))
    else:
        collect()
