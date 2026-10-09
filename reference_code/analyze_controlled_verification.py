#!/usr/bin/env python3
"""Independent CPU analysis of a construction-labelled verification benchmark.

Synthetic gold is never a human outcome. Replacement acceptance is evaluated
as a full-candidate gate, not scored as fragment-classification accuracy when
no separately certified fragment gold exists. Malformed responses fail closed
operationally but count as classification errors, never correct rejections.
"""
from __future__ import annotations

import argparse
import csv
import os
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("replacement_factual", "whole_factual", "whole_factual_prominence")
SPLITS = ("development", "evaluation")
BOOTSTRAP_SEED = 20261008
BOOTSTRAP_REPETITIONS = 5000


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def equal(actual, expected, message):
    if actual != expected:
        raise ValueError(message)


def strict_jsonl(path):
    data = Path(path).read_bytes() if Path(path).exists() else b""
    lines = data.splitlines(keepends=True)
    if any(not line.strip() for line in lines) or (lines and not lines[-1].endswith(b"\n")):
        raise ValueError(f"Blank or unfinished JSONL line: {path}")
    return [json.loads(line) for line in lines]


def snapshot(paths):
    return {str(p): sha(p) if p.exists() else None for p in paths}


def assert_unchanged(hashes):
    for name, expected in hashes.items():
        path = Path(name)
        equal(sha(path) if path.exists() else None, expected, f"File changed during analysis: {name}")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonstandard_json_constant")


def parse_primary_response(raw):
    """Only exact one-key JSON; whitespace is allowed, wrappers are not."""
    if not isinstance(raw, str):
        raise ValueError("Raw judgment response must be text")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    except (ValueError, TypeError):
        return {"format_valid": False, "verdict": None, "operational_accept": False}
    valid = isinstance(value, dict) and set(value) == {"verdict"} and value["verdict"] in ("yes", "no")
    if valid and not isinstance(value["verdict"], str):
        valid = False
    verdict = value["verdict"] if valid else None
    return {"format_valid": valid, "verdict": verdict, "operational_accept": verdict == "yes"}


def construction_class(label):
    factual, prominence = label.get("factual_valid"), label.get("prominence_valid")
    if not isinstance(factual, bool) or not isinstance(prominence, bool):
        raise ValueError("Construction labels must be explicit Boolean factual_valid and prominence_valid")
    if not factual and not prominence:
        return "both_invalid"
    if not factual:
        return "factual_invalid"
    if not prominence:
        return "prominence_invalid"
    return "acceptable"


def join_predictions(predictions, cases, labels):
    cmap = {r["case_id"]: r for r in cases}
    lmap = {r["case_id"]: r for r in labels}
    if len(cmap) != len(cases) or len(lmap) != len(labels) or set(cmap) != set(lmap):
        raise ValueError("Cases and gold sidecar must be a unique exact one-to-one mapping")
    out, seen = [], set()
    for prediction in predictions:
        cid = prediction["case_id"]
        if cid not in cmap or prediction["condition_id"] not in CONDITIONS:
            raise ValueError("Unknown case or condition")
        key = (prediction["model_key"], cid, prediction["condition_id"])
        if key in seen:
            raise ValueError("Duplicate model/case/condition judgment")
        seen.add(key)
        case, label = cmap[cid], lmap[cid]
        for field in ("base_scenario_id", "template_family", "split"):
            equal(prediction[field], case[field], f"Prediction differs from source {field}")
        parsed = parse_primary_response(prediction["raw_response"])
        for field, value in parsed.items():
            if field in prediction:
                if field != "verdict" and not isinstance(prediction[field], bool):
                    raise ValueError(f"Prediction {field} must be a Boolean")
                equal(prediction[field], value, f"Recorded {field} differs from independent parsing")
        kind = construction_class(label)
        row = dict(prediction, **{k: v for k, v in parsed.items() if k not in prediction})
        row["family_id"] = case["template_family"]
        row["construction_type"] = label.get("construction_type")
        row.update({"factual_valid": label["factual_valid"], "prominence_valid": label["prominence_valid"],
                    "joint_valid": label["factual_valid"] and label["prominence_valid"], "construction_class": kind})
        row["declared_target_gold"] = (None if row["condition_id"] == "replacement_factual"
            else row["factual_valid"] if row["condition_id"] == "whole_factual" else row["joint_valid"])
        # Invalid/missing output is always an error for classification diagnostics.
        row["joint_classification_error"] = not row["format_valid"] or (row["operational_accept"] != row["joint_valid"])
        row["declared_target_error"] = (None if row["declared_target_gold"] is None else
            not row["format_valid"] or row["operational_accept"] != row["declared_target_gold"])
        out.append(row)
    return out


def metric_parts(row, metric):
    """Return numerator, denominator; no prediction-dependent primary eligibility."""
    factual, prominence = row["factual_valid"], row["prominence_valid"]
    accepted, valid = row["operational_accept"], row["format_valid"]
    joint = factual and prominence
    mapping = {
        "assembled_factual_error_admission": (accepted, not factual),
        "pure_prominence_error_admission": (accepted, factual and not prominence),
        "joint_error_admission": (accepted, not joint),
        "acceptable_false_reject_operational": (not accepted, joint),
        "acceptable_valid_no": (valid and row["verdict"] == "no", joint),
        "factual_valid_false_reject_operational": (not accepted, factual),
        "factual_valid_valid_no": (valid and row["verdict"] == "no", factual),
        "pure_prominence_valid_no": (valid and row["verdict"] == "no", factual and not prominence),
        "malformed": (not valid, True),
        "coverage": (valid, True),
        "joint_classification_error": (row["joint_classification_error"], True),
        "declared_target_error": (row["declared_target_error"], row["declared_target_error"] is not None),
        "parsed_only_factual_error_admission": (accepted, not factual and valid),
        "parsed_only_prominence_error_admission": (accepted, factual and not prominence and valid),
        "parsed_only_acceptable_false_reject": (row["verdict"] == "no", joint and valid),
    }
    if metric not in mapping:
        raise ValueError(f"Unknown metric: {metric}")
    event, eligible = mapping[metric]
    return int(bool(event) and eligible), int(eligible)


def rate_record(rows, metric):
    parts = [metric_parts(r, metric) for r in rows]
    numerator, denominator = sum(p[0] for p in parts), sum(p[1] for p in parts)
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def summarize(rows, *, model, split, condition, output_class="all"):
    selected = [r for r in rows if r["model_key"] == model and r["split"] == split and r["condition_id"] == condition
                and (output_class == "all" or r["construction_class"] == output_class)]
    metrics = ("assembled_factual_error_admission", "pure_prominence_error_admission", "joint_error_admission",
               "acceptable_false_reject_operational", "acceptable_valid_no", "factual_valid_false_reject_operational",
               "factual_valid_valid_no", "pure_prominence_valid_no", "malformed", "coverage",
               "joint_classification_error", "declared_target_error", "parsed_only_factual_error_admission",
               "parsed_only_prominence_error_admission", "parsed_only_acceptable_false_reject")
    result = {"model": model, "split": split, "condition": condition, "construction_class": output_class,
              "cases": len(selected), "base_scenarios": len({r["base_scenario_id"] for r in selected}),
              "families": len({r["family_id"] for r in selected}),
              "valid_yes": sum(r["format_valid"] and r["verdict"] == "yes" for r in selected),
              "valid_no": sum(r["format_valid"] and r["verdict"] == "no" for r in selected),
              "malformed_count": sum(not r["format_valid"] for r in selected),
              "input_tokens": sum(r["input_tokens"] for r in selected),
              "output_tokens": sum(r["output_tokens"] for r in selected),
              "token_limit_count": sum(r["hit_token_limit"] for r in selected)}
    for metric in metrics:
        value = rate_record(selected, metric)
        result.update({metric + "_" + k: v for k, v in value.items()})
    errors = [rate_record([r for r in selected if r["construction_class"] == kind], "joint_classification_error")["rate"]
              for kind in ("acceptable", "factual_invalid", "prominence_invalid")]
    result["balanced_joint_classification_error"] = sum(errors) / 3 if all(x is not None for x in errors) else None
    result["declared_target_scope"] = {
        "replacement_factual": "fragment support; nominal accuracy omitted because only assembled-candidate gold is certified",
        "whole_factual": "complete-candidate factual validity only; pure prominence acceptance is not a factual-task error",
        "whole_factual_prominence": "complete-candidate factual validity and explicit editorial ordering-policy compliance",
    }[condition]
    return result


def paired_cluster_contrast(rows, before, after, metric, *, cluster="family_id", seed=BOOTSTRAP_SEED,
                            repetitions=BOOTSTRAP_REPETITIONS):
    """Paired ratio-of-sums bootstrap, preserving all variants within a cluster."""
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    if cluster not in ("family_id", "base_scenario_id"):
        raise ValueError("Only declared family or base-scenario clusters are allowed")
    pairs = defaultdict(dict)
    for row in rows:
        if row["condition_id"] in (before, after):
            if row["condition_id"] in pairs[row["case_id"]]:
                raise ValueError("Duplicate condition within a paired case")
            pairs[row["case_id"]][row["condition_id"]] = row
    sums = defaultdict(lambda: [0, 0])
    for pair in pairs.values():
        if set(pair) != {before, after}:
            raise ValueError("Paired contrast has missing condition outputs")
        left, right = pair[before], pair[after]
        for field in ("family_id", "base_scenario_id", "split", "model_key", "factual_valid", "prominence_valid"):
            equal(left[field], right[field], f"Mismatched paired {field}")
        a, na = metric_parts(left, metric)
        b, nb = metric_parts(right, metric)
        if na != nb:
            raise ValueError("Primary contrast cannot condition on differing predicted-output eligibility")
        if na:
            sums[left[cluster]][0] += b - a
            sums[left[cluster]][1] += na
    numerator, denominator = sum(x[0] for x in sums.values()), sum(x[1] for x in sums.values())
    estimate = numerator / denominator if denominator else None
    values = list(sums.values())
    interval, degenerate = None, None
    if len(values) >= 2:
        rng = random.Random(seed)
        draws = []
        for _ in range(repetitions):
            selected = rng.choices(values, k=len(values))
            draws.append(sum(v[0] for v in selected) / sum(v[1] for v in selected))
        draws.sort()
        interval = [draws[int(.025 * repetitions)], draws[min(repetitions - 1, int(.975 * repetitions))]]
        degenerate = len(set(draws)) == 1
    return {"before": before, "after": after, "contrast": "after_minus_before", "metric": metric,
            "estimate": estimate, "numerator_difference": numerator, "eligible_cases": denominator,
            "cluster": cluster, "clusters": len(values), "ci95_descriptive": interval,
            "bootstrap_seed": seed, "bootstrap_repetitions": repetitions, "interval_degenerate": degenerate,
            "scope": ("primary descriptive resampling of four held-out template families; very limited generalizability"
                      if cluster == "family_id" else "secondary sensitivity conditional on the selected templates; bases are not independent narrative families")}


def exact_view_bound(rows, target="factual_valid"):
    """Finite uniform-case information bound, not fragment-task accuracy."""
    if target not in ("factual_valid", "joint_valid"):
        raise ValueError("Bound targets complete-candidate factual or joint validity")
    groups, seen = defaultdict(list), set()
    for row in rows:
        if row["case_id"] in seen:
            raise ValueError("One model/condition/split is required for exact-view bound")
        seen.add(row["case_id"])
        groups[row["visible_messages_sha256"]].append(row)
    conflicts, total_errors, disagreements = [], 0, 0
    for key, values in groups.items():
        token_hashes = {r["input_token_ids_sha256"] for r in values}
        if len(token_hashes) != 1:
            raise ValueError("Identical visible messages have differing input-token hashes")
        positives = sum(r[target] for r in values)
        negatives = len(values) - positives
        total_errors += min(positives, negatives)
        if len({(r["format_valid"], r["verdict"]) for r in values}) > 1:
            disagreements += 1
        if positives and negatives:
            conflicts.append({"visible_messages_sha256": key, "case_ids": [r["case_id"] for r in values],
                              "positive_labels": positives, "negative_labels": negatives,
                              "unavoidable_empirical_errors": min(positives, negatives)})
    return {"target": target, "cases": len(rows), "unique_views": len(groups),
            "contradictory_views": len(conflicts), "unavoidable_empirical_errors": total_errors,
            "minimum_empirical_error": total_errors / len(rows) if rows else None,
            "groups_with_verdict_disagreement": disagreements, "contradictory_groups": conflicts,
            "scope": "uniform finite-case complete-candidate target from this exact visible view; not population Bayes risk or a fragment-task accuracy label"}


def object_sha(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_dataset(root):
    """Replay deterministic gold, and independently check assembly and labels."""
    import controlled_editing_data as data
    folder = root / "data/controlled_editing"
    manifest = read_json(folder / "manifest.json")
    names = ("worlds.jsonl", "tasks.jsonl", "gold.jsonl", "stress_spans.jsonl")
    collections = []
    for name in names:
        equal(sha(folder / name), manifest["files"][name]["sha256"], f"Dataset file hash mismatch: {name}")
        rows = strict_jsonl(folder / name)
        equal(len(rows), manifest["files"][name]["rows"], f"Dataset row count mismatch: {name}")
        collections.append(rows)
    equal(sha(root / "scripts/controlled_editing_data.py"), manifest["script_sha256"], "Construction source changed")
    equal(collections, list(data.build_records()), "Construction files differ from deterministic replay")
    worlds, cases, labels, stress = collections
    data.validate_records(worlds, cases, labels, stress)
    equal((len(worlds), len(cases), len(labels), len(stress)), (24, 144, 144, 48), "Wrong construction lattice")
    equal(manifest["human_labels"], 0, "Synthetic construction cannot contain human labels")
    wmap = {w["base_scenario_id"]: w for w in worlds}
    lmap = {x["case_id"]: x for x in labels}
    if len(lmap) != 144 or set(lmap) != {x["case_id"] for x in cases}:
        raise ValueError("Missing, extra, or duplicate construction gold")
    for case in cases:
        world, label = wmap[case["base_scenario_id"]], lmap[case["case_id"]]
        equal(label["task_sha256"], object_sha(case), "Gold detached from task")
        equal(case["source_sha256"], object_sha(case["evidence"]), "Evidence hash mismatch")
        equal(case["evidence"], world["evidence"], "Task source differs from world")
        equal(label["human_label"], False, "Synthetic gold mislabelled as human")
        equal(label["label_source"], "deterministic_controlled_construction", "Unrecognized gold origin")
        a, b = case["local_span"]["start_char"], case["local_span"]["end_char"]
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(case["canonical_text"]):
            raise ValueError("Invalid primary source span")
        equal(case["canonical_text"][a:b], case["local_span"]["text"], "Primary span mismatch")
        equal(case["candidate_text"], case["canonical_text"][:a] + case["replacement_text"] + case["canonical_text"][b:], "Candidate assembly mismatch")
        equal(case["candidate_sha256"], digest(case["candidate_text"].encode()), "Candidate hash mismatch")
        clauses = [s.strip() + "." for s in case["candidate_text"].split(".") if s.strip()]
        # This is finite grammar integrity, not independent semantic annotation.
        certificates = [world["clause_certificates"][clause] for clause in clauses]
        equal(label["factual_valid"], all(c["factual_valid"] for c in certificates), "Factual certificate mismatch")
        equal(label["prominence_valid"], certificates[0]["event_class"] == "central_work", "Ordering certificate mismatch")
        construction_class(label)
    per_base = defaultdict(Counter)
    for case in cases:
        per_base[case["base_scenario_id"]][construction_class(lmap[case["case_id"]])] += 1
    equal(set(tuple(sorted(c.items())) for c in per_base.values()),
          {(('acceptable', 2), ('factual_invalid', 2), ('prominence_invalid', 2))}, "Wrong gold class balance")
    return cases, labels, manifest


def validate_protocol(root, cases):
    import run_controlled_verification as run
    path = root / "results/controlled_verification/run_protocol.json"
    protocol = read_json(path)
    for name, expected in protocol["code_files"].items():
        equal(sha(root / name), expected, f"Frozen runner/helper changed: {name}")
    for name, expected in protocol["source_files"].items():
        equal(sha(root / "data/controlled_editing" / name), expected, f"Frozen source changed: {name}")
    equal(sha(root / "notes/no_human_experiment_protocol.md"), protocol["analysis_plan_sha256"], "Frozen analysis plan changed")
    # Runtime package versions, every message and source binding are replayed.
    equal(protocol, json.loads(json.dumps(run.protocol_payload(cases))), "Runtime protocol differs from replay")
    equal(protocol["config"], {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "max_tokens": 128,
          "batch_size": 4, "repetitions": 1, "retry_count": 0, "base_seed": 281905}, "Unexpected frozen generation configuration")
    stages = {k: [x for x in cases if x["case_id"] in protocol["stage_case_ids"][k]]
              for k in ("smoke", "development", "evaluation")}
    equal({k: len(v) for k, v in stages.items()}, {"smoke": 6, "development": 42, "evaluation": 96}, "Stage lattice mismatch")
    equal([c["case_id"] for v in stages.values() for c in v], [c["case_id"] for c in cases], "Stages are not a disjoint ordered partition")
    return protocol, stages


def expected_entry(case, condition, tokenizer):
    import run_controlled_verification as run
    from generation_contract import render_verified_nonthinking
    messages = run.make_messages(case, condition)
    ids, metadata = render_verified_nonthinking(tokenizer, messages)
    return {"messages": messages, "messages_sha256": object_sha(messages), "input_token_ids": ids,
            "input_token_ids_sha256": object_sha(ids), "render_metadata": metadata}


def validate_model(root, key, protocol, cases):
    import run_controlled_verification as run
    from transformers import AutoTokenizer
    folder = root / "results/controlled_verification" / key
    manifest = read_json(folder / "model_manifest.json")
    directory = run.model_directory(key)
    equal({k: manifest[k] for k in ("model_id", "revision", "weight_sha256")}, protocol["models"][key], "Model identity mismatch")
    equal(manifest["run_protocol_sha256"], sha(folder.parent / "run_protocol.json"), "Model/protocol binding mismatch")
    equal(set(manifest["files"]), {"model.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json"}, "Missing frozen model files")
    for name, expected in manifest["files"].items():
        equal(sha(directory / name), expected, f"Model/tokenizer file changed: {key}/{name}")
    equal(manifest["files"]["model.safetensors"], protocol["models"][key]["weight_sha256"], "Weight differs from pinned checkpoint")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    actual = read_json(folder / "rendered_inputs.json")
    replay = {f"{case['case_id']}:{condition}": expected_entry(case, condition, tokenizer)
              for case in cases for condition in CONDITIONS}
    equal(actual, replay, "Frozen rendering differs from CPU tokenizer replay")
    return tokenizer, actual, manifest


def validate_stage(rows, cases, stage, model, protocol, rendered, tokenizer, folder, *, allow_incomplete=False):
    """Validate all rows, exact prefix, pre-call inputs, seeds and cost without GPU."""
    import run_controlled_verification as run
    jobs = [(f"{model}:{case['case_id']}:{condition}:0", case, condition)
            for case in cases for condition in CONDITIONS]
    expected_ids = [j[0] for j in jobs]
    ids = [row.get("judgment_id") for row in rows]
    if len(set(ids)) != len(ids) or ids != expected_ids[:len(ids)] or len(ids) > len(jobs):
        raise ValueError("Judgment cache is not a unique exact prefix of its frozen lattice")
    if len(rows) != len(jobs) and (not allow_incomplete or len(rows) % protocol["config"]["batch_size"]):
        raise ValueError("Incomplete lattice or interrupted partial batch")
    config = protocol["config"]
    batch_size = config["batch_size"]
    for batch_start in range(0, len(rows), batch_size):
        batch_index = batch_start // batch_size
        group = jobs[batch_start:batch_start + batch_size]
        seed = config["base_seed"] + {"smoke": 0, "development": 10000, "evaluation": 20000}[stage] + batch_index
        precall = folder / f"{stage}_batch_{batch_index:03d}_inputs.json"
        expected = {"judgment_ids": [j[0] for j in group], "batch_index": batch_index, "batch_seed": seed,
                    "config": config, "run_protocol_sha256": sha(folder.parent / "run_protocol.json"),
                    "inputs": [rendered[f"{case['case_id']}:{condition}"] for _, case, condition in group]}
        equal(read_json(precall), expected, "Pre-call batch differs from frozen inputs, seed, or job order")
        timing_path = folder / f"{stage}_batch_{batch_index:03d}_timing.json"
        if not timing_path.exists():
            raise ValueError("Completed cache batch lacks its timing transaction record")
        timing = read_json(timing_path)
        equal(timing["judgment_ids"], expected["judgment_ids"], "Timing record belongs to another batch")
        equal(timing["scope"], "batch_generation_only_not_serving_latency", "Incorrect timing scope")
        if not isinstance(timing["wall_seconds"], (int, float)) or not math.isfinite(timing["wall_seconds"]) or timing["wall_seconds"] < 0:
            raise ValueError("Invalid generation batch timing")
        for index in range(batch_start, min(len(rows), batch_start + batch_size)):
            row, (jid, case, condition) = rows[index], jobs[index]
            entry = rendered[f"{case['case_id']}:{condition}"]
            expected_row = {"judgment_id": jid, "case_id": case["case_id"], "pair_id": case["pair_id"],
                "base_scenario_id": case["base_scenario_id"], "template_family": case["template_family"],
                "split": case["split"], "run_stage": stage, "condition_id": condition, "model_key": model,
                "model_id": protocol["models"][model]["model_id"], "model_revision": protocol["models"][model]["revision"],
                "config": config, "batch_index": batch_index, "batch_seed": seed, "attempt_index": 0,
                "case_sha256": object_sha(case), "messages_sha256": entry["messages_sha256"],
                "precall_manifest_sha256": sha(precall), "render_metadata": entry["render_metadata"],
                "input_token_ids": entry["input_token_ids"], "input_tokens": len(entry["input_token_ids"]),
                "prediction_source": "local_model", "human_label": False, "api_spend_usd": 0,
                "semantic_certification": False}
            for field, value in expected_row.items():
                equal(row.get(field), value, f"Judgment mismatch {jid}: {field}")
            for field in ("input_tokens", "output_tokens", "attempt_index", "batch_index", "batch_seed"):
                if type(row.get(field)) is not int or row[field] < 0:
                    raise ValueError(f"Invalid integer count/ID: {field}")
            for field in ("format_valid", "operational_accept", "normalized_format_valid", "human_label", "semantic_certification", "hit_token_limit"):
                if type(row.get(field)) is not bool:
                    raise ValueError(f"Expected JSON Boolean: {field}")
            generated = row.get("output_token_ids")
            if not isinstance(generated, list) or any(type(t) is not int or t < 0 for t in generated + row["input_token_ids"]):
                raise ValueError("Invalid saved token IDs")
            equal(row["output_tokens"], len(generated), "Cost differs from generated token IDs")
            if row["output_tokens"] > config["max_tokens"]:
                raise ValueError("Output token count exceeds frozen budget")
            equal(row["hit_token_limit"], len(generated) >= config["max_tokens"], "Incorrect truncation flag")
            equal(tokenizer.decode(generated), row["raw_response"], "Response does not decode from saved output token IDs")
            for field, value in run.inspect_response(row["raw_response"]).items():
                equal(row.get(field), value, f"Primary/secondary response replay mismatch: {field}")
            for field, value in parse_primary_response(row["raw_response"]).items():
                equal(row[field], value, f"Independent strict parser disagrees: {field}")
            if any(row.get(field) is not None for field in ("human_ratings", "usefulness_gain", "expectation_risk", "gold")):
                raise ValueError("Unsupported human/outcome/gold field in raw model prediction")
    completion_path = folder / f"{stage}_completion.json"
    complete = len(rows) == len(jobs)
    cache = folder / f"{stage}_judgments.jsonl"
    if complete:
        expected_completion = {"model_key": model, "stage": stage, "expected": len(jobs), "completed": len(rows),
            "case_count": len(cases), "format_valid": sum(r["format_valid"] for r in rows),
            "malformed": sum(not r["format_valid"] for r in rows), "generation_calls": len(rows),
            "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
            "cache_sha256": sha(cache), "run_protocol_sha256": sha(folder.parent / "run_protocol.json"),
            "api_spend_usd": 0, "human_labels": 0, "semantic_certification": False}
        equal(read_json(completion_path), expected_completion, "Completion marker differs from actual calls/cost/cache")
    elif completion_path.exists():
        raise ValueError("Completion marker exists for an incomplete cache")
    return {"stage": stage, "expected": len(jobs), "observed": len(rows), "complete": complete,
            "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
            "cache_sha256": sha(cache) if cache.exists() else None}


def enrich_predictions(rows, rendered):
    result = []
    for row in rows:
        entry = rendered[f"{row['case_id']}:{row['condition_id']}"]
        result.append(dict(row, visible_messages_sha256=entry["messages_sha256"],
                           input_token_ids_sha256=object_sha(row["input_token_ids"])))
    return result


def analytic_controls(cases, labels):
    rows = []
    for name, verdict in (("analytic_always_accept", "yes"), ("analytic_always_reject", "no")):
        for case in cases:
            for condition in CONDITIONS:
                rows.append({**{k: case[k] for k in ("case_id", "base_scenario_id", "template_family", "split")},
                    "model_key": name, "condition_id": condition, "raw_response": json.dumps({"verdict": verdict}),
                    "input_tokens": 0, "output_tokens": 0, "hit_token_limit": False})
    return join_predictions(rows, cases, labels)


def all_summaries(rows, models):
    return [summarize(rows, model=model, split=split, condition=condition, output_class=kind)
            for model in models for split in SPLITS for condition in CONDITIONS
            for kind in ("all", "acceptable", "factual_invalid", "prominence_invalid")]


def build_contrasts(rows, models):
    specifications = (("replacement_factual", "whole_factual", "assembled_factual_error_admission"),
                      ("whole_factual", "whole_factual_prominence", "pure_prominence_error_admission"),
                      ("whole_factual", "whole_factual_prominence", "acceptable_false_reject_operational"))
    return [dict(paired_cluster_contrast([r for r in rows if r["model_key"] == model and r["split"] == "evaluation"],
                                        before, after, metric, cluster=cluster), model=model, split="evaluation")
            for model in models for before, after, metric in specifications for cluster in ("family_id", "base_scenario_id")]


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def note_text(report):
    lines = ["# Controlled verification: independent computational audit", "", f"Status: **{report['status']}**. "
             "Gold is deterministic fictional construction, not human annotation. Human ratings: **0**. API spend: **$0**.", "",
             "All counts include malformed responses. A malformed response rejects operationally but is always a classification error. "
             "Replacement acceptance is evaluated as a complete-candidate gate; nominal fragment accuracy is undefined because fragment gold was not independently certified.", "",
             "| Model | Split | Check | Calls | Factual errors admitted | Ordering errors admitted | Acceptable rejected | Factual-valid rejected | Malformed | Output tokens |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summaries_primary"]:
        if row["construction_class"] != "all":
            continue
        def fraction(metric):
            return f"{row[metric + '_numerator']}/{row[metric + '_denominator']}"
        lines.append(f"| {row['model']} | {row['split']} | {row['condition']} | {row['cases']} | "
            f"{fraction('assembled_factual_error_admission')} | {fraction('pure_prominence_error_admission')} | "
            f"{fraction('acceptable_false_reject_operational')} | {fraction('factual_valid_false_reject_operational')} | "
            f"{row['malformed_count']} | {row['output_tokens']} |")
    lines += ["", "The factual-valid rejection column is a declared-task diagnostic for the whole-factual condition only. "
              "Its rejection of a factually true ordering violation is an error for its factual-only target, even when that rejection helps a joint gate. "
              "For the joint condition, that same rejection implements its explicit policy criterion. Possible criterion bleed describes behavior, not known internal reasoning.", ""]
    if report["paired_contrasts"]:
        lines += ["Paired evaluation contrasts (after minus before; family bootstrap):", ""]
        for row in report["paired_contrasts"]:
            if row["cluster"] != "family_id":
                continue
            ci = row["ci95_descriptive"]
            lines.append(f"- {row['model']}, {row['after']} − {row['before']}, {row['metric']}: "
                         f"{row['estimate']:.3f}; descriptive 95% interval {ci}; {row['clusters']} families.")
    else:
        lines += ["Incomplete diagnostics: final paired contrasts are suppressed."]
    lines += ["", "These six lexical contexts share a finite sentence grammar. Four held-out families provide very limited coverage; "
              "bootstrap intervals, including degenerate intervals, do not establish equivalence, rare-risk safety, or generalization to arbitrary syntax or real films. "
              "The 16-base sensitivity is conditional on these fixed templates. Class proportions are deliberately balanced, not natural error prevalence.", "",
              "All conditions receive the same evidence, including a stipulated opening-sentence rule. Acceptable and violating minor-event edits also differ in retained factual coverage; this is not a pure permutation or an isolated causal reordering experiment. Enforcing the source rule is not a measure of viewers' expectations, tone fidelity, or personalization usefulness. "
              "No learned personalization policy, audience study, diffusion comparison, or serving approval is supplied by this experiment.", "",
              "The runtime was frozen at greedy decoding, 128 returned output tokens, one call per case/condition, and no retries. "
              "Profiles remain dataset metadata and do not enter prompts. The additive runtime note records these clarifications without changing the original protocol hash.", "",
              "Exact-view information bounds concern a uniform finite collection and complete-candidate targets. "
              "They diagnose omitted context, not nominal fragment-task mistakes or a new population impossibility theorem."]
    return "\n".join(lines) + "\n"


def analyze(root=ROOT, models=None, *, allow_incomplete=False):
    root = Path(root)
    out = root / "results/controlled_verification"
    cases, labels, dataset_manifest = validate_dataset(root)
    protocol, stages = validate_protocol(root, cases)
    if models is None:
        models = [key for key in protocol["models"] if any((out / key).glob("*_judgments.jsonl"))]
    if not models or len(set(models)) != len(models) or any(m not in protocol["models"] for m in models):
        raise ValueError("Select one or more unique actually run frozen models")
    # Preserve old experimental records; importing their stored hash map is a
    # provenance comparison, never a rerun or an empirical sample in this study.
    preserved = {}
    previous = root / "results/generation_repair_v2/engineering_audit.json"
    if previous.exists():
        old = read_json(previous)
        preserved.update(old.get("verified_file_sha256", {}))
        if not preserved:
            raise ValueError("Earlier experiment audit lacks its preserved-file bindings")
        assert_unchanged(preserved)
    frozen_paths = [root / "notes/no_human_experiment_protocol.md", out / "run_protocol.json"]
    frozen_paths += [root / p for p in protocol["code_files"]]
    frozen_paths += list((root / "data/controlled_editing").glob("*"))
    for model in models:
        frozen_paths += list((out / model).glob("*.json")) + list((out / model).glob("*.jsonl"))
    hashes = snapshot([p for p in frozen_paths if p.is_file()])
    all_rows, model_audits, normalization_rows = [], {}, []
    for model in models:
        tokenizer, rendered, manifest = validate_model(root, model, protocol, cases)
        stage_audits = []
        for stage, selected in stages.items():
            folder = out / model
            raw = strict_jsonl(folder / f"{stage}_judgments.jsonl")
            stage_audits.append(validate_stage(raw, selected, stage, model, protocol, rendered, tokenizer, folder,
                                              allow_incomplete=allow_incomplete))
            enriched = enrich_predictions(raw, rendered)
            all_rows.extend(join_predictions(enriched, cases, labels))
            secondary = []
            for row in enriched:
                normalized = dict(row, raw_response=row["normalized_response"])
                normalized.update(parse_primary_response(normalized["raw_response"]))
                secondary.append(normalized)
            normalization_rows.extend(join_predictions(secondary, cases, labels))
        model_audits[model] = {"stages": stage_audits, "generation_calls": sum(x["observed"] for x in stage_audits),
            "complete": all(x["complete"] for x in stage_audits), "manifest": manifest,
            "rendered_inputs_replayed": len(rendered)}
    complete = all(m["complete"] for m in model_audits.values())
    if not complete and not allow_incomplete:
        raise ValueError("Incomplete selected-model experiment")
    final = complete and not allow_incomplete
    summaries = all_summaries(all_rows, models)
    bounds = []
    if final:
        for model in models:
            for split in SPLITS:
                for condition in CONDITIONS:
                    selected = [r for r in all_rows if r["model_key"] == model and r["split"] == split and r["condition_id"] == condition]
                    bounds += [dict(exact_view_bound(selected, target), model=model, split=split, condition=condition)
                               for target in ("factual_valid", "joint_valid")]
    report = {"experiment": protocol["experiment"], "status": "complete" if final else "incomplete_diagnostic",
        "selected_models": models, "models": model_audits, "generation_calls": len(all_rows),
        "input_tokens": sum(r["input_tokens"] for r in all_rows), "output_tokens": sum(r["output_tokens"] for r in all_rows),
        "human_ratings": 0, "api_spend_usd": 0, "gold_scope": protocol["gold_scope"],
        "dataset_id": dataset_manifest["dataset_id"], "frozen_protocol_sha256": sha(out / "run_protocol.json"),
        "analysis_script_sha256": sha(Path(__file__)), "current_file_sha256": hashes,
        "preserved_prior_file_count": len(preserved), "summaries_primary": summaries,
        "summaries_normalized_secondary": all_summaries(normalization_rows, models),
        "analytic_controls": all_summaries(analytic_controls(cases, labels), ["analytic_always_accept", "analytic_always_reject"]),
        "paired_contrasts": build_contrasts(all_rows, models) if final else [], "exact_view_bounds": bounds,
        "scope": "constructed lexical-context benchmark sharing one sentence grammar; no human study or natural-error prevalence",
        "runtime_clarifications": {"max_tokens": 128, "profile_text_in_prompt": False,
            "addendum": "notes/no_human_experiment_protocol_runtime_addendum.md"}}
    assert_unchanged(hashes)
    assert_unchanged(preserved)
    suffix = "" if final else "_incomplete"
    # Prefix selected models to prevent a later one-model report silently
    # replacing a completed multi-model result or vice versa.
    prefix = "_".join(models) + "_audit" + suffix
    path = out / (prefix + ".json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(out / (prefix + "_summary.csv"), summaries)
    write_csv(out / (prefix + "_paired.csv"), report["paired_contrasts"])
    detail_keys = ("model_key", "case_id", "base_scenario_id", "family_id", "split", "condition_id", "construction_type",
                   "factual_valid", "prominence_valid", "format_valid", "verdict", "operational_accept",
                   "joint_classification_error", "declared_target_gold", "declared_target_error", "input_tokens", "output_tokens")
    write_csv(out / (prefix + "_cases.csv"), [{k: r[k] for k in detail_keys} for r in all_rows])
    (out / (prefix + "_note.md")).write_text(note_text(report))
    return report, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("qwen3_8b", "qwen3_17b"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report, path = analyze(models=args.models, allow_incomplete=args.allow_incomplete)
    print(json.dumps({"status": report["status"], "generation_calls": report["generation_calls"],
                      "output_tokens": report["output_tokens"], "path": str(path)}))


if __name__ == "__main__":
    main()
