#!/usr/bin/env python3
"""Frozen analysis for the separate checklist replication (CPU, no inference).

Each source contributes exactly one valid and one invalid candidate. All methods
and both model outcomes stay together when resampling a source. This module never
imports inference code or rewrites source data, responses, or previous studies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen3_8b", "llama32_3b")
CONDITIONS = ("matched", "matched_checklist", "delta_prompt")
CONTRASTS = (("delta_prompt", "matched_checklist", "primary"),
             ("delta_prompt", "matched", "secondary"),
             ("matched_checklist", "matched", "secondary"))
METRICS = ("accuracy", "invalid_accept_rate", "valid_reject_rate")
SEED = 20261009
BOOTSTRAP_REPETITIONS = 10000
PERMUTATION_REPETITIONS = 100000
EXPECTED_SOURCES = {"dev": 10, "test": 50}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    require(all(line.strip() for line in lines), f"Blank row in {Path(path).name}")
    return [json.loads(line) for line in lines]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_raw(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate_json_key")
            out[key] = value
        return out
    try:
        obj = json.loads(raw.strip(), object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None
    if (isinstance(obj, dict) and set(obj) == {"verdict"}
            and isinstance(obj["verdict"], str) and obj["verdict"] in ("yes", "no")):
        return obj["verdict"]
    return None


def correct(row):
    # Recognizable semantic verdicts do not rescue a malformed pipeline.
    return int(row["parse_error"] is None and row["verdict"] == ("yes" if row["label"] else "no"))


def validate_matrix(data, responses, split, expected_sources=None):
    """Reject incomplete/duplicate matrices and inconsistent gold/source metadata."""
    require(split in EXPECTED_SOURCES, "Unknown split")
    cases, groups = {}, defaultdict(list)
    for row in data:
        cid, gid = row.get("case_id"), row.get("group_id")
        require(isinstance(cid, str) and cid and cid not in cases, "Missing/duplicate data case ID")
        require(isinstance(gid, str) and gid, "Missing source group")
        require(type(row.get("label")) is int and row["label"] in (0, 1), "Gold label must be original binary integer")
        require(row.get("split", split) == split, "Data split differs from analysis split")
        cases[cid] = row
        groups[gid].append(row)
    require(bool(groups), "No source groups")
    if expected_sources is not None:
        require(len(groups) == expected_sources, "Wrong frozen source-group count")
    for gid, pair in groups.items():
        require(len(pair) == 2 and Counter(r["label"] for r in pair) == {0: 1, 1: 1},
                "Each source must contain exactly one valid and one invalid candidate")
        for field in ("source", "original"):
            if any(field in r for r in pair):
                require(all(field in r for r in pair) and pair[0][field] == pair[1][field],
                        f"Paired source/seed mismatch: {gid}, {field}")
    indexed, call_ids = {}, set()
    required = {"call_id", "case_id", "source_group", "label", "split", "model", "condition", "verdict",
                "parse_error", "input_tokens", "output_tokens", "seconds_unrounded", "finish_reason", "raw"}
    for row in responses:
        require(required <= row.keys(), f"Missing response fields: {sorted(required - row.keys())}")
        cid = row["case_id"]
        require(cid in cases, "Unknown response case")
        require(row["model"] in MODELS and row["condition"] in CONDITIONS, "Unknown model/condition")
        require(row["split"] == split, "Mixed response splits")
        require(type(row["label"]) is int and row["label"] == cases[cid]["label"], "Response gold differs from original data")
        require(row["source_group"] == cases[cid]["group_id"], "Response source group differs from data")
        key = row["model"], cid, row["condition"]
        require(key not in indexed, "Duplicate model/case/condition response")
        require(isinstance(row["call_id"], str) and row["call_id"] and row["call_id"] not in call_ids,
                "Missing or duplicate call ID")
        require(row["verdict"] is None or row["verdict"] in ("yes", "no"), "Unknown verdict")
        error = row["parse_error"]
        require(error is None or (isinstance(error, str) and bool(error)), "parse_error must be null or nonempty code")
        require(isinstance(row["raw"], str), "Raw response must be text")
        if error is None:
            require(parse_raw(row["raw"]) == row["verdict"] and row["verdict"] is not None,
                    "Cached valid verdict differs from independent strict parsing")
            require(row["finish_reason"] in ("stop", "length"), "Missing recognized terminal state")
        for field in ("input_tokens", "output_tokens"):
            require(row[field] is None or (type(row[field]) is int and row[field] >= 0), "Invalid token accounting")
        value = row["seconds_unrounded"]
        require(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0), "Invalid time accounting")
        if error is None:
            require(all(row[k] is not None for k in ("input_tokens", "output_tokens", "seconds_unrounded")),
                    "Successful response missing usage")
        indexed[key] = row
        call_ids.add(row["call_id"])
    expected = {(model, cid, condition) for model in MODELS for cid in cases for condition in CONDITIONS}
    require(set(indexed) == expected, f"Incomplete response matrix: {len(expected-set(indexed))} missing, {len(set(indexed)-expected)} extra")
    # Sorting hashes, rather than input arrival order, fixes resampling identities.
    return cases, {gid: sorted(pair, key=lambda r: r["label"]) for gid, pair in sorted(groups.items())}, indexed


def group_vectors(groups, indexed):
    """Each vector element is one complete source, never one candidate row."""
    vectors = {}
    for model in MODELS:
        for condition in CONDITIONS:
            values = {metric: [] for metric in METRICS}
            for pair in groups.values():
                invalid, valid = (indexed[model, d["case_id"], condition] for d in pair)
                values["accuracy"].append((correct(invalid) + correct(valid)) / 2)
                values["invalid_accept_rate"].append(int(invalid["verdict"] == "yes"))
                values["valid_reject_rate"].append(int(valid["verdict"] != "yes"))
            for metric in METRICS:
                vectors[model, condition, metric] = values[metric]
    return vectors


def paired_vectors(vectors):
    return {(model, first, second, metric): [a-b for a, b in zip(vectors[model, first, metric], vectors[model, second, metric])]
            for model in MODELS for first, second, _ in CONTRASTS for metric in METRICS}


def cluster_intervals(vectors, repetitions=BOOTSTRAP_REPETITIONS, seed=SEED):
    """Shared source-index draws for all model, condition and contrast endpoints."""
    require(repetitions >= 1 and bool(vectors), "Positive bootstrap repetitions and vectors required")
    n = len(next(iter(vectors.values())))
    require(n > 0 and all(len(v) == n for v in vectors.values()), "Mismatched source vectors")
    rng = random.Random(seed)
    draws = {key: [] for key in vectors}
    for _ in range(repetitions):
        selected = [rng.randrange(n) for _ in range(n)]
        for key, values in vectors.items():
            draws[key].append(sum(values[i] for i in selected) / n)
    result = {}
    for key, values in draws.items():
        values.sort()
        result[key] = [values[int(.025*(repetitions-1))], values[int(.975*(repetitions-1))]]
    return result


def source_randomization(contrasts, repetitions=PERMUTATION_REPETITIONS, seed=SEED):
    """Two-sided whole-source method-label swaps, with Monte Carlo +1 correction.

    Both candidate rows within a source receive the same sign. One sign vector
    is shared by every model/contrast on each iteration. This is an exchangeable
    paired-source test, not a claim of randomized treatment assignment.
    """
    require(repetitions >= 1, "Positive randomization repetitions required")
    n = len(next(iter(contrasts.values())))
    require(n > 0 and all(len(v) == n for v in contrasts.values()), "Mismatched randomization vectors")
    # With exactly two candidates/source, multiplying accuracy differences by two
    # yields integer correct-count differences, avoiding floating threshold error.
    counts = {k: [int(round(2*x)) for x in values] for k, values in contrasts.items()}
    require(all(all(abs(2*x-y) < 1e-12 for x, y in zip(contrasts[k], values)) for k, values in counts.items()),
            "Accuracy vectors violate two-candidate source design")
    observed = {k: abs(sum(v)) for k, v in counts.items()}
    extreme = {key: 0 for key in counts}
    rng = random.Random(seed)
    for _ in range(repetitions):
        bits = rng.getrandbits(n)
        signs = [1 if (bits >> i) & 1 else -1 for i in range(n)]
        for key, values in counts.items():
            statistic = abs(sum(s*v for s, v in zip(signs, values)))
            extreme[key] += statistic >= observed[key]
    return {k: {"p_two_sided_unadjusted": (extreme[k]+1)/(repetitions+1), "extreme_draws": extreme[k],
                "repetitions": repetitions, "seed": seed, "source_groups": n,
                "observed_correct_count_difference": sum(counts[k])} for k in counts}


def holm(pvalues):
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], str(key)))
    adjusted, running = {}, 0.
    for rank, key in enumerate(ordered):
        running = max(running, (len(ordered)-rank)*pvalues[key])
        adjusted[key] = min(1., running)
    return adjusted


def usage_summary(records):
    fields = ("input_tokens", "output_tokens", "seconds_unrounded")
    known = {field: sum(r[field] for r in records if r[field] is not None) for field in fields}
    complete = {field: all(r[field] is not None for r in records) for field in fields}
    inp = known["input_tokens"] if complete["input_tokens"] else None
    out = known["output_tokens"] if complete["output_tokens"] else None
    return {"model_calls": len(records), "usage_complete": all(complete.values()),
            "incomplete_usage_calls": sum(any(r[k] is None for k in fields) for r in records),
            "token_total_input": inp, "token_total_output": out,
            "token_total": inp+out if inp is not None and out is not None else None,
            "seconds_total_unrounded": known["seconds_unrounded"] if complete["seconds_unrounded"] else None,
            "known_input_tokens_lower_bound": known["input_tokens"], "known_output_tokens_lower_bound": known["output_tokens"],
            "known_generation_seconds_lower_bound": known["seconds_unrounded"],
            "finish_reason_counts": dict(Counter(str(r["finish_reason"]) for r in records))}


def analyze_records(data, responses, split, expected_sources=None,
                    bootstrap_repetitions=BOOTSTRAP_REPETITIONS, permutation_repetitions=PERMUTATION_REPETITIONS):
    cases, groups, indexed = validate_matrix(data, responses, split, expected_sources)
    vectors = group_vectors(groups, indexed)
    differences = paired_vectors(vectors)
    intervals = cluster_intervals({**vectors, **differences}, bootstrap_repetitions)
    randomization = source_randomization({k: v for k, v in differences.items() if k[-1] == "accuracy"}, permutation_repetitions)
    primary_p = {model: randomization[model, "delta_prompt", "matched_checklist", "accuracy"]["p_two_sided_unadjusted"] for model in MODELS}
    primary_holm = holm(primary_p)
    conditions, comparisons, disagreements = [], [], []
    n_sources = len(groups)
    for model in MODELS:
        for condition in CONDITIONS:
            records = [indexed[model, cid, condition] for cid in cases]
            condition_result = {"model": model, "condition": condition, "split": split,
                                "n": len(records), "source_groups": n_sources, "n_valid": n_sources, "n_invalid": n_sources,
                                "correct": sum(correct(r) for r in records),
                                "invalid_accept_count": sum(r["label"] == 0 and r["verdict"] == "yes" for r in records),
                                "valid_reject_count": sum(r["label"] == 1 and r["verdict"] != "yes" for r in records),
                                "malformed_count": sum(r["parse_error"] is not None for r in records),
                                "length_terminated_calls": sum(r["finish_reason"] == "length" for r in records), **usage_summary(records)}
            for metric in METRICS:
                condition_result[metric] = sum(vectors[model, condition, metric]) / n_sources
                condition_result[metric+"_source_cluster_bootstrap_95"] = intervals[model, condition, metric]
            condition_result["strict_accuracy"] = condition_result["accuracy"]
            conditions.append(condition_result)
        for first, second, scope in CONTRASTS:
            for metric in METRICS:
                key = model, first, second, metric
                comparison = {"model": model, "first": first, "second": second, "metric": metric,
                              "scope": scope if metric == "accuracy" else "secondary",
                              "source_groups": n_sources, "eligible_rows": len(cases) if metric == "accuracy" else n_sources,
                              "first_minus_second_rate": sum(differences[key])/n_sources,
                              "source_cluster_bootstrap_95": intervals[key]}
                if metric == "accuracy":
                    comparison["source_randomization"] = randomization[key]
                    comparison["multiplicity"] = "Holm over two model primary accuracy tests" if scope == "primary" else "exploratory unadjusted"
                    if scope == "primary": comparison["p_holm_two_model_primary"] = primary_holm[model]
                comparisons.append(comparison)
            for cid in sorted(cases):
                a, b = indexed[model, cid, first], indexed[model, cid, second]
                if a["verdict"] == b["verdict"] and a["parse_error"] == b["parse_error"]:
                    continue
                disagreements.append({"model": model, "case_id": cid, "source_group": cases[cid]["group_id"],
                    "label": cases[cid]["label"], "first": first, "second": second,
                    "first_verdict": a["verdict"], "second_verdict": b["verdict"],
                    "first_parse_error": a["parse_error"], "second_parse_error": b["parse_error"],
                    "first_correct": correct(a), "second_correct": correct(b), "correct_difference": correct(a)-correct(b),
                    **{prefix+"_"+field: row[field] for prefix, row in (("first", a), ("second", b))
                       for field in ("input_tokens", "output_tokens", "seconds_unrounded")}})
    return {"study": "checklist_replication_v1", "split": split, "cases": len(cases), "source_groups": n_sources,
            "source_group_order": list(groups), "model_calls": len(responses), "conditions": conditions,
            "paired_comparisons": comparisons, "disagreements": disagreements,
            "analysis_plan": {"primary": "Delta-prompt minus matched_checklist accuracy, separately in each of two prespecified models",
                "unit": "normalized source; retain its valid/invalid pair and all six model-condition outcomes per candidate",
                "point_weighting": "micro-average across candidates; equals equal-source macro-average because each source has exactly one row per label",
                "bootstrap": "unstratified paired source resampling with replacement; same source indices across all models/conditions/metrics",
                "bootstrap_repetitions": bootstrap_repetitions, "seed": SEED,
                "percentile_endpoints": "sorted_draws[int(q*(B-1))] at q=.025,.975",
                "randomization": "two-sided whole-source method-label swaps for accuracy; fixed common sign vector per draw; plus-one Monte Carlo p",
                "randomization_repetitions": permutation_repetitions,
                "multiplicity": "Holm correction across the two prespecified model primary accuracy p-values only; secondary contrasts exploratory/unadjusted; no pooled cross-model test",
                "development_status": "Development checks execution and formatting only; all prompts/settings frozen before any inference; no tuning or method selection." if split == "dev" else "Test protocol must have been frozen before these responses; no outcome-based method selection.",
                "interval_scope": "conditional descriptive source-cluster intervals for selected source sample; no domain-level generalization, multiplicity-adjusted CI, or randomized causal claim",
                "cost_scope": "all recorded calls; output tokens exclude EOS by collection contract; unknown usage stays unknown, with known lower bounds",
                "label_policy": "Original benchmark labels only; label-review flags never used to rescore"}}


def write_csv(path, records, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/checklist_replication")
    parser.add_argument("--results-dir", type=Path, default=ROOT/"results/checklist_replication")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    args = parser.parse_args()
    data_path = args.data_dir/f"{args.split}.jsonl"
    data = read_jsonl(data_path)
    responses, response_hashes = [], {}
    for model in MODELS:
        path = args.results_dir/args.split/model/"responses.jsonl"
        records = read_jsonl(path)
        require(all(r.get("model") == model for r in records), "Response file contains a different model")
        responses += records
        response_hashes[f"{args.split}/{model}/responses.jsonl"] = sha(path)
    report = analyze_records(data, responses, args.split, EXPECTED_SOURCES[args.split])
    report["provenance"] = {"data_file": f"{args.split}.jsonl", "data_sha256": sha(data_path),
                            "responses_sha256": response_hashes, "analyzer_sha256": sha(__file__)}
    output = args.results_dir/args.split/"analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    write_csv(output/"summary.csv", report["conditions"], ["model", "condition", "split", "n", "source_groups", "correct", "strict_accuracy",
        "invalid_accept_count", "invalid_accept_rate", "valid_reject_count", "valid_reject_rate", "malformed_count", "model_calls",
        "token_total_input", "token_total_output", "token_total", "seconds_total_unrounded", "usage_complete"])
    write_csv(output/"costs.csv", report["conditions"], ["model", "condition", "model_calls", "token_total_input", "token_total_output", "token_total",
        "seconds_total_unrounded", "usage_complete", "incomplete_usage_calls", "known_input_tokens_lower_bound", "known_output_tokens_lower_bound", "known_generation_seconds_lower_bound"])
    write_csv(output/"disagreements.csv", report["disagreements"], ["model", "case_id", "source_group", "label", "first", "second", "first_verdict", "second_verdict",
        "first_parse_error", "second_parse_error", "first_correct", "second_correct", "correct_difference", "first_input_tokens", "first_output_tokens",
        "first_seconds_unrounded", "second_input_tokens", "second_output_tokens", "second_seconds_unrounded"])
    (output/"paired_comparisons.json").write_text(json.dumps(report["paired_comparisons"], indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "split": args.split, "source_groups": report["source_groups"], "model_calls": report["model_calls"],
                      "conditions": len(report["conditions"]), "paired_metric_comparisons": len(report["paired_comparisons"]), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
