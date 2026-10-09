#!/usr/bin/env python3
"""Prepare source-disjoint, paired SummEdits data without network or inference.

Selection uses labels, source/domain identity, and hashes, never model outputs or
manual error inspection. Original strings and benchmark labels are preserved.
The output directory is write-once: reruns verify identical serialized data.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import unicodedata
from pathlib import Path


DATASET_REVISION = "ce0c479aaf59259abb6b67e42248b2f49004b7d5"
DEFAULT_SEED = 20261009


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_source(text: str) -> str:
    """NFKC plus collapse Unicode whitespace; case/punctuation retained."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def source_id(text: str) -> str:
    return digest(canonical_source(text))


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def jsonl_text(rows) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n" for r in rows)


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Refusing to overwrite frozen data: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def protocol(seed: int, dev_sources: int, test_sources: int) -> dict:
    return {
        "study": "checklist_replication_v1",
        "selection_seed": seed,
        "requested_dev_sources": dev_sources,
        "requested_test_sources": test_sources,
        "examples_per_source": 2,
        "labels_per_source": [0, 1],
        "source_normalization": "Unicode NFKC; collapse all Unicode whitespace to one ASCII space; strip; preserve case and punctuation",
        "exclusion": "Exclude an entire normalized-source group if it occurs in any supplied prior split, across every domain. Prior predictions must all belong to supplied prior splits.",
        "eligibility": "Nonempty source, original, edited text and original binary benchmark label; edited and original differ after NFKC and whitespace normalization. Require label-0 and label-1 candidates with the identical stripped source text and original seed summary. Exclude candidate text with conflicting benchmark labels within that exact source/seed group, without changing any label.",
        "grouping": "Cluster by normalized source across domains. Choose one exact source/seed pairing group per cluster by seeded hash. No fallback to different seeds.",
        "domain_strata": "A source's stratum is the sorted unique domain list joined with |. Select at least one dev source per stratum, then allocate any remaining dev quota by largest remainder proportional to residual capacity. Allocate test quota by largest remainder proportional to residual source capacity after dev selection. Hash-break all ties.",
        "within_stratum_selection": "Order source IDs by SHA256(selection_seed|source|normalized_source_id); take dev sources first, then test sources, leaving unused remainder. Choose seed and each label's candidate by independently salted hashes of immutable identities.",
        "row_order": "Independently hash-shuffle each split's rows after pairing, so label ordering is not fixed.",
        "selection_information": "Only structural eligibility, original labels, domain, source/seed/candidate identity, and deterministic hashes; no model predictions, manual error categories, or candidate quality scoring.",
        "length_filter": None,
        "label_policy": "Benchmark labels retained verbatim as integers; structural label conflicts flagged separately and never corrected.",
        "uncertainty_unit": "Normalized source document, not individual paired edits.",
        "split_policy": "Freeze prompts/settings on development only before executing test. Existing study data and results are unchanged. Do not tune or select methods on this new test split.",
        "model_calls": 0,
        "network_calls": 0,
    }


def normalize_rows(raw: list[dict]) -> tuple[list[dict], dict]:
    rows, rejects = [], collections.Counter()
    seen_ids = set()
    for r in raw:
        try:
            if not all(isinstance(r[k], str) for k in ("doc", "seed_summary", "summary")):
                raise ValueError("nontext")
            src, original, edited = (r[k].strip() for k in ("doc", "seed_summary", "summary"))
            label = r["label"]
            if label not in (0, 1) or isinstance(label, bool):
                raise ValueError("label")
            if not all((src, original, edited)):
                raise ValueError("empty")
            cid = str(r["id"])
            if cid in seen_ids:
                raise ValueError(f"Duplicate raw case identity: {cid}")
            seen_ids.add(cid)
            rows.append({"case_id": cid, "group_id": source_id(src),
                         "original_source_group": digest(src), "source": src,
                         "original": original, "edited": edited,
                         "label": int(label), "domain": str(r.get("domain", "")),
                         "edit_types": r.get("edit_types") or [],
                         "changed": canonical_source(original) != canonical_source(edited)})
        except (KeyError, TypeError, ValueError) as exc:
            if str(exc).startswith("Duplicate raw case"):
                raise
            rejects[str(exc)] += 1
    return rows, dict(rejects)


def allocate(capacity: dict[str, int], n: int, seed: int, salt: str) -> dict[str, int]:
    """Hamilton allocation proportional to remaining source capacity."""
    total = sum(capacity.values())
    if n < 0 or n > total:
        raise ValueError(f"Requested {n} sources from {total} available")
    if not total:
        return {k: 0 for k in capacity}
    quota = {k: n * v // total for k, v in capacity.items()}
    order = sorted(capacity, key=lambda k: (-(n * capacity[k] % total),
                                            digest(f"{seed}|{salt}|{k}")))
    for key in order[:n - sum(quota.values())]:
        quota[key] += 1
    assert sum(quota.values()) == n
    assert all(quota[k] <= capacity[k] for k in capacity)
    return quota


def make_pairs(rows: list[dict], excluded: set[str], seed: int):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    pairs, conflicts, ineligible = {}, [], []
    for gid, group_rows in groups.items():
        if gid in excluded:
            continue
        by_seed = collections.defaultdict(list)
        for r in group_rows:
            if r["changed"]:
                by_seed[(r["source"], r["original"])].append(r)
        possible = []
        for (src, original), variants in by_seed.items():
            by_edited = collections.defaultdict(list)
            for r in variants:
                by_edited[canonical_source(r["edited"])].append(r)
            clean = []
            for edited_variants in by_edited.values():
                if len({r["label"] for r in edited_variants}) > 1:
                    conflicts.append({"source_group": gid,
                                      "case_ids": sorted(r["case_id"] for r in edited_variants),
                                      "flag": "identical_normalized_candidate_has_conflicting_labels",
                                      "action": "exclude candidate from new paired selection; preserve all original labels"})
                else:
                    clean.extend(edited_variants)
            if {r["label"] for r in clean} == {0, 1}:
                key = digest(f"{seed}|seed|{gid}|{digest(src)}|{digest(original)}")
                possible.append((key, clean))
        if not possible:
            ineligible.append(gid)
            continue
        variants = min(possible, key=lambda x: x[0])[1]
        pair = [min((r for r in variants if r["label"] == label),
                    key=lambda r: digest(f"{seed}|candidate|{label}|{r['case_id']}"))
                for label in (0, 1)]
        stratum = "|".join(sorted({r["domain"] for r in group_rows}))
        pairs[gid] = (stratum, pair)
    return pairs, conflicts, ineligible


def choose_splits(pairs: dict, dev_n: int, test_n: int, seed: int):
    strata = collections.defaultdict(list)
    for gid, (stratum, pair) in pairs.items():
        strata[stratum].append(gid)
    if dev_n < len(strata):
        raise ValueError("Development quota must represent every available domain stratum")
    capacity = {k: len(v) for k, v in strata.items()}
    residual = {k: v - 1 for k, v in capacity.items()}
    dev_quota = {k: v + 1 for k, v in allocate(residual, dev_n - len(strata), seed, "dev-quota").items()}
    test_quota = allocate({k: capacity[k] - dev_quota[k] for k in capacity}, test_n, seed, "test-quota")
    chosen = {"dev": [], "test": []}
    for stratum, gids in strata.items():
        order = sorted(gids, key=lambda gid: digest(f"{seed}|source|{gid}"))
        dev_ids = order[:dev_quota[stratum]]
        test_ids = order[dev_quota[stratum]:dev_quota[stratum] + test_quota[stratum]]
        for split, ids in (("dev", dev_ids), ("test", test_ids)):
            for gid in ids:
                for r in pairs[gid][1]:
                    chosen[split].append({**r, "split": split, "pair_id": gid,
                                          "source_stratum": stratum,
                                          "original_seed_hash": digest(r["original"])})
    for split, records in chosen.items():
        records.sort(key=lambda r: digest(f"{seed}|row-order|{split}|{r['case_id']}"))
    return chosen, {"eligible": capacity, "dev": dev_quota, "test": test_quota}


def verify_splits(chosen, prior_sources):
    seen = set(prior_sources)
    for split in ("dev", "test"):
        grouped = collections.defaultdict(list)
        for row in chosen[split]:
            assert row["group_id"] == source_id(row["source"])
            assert row["changed"]
            grouped[row["group_id"]].append(row)
        assert not (set(grouped) & seen), "Source leakage"
        seen.update(grouped)
        for pair in grouped.values():
            assert len(pair) == 2 and {r["label"] for r in pair} == {0, 1}
            assert len({r["source"] for r in pair}) == 1
            assert len({r["original"] for r in pair}) == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, required=True, help="Existing local SummEdits Arrow cache; never downloads")
    parser.add_argument("--prior-split", type=Path, action="append", required=True)
    parser.add_argument("--prior-predictions", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("data/checklist_replication"))
    parser.add_argument("--dev-sources", type=int, default=10)
    parser.add_argument("--test-sources", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    # Save the selection protocol before loading any dataset records.
    write_once(args.output / "selection_protocol.json", json_text(protocol(args.seed, args.dev_sources, args.test_sources)))
    import pyarrow.ipc as ipc
    with args.arrow.open("rb") as stream:
        raw = ipc.open_stream(stream).read_all().to_pylist()
    rows, rejects = normalize_rows(raw)
    prior = [r for path in args.prior_split for r in read_jsonl(path)]
    prior_ids = {str(r["case_id"]) for r in prior}
    prior_sources = {source_id(r["source"]) for r in prior}
    prediction_checks = []
    for path in args.prior_predictions:
        records = read_jsonl(path)
        extras = {str(r["case_id"]) for r in records} - prior_ids
        if extras:
            raise ValueError(f"Prior predictions contain {len(extras)} cases missing from prior splits")
        prediction_checks.append({"name": path.name, "sha256": file_digest(path),
                                  "records": len(records), "all_cases_in_prior_splits": True})
    pairs, conflicts, ineligible = make_pairs(rows, prior_sources, args.seed)
    chosen, strata = choose_splits(pairs, args.dev_sources, args.test_sources, args.seed)
    verify_splits(chosen, prior_sources)
    for split, records in chosen.items():
        write_once(args.output / f"{split}.jsonl", jsonl_text(records))
        metadata = [{k: v for k, v in r.items() if k not in {"source", "original", "edited", "edit_types"}}
                    for r in records]
        write_once(args.output / f"{split}.metadata.jsonl", jsonl_text(metadata))
    write_once(args.output / "label_flags.json", json_text({"structural_flags_only": True,
                "manual_review_performed": False, "flags": sorted(conflicts, key=lambda x: x["case_ids"])}))
    write_once(args.output / "prior_source_exclusions.json", json_text({
        "source_normalization": protocol(args.seed, args.dev_sources, args.test_sources)["source_normalization"],
        "excluded_source_ids": sorted(prior_sources),
        "prior_case_ids": sorted(prior_ids),
    }))
    split_stats = {}
    for split, records in chosen.items():
        lengths = [len(r["source"]) for r in records[::1]]
        split_stats[split] = {"sources": len({r["group_id"] for r in records}), "examples": len(records),
                             "class_counts": dict(collections.Counter(r["label"] for r in records)),
                             "domain_source_counts": strata[split],
                             "source_characters": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)},
                             "same_source_and_seed_per_pair": True}
    manifest = {
        "study": "checklist_replication_v1", "status": "frozen_data_prepared_without_inference",
        "dataset": "Salesforce/summedits", "dataset_revision": DATASET_REVISION,
        "raw_cache_sha256": file_digest(args.arrow), "raw_rows": len(raw),
        "valid_rows": len(rows), "malformed_rows": rejects,
        "normalized_source_count": len({r["group_id"] for r in rows}),
        "prior_sources_excluded": len(prior_sources),
        "prior_rows": len(prior), "prior_unique_case_ids": len(prior_ids),
        "prior_split_files": [{"name": p.name, "sha256": file_digest(p)} for p in args.prior_split],
        "prior_prediction_checks": prediction_checks,
        "unused_sources": len({r["group_id"] for r in rows} - prior_sources),
        "eligible_paired_sources": len(pairs), "ineligible_unused_sources": len(ineligible),
        "unused_eligible_sources_remaining": len(pairs) - args.dev_sources - args.test_sources,
        "structural_conflict_groups": len(conflicts),
        "selection_seed": args.seed, "eligible_domain_sources": strata["eligible"],
        "splits": split_stats, "source_disjoint_from_all_prior_splits": True,
        "dev_test_source_disjoint": True, "labels_unchanged": True,
        "model_calls": 0, "network_calls": 0,
        "files": {name: file_digest(args.output / name) for name in (
            "dev.jsonl", "dev.metadata.jsonl", "test.jsonl", "test.metadata.jsonl",
            "selection_protocol.json", "label_flags.json", "prior_source_exclusions.json")},
        "preparation_script_sha256": file_digest(Path(__file__)),
    }
    write_once(args.output / "manifest.json", json_text(manifest))
    print(json_text({k: manifest[k] for k in ("status", "normalized_source_count", "prior_sources_excluded",
                                             "unused_sources", "eligible_paired_sources", "splits", "structural_conflict_groups")}))


if __name__ == "__main__":
    main()
