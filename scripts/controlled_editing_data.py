#!/usr/bin/env python3
"""Deterministic fictitious-world verification diagnostic; no human labels.

Gold is computed from the exact controlled construction, not an LLM judge or
an assertion about actual viewers. This is not a general free-text fact parser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "controlled-editing-verification-v1"
PROFILES = {
    "relationships": "I enjoy stories about family, friendship, and close personal relationships.",
    "suspense_mystery": "I enjoy suspense, mysteries, investigations, and uncovering secrets.",
    "comedy_warmth": "I enjoy humor, comic situations, and warm or uplifting stories.",
}
# All families instantiate every construction type. Family assignment is fixed
# before inference; entity variants do not become additional template families.
FAMILIES = [
    ("harbor", "searches for", "search for", "missing navigation charts", "watches", "watch", "sealed cargo crates", "at the old harbor."),
    ("archive", "sorts", "sort", "damaged history books", "catalogs", "catalog", "private case files", "during the evening shift."),
    ("orchard", "repairs", "repair", "damaged water pumps", "counts", "count", "packed apple crates", "beside the village orchard."),
    ("observatory", "studies", "study", "faint radio signals", "protects", "protect", "sealed supply lockers", "throughout the long winter."),
    ("ferry", "tracks", "track", "missing ferry tickets", "cleans", "clean", "empty engine compartments", "before the morning crossing."),
    ("theater", "restores", "restore", "damaged stage costumes", "secures", "secure", "locked storage cabinets", "after the final rehearsal."),
]
NAMES = [
    ("Mira", "Oren", "Tavi"), ("Lina", "Pavel", "Suri"), ("Nadia", "Emil", "Rafi"), ("Dara", "Tomas", "Vera"),
    ("Asha", "Bram", "Cleo"), ("Esme", "Hugo", "Jori"), ("Kira", "Luca", "Mavi"), ("Nola", "Otis", "Pia"),
    ("Rina", "Soren", "Tala"), ("Uma", "Viktor", "Willa"), ("Xena", "Yuri", "Zola"), ("Anya", "Basil", "Cora"),
    ("Dina", "Evan", "Faye"), ("Gita", "Hans", "Iris"), ("Jaya", "Klaus", "Lora"), ("Mina", "Nils", "Opal"),
    ("Petra", "Quinn", "Rosa"), ("Sana", "Theo", "Una"), ("Vina", "Wes", "Yara"), ("Zara", "Arlo", "Bina"),
    ("Celia", "Damon", "Elia"), ("Fina", "Galen", "Hana"), ("Ida", "Jonas", "Kali"), ("Lea", "Milan", "Nera"),
]


def encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def minor_fact(profile: str, actor: str, helper: str) -> str:
    return {
        "relationships": f"{actor} and {helper} share a close friendship.",
        "suspense_mystery": f"{actor} notices a mysterious sealed envelope.",
        "comedy_warmth": f"{actor} and {helper} share a warm joke.",
    }[profile]


def make_world(index: int, family_index: int, variant_index: int) -> dict:
    family, verb_a, infinitive_a, object_a, verb_b, infinitive_b, object_b, tail = FAMILIES[family_index]
    actor, other, helper = NAMES[index]
    sentence_a = f"{actor} {verb_a} {object_a} {tail}"
    sentence_b = f"{other} {verb_b} {object_b} {tail}"
    closing = f"The main storyline follows {actor}'s work."
    central_profile = list(PROFILES)[family_index % 3]
    minor_profiles = [p for p in PROFILES if p != central_profile]
    minor_sentences = [minor_fact(p, actor, helper) for p in minor_profiles]
    helped_a = f"{actor} {verb_a} {object_a} with {helper}'s help."
    helped_b = f"{other} {verb_b} {object_b} with {helper}'s help."
    negated_a = f"{actor} does not {infinitive_a} {object_a} {tail}"
    evidence_texts = [
        sentence_a,
        sentence_b,
        f"{helper} helps {actor} {infinitive_a} {object_a}. {helper} does not help {other} {infinitive_b} {object_b}.",
        minor_sentences[0], minor_sentences[1],
        f"The central storyline follows {actor}'s work ({actor} {verb_a} {object_a}); its declared theme is {central_profile.replace('_', ' ')}. "
        f"The {minor_profiles[0].replace('_', ' ')} scene and the {minor_profiles[1].replace('_', ' ')} scene are each minor subplots.",
        f"Explicit editorial prominence rule: the opening sentence must discuss {actor}'s central work activity. "
        "Either true minor-subplot scene may be mentioned only in a later sentence. "
        "This is a stipulated sentence-position rule, not evidence about audience impressions.",
    ]
    evidence = [{"evidence_id": f"E{i}", "text": s} for i, s in enumerate(evidence_texts)]
    facts = [
        {"subject": actor, "predicate": infinitive_a, "object": object_a, "polarity": True},
        {"subject": other, "predicate": infinitive_b, "object": object_b, "polarity": True},
        {"subject": helper, "predicate": "help", "object": [actor, infinitive_a, object_a], "polarity": True},
        {"subject": helper, "predicate": "help", "object": [other, infinitive_b, object_b], "polarity": False},
    ]
    # Exact surface realizations bind to declared facts. Unknown realizations are
    # errors in this construction audit, never silently scored as true or false.
    clauses = {
        sentence_a: {"factual_valid": True, "event_class": "central_work"},
        sentence_b: {"factual_valid": True, "event_class": "secondary_work"},
        closing: {"factual_valid": True, "event_class": "central_summary"},
        helped_a: {"factual_valid": True, "event_class": "central_work"},
        helped_b: {"factual_valid": False, "event_class": "secondary_work", "violated_fact_index": 3},
        negated_a: {"factual_valid": False, "event_class": "central_work", "violated_fact_index": 0},
        **{s: {"factual_valid": True, "event_class": "minor_scene"} for s in minor_sentences},
    }
    return {
        "base_scenario_id": f"b{index + 1:02d}", "template_family": f"f{family_index + 1:02d}_{family}",
        "variant_index": variant_index, "split": "development" if family_index < 2 else "evaluation",
        "fictional": True, "central_profile": central_profile, "minor_profiles": minor_profiles,
        "actors": [actor, other, helper], "canonical_text": " ".join([sentence_a, sentence_b, closing]),
        "sentences": {"a": sentence_a, "b": sentence_b, "closing": closing, "helped_a": helped_a,
                      "helped_b": helped_b, "negated_a": negated_a, "minor_1": minor_sentences[0], "minor_2": minor_sentences[1]},
        "tail": tail, "objects": [object_a, object_b], "evidence": evidence, "signed_facts": facts,
        "clause_certificates": clauses, "source_sha256": sha(encoded(evidence)),
        "provenance": "Deterministic fictitious construction; no external title, human rating, or model-generated gold.",
    }


def construction_certificate(world: dict, candidate: str) -> dict:
    """Certify only strings in the finite hand-specified surface construction."""
    clauses = [s.strip() + "." for s in candidate.split(".") if s.strip()]
    if " ".join(clauses) != candidate:
        raise ValueError("Candidate is outside the exact controlled sentence grammar")
    if any(s not in world["clause_certificates"] for s in clauses):
        raise ValueError("Unknown clause cannot receive a construction gold label")
    certificates = [world["clause_certificates"][s] for s in clauses]
    return {
        "factual_valid": all(c["factual_valid"] for c in certificates),
        "prominence_valid": certificates[0]["event_class"] == "central_work",
        "first_sentence_class": certificates[0]["event_class"],
        "clause_factual_validity": [c["factual_valid"] for c in certificates],
        "clause_event_classes": [c["event_class"] for c in certificates],
        "violated_fact_indices": [c["violated_fact_index"] for c in certificates if "violated_fact_index" in c],
        "rule": "Opening sentence discusses central work; minor scenes occur only later.",
    }


def make_case(world: dict, number: int, old: str, replacement: str, start: int, kind: str, profile: str, pair: str, span_kind: str) -> tuple[dict, dict]:
    canonical = world["canonical_text"]
    end = start + len(old)
    if canonical[start:end] != old:
        raise ValueError("Marked canonical span mismatch")
    candidate = canonical[:start] + replacement + canonical[end:]
    certificate = construction_certificate(world, candidate)
    case_id = f"{world['base_scenario_id']}_c{number:02d}"
    task = {"schema": SCHEMA, "case_id": case_id, "pair_id": f"{world['base_scenario_id']}:{pair}",
            **{k: world[k] for k in ("base_scenario_id", "template_family", "split", "evidence", "source_sha256")},
            "profile_id": profile, "profile_text": PROFILES[profile], "canonical_text": canonical,
            "local_span": {"start_char": start, "end_char": end, "text": old, "offset_unit": "unicode_codepoints"},
            "replacement_text": replacement, "candidate_text": candidate, "candidate_sha256": sha(candidate.encode())}
    gold = {"schema": SCHEMA, "case_id": case_id, "task_sha256": sha(encoded(task)),
            "factual_valid": certificate["factual_valid"], "prominence_valid": certificate["prominence_valid"],
            "construction_type": kind, "primary_span_kind": span_kind, "certificate": certificate,
            "label_source": "deterministic_controlled_construction", "human_label": False}
    return task, gold


def build_records() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    worlds, tasks, gold, stress = [], [], [], []
    for fi in range(len(FAMILIES)):
        for vi in range(4):
            world = make_world(fi * 4 + vi, fi, vi)
            worlds.append(world)
            canonical, s, tail = world["canonical_text"], world["sentences"], world["tail"]
            central, (minor_1, minor_2) = world["central_profile"], world["minor_profiles"]
            helper = world["actors"][2]
            start_a = canonical.index(tail)
            start_b = canonical.index(tail, start_a + len(tail))
            specs = [
                (tail, f"with {helper}'s help.", start_a, "acceptable_role", central, "role", "prepositional_phrase"),
                (s["closing"], s["minor_1"], canonical.index(s["closing"]), "acceptable_minor", minor_1, "prominence", "complete_sentence"),
                (tail, f"with {helper}'s help.", start_b, "wrong_actor", minor_2, "role", "prepositional_phrase"),
                (s["a"], s["negated_a"], 0, "reversed_polarity", central, "polarity", "complete_sentence"),
                (s["a"], s["minor_1"], 0, "foregrounded_minor_1", minor_1, "prominence", "complete_sentence"),
                (s["a"], s["minor_2"], 0, "foregrounded_minor_2", minor_2, "prominence", "complete_sentence"),
            ]
            for i, spec in enumerate(specs, 1):
                task, label = make_case(world, i, *spec)
                tasks.append(task)
                gold.append(label)
                if i in (1, 3):
                    obj = world["objects"][0 if i == 1 else 1]
                    obj_start = canonical.index(obj)
                    noun_start = obj_start + obj.rindex(" ") + 1
                    end = task["local_span"]["end_char"]
                    replacement = obj.split()[-1] + " " + task["replacement_text"]
                    assert canonical[:noun_start] + replacement + canonical[end:] == task["candidate_text"]
                    stress.append({"case_id": task["case_id"], "stress_arm": "artificial_noun_phrase_bisection_metadata_only",
                                   "primary_experiment": False, "noun_phrase": obj,
                                   "local_span": {"start_char": noun_start, "end_char": end, "text": canonical[noun_start:end], "offset_unit": "unicode_codepoints"},
                                   "replacement_text": replacement, "candidate_sha256": task["candidate_sha256"],
                                   "note": "Alternative starts inside a noun phrase but preserves the noun head and identical complete text; not an automatic error label."})
    return worlds, tasks, gold, stress


def validate_records(worlds: list[dict], tasks: list[dict], labels: list[dict], stress: list[dict]) -> dict:
    if len(worlds) != 24 or len(tasks) != 144 or len(labels) != 144 or len(stress) != 48:
        raise ValueError("Wrong frozen construction counts")
    world_map = {w["base_scenario_id"]: w for w in worlds}
    task_map = {t["case_id"]: t for t in tasks}
    label_map = {g["case_id"]: g for g in labels}
    if len(world_map) != 24 or len(task_map) != 144 or set(task_map) != set(label_map):
        raise ValueError("Duplicate/missing world, task, or label ID")
    families = {}
    for task in tasks:
        world = world_map[task["base_scenario_id"]]
        label = label_map[task["case_id"]]
        families.setdefault(task["template_family"], set()).add(task["split"])
        if any(task[k] != world[k] for k in ("template_family", "split", "evidence", "source_sha256", "canonical_text")):
            raise ValueError("World/source binding drift")
        if task["source_sha256"] != sha(encoded(task["evidence"])) or label["task_sha256"] != sha(encoded(task)):
            raise ValueError("Source/task hash drift")
        span = task["local_span"]
        lo, hi = span["start_char"], span["end_char"]
        canonical = task["canonical_text"]
        if not (type(lo) is int and type(hi) is int and 0 <= lo < hi <= len(canonical)) or span["text"] != canonical[lo:hi]:
            raise ValueError("Invalid span offsets")
        if (lo and not canonical[lo-1].isspace()) or (hi < len(canonical) and not canonical[hi].isspace()):
            raise ValueError("Primary span cuts a word")
        permitted = [world["tail"], world["sentences"]["a"], world["sentences"]["closing"]]
        if span["text"] not in permitted:
            raise ValueError("Primary span is not a declared grammatical phrase or complete sentence")
        assembled = canonical[:lo] + task["replacement_text"] + canonical[hi:]
        if assembled != task["candidate_text"] or sha(assembled.encode()) != task["candidate_sha256"]:
            raise ValueError("Candidate reconstruction/hash mismatch")
        if not 20 <= len(canonical.split()) <= 40 or not 20 <= len(assembled.split()) <= 40:
            raise ValueError("Canonical/candidate outside 20–40 word contract")
        cert = construction_certificate(world, assembled)
        if cert != label["certificate"] or any(cert[k] != label[k] for k in ("factual_valid", "prominence_valid")):
            raise ValueError("Gold does not follow declared construction")
        if any(k in task for k in ("gold", "certificate", "construction_type", "factual_valid", "prominence_valid")):
            raise ValueError("Gold leaked into candidate task")
    if len(families) != 6 or any(len(s) != 1 for s in families.values()):
        raise ValueError("Template family split leakage")
    for world in worlds:
        a, b = (task_map[world["base_scenario_id"] + suffix] for suffix in ("_c01", "_c03"))
        if a["evidence"] != b["evidence"] or a["replacement_text"] != b["replacement_text"]:
            raise ValueError("Role matched pair lost identical replacement-only input")
        if not label_map[a["case_id"]]["factual_valid"] or label_map[b["case_id"]]["factual_valid"]:
            raise ValueError("Role pair no longer has opposite factual labels")
    for row in stress:
        task = task_map[row["case_id"]]
        lo, hi = row["local_span"]["start_char"], row["local_span"]["end_char"]
        canonical = task["canonical_text"]
        if row["primary_experiment"] or canonical[lo:hi] != row["local_span"]["text"] or canonical[:lo] + row["replacement_text"] + canonical[hi:] != task["candidate_text"]:
            raise ValueError("Stress metadata must preserve complete candidate and stay outside primary experiment")
    return {"schema": SCHEMA, "status": "construction_checks_passed_not_semantic_human_validation", "worlds": 24,
            "candidates": 144, "development_candidates": 48, "evaluation_candidates": 96,
            "families": 6, "evaluation_families": 4, "identical_fragment_role_pairs": 24,
            "stress_metadata_rows": 48, "human_labels": 0}


def write_dataset(out: Path) -> dict:
    worlds, tasks, labels, stress = build_records()
    audit = validate_records(worlds, tasks, labels, stress)
    artifacts = {"worlds.jsonl": worlds, "tasks.jsonl": tasks, "gold.jsonl": labels, "stress_spans.jsonl": stress}
    out.mkdir(parents=True, exist_ok=True)
    encoded_artifacts = {name: b"".join(encoded(row) + b"\n" for row in records) for name, records in artifacts.items()}
    # Idempotent replay is allowed; changing an existing frozen artifact is not.
    for name, data in encoded_artifacts.items():
        if (out / name).exists() and (out / name).read_bytes() != data:
            raise ValueError(f"Refusing to replace frozen {name}; create a new version")
    manifest = {"schema": SCHEMA, "dataset_id": sha(encoded(tasks))[:20], **audit,
                "files": {name: {"sha256": sha(data), "rows": len(artifacts[name])} for name, data in encoded_artifacts.items()},
                "smoke_case_ids": [t["case_id"] for t in tasks[:6]],
                "split_rule": "First two of six fixed contextual template families are development; last four evaluation.",
                "source_status": "Entirely fictitious and constructed; no corpus or real movie sampling.",
                "primary_span_rule": "Meaningful prepositional phrase or complete sentence; stress spans excluded.",
                "prominence_endpoint": "Compliance with declared first-sentence editorial rule; not actual viewer expectations.",
                "prespecified_primary_conditions": ["replacement_factual", "whole_factual", "whole_factual_prominence"],
                "primary_calls_per_model": 432, "independent_source_review": False, "human_identity_claim": False,
                "generator_instruction": "Never supply worlds, gold, stress metadata, case IDs, split, or construction names to verifier.",
                "script_sha256": sha(Path(__file__).read_bytes())}
    for name, data in encoded_artifacts.items():
        (out / name).write_bytes(data)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "construction_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "README.md").write_text(README)
    return manifest


README = """# Controlled verification diagnostic v1

This dataset contains **24 fictional micro-worlds**, not real films or human
ratings. Six contextual templates each have four entity variants. The first
two families (8 worlds) are development; the remaining four (16 worlds) are
evaluation. Entity variants share a template and are not independent families.

Each world has six fixed candidate edits: two acceptable, one changed helper
relation, one reversed polarity, and two truthful minor-scene foregroundings.
Every canonical and candidate contains 20–40 whitespace-separated words.
Primary spans are complete sentences or meaningful prepositional phrases.
The optional 48 noun-phrase-bisecting spans are metadata only and preserve the
same full candidate; they are not included in the primary experiment.

The role pair has identical source evidence and replacement text, but different
placement changes whose work is helped. A fragment-only verifier therefore
lacks information that the complete-description verifier receives. This is an
explicit information restriction, not a novel general impossibility result.

Prominence uses a **declared sentence-position policy**: the opening sentence
discusses the central work; true minor scenes may occur later. Negative examples
foreground a true minor scene, never falsely claim that it is dominant. This
separates factual validity from a stipulated editorial constraint, but cannot
measure actual misleading expectations or usefulness. Profiles are the same
three supplied interest categories; no individual preference claims follow.

`tasks.jsonl` is the candidate cache. `gold.jsonl` and `worlds.jsonl` are
evaluator-only sidecars. Prompts may contain only the evidence and the candidate
view permitted by their condition. No labels, IDs, split, construction subtype,
profile, canonical, or stress metadata enter verifier prompts. The source
dossier, including the rule, is identical across verifier conditions.

Gold certificates are computed from a finite construction grammar with explicit
signed facts. The grammar is an integrity check for these authored candidates;
it is **not** a semantic parser for arbitrary generated language. Unknown
sentences fail construction validation rather than being silently labeled.

Reproduce: `python3 scripts/controlled_editing_data.py build`.
Validate: `python3 scripts/controlled_editing_data.py validate`.
Test: `python3 -m unittest discover -s scripts -p test_controlled_editing_data.py`.
Existing changed artifacts cannot be overwritten; use a new version directory.
Generation and frozen verifier analysis are separate scripts. A full primary
pass uses 144 candidates × 3 checks = 432 first-attempt calls per model. The six
first-development-world candidates are smoke tasks and must not be resampled.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["build", "validate"])
    p.add_argument("--out", type=Path, default=ROOT / "data/controlled_editing")
    args = p.parse_args()
    if args.command == "build":
        report = write_dataset(args.out)
    else:
        report = validate_records(*(read_rows(args.out / name) for name in ("worlds.jsonl", "tasks.jsonl", "gold.jsonl", "stress_spans.jsonl")))
        manifest = json.loads((args.out / "manifest.json").read_text())
        for name, meta in manifest["files"].items():
            if sha((args.out / name).read_bytes()) != meta["sha256"]:
                raise ValueError(f"Frozen file hash mismatch: {name}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
