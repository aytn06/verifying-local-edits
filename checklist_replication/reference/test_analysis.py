"""Tests use wholly invented examples; no experimental response files are read."""
import copy
import importlib.util
from pathlib import Path
import random
import unittest

SPEC = importlib.util.spec_from_file_location("checklist_analysis", Path(__file__).with_name("analyze.py"))
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def fixture(n=4):
    data, responses = [], []
    for group in range(n):
        for label in (0, 1):
            cid = f"invented_{group}_{label}"
            row = {"case_id": cid, "group_id": f"fictional_source_{group}", "label": label,
                   "source": f"Invented source {group}.", "original": f"Invented seed {group}.", "split": "dev"}
            data.append(row)
            for model in a.MODELS:
                for condition in a.CONDITIONS:
                    verdict = "yes" if label else "no"
                    responses.append({"call_id": f"{model}:{cid}:{condition}", "case_id": cid, "source_group": row["group_id"],
                        "label": label, "split": "dev", "model": model, "condition": condition, "verdict": verdict, "parse_error": None,
                        "input_tokens": 100+group, "output_tokens": 6, "seconds_unrounded": .25, "finish_reason": "stop", "raw": '{"verdict":"'+verdict+'"}'})
    return data, responses


def change(row, verdict, error=None):
    row["verdict"] = verdict
    row["parse_error"] = error
    row["raw"] = '{"verdict":"'+verdict+'"}' if verdict else "malformed"


class AnalysisTests(unittest.TestCase):
    def test_complete_matrix_and_source_pairs(self):
        data, responses = fixture()
        _, groups, indexed = a.validate_matrix(data, responses, "dev", 4)
        self.assertEqual(len(indexed), 4*2*3*2)
        self.assertEqual([r["label"] for r in next(iter(groups.values()))], [0, 1])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            a.validate_matrix(data, responses[:-1], "dev")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            a.validate_matrix(data, responses+[responses[0]], "dev")
        with self.assertRaisesRegex(ValueError, "source-group count"):
            a.validate_matrix(data, responses, "dev", 50)

    def test_gold_and_source_disagreement_rejected(self):
        data, responses = fixture()
        bad = copy.deepcopy(responses); bad[0]["label"] = 1
        with self.assertRaisesRegex(ValueError, "gold"):
            a.validate_matrix(data, bad, "dev")
        bad = copy.deepcopy(responses); bad[0]["source_group"] = "another"
        with self.assertRaisesRegex(ValueError, "source group"):
            a.validate_matrix(data, bad, "dev")
        bad_data = copy.deepcopy(data); bad_data[1]["original"] = "Different seed"
        with self.assertRaisesRegex(ValueError, "source/seed"):
            a.validate_matrix(bad_data, responses, "dev")

    def test_each_source_needs_one_of_each_label(self):
        data, responses = fixture()
        data[1]["label"] = 0
        with self.assertRaisesRegex(ValueError, "one valid and one invalid"):
            a.validate_matrix(data, responses, "dev")

    def test_cached_parser_disagreement_rejected(self):
        data, responses = fixture()
        responses[0]["raw"] = '{"verdict":"yes"}'
        with self.assertRaisesRegex(ValueError, "strict parsing"):
            a.validate_matrix(data, responses, "dev")
        responses[0]["raw"] = '{"verdict":"no","verdict":"no"}'
        with self.assertRaisesRegex(ValueError, "strict parsing"):
            a.validate_matrix(data, responses, "dev")

    def test_malformed_is_wrong_and_usage_not_zero_filled(self):
        data, responses = fixture(2)
        target = next(r for r in responses if r["model"] == a.MODELS[0] and r["condition"] == "matched" and r["label"] == 1)
        change(target, "yes", "instrumentation_error")
        self.assertEqual(a.correct(target), 0)
        target["output_tokens"] = None; target["seconds_unrounded"] = None
        report = a.analyze_records(data, responses, "dev", bootstrap_repetitions=100, permutation_repetitions=100)
        result = next(r for r in report["conditions"] if r["model"] == a.MODELS[0] and r["condition"] == "matched")
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["malformed_count"], 1)
        self.assertEqual(result["valid_reject_count"], 0)  # Semantic yes; accuracy still receives no credit.
        self.assertIsNone(result["token_total"])
        self.assertIsNone(result["seconds_total_unrounded"])
        self.assertEqual(result["known_output_tokens_lower_bound"], 18)
        change(target, None, "not_strict_json")
        report = a.analyze_records(data, responses, "dev", bootstrap_repetitions=100, permutation_repetitions=100)
        result = next(r for r in report["conditions"] if r["model"] == a.MODELS[0] and r["condition"] == "matched")
        self.assertEqual(result["valid_reject_count"], 1)

    def test_bootstrap_retains_anticorrelated_pair(self):
        # Delta fixes every invalid candidate but regresses on the paired valid
        # candidate. Every complete source has zero accuracy difference. Rowwise
        # resampling would wrongly invent nonzero accuracy uncertainty here.
        data, responses = fixture(4)
        for row in responses:
            if row["condition"] == "matched_checklist" and row["label"] == 0:
                change(row, "yes")
            if row["condition"] == "delta_prompt" and row["label"] == 1:
                change(row, "no")
        report = a.analyze_records(data, responses, "dev", bootstrap_repetitions=200, permutation_repetitions=200)
        primary = [r for r in report["paired_comparisons"] if r["scope"] == "primary"]
        self.assertEqual(len(primary), 2)
        for row in primary:
            self.assertEqual(row["first_minus_second_rate"], 0)
            self.assertEqual(row["source_cluster_bootstrap_95"], [0, 0])
            self.assertEqual(row["source_randomization"]["p_two_sided_unadjusted"], 1)
        ia = next(r for r in report["paired_comparisons"] if r["first"] == "delta_prompt" and r["second"] == "matched_checklist" and r["metric"] == "invalid_accept_rate")
        self.assertEqual(ia["source_cluster_bootstrap_95"], [-1, -1])

    def test_shared_draws_floor_percentiles_and_reproducibility(self):
        values = [0., .5, 1.]
        intervals = a.cluster_intervals({"first": values, "same": values, "twice": [2*x for x in values]}, repetitions=101)
        rng = random.Random(a.SEED)
        expected = sorted(sum(values[rng.randrange(3)] for _ in values)/3 for _ in range(101))
        self.assertEqual(intervals["first"], [expected[int(.025*100)], expected[int(.975*100)]])
        self.assertEqual(intervals["first"], intervals["same"])
        self.assertEqual(intervals["twice"], [2*x for x in intervals["first"]])
        self.assertEqual(intervals, a.cluster_intervals({"first": values, "same": values, "twice": [2*x for x in values]}, repetitions=101))

    def test_signs_apply_to_source_and_plus_one_is_used(self):
        # Two sources each contribute two corrected candidates. Source swaps have
        # four equiprobable possibilities, two at least as extreme as observed;
        # independent row swaps would have sixteen and only two as extreme.
        result = a.source_randomization({"a": [1., 1.], "same": [1., 1.]}, repetitions=10000)
        self.assertEqual(result["a"], result["same"])
        self.assertAlmostEqual(result["a"]["p_two_sided_unadjusted"], .5, delta=.025)
        self.assertEqual(result["a"]["p_two_sided_unadjusted"], (result["a"]["extreme_draws"]+1)/10001)

    def test_holm_and_all_pairs_disagreements(self):
        self.assertEqual(a.holm({"one": .01, "two": .04}), {"one": .02, "two": .04})
        data, responses = fixture(2)
        for row in responses:
            if row["condition"] == "matched" and row["label"] == 0: change(row, "yes")
        report = a.analyze_records(data, responses, "dev", bootstrap_repetitions=100, permutation_repetitions=100)
        self.assertEqual(len(report["paired_comparisons"]), 18)
        self.assertEqual(len(report["conditions"]), 6)
        self.assertEqual(len(report["disagreements"]), 8)
        self.assertEqual({(r["first"], r["second"]) for r in report["disagreements"]},
                         {("delta_prompt", "matched"), ("matched_checklist", "matched")})
        self.assertEqual(sum("p_holm_two_model_primary" in r for r in report["paired_comparisons"]), 2)
        self.assertFalse(any("raw" in r for r in report["disagreements"]))


if __name__ == "__main__":
    unittest.main()
