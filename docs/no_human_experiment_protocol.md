# Controlled verification study: frozen computational scope

Protocol date: 8 October 2026. This replaces the human-benefit experiment for the present paper under the user's constraints: no human study, one available author, and no paid API calls. The earlier audience-usefulness, expectation-validity, and learned-personalization claims remain unmeasured; this experiment does not complete those claims by assigning synthetic or model judgments their names. No recruitment, human ratings, paid model calls, or model fine-tuning are part of this study.

## Research question and permitted claim

Can a frozen local language model detect controlled factual and explicit editorial-policy violations when verification sees (i) a replacement fragment, (ii) the complete candidate, or (iii) the complete candidate plus the declared prominence criterion? Can additional checks reduce false acceptance without merely rejecting acceptable descriptions?

The material is constructed natural language, not sampled movie descriptions. Event roles and polarity have exact construction labels. Prominence is an explicit, mechanical source-dossier rule about which event class may appear in the first sentence. A true minor-event statement placed first can violate that rule without making any event proposition false. This is **editorial ordering-policy compliance**, not validated audience expectations, tone measurement, misleadingness judgments, or semantic-error prevalence in ordinary generation.

Replacement-focused verification is a deliberate information ablation, not a claim that a competent production verifier should omit context. Its prompt judges the visible fragment. Using its acceptance as a gate for the complete candidate is the experimental intervention; an admitted complete-candidate error is not necessarily a mistaken fragment judgment. Factual-only verification is not asked to enforce the prominence rule; accepting a pure prominence violation is correct under its factual target. Comparisons to a joint gate describe the consequences of adding a criterion and its information, not an architecture improvement or a factuality mistake by the factual-only condition.

## Fixed construction and splits

Use six narrative/template families, four entity/context instances per family, and six candidate variants per instance: **24 base scenarios and 144 cases**. Two families (eight base scenarios, 48 cases) are development; the other four (16 base scenarios, 96 cases) are held out for evaluation. Split whole families before any model outputs. Families, scenarios, profiles, and case IDs remain fixed in the dataset manifest. The primary experiment uses meaningful phrase/clause boundaries. A bisected-boundary stress condition is outside the primary run and must not be silently pooled with it.

Each base contains two acceptable controls, two factual violations (controlled actor/role and polarity changes), and two factually valid prominence-policy violations. Natural-language content, candidate assembly, gold derivation, and exact construction certificates are deterministic. Case and label IDs have an exact one-to-one mapping. Labels are kept in a separate sidecar with `factual_valid` and `prominence_valid`; a joint acceptable case satisfies both. Ambiguous constructions must be fixed or excluded with a recorded rule before outputs, never assigned convenient labels after inspection. The dataset is author-constructed and programmatically checked, not independently human-validated.

Some paired cases intentionally have identical evidence/profile/replacement views but different full-candidate factual labels because the omitted edit location or surrounding context differs. The complete candidate and gold certificates must not leak into replacement-only prompts. IDs, split/family names, variant types, label fields, and construction certificates must never enter any model prompt. All output-independent dataset checks and hashes are frozen before evaluation.

## Paired model experiment

Run the same cases under three conditions:

| Condition ID | Visible candidate view | Declared decision target |
|---|---|---|
| `replacement_factual` | Replacement fragment and source facts/profile; omitted surrounding candidate context | Whether the visible replacement fragment is supported; full-candidate acceptance is a separate gate-use diagnostic |
| `whole_factual` | Complete assembled candidate and the same source facts/profile | Factual support only; ignore the editorial prominence rule |
| `whole_factual_prominence` | Complete assembled candidate and source facts/profile with the explicit editorial criterion | Factual support **and** compliance with the source-declared first-sentence prominence rule |

Each model produces **432 judgments** (144 cases × three conditions): 144 development and 288 evaluation. Any smoke batch is an identified prefix/subset, not an additional replicate. Root freezes the exact executed local model(s); an unrun model is not fabricated or required for completion. The primary model is the already cached four-bit Qwen3-8B. Any second local model is a separately identified system comparison, with the same case/condition lattice and no architecture claim.

Freeze prompts, case order, model and tokenizer revisions, runtime versions, seeds, temperature, decoding budget, nonthinking chat rendering, output schema, and parsing before evaluation. The execution agent plans greedy decoding, a 256-token limit, batches of four, and no retries or output repairs. A single strict JSON object with exactly `verdict: "yes"` or `verdict: "no"` is the primary format. Store raw response, input/output token IDs, input rendering metadata, tokens, batch seed, source/prompt hashes, and pre-call manifest hash. A predeclared normalization diagnostic may strip only a specified empty thought prefix or complete Markdown fence; it is secondary and may not replace the strict primary outcome after seeing results.

## Primary outcomes and denominators

For each model, condition, split, and construction class, report the full number of cases, valid verdicts, malformed responses, yes/no verdicts, returned tokens, and input tokens. No candidate is discarded for a model failure.

1. **Assembled factual error admitted (operational false acceptance):** number of valid `yes` responses on `factual_valid=false` cases, divided by all such cases. A malformed response is an operational rejection but does not count as correct semantic classification.
2. **Prominence-rule false acceptance:** number of valid `yes` responses on `factual_valid=true, prominence_valid=false` cases, divided by all such cases. For factual-only conditions this is a joint-pipeline acceptance diagnostic, not a factual error or violation of their declared target.
3. **False rejection on acceptable controls:** report valid `no` responses separately from operational rejection (`no` or malformed), each divided by all cases satisfying both gold criteria.
4. **Malformed rate and semantic classification error:** malformed outputs count as incorrect regardless of the gold label. Declared-target accuracy uses complete-candidate factual validity for `whole_factual` and joint validity for `whole_factual_prominence`. Omit nominal fragment-classification accuracy for `replacement_factual` unless a separately certified fragment gold label is supplied; full-candidate factual validity is not its fragment-task gold. Report full-candidate joint-target error for all conditions only as an operational gate-use diagnostic, explicitly distinct from the prompts' declared targets.
5. **Coverage and conditional rates:** valid-verdict coverage and parsed-only error rates are secondary, with their denominators. Zero coverage has an undefined conditional rate, never zero error.

Report balanced classification error as the equal-weight average of errors on acceptable, factual-invalid, and prominence-invalid classes. This is a constructed benchmark objective; its class weights are not population frequencies. Add deterministic always-accept and always-reject reference points, labeled as analytic controls requiring no model invocation. Rejecting everything cannot support an improved-verifier claim because it fails every acceptable control.

## Predeclared contrasts and uncertainty

Use all held-out evaluation cases for the following paired contrasts, with signs always **second minus first**:

- `whole_factual − replacement_factual`: assembled factual-error admission rate among factual-invalid cases, as a gate-use comparison (not a fragment-classification accuracy contrast).
- `whole_factual_prominence − whole_factual`: joint false-acceptance rate among factually valid prominence-rule violations.
- `whole_factual_prominence − whole_factual`: operational false-rejection rate among acceptable controls.

The desired finite-benchmark pattern is lower false acceptance without an increased acceptable-control rejection rate; no positive outcome is assumed. Report both effects even if the added rule only trades false acceptance for false rejection. Do not convert an interval overlapping zero into evidence of equivalence or noninferiority. No empirically unsupported population safety margin, human-quality threshold, or research success probability is introduced.

Primary descriptive 95% percentile intervals use **5,000 paired bootstrap resamples of the four held-out narrative families**, seed **20261008**. Preserve every base instance, variant, and condition within each selected family; compute ratios of resampled sums to resampled eligible counts, rather than averaging unequal-size cluster rates. The effective family count is four, not 96 cases or 288 judgments. This is extremely limited family coverage; intervals do not certify generalization or rare-risk bounds. A bootstrap over the 16 base scenarios is an explicitly conditional sensitivity analysis, not 16 independent narrative families. Report degenerate empirical intervals without claiming equivalence or a perfect population error rate. Development summaries are diagnostic and separate; they do not contribute to held-out effects. No p-value or multiple-comparison winner claim is primary.

## Exact-view information limit

Partition the fixed case set by the exact serialized visible messages (and verify matching rendered input-token hashes). For a chosen target, a group with `n_yes` and `n_no` gold labels contributes at least `min(n_yes,n_no)` errors to any deterministic binary classifier using only that view. Summing those counts and dividing by the total case count gives the best attainable empirical error under uniform case weighting. The same lower bound applies to expected error for a stochastic classifier whose distribution depends only on that identical view. Treat malformed/abstaining predictions as errors for this comparison.

Report this bound, the number and sizes of contradictory groups, and representative case IDs separately for each view and target. A zero bound is not proof that a model can classify perfectly. A positive bound shows an intentional information deficit in this constructed benchmark; it is an elementary finite-set consequence, not a new theorem or a population Bayes-risk estimate. It does not measure how frequently real editing requests create such collisions. Also report exact-view verdict disagreements as a runtime/batching diagnostic, not extra independent evidence.

## Independent audit and completion

The analyzer must verify dataset and gold provenance, exact split/family lattice, meaningful source spans and candidate assembly, frozen runner/helper/model/source hashes, exact expected judgment IDs, duplicates and completion markers, prompt and local-tokenizer replay, pre-call records, strict parsing, and token accounting. Never read a model verdict as gold, replace missing outputs with fabricated verdicts, call synthetic labels human ratings, or report unfinished evaluation as complete. `--allow-incomplete`, if provided, must produce separately named incomplete diagnostics and no final paired claims.

The result is complete when the selected frozen model's expected judgments and independent audit are complete, including unfavorable outcomes. It is not contingent on one verifier winning. Preserve all earlier source data and generation studies. Save new machine-readable tables and a compact interpretation note under `results/controlled_verification/`; do not overwrite prior experiments or the active paper from the analyzer. Human ratings remain **0**, audience benefit remains **unmeasured**, and no output is approved for real users.
