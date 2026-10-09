# Verifying Local Edits Under Factual and Editorial Constraints

Research artifact for the empirical preprint by Achintya Raya Polavarapu and Ahmedijad Youssef. **Research preprint and reproducibility package.**

[Read the paper](https://aytn06.github.io/verifying-local-edits/) · [PDF](https://aytn06.github.io/verifying-local-edits/final_paper.pdf) · [Source repository](https://github.com/aytn06/verifying-local-edits) The main contribution is a controlled empirical account of verification behavior, with limited external factual-consistency evidence. A separate oracle analysis characterizes restricted rechecking under explicit coverage and retention assumptions. The empirical methods remain fixed prompt variants; no new learned verifier or unmeasured production benefit is claimed.

## Reproduce the results on CPU

From this directory, use Python 3.10 or later. These commands use only the standard library and require **no installation, network, tokenizer, weights, or model inference**:

```sh
python3 scripts/reproduce_tables.py
python3 scripts/validate_release.py
python3 -m unittest discover -s tests -v
```

The first command recomputes the ten original numerical manuscript tables, including appendix construction/family counts and descriptive paired intervals. It writes `generated/tables.json`, `generated/tables.md`, and `generated/tables.tex`. The second checks frozen-file hashes, deterministic construction, case/condition coverage, labels, source disjointness, exact owned prompt fixtures, observed counts/costs, and table replay. Regeneration never changes the recorded labels or predictions.

This is replay of observed data, not a new model experiment. It reproduces statistics and tables; it does not claim independent reproduction of omitted external source text, tokenizer decoding, model behavior, or historical wall-clock measurements. Original full audits and public projection checks have different scopes, detailed in [PROVENANCE.md](PROVENANCE.md).

## Recorded results

The primary controlled collection contains **864 judgments**: 144 constructed candidates × three conditions × two models. Development uses 48 candidates from two lexical families; evaluation uses 96 from four other families. The six initial smoke candidates are part of development and are counted once. The separate rule-deletion diagnostic contains **48 additional development judgments** and is exploratory.

On 96 controlled evaluation candidates, Qwen3-8B F/W/J admits 16/32, 6/32, and 2/32 factual errors. Its joint checker retains all 32 valid controls. Qwen3-1.7B's joint checker admits no constructed errors but rejects 26/32 valid controls. F, W, and J have different declared targets; gate outcomes must not be presented as a same-task accuracy leaderboard.

The initial external test contains 100 source-disjoint SummEdits cases: 50 valid and 50 invalid edited summaries. All three methods use Qwen3-8B and the same factual-consistency target. The historical frozen test was executed once.

| Method | Accuracy | Invalid accepted | Valid rejected | Input + output tokens | Generation seconds |
|---|---:|---:|---:|---:|---:|
| Whole | 74% | 23/50 | 3/50 | 107,762 | 1,066.59 |
| Matched | 75% | 22/50 | 3/50 | 122,275 | 1,194.95 |
| Delta-prompt | 82% | 14/50 | 4/50 | 124,975 | 1,217.82 |

Delta-prompt minus Matched is +7 percentage points in accuracy, with a descriptive paired 95% interval of [2,13]. It corrects eight invalid acceptances and adds one valid rejection. There is no multiplicity-adjusted significance or universal-superiority claim. Times are observed sequential generation costs, not randomized latency comparisons. Token totals exclude EOS; EOS counts remain in metadata.

**Development is separate from test.** The two-call `delta_claims_v1` ablation ties Delta-prompt at 31/40 correct, using 74,995 versus 46,494 tokens and 1,026.39 versus 375.30 generation seconds. It is a negative development-only ablation and was not run on test. Seven development label-ambiguity flags remain unadjudicated; all reported scores use the original labels, with no relabeling or selective exclusion.

The original collection has 1,412 model calls in distinct scopes: 864 controlled primary, 48 controlled diagnostic, 120 external baseline development, 80 two-stage development, and 300 final test. These are not 1,412 independent examples. The external data include heterogeneous sources and benchmark-generated edited summaries; synthetic sales-call and sales-email domains account for 18/100 selected test sources. The study does not measure editorial quality, audience benefit, or personalization on external data. No new human ratings were collected; external API spend was $0.

## Separate checklist and second-model replication

The completed follow-up contains **120 development calls and 600 test calls**. Its test has 100 edits from **50 previously unused sources**, with one valid and one invalid edit of the same original summary per source; 20 development edits come from ten other unused sources. All earlier source documents are excluded. Prompts, model snapshots, all 720 requests, and source-cluster analysis were frozen before inference; no prompts were tuned from development or test outcomes.

| Model | Method | Accuracy | Invalid accepted | Valid rejected |
|---|---|---:|---:|---:|
| Qwen3-8B | Matched | 81% | 18/50 | 1/50 |
| Qwen3-8B | Checklist | 80% | 18/50 | 2/50 |
| Qwen3-8B | Delta-prompt | 83% | 15/50 | 2/50 |
| Llama-3.2-3B | Matched | 54% | 7/50 | 39/50 |
| Llama-3.2-3B | Checklist | 61% | 10/50 | 29/50 |
| Llama-3.2-3B | Delta-prompt | 51% | 0/50 | 49/50 |

The primary Delta–Checklist accuracy contrast is **+3 points [0,7]** for Qwen (Holm p=0.250) and **−10 points [−17,−3]** for Llama (Holm p=0.041). These intervals resample whole sources and retain both edits. The new results do not establish a general advantage for Delta across the two checkpoints. Llama's low invalid acceptance conceals severe valid-edit rejection. Family, size, tokenizer, template, and quantization vary together. The original 82% versus 75% result remains a separate historical test.

Replay all new scores, clustered intervals, paired comparisons, disagreements, and costs on CPU without inference:

```sh
python3 checklist_replication/scripts/reproduce.py --split all
python3 checklist_replication/scripts/validate.py
```

See [the extension README](checklist_replication/README.md) for frozen metadata, exact analysis code, independently downloaded inputs, and optional fresh inference. The extension adds 16 CPU tests (nine analysis and seven extension tests); together with the eight original-release tests, the distributed package provides 24 tests. Combined with the original 1,412 calls, the collection contains **2,132 model calls**, not independent observations. No human ratings or scope-coverage estimates are claimed: a separate local blinded review form is prepared, but genuine completed annotations have not been collected. Model weights and third-party source/summary prose are excluded from this archive.

## Separate oracle analysis

The manuscript's “Scope Coverage and Re-verification Cost” section and proof appendix analyze perfect individual-claim queries with additive costs. Given nested scopes and conditional bounds on missing every false assertion, the optimum worst-case expected cost is the valid-retention target multiplied by the lower convex frontier of scope miss-bound/cost points. An optimal policy mixes at most two scopes. The shortlist corollary shows when queries outside a predicted scope are necessary.

This is a mathematical result under stated assumptions, not an additional model experiment. The recorded studies do not estimate those scope-coverage bounds, provide perfect claim oracles, or measure costs in those units. The theorem neither certifies Delta-prompt nor explains its measured improvement. The complete assumptions, proofs, and relevant prior work are in `paper/main.tex`. The ten original empirical tables, labels, prompts, and inference records are unchanged; the separate replication adds three tables and one figure.

## Repository contents

- `data/controlled_editing/`: complete authored fictional worlds, tasks, exact span assembly, labels, construction certificates, and split manifest.
- `results/controlled_verification/`, `results/controlled_rule_ablation/`: constructed-data receipts, rendered inputs, raw verdicts, token traces, timing, frozen protocols, and original audits. Path-bearing files are explicitly mapped as transformed copies.
- `data/external/`: source hashes, original selected IDs/labels/order/domains and sampling metadata. No source or summary text.
- `results/external/`: per-case verdicts and per-call token/timing metadata, original aggregate result summaries, and paired comparisons. No text-recoverable natural-data token IDs, responses, claims, or rendered prompts.
- `prompts/`: exact controlled and external prompt strings, including the two-stage development specification. `provenance/dependencies.json` records exact runtime versions, model revisions, file hashes, and decoding settings.
- `scripts/`: portable CPU replay/validation, independent prompt adapter, opt-in hydration, and optional baseline replication.
- `reference_code/`: original project controlled construction/analysis/collection code for methodological inspection and pure-function reuse. Its historical full-audit entrypoints assume the original tree and private provenance; use the public commands above for this release.
- `provenance/`: original/public hash mapping, text-free request hashes, label-review status, and packaging checks.
- `tests/`: eight substantive standard-library preservation and scoring tests.
- `checklist_replication/`: the separate completed 720-call comparison, source-paired metadata, frozen protocol, portable replay/hydration and optional fresh inference.
- `paper/`: the final editable manuscript and verified references. LaTeX/PDF packaging is separate from CPU numerical replay.

## Exact inputs and optional fresh replication

Read [THIRD_PARTY.md](THIRD_PARTY.md) before obtaining upstream material. Full SummEdits texts, model/tokenizer assets, and the unlicensed research-kit implementation are excluded. The independent adapter preserves the recorded prompt serialization, word differences, nonthinking rendering, strict verdict contract, and decoding settings. It was compared on CPU against all 420 completed baseline requests and their original rendered hashes. Eighteen owned constructed prompt fixtures are distributed; no natural source text is needed to run them.

Download dependencies only if you want to inspect external inputs or perform a **new, separately recorded replication**:

```sh
python3 -m venv .local/venv
source .local/venv/bin/activate  # macOS/Linux shells
python3 -m pip install -r requirements-download.txt
python3 scripts/hydrate_external.py --download
python3 scripts/hydrate_external.py --models qwen3_8b qwen3_17b
```

Hydration pins SummEdits revision `dd891c290baa301b61040b0be75cadc482f5108e`, restores the previously selected IDs in their original order, and checks original labels, normalized source hashes, all three prepared-file hashes, and all 420 baseline request/render hashes. It does not resample or retune the test. The original research-kit sampler is retained only in the local execution records and is not distributed here. In contrast, `checklist_replication/` includes the selection implementation for the separate source-paired follow-up. An already obtained pinned file can be used offline:

```sh
python3 scripts/hydrate_external.py --source-json PATH_TO_SUMMEDITS_JSON
```

Model revisions and four artifact hashes per model are pinned in `provenance/dependencies.json`. Downloads go to ignored `.local/`. No data or model download was needed or performed for release validation. The hydration adapter passed offline checks using the original normalized rows and, independently, the existing pinned raw upstream artifact: all 140 cases, three prepared-file hashes, and 420 request/render hashes match. No network transfer was performed; the downloader also checks the pinned raw artifact SHA-256.

Fresh baseline inference requires Apple Silicon and the pinned MLX environment. See [docs/REPLICATION.md](docs/REPLICATION.md). It is explicitly opt-in, writes a new directory, and cannot replace the completed ledger. Output differences across hardware/runtime are possible even with greedy decoding. No fresh model inference was performed while preparing this release.

## License and attribution

No blanket open-source license is asserted. [LICENSE](LICENSE) reserves author-controlled rights pending the author's release choice; upstream data, model, runtime, and research-kit rights are separate. [THIRD_PARTY.md](THIRD_PARTY.md) provides sources and exclusions. The repository is hosted at [aytn06/verifying-local-edits](https://github.com/aytn06/verifying-local-edits).

## Publish this package

[docs/PUBLISHING.md](docs/PUBLISHING.md) gives the exact arXiv file selection and GitHub publication steps. The paper and reproducibility package have separate archives. The repository exposes the extracted code and the paper; archived release copies preserve the distributable snapshot.
