# Checklist-control replication artifact

This separate extension compares Matched, Matched-checklist, and Delta-prompt on Qwen3-8B and Llama-3.2-3B. It does not replace or pool with the original study. This repository is prepared for sharing but has not been published; the local freeze is not a public preregistration. See `provenance/status.json` for whether completed response projections have been added.

From the repository root, replay all extension statistics without network access, models, or third-party Python packages (Python 3.10+):

```sh
python3 checklist_replication/scripts/reproduce.py --split all
python3 checklist_replication/scripts/validate.py
```

Replay uses the exact frozen `reference/analyze.py`, with explicit paths to avoid its original default layout. Output goes to ignored `checklist_replication/.local/replay/`. Every numerical field, disagreement, bootstrap endpoint and randomization p-value must equal the preserved original analysis. The original repository's ten-table reproducer and validator remain separate and unchanged. A preparation-only artifact fails closed for replay; validate its fixtures using `python3 checklist_replication/scripts/validate.py --prepared-only`.

## Design and interpretation

There are ten development source groups (20 cases, 120 calls) and fifty test source groups (100 cases, 600 calls). Each group contributes exactly one benchmark-valid and one benchmark-invalid changed summary with the identical source and seed summary. Source identity uses Unicode NFKC and collapsed whitespace while preserving case and punctuation. The 140 prior source groups are excluded. No semantic near-duplicate or training-contamination guarantee follows from exact source separation.

Selection uses original labels, structural eligibility, domain, immutable identities and deterministic hashes. It selects one development source per domain stratum and allocates test sources proportionally by Hamilton allocation from the remaining eligible groups. `reference/prepare.py` is an exact copy of the deterministic selection code; `data/prior_source_exclusions.json` permits reconstructing selection without distributing prior source text. All labels and row order are preserved.

All prompts, settings, ordering, data, and 720 rendered requests were frozen before any inference. Development checks execution and formatting only; no tuning or method selection is permitted. The earlier wording in the byte-identical `data/selection_protocol.json` is superseded by this stricter execution policy in `provenance/freeze.public.json`. Test execution followed both development blocks and their analysis, with no accuracy-based gate. No retries, prompt cache reuse, truncation, or candidate regeneration are used.

The three conditions receive identical source, original summary, edited summary and token-difference payloads. Matched-checklist appends the recorded factual checklist to Matched. Delta-prompt uses its original boundary-oriented instruction. The primary contrast is Delta-prompt minus Matched-checklist strict accuracy, separately by model; the other two contrasts are secondary. The checklist contrast does not isolate prompt length or establish an internal reasoning mechanism. Model configurations also differ in parameter count, quantization and native tokenizer/template, so this is not an isolated model-family causal comparison or a new algorithm claim.

Strict accuracy counts malformed outputs as incorrect. Invalid acceptance means a recorded `yes` verdict on an invalid item; valid rejection means anything other than `yes` on a valid item. Original parse errors and token/timing values are retained. `decode_matches_raw` is a diagnostic: mismatches are reported without retrying, changing the output, or rescoring it. Raw streamed JSON is authoritative.

Uncertainty uses 10,000 unstratified source-cluster bootstrap draws, seed 20261009; both paired candidates and every model/condition outcome stay together. Percentile endpoints use floor indices `int(q*(B-1))`. Accuracy comparisons additionally use 100,000 whole-source sign flips with a common sign vector, two-sided plus-one Monte Carlo p-values. Holm correction applies only across the two primary model-specific accuracy tests; other p-values are exploratory and unadjusted. The analysis is conditional on this selected sample; no model pooling occurs. Equal pair sizes make micro accuracy equal the mean source accuracy.

## Distributed files and provenance

`data/dev.jsonl` and `data/test.jsonl` are text-free projections, not the original full benchmark files. They retain IDs, original labels, group identities, domains, pairing, and original ordering. `provenance/requests/` retains every request's settings, counts, call ID, and SHA-256 commitments while omitting source-bearing messages, rendered strings and token sequences. `prompts/spec.json` contains the exact generic instruction strings. No upstream research-kit implementation is copied.

After execution completes, `results/{dev,test}/{model}/responses.jsonl` contains all 720 result projections. Only strict one-key `yes`/`no` JSON may remain in `raw`; malformed free text is replaced with an empty string and its SHA-256. Generated token sequences and decoded text are omitted with hashes/counts. Verdicts, parse errors, costs, times, completion states, and decode diagnostics remain unchanged. `results/expected_analysis/` holds the exact original numerical reports without private-file provenance. `provenance/result_diagnostics.json` lists every malformed, redacted and decode-mismatch call ID. No human-review tasks, answer keys, annotations or completed human analysis are claimed or distributed.

`provenance/source_mapping.json` distinguishes byte-identical copies from transformations and records both original and public file hashes. The original freeze hash commits to private absolute paths as well as scientific settings; `freeze.public.json` is explicitly a projection with different bytes. `provenance/checksums.json` checks this extension only. The author's original study and protected earlier scientific artifacts are verified unchanged by the private builder.

The frozen dataset revision `ce0c479aaf59259abb6b67e42248b2f49004b7d5` was independently verified as an actual upstream Hugging Face revision. The Arrow cache's `dataset_info` records the earlier raw download revision `dd891c290baa301b61040b0be75cadc482f5108e`. Both upstream revisions supply identical `summedits.json` bytes with SHA-256 `21e34593330020d02b7d92f4651fdcc550f7d79a2f899392f106fd816c215a89`. The independent download below pins the frozen `ce0c479…` revision and that raw-file hash. `provenance/dataset_revision_verification.json` records the upstream verification added after the execution freeze; it changes no scientific data or frozen setting. Hydration also reconstructs the exact full selected-file hashes.

## Independent hydration and optional fresh inference

Read [THIRD_PARTY.md](THIRD_PARTY.md) and the repository [LICENSE](../LICENSE) before independently obtaining dependencies. Downloaded benchmark text, weights and tokenizer assets remain in ignored `.local/`. Nothing below is needed for numerical replay.

```sh
python3 -m pip install -r checklist_replication/requirements-download.txt
python3 checklist_replication/scripts/hydrate.py --download
python3 checklist_replication/scripts/hydrate.py --models qwen3_8b llama32_3b
```

For an already obtained raw JSON, use `--source-json path/to/summedits.json` instead of `--download`. Hydration reruns deterministic selection, preserves all original labels, verifies both full selected-file hashes, and verifies all 720 exact message and rendered-string hashes. `--source-arrow` also accepts the original Arrow stream, with optional `pyarrow` installed and its recorded raw-cache hash verified.

Fresh inference requires Apple Silicon and the pinned environment in `requirements-inference.txt` (observed Python 3.13.7). Qwen uses the recorded no-thinking native template. Llama uses its own native template with the fixed date `09 Oct 2026`, exactly one BOS token and no Qwen think prefill. Both encode rendered prompts with `add_special_tokens=False`; full token IDs and original call IDs must match before inference. Greedy decoding uses temperature zero and at most 64 output tokens. Generation seconds exclude model loading. Finite-precision implementations and hardware can affect fresh outputs or timing.

```sh
python3 -m pip install -r checklist_replication/requirements-inference.txt
python3 checklist_replication/scripts/replicate.py --run-inference \
  --output checklist_replication/.local/replications/new-run
```

The output directory must not exist. This independent adapter preserves frozen messages, native templates, request order, parser semantics and decoding; it refuses replacement of distributed caches and performs no automatic retry or resume. Both development blocks and their frozen analysis run before either test block, with no accuracy threshold. It does not run the historical study again. Fresh raw logs may contain third-party prose and must not be added to the repository. To analyze a fresh complete replication, explicitly pass its output directory and the hydrated data to the exact analyzer:

```sh
python3 checklist_replication/reference/analyze.py \
  --data-dir checklist_replication/.local/data \
  --results-dir checklist_replication/.local/replications/new-run --split test
```

For separate test blocks, `--development-results` must point to both completed independent development blocks and their analysis. Neither historical response caches nor a favorable development accuracy may substitute for this execution gate.
