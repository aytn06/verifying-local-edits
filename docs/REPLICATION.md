# Optional fresh replication

The completed historical test was executed once. CPU replay uses its original metadata. Any inference below is a new independent replication; it does not change the paper's recorded findings and may differ across runtime/hardware.

First hydrate the pinned inputs and independently obtain the 8B model as described in the README. On Apple Silicon, create/activate a separate environment and install the exact observed runtime versions:

```sh
python3 -m pip install -r requirements-inference.txt
python3 scripts/replicate_baselines.py --split dev --run-inference --output replications/dev-new
# Optional separate test replication, only after freezing your own protocol:
python3 scripts/replicate_baselines.py --split test --run-inference --output replications/test-new
```

The script rejects an existing output directory and paths inside frozen data/results/provenance/prompts. Each request is durably recorded before generation. An interrupted or unknown call stops; the script does not silently resume or retry it. Both case order and Whole/Matched/Delta-prompt order remain fixed, as do all source inputs, original labels, exact messages, tokenizer rendering, greedy temperature-zero decoding, 64-token cap, and 40,960-token context check. The script validates pinned version/model/input hashes, rendered hashes, and exact input-token hashes before inference. Tokens exclude EOS for reported output costs, with EOS counted separately; time covers stream generation only. No new inference was run to prepare this package.

The optional baseline runner is independently implemented release glue, not a byte-identical copy of the omitted research kit. Its prompt/rendering equivalence is covered by the CPU checks. It requires the exact pinned MLX stream interface and checks terminal response/count semantics. Real inference on a fresh installation has not been tested in this release-preparation work.

The controlled collection and rule-deletion runner sources are retained in `reference_code/` for inspection. Their historical strict entrypoints expect the original project tree, original code hashes, assets, and provenance. They are not advertised as portable public entrypoints. Public controlled reproduction uses the complete authored fixtures and cached judgments through `scripts/reproduce_tables.py` and `scripts/validate_release.py`.

The two-stage `delta_claims_v1` condition is development-only. Its exact extraction/verification prompts, schemas, limits (768 and 512 tokens), parsing contract, original verdict metadata, and both-stage costs are preserved. This release provides no automatic test execution for that ablation. Its extraction prose/claims are omitted with the natural-data texts; optional baseline replication does not regenerate those claims.

## Environment

CPU replay is standard-library Python 3.10+; the observed experiment used Python 3.13.7 on arm64. Historical runtime: MLX 0.32.3, MLX-LM 0.32.0, Transformers 5.19.0, Tokenizers 0.23.2, Datasets 4.8.5, Hugging Face Hub 1.33.0. Package availability/platform compatibility must be satisfied independently. Version pins identify the executed environment and do not imply cross-hardware numerical identity.

## Maintainer packaging

After intentional final edits (for example adding the final manuscript), refresh only the public release inventory and rerun checks:

```sh
python3 scripts/validate_release.py --write-checksums
python3 -m unittest discover -s tests -v
```

This changes the public file inventory, not the source-mapping hashes. Original/projected source files still must match `provenance/source_mapping.json`. Exclude `.local/`, `replications/`, `.git/`, `__pycache__/`, and compiled Python files from release archives. `generated/` contains reproducible outputs and is ignored by Git.
