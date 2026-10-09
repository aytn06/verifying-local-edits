# Sources, attribution, and redistribution boundary

## SummEdits

The original benchmark is *SummEdits: Measuring LLM Ability at Factual Reasoning Through The Lens of Summarization*, by Philippe Laban, Wojciech Kryscinski, Divyansh Agarwal, Alexander R. Fabbri, Caiming Xiong, Shafiq Joty, and Chien-Sheng Wu, EMNLP 2023, pages 9662–9676. See the [paper](https://aclanthology.org/2023.emnlp-main.600/) and [dataset](https://huggingface.co/datasets/Salesforce/summedits). The upstream dataset card declares CC-BY-4.0. The frozen experiment used revision `dd891c290baa301b61040b0be75cadc482f5108e` and its `summedits.json` artifact.

The collection combines third-party source corpora, benchmark-generated edited summaries, and synthetic sales domains. The dataset-level declaration is not treated here as an independently verified blanket permission to redistribute all underlying source materials. This package therefore distributes **no source documents, seed summaries, edited summaries, extracted claims, natural-data response prose, full natural-data prompts, or natural-data token-ID arrays**. Selected IDs, original labels, domain/edit metadata, source hashes, verdicts, usage, and timings are retained for reproducibility and attributed to the benchmark. The original labels are unchanged.

Readers obtain the pinned dataset independently, review its terms and underlying source provenance, and run the opt-in hydration script locally. Hydrated texts are ignored by Git. An exact normalized-row/file hash check rejects altered or mismatched data.

## Qwen model/tokenizer artifacts

The exact pinned repositories expose Apache-2.0 licenses:

- [Qwen3-8B-MLX-4bit, revision 383413e909f3bc5303ce195ebbdf0339c5a1a2a3](https://huggingface.co/Qwen/Qwen3-8B-MLX-4bit/blob/383413e909f3bc5303ce195ebbdf0339c5a1a2a3/LICENSE).
- [Qwen3-1.7B-MLX-4bit, revision 21457c6f51ed54a7c16e988c0844db973815c137](https://huggingface.co/Qwen/Qwen3-1.7B-MLX-4bit/blob/21457c6f51ed54a7c16e988c0844db973815c137/LICENSE).

No weights, tokenizer JSON/configuration, chat-template asset, or model configuration asset is bundled. The repository stores hashes and the experimentally observed rendering protocol only. The opt-in downloader retrieves pinned artifacts and their upstream LICENSE/README; all four measured artifact hashes are verified. Upstream terms apply to downloaded assets.

## Research-kit implementation and dependencies

The original research-kit `paper_study.py` had no established redistribution license in the supplied material. Its implementation is excluded; the original hash is recorded in `provenance/source_mapping.json`. Frozen experimental instructions/settings remain documented, and a separate public adapter reproduces their behavior. There is no blanket relicensing of that kit.

Python, Hugging Face Hub, Datasets, Transformers, Tokenizers, MLX, and MLX-LM are dependencies with their own upstream terms; their implementations are not vendored. Runtime versions are recorded in `provenance/dependencies.json`. Standard-library CPU replay requires none of the optional downloaded dependencies.

The manuscript may contain brief attributed scientific illustrative quotations. It is distinct from the underlying document collection and does not expand the data/code licenses.
