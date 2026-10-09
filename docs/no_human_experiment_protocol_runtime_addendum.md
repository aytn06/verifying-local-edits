# Execution clarification for controlled verification v1

8 October 2026. This additive record preserves the original design-note hash and the frozen runner; it does not change any cases, labels, splits, endpoints, contrasts, or prompts. The authoritative pre-call runtime protocol is `results/controlled_verification/run_protocol.json` (SHA-256 `ff1b5744f1b7024d2e0b0347338cb927e575ac2df016c386434b04f3448abf87`).

The executed completion limit is **128 tokens**, not the tentative 256-token plan described in the design note. It was frozen before the smoke calls. Actual prompts include source evidence and the permitted candidate view; they **do not include profile text**. Profiles are dataset metadata only. Exact-view collisions therefore compare evidence and replacement strings, without a profile component. All three conditions receive the same source evidence, including the ordering rule, while only the joint condition is instructed to enforce that rule.

The six families are six lexical narrative contexts sharing the same sentence-construction grammar. Held-out contexts do not test arbitrary syntactic generalization. Primary descriptive uncertainty remains clustered by the four evaluation families; the effective family count is four.

The smoke batch showed that the factual-only verifier can reject factually true ordering violations despite its instruction to exclude ordering. In addition to the unchanged predeclared gate-use contrasts, report this as a **declared-target factual false rejection**. It is a diagnostic of possible criterion bleed, not proof of the model's internal reasoning. These additional target-specific summaries do not replace, select, or redefine a primary endpoint. The frozen prompts remain unchanged for the full run.
