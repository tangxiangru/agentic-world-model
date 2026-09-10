# Additional 2,000 combinations: combined 3,000-exp_id specification

Status: packaged experiment specification, **not verified launch readiness or authorization to spend compute**. The existing GPU execution agent must validate the added settings and checkpoint artifacts before scheduling them. Nothing in this package trains new weights.

## One location, three views

- `experiment_matrix_all_3k.jsonl`: the full 3,000-exp_id input, with original rows first.
- `experiment_matrix_extension_2k.jsonl`: only the 2,000 additions, for a runner that already scheduled the original work.
- `experiment_matrix.jsonl`: the original 1,000 rows, retained byte-for-byte.

These are overlapping views, not three workloads to concatenate. Original exp_ids retain the `poc1k-` prefix. New exp_ids use `poc3k-<checkpoint_id>-<generation_config_id>`. Deduplicate work by exp_id and full artifact/protocol identity. Relative paths in every view resolve from this bundle directory.

`execution_plan.json` specifies the stage order. The original phase filenames are preserved, so do not infer launch order by sorting all seven filenames: the locked test must remain after **all** development work, including the extension.

## Exact additional allocation

| Added coverage | Calculation | Additional combinations |
|---|---:|---:|
| Remaining GSM8K candidates under G01/G02 | 76 × 2 | 152 |
| Remaining AIME candidates under A01/A02/A03 | 40 × 3 | 120 |
| Complete A04/A05 coverage across all AIME candidates | 180 × 2 | 360 |
| G03/GX01/GX02 on every GSM8K candidate | 316 × 3 | 948 |
| AX01/AX02 on every AIME candidate | 200 × 2 | 400 |
| AX03 on the original 20 development AIME diagnostic checkpoints | 20 × 1 | 20 |
| **Additional total** | | **2,000** |

The full matrix is 316 GSM8K candidates × five policies, plus 200 AIME candidates × seven policies, plus 20 medium-cap diagnostic combinations: **1,580 + 1,400 + 20 = 3,000**. All original 1,000 combinations are included exactly once. There are 516 candidate checkpoint IDs, not yet 516 tensor-hash-certified distinct weight sets.

## Six added policy files

| Config ID | Benchmark | Temperature | top_k | top_p | Repetition penalty | Output-token cap |
|---|---|---:|---:|---:|---:|---:|
| G03 | GSM8K | 1.0 | 0 | 1.0 | 1 | 4,000 |
| GX01 | GSM8K | 0.4 | 64 | 0.95 | 1 | 4,000 |
| GX02 | GSM8K | 0.4 | 0 | 1.0 | 1 | 4,000 |
| AX01 | AIME | 0.6 | 0 | 1.0 | 1 | 16,000 |
| AX02 | AIME | 1.0 | 20 | 0.95 | 1 | 16,000 |
| AX03 | AIME, diagnostic subset | 1.0 | 0 | 1.0 | 1 | 4,096 |

Use the complete `configs/<config_id>/request_template.json`, not only temperature. All new policies retain `min_p=0`, one completion per request, the same benchmark-specific intended stop-token union, and the unchanged `protocol.json` seed formula for repeats 0–9. `top_k=0, top_p=1` denotes no top-k/top-p filtering. The executor must prevent inherited generation settings from altering these requests and record actual server-resolved settings.

The original G01/G02 and A01–A05 files are unchanged. `generation_policies_all_13.json` is the complete runner catalog; the original seven-policy catalog is preserved. New GX/AX names avoid collisions with unrelated frequency-ranked policy IDs in the historical config audit. G03 agrees with the existing audit's unfiltered GSM definition. GX01/GX02 correspond to the audit's rare G07/G06 patterns; AX02/AX03 to its rare A15/A20 patterns. AX01 is a deliberate new controlled combination, not a claimed common historical policy.

## What the extra measurements test

GSM8K now compares temperatures 0.4 and 1.0 with filtering either off or set to `top_k=64, top_p=0.95`, plus greedy. AIME compares temperatures 0.6 and 1.0 with filtering off or set to `top_k=20, top_p=0.95`, alongside the original greedy, short-cap and repetition-penalty contrasts. This separates temperature from the **filtering bundle**; it does not separately identify top-k and top-p effects.

AX03 creates matched 2,048 / 4,096 / 16,000-token AIME comparisons on the same 20 checkpoints already assigned original A04/A05 diagnostics. Those IDs are fixed before new outcomes; no new score-based subset is selected. These are mechanism tests, not recommendations for optimal settings or evidence that rare policies improve performance.

For the predictor PoC, preserve the original shared-policy target: G01/G02 and A01/A02/A03. That gives **1,232 primary combinations** across 516 candidates, versus the original 960 across 400. The other **1,768 combinations** support separately reported serving/diagnostic analyses. Preserve the original 400-checkpoint scorecard too, so broader recipe coverage is distinguishable from a changed evaluation population. If defining an expanded-policy headline metric, freeze and name it separately before test outcomes are exposed.

## Scheduling and holdouts

| Combined stage | Original work | Added work | Total |
|---|---:|---:|---:|
| Pilot | 100 | 96 | 196 |
| Remaining development | 613 | 1,171 | 1,784 |
| Locked test | 287 | 733 | 1,020 |
| **Total** | **1,000** | **2,000** | **3,000** |

The extension pilot applies all six new policies where applicable to the original 20 GSM8K and 12 AIME pilot checkpoints: 20 × 3 + 12 × 3 = 96 new exp_ids. It is part of the extension budget. All original AIME pilot checkpoints already belong to the fixed diagnostic subset. The combined pilot covers all 13 policies and uses the same 32 checkpoint candidates.

Keep the original 40 held-out sessions unchanged. Expanded checkpoint counts are GSM8K 203 development / 113 locked and AIME 135 development / 65 locked. Across all policies, there are **1,980 development combinations and 1,020 locked combinations**. None of the added policies of a locked checkpoint can enter predictor training or retrieval; the same exclusion applies to old native labels and outcome-bearing trajectory content.

This does not add independent scientist sessions: there are still 124 total and 40 locked. The 116 additional checkpoint candidates comprise 81 continuations and 35 parameter merges; no extra from-base training outputs are introduced. Evaluating old weights under new settings does not demonstrate pre-training prediction of genuinely new recipes.

## Readiness and cost gates

Among the 116 added candidates, the historical script audit has 92 reconstructed records, 14 unavailable and 10 blocked. Even reconstructed code needs launch/checkpoint binding and leakage review. All 516 need weight and tokenizer verification. If a candidate remains unusable or proves to be an alias, report it and refreeze any approved selection/split change; do not silently count duplicate weights as new recipes, omit failed rows, or invent replacements to maintain the budget.

The added work is 1,100 GSM8K combinations and 900 AIME combinations. At ten complete passes over 1,319 or 30 questions, respectively, that is **14,779,000 additional completions**; the full 3,000 matrix is **21,266,200 completions** if evaluated afresh. These are not GPU-hour estimates. Use the pilot to measure token lengths, throughput, failures, transfers and real cost before releasing later stages.

The target remains the arithmetic mean of ten complete pass rates, equivalently total correct divided by `10 × N` on the fixed question set. It is mean pass@1, not pass@10. Keep per-question outputs and errors; missing passes are invalid, not zeros or silently dropped observations. Historical sampling/cap matches require full protocol/artifact equivalence before any label reuse.

## Reproduction and verification

The deterministic extension builder consumes only tracked original bundle files and does not require private analysis data. `extension_summary.json` records hashes of its original inputs; `bundle_files.sha256.json` covers the published extension assets. The base protocol and phase files remain intact, so an existing runner can reconcile already scheduled work without remapping old exp_ids.

Run `python3 tools/outcome_prediction/verify_eval_matrix_extension.py` before consuming the combined manifest. Static success confirms the packaged assignments and identities, not model loadability or actual serving behavior. See [README.md](README.md) for metadata-fetch commands and [AGENT_HANDOFF.md](AGENT_HANDOFF.md) for runner integration requirements.
