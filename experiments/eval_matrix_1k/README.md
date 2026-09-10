# 3,000-combination evaluation handoff: original 1,000 + additional 2,000

This directory is the single handoff location for the original 1,000 combinations and the approved-to-package 2,000-combination extension. The directory name is retained for compatibility. The full selection is **3,000 exp_ids, 516 candidate checkpoints, 13 generation policies, ten benchmark passes per exp_id**. See [EXTENSION_2K.md](EXTENSION_2K.md) for the added design and [the original study proposal](../../doc/experiments/eval_matrix_1k_proposal.md) for its motivation.

**Runner input: `experiment_matrix_all_3k.jsonl`. Scheduling input: `execution_plan.json`.** The combined manifest contains the original 1,000 rows unchanged followed by the 2,000 additions. Do not concatenate it with the original or extension manifest: that would duplicate work. Use the extension-only manifest if the original work is already scheduled, and reconcile completed work by exp_id plus artifact/protocol identity.

Start or update the execution agent with [AGENT_HANDOFF.md](AGENT_HANDOFF.md). This package contains validation and small-asset download tools, **not a GPU executor**. It does not itself authorize compute. The first combined pilot is 196 combinations, including the original 100; never launch all 3,000 simply because they share one file.

## What is authoritative?

| File | Purpose |
|---|---|
| `experiment_matrix_all_3k.jsonl` | Authoritative combined runner input: all 3,000 combinations |
| `experiment_matrix_extension_2k.jsonl` | Only the 2,000 additions; no overlap with original exp_ids or checkpoint/policy pairs |
| `experiment_matrix.jsonl` | Preserved original 1,000 combinations; not silently enlarged |
| `execution_plan.json` | Explicit pilot, development, and locked-test stage order and phase-file membership |
| `configs/<config_id>/generation_config.json` | All 13 generation-config payloads: original seven plus six additions |
| `configs/<config_id>/request_template.json` | Explicit per-request settings; these must actually be applied |
| `generation_policies_all_13.json` | Complete policy catalog; the original `generation_policies.json` remains seven-policy-only |
| `protocol.json` | Seeds, runtime reference, benchmark assets, stopping and completeness contract |
| `phases/*.jsonl` | Seven disjoint phase files, ordered by `execution_plan.json`, not filename sorting |
| `splits.json` | Forty reserved scientist sessions; all their policies and historical labels stay excluded from predictor training/retrieval |
| `selected_checkpoints_all_516.jsonl`, `selected_ids_516.json` | All selected checkpoint locations/IDs and unresolved readiness flags |
| `selected_checkpoints.jsonl`, `checkpoint_inventory.json` | Original 400 checkpoint records and complete source inventory/provenance |
| `extension_summary.json` | Extension counts, preserved-source hashes, settings and analysis roles |
| `bundle_files.sha256.json` | Checksums for the packaged specification/audit files |

The combined matrix, execution plan, configuration files and protocol take precedence over intermediate allocation discussions in background audit notes. Original `matrix_summary.json` and earlier audit notes still describe the original 1,000. All jobs remain `proposal_only_preflight_required`; static validation does **not** change that status.

## Commands that work now: no GPU execution

From a checkout of branch `jerry-wm`, run:

```sh
python3 tools/outcome_prediction/verify_eval_matrix_extension.py
git submodule update --init third_party/PostTrainBench
git -C third_party/PostTrainBench rev-parse HEAD
```

The submodule SHA must be `882eb90fef88f255374f9d30b2c16ba7e3ae5c56`. The extension validator also checks the original bundle. It uses Python's standard library and does not require the private `data/` directory, Hugging Face cache, credentials or a GPU.

Preview the small-asset fetch, then fetch if the dataset credentials are available:

```sh
python3 tools/outcome_prediction/fetch_eval_matrix_assets.py \
  --phase combined-all --out data/eval_matrix_1k_assets --dry-run

uv run --frozen python tools/outcome_prediction/fetch_eval_matrix_assets.py \
  --phase combined-all --out data/eval_matrix_1k_assets
```

`combined-all` fetches the historical evaluator kit/runtime README and all 516 checkpoints' `config.json` / archived `generation_config.json` from pinned HF revisions. For only the 32 pilot checkpoints, use `--phase combined-pilot`. `extension-all`, `extension-pilot`, `extension-development` and `extension-test` select the new work. For backward compatibility, `--phase all` still means the original 1,000, not all 3,000.

The fetcher does **not** download weights, model evaluation results, scientific trajectories, `.env` files or the whole HF repository. Its receipts record paths, revisions and hashes, and it does not overwrite conflicting assets. Benchmark source data may include answer keys; keep this operator cache separate from predictor-agent inputs.

Use existing credential managers or environment configuration for private HF access. Later weight staging requires read access to the exact GCS URIs in the manifest. Never commit credentials or print their values. A `--dry-run` needs no network or credentials.

## What to launch, and in what order

| Stage | Phase files under `phases/` | Combinations |
|---|---|---:|
| Pilot, after preflight and compute approval | `1_operational_pilot.jsonl` + `5_extension_pilot.jsonl` | 196 = 100 + 96 |
| Development, after pilot correctness/cost review | `2_development_core.jsonl` + `3_diagnostic_extensions.jsonl` + `6_extension_development.jsonl` | 1,784 = 597 + 16 + 1,171 |
| Locked test, only after freezing predictor and analysis | `4_locked_test.jsonl` + `7_extension_locked_test.jsonl` | 1,020 = 287 + 733 |
| **Total** | Every exp_id appears exactly once across these stages | **3,000** |

The combined pilot uses the same **20 GSM8K and 12 AIME checkpoint candidates** as before, now under five and eight policies respectively: 20 × 5 + 12 × 8 = 196. It covers all 13 policy files and is included in the 3,000 budget. If the original pilot is already validly completed, only the 96 extension-pilot exp_ids are new. All seven phase files are disjoint; their numerical filenames preserve compatibility, not launch order.

There is intentionally **no claimed `run-pilot` command yet**. The execution agent must implement/test the adapter described in [AGENT_HANDOFF.md](AGENT_HANDOFF.md), then provide its exact invocation and preflight report before compute approval. The old `evaluate_epochs.py` and cluster wrapper do not implement this explicit-policy protocol unchanged.

## Pinned sources for the execution agent

- HF dataset: `JerrrrryL/awm-gsm8k-trajectories`.
- Historical evaluator/assets: revision `cc2ac9d884a7d962a6024ab0d5cd8ed3370070de`, paths `rescore10/eval/**` and `rescore10/trajectories/README.md`.
- Archived model/generation metadata: revision `446127629d7b271d537390e69bfb2d960a3aa515`, paths `checkpoints_meta/<checkpoint_id>/{config,generation_config}.json`.
- PostTrainBench submodule: `882eb90fef88f255374f9d30b2c16ba7e3ae5c56`. AIME's missing-from-HF-kit `task.py`, `score.py` and scorer tests are under `src/eval/tasks/aime2025/`.
- The PTB container source pins `inspect_evals` to `06001a83e6d7c709c2ede0570dce7f1031a0bad8`. Pin this task implementation, not just Inspect itself.
- Recorded evaluation runtime: vLLM `0.11.0`, Transformers `4.57.3`, Inspect `0.3.150`, PyTorch `2.8.0+cu129`. A complete verified image digest/hardware configuration remains to be established.

Do not run PTB's full `run_task.sh`: it includes training/judging outside this evaluation-only study. Do not reuse the historical cluster shell wrapper as-is: it has machine-specific paths and deletes temporary logs. Do not build `vllm_debug.def` unchanged and call it reproducible: it installs an unpinned Inspect fork. Do not follow historical relay/delete-slot workflows; stage weights read-only from the manifest's persistent archive URIs.

## Outstanding checks before spending

All 516 weight identities still need shard/tokenizer verification. The historical script audit has 53 unresolved entries (34 unavailable, 19 blocked), including 24 among the 116 added checkpoint candidates. Clean launch-only predictor bundles are not included or certified complete here. Newer missing-code/content-review flags have a different scope; do not add those counts to the historical audit.

A matching historical sampling/cap pattern is not automatic label reuse; the original 1,000 alone had 368 candidates for an equivalence audit. Preserve the frozen selection/splits if possible; any necessary replacement or alias-driven regrouping must be documented and refrozen before new labels are exposed. Do not silently omit blocked experiments or substitute nearby checkpoints. All 516 candidates are from the same 124 sessions: more serving configurations do not create independent recipes or sessions.

The package contains operator/research-design material. It is **not** a leakage-clean predictor input corpus: background audit notes include historical aggregate results and a few known diagnostic examples. Predictor agents must receive separately frozen training-launch-only inputs and permitted TRAIN-only labels, with test outcomes inaccessible.

## Reproducibility tools

`tools/outcome_prediction/select_eval_matrix_checkpoints.py` and `propose_eval_matrix.py` preserve the original selection/generation code. Re-running them requires the original private corpus and derived audit inputs under `data/`; their historical paths are provenance, not dependencies of the portable checker. Consume this frozen package for execution rather than attempting to regenerate it from a fresh checkout.

`tools/outcome_prediction/extend_eval_matrix.py` builds the extension using only this tracked bundle. It does not change the original 1,000 matrix rows, their seven configurations, their phase files, the protocol or session splits. The extension adds its own rows/configs and provides a combined view. Prefer consuming the checked-in files; rebuilding is for reproducibility, not a launch step.

Tests for the portable handoff:

```sh
uv run --frozen --extra dev python -m pytest -q \
  tests/test_eval_matrix_bundle.py \
  tests/test_eval_matrix_extension.py \
  tests/test_fetch_eval_matrix_assets.py \
  tests/test_propose_eval_matrix.py
```

Full-corpus regeneration tests skip explicitly when those external source artifacts are absent. Static validation and the portable bundle tests must still pass.
