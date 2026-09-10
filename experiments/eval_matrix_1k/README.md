# 1,000-combination evaluation handoff

This directory is the **portable, frozen experiment specification** for the [study proposal](../../doc/experiments/eval_matrix_1k_proposal.md). It contains the exact checkpoint/config assignments, not just a suggested grid. It does **not** contain a GPU evaluation launcher or grant permission to spend compute.

Start a preparation/executor-implementation agent with [AGENT_HANDOFF.md](AGENT_HANDOFF.md). The commands below perform static validation and download small source/metadata assets only. After implementation and preflight, the first evaluation batch is **`phases/1_operational_pilot.jsonl`**, not the full matrix.

## What is authoritative?

| File | Purpose |
|---|---|
| `experiment_matrix.jsonl` | The exact 1,000 `exp_id` / checkpoint / generation-policy combinations |
| `configs/<config_id>/generation_config.json` | The seven proposed generation-config payloads |
| `configs/<config_id>/request_template.json` | Explicit per-request settings; these must actually be applied |
| `protocol.json` | Seeds, runtime reference, benchmark assets, stopping and completeness contract |
| `phases/*.jsonl` | Exact phase membership; the first phase contains 100 combinations |
| `splits.json` | Forty reserved scientist sessions; all their policies and historical labels stay excluded from predictor training/retrieval |
| `selected_checkpoints.jsonl`, `checkpoint_inventory.json` | Checkpoint locations, selection/provenance evidence and unresolved readiness flags |
| `bundle_files.sha256.json` | Checksums for the packaged specification/audit files |

The matrix, configuration files, protocol and phase files take precedence over intermediate allocation discussions in the background audit notes. All jobs intentionally remain `proposal_only_preflight_required`. Passing the static bundle checker does **not** change that status.

## Commands that work now: no GPU execution

From a checkout of branch `jerry-wm`, run:

```sh
python3 tools/outcome_prediction/verify_eval_matrix_bundle.py
git submodule update --init third_party/PostTrainBench
git -C third_party/PostTrainBench rev-parse HEAD
```

The submodule SHA must be `882eb90fef88f255374f9d30b2c16ba7e3ae5c56`. The validator uses Python's standard library and does not require the original `data/` directory, Hugging Face cache, credentials or a GPU.

Preview the small-asset fetch, then fetch if the dataset credentials are available:

```sh
uv run --frozen python tools/outcome_prediction/fetch_eval_matrix_assets.py \
  --phase pilot --out data/eval_matrix_1k_assets --dry-run

uv run --frozen python tools/outcome_prediction/fetch_eval_matrix_assets.py \
  --phase pilot --out data/eval_matrix_1k_assets
```

The fetcher downloads the historical evaluator kit/runtime README and the selected pilot checkpoints' `config.json` / archived `generation_config.json` from pinned HF revisions. It does **not** download weights, labels, scientific trajectories, `.env` files or the whole HF repository. Its receipt records paths, source revisions and hashes. It does not overwrite conflicting local assets.

Use existing credential managers or environment configuration for private HF access. Later weight staging requires read access to the exact GCS URIs in the manifest. Never commit credentials or print their values. A `--dry-run` needs no network or credentials.

## What to launch, and in what order

| Phase | Manifest | Combinations |
|---|---|---:|
| First, after preflight and explicit compute approval | `phases/1_operational_pilot.jsonl` | 100 |
| After pilot correctness/cost review | `phases/2_development_core.jsonl` | 597 |
| Remaining diagnostic settings | `phases/3_diagnostic_extensions.jsonl` | 16 |
| Only after freezing the predictor and test protocol | `phases/4_locked_test.jsonl` | 287 |

The pilot is **20 GSM8K checkpoints × two policies plus 12 AIME checkpoints × five policies**: 32 candidate weight sets, all seven policies, 100 combinations, ten full benchmark passes each. It includes 24 of the 40 diagnostic combinations; the other 16 are phase 3. Do not add the pilot on top of the 1,000-combination budget.

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

All 400 weight identities still need shard/tokenizer verification. The historical script audit has 29 unresolved entries (20 unavailable, nine blocked); three selected records have newer missing-declared-code flags, and 177 need content review. These are different audits, not counts to add together. Clean launch-only predictor bundles are not included or certified complete here.

A matching historical sampling/cap pattern is not automatic label reuse: 368 combinations are candidates for an equivalence audit. Preserve the frozen selection/splits if possible; any necessary replacement or alias-driven regrouping must be documented and refrozen before new labels are exposed. Do not silently omit blocked experiments or substitute nearby checkpoints.

The package contains operator/research-design material. It is **not** a leakage-clean predictor input corpus: background audit notes include historical aggregate results and a few known diagnostic examples. Predictor agents must receive separately frozen training-launch-only inputs and permitted TRAIN-only labels, with test outcomes inaccessible.

## Reproducibility tools

`tools/outcome_prediction/select_eval_matrix_checkpoints.py` and `propose_eval_matrix.py` preserve the original selection/generation code. Re-running them requires the original private corpus and derived audit inputs under `data/`; their historical paths are provenance, not dependencies of the portable checker. Consume this frozen package for execution rather than attempting to regenerate it from a fresh checkout.

Tests for the portable handoff:

```sh
uv run --frozen --extra dev python -m pytest -q \
  tests/test_eval_matrix_bundle.py \
  tests/test_fetch_eval_matrix_assets.py \
  tests/test_propose_eval_matrix.py
```

Full-corpus regeneration tests skip explicitly when those external source artifacts are absent. Static validation and the portable bundle tests must still pass.
