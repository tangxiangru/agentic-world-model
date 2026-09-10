# Execution-agent task: prepare and validate the 100-combination pilot

## Scope and authority

Read `experiments/eval_matrix_1k/README.md`, the study proposal and the frozen manifest/configuration files. This is evaluation of **existing checkpoint weights**, not new training and not a search for better decoding settings.

**Default task: implement the executor, perform non-GPU preparation/dry-run checks, and return a preflight report plus an exact proposed pilot command. Stop before GPU execution unless the operator explicitly approves it.** This document alone does not approve the pilot or later phases. Never launch all 1,000 combinations as a convenience.

## Inputs and preparation

1. Run `python3 tools/outcome_prediction/verify_eval_matrix_bundle.py`. The frozen batch is `phases/1_operational_pilot.jsonl`: 100 exp_ids, 32 checkpoint IDs, ten passes each. Do not choose a new subset.
2. Initialize and verify the pinned PostTrainBench submodule. Use `fetch_eval_matrix_assets.py --phase pilot --out <cache> --dry-run`, then its actual fetch, for the small pinned assets. No benchmark outcomes or large weight downloads are needed for this first step.
3. Read credentials from existing credential providers; do not log or commit secrets. Confirm read access to the manifest's exact GCS archive locations before proposing weight transfers. Request missing GPU/storage/access details rather than guessing an account or machine.
4. Resolve the selected pilot checkpoint/provenance flags. The frozen inventory is evidence, not proof of actual weight identity or launch-script completeness. Verify training-launch code/data/config binding and report unresolved items. Do not pass the broad historical v6 code prefix or raw scientific trajectories to the predictor.

## Complete the evaluation source package

- Use the HF-pinned `evaluate_epochs.py`, templates and benchmark reference data.
- Supply AIME `task.py`, `score.py` and scorer tests from PTB submodule commit `882eb90fef88f255374f9d30b2c16ba7e3ae5c56`. The recorded trajectory and this submodule agree on task SHA `5b723072d760496ae964acf8b46d1776d9ed1585080257c5879e2a397603d412` and scorer SHA `9720223a2573b5d495539f0aa691cbb65f75aadeeccd16112832c6139b5f63d0`.
- Pin `inspect_evals` commit `06001a83e6d7c709c2ede0570dce7f1031a0bad8`, including GSM8K's task implementation and AIME's imported `aime2024_solver`.
- Inspect the actual task loaders: the historical scripts can load HF datasets rather than the adjacent `test_data.json`. Freeze and verify the question IDs/content, few-shot examples, prompts, solver and scorer. Merely hashing an unused local data file is not sufficient.
- Establish a reproducible image using the runtime versions in `protocol.json`, and record its digest, GPU type, dtype/quantization, context limit, concurrency and caching settings. Verify actual imported package versions. Do not blindly execute old wrappers/container recipes or PTB's full training workflow.

## Implement the matrix executor

The existing proposal generator is **not** the executor. Implement an adapter that consumes a phase manifest and fails closed when requirements are unmet:

1. Stage a checkpoint and its tokenizer/config artifacts read-only from `checkpoint_uri`. Cache by verified artifact identity; hash all relevant shards and tokenizer files; detect aliases and verify loadability. Do not mutate the archive or apply an in-place checkpoint fixup.
2. Check tokenizer vocabulary/special-token compatibility with weights. Apply the common intended stop sets from the requested policy; require the tokenizer's primary EOS to be within that set. Do not silently swap a tokenizer.
3. Use vLLM's `generation_config=vllm` mode (or demonstrate an equivalent fully controlled route), then supply the complete corresponding `request_template.json`. These files are not applied merely because their paths appear in a manifest. Disable accidental inherited caps/defaults; explicitly supply stop IDs as well as sampling settings.
4. Generate exactly one completion per question per repeat. Use the `protocol.json` question/repeat seed formula with repeat IDs 0–9. Keep stable question IDs and record each actual request seed. `n_passes=10` does not mean a best-of-ten score.
5. Capture and assert the **server-resolved** sampling parameters, actual output caps, stop behavior and loaded artifact identities. An echoed request payload is not evidence of the server's resolved state. Add instrumentation or a verified interception point as necessary.
6. Reuse a loaded checkpoint across its requested policies where safe. Preserve fixed cache/concurrency settings and request/policy order. Ten repeats do not require ten checkpoint downloads.
7. Support a no-model dry run, bounded pilot-only scheduling, per-exp_id status, retries/resume without duplicate samples, and immutable result/provenance records. Require explicit authorization for a phase; do not auto-advance to the next one.
8. Preserve full logs, completions, per-question correctness, finish reasons, token counts and timing. Infrastructure errors, missing scores and missing passes are invalid—not zero and not silently excluded. Do not inherit the old wrapper's temporary-log deletion behavior.

The intended benchmark sizes are 1,319 GSM8K questions and 30 AIME questions. Validate the complete `(exp_id, question_id, repeat_id)` Cartesian product for each completed exp_id before producing its label.

## Result contract

For each exp_id, emit at least:

- `exp_id`, `checkpoint_id`, `generation_config_id`, benchmark and split;
- artifact, input-bundle, container, evaluator, template and effective-policy digests;
- ten run records with repeat ID, question count, correct count and pass rate;
- `avg_pass_rate = sum(per_run_pass_rate) / 10`, in the 0–1 scale;
- a per-question/run log reference, actual request/resolved settings, seeds and stop/token/timing details;
- an explicit completion/validation status and any errors.

The output target is mean pass@1, not pass@10. Produce a separate validation report. Resume existing work only when its full exp_id/artifact/protocol identity matches. Reuse old labels only after complete equivalence verification; matching temperature or output length alone is insufficient.

## Stop and report before the pilot

Return:

1. Successful portable-bundle validation and small-asset fetch receipts.
2. The exact evaluator sources, container/runtime and compute/storage configuration.
3. A checkpoint/input readiness table, including any failed or unresolved items; no silent replacements.
4. A dry-run schedule for exactly the 100 pilot exp_ids, confirming seven policies and 32 checkpoint candidates.
5. Tests for policy application, seed/sample identity, cap/EOS handling, result completeness and safe resume. Distinguish CPU/mock checks from anything actually tested on a GPU.
6. The exact pilot invocation using the **newly implemented** executor, a concurrency limit, expected storage/transfer needs and the initial cost-measurement plan.

Ask for explicit compute approval if it has not already been given. A static validation pass is not a reason to mark the unresolved jobs ready.

## If the operator approves the pilot

Run only `phases/1_operational_pilot.jsonl`, ten complete passes per exp_id, using the approved runtime/resources. Stop after the pilot and report valid/failed/reused cells, resolved-policy checks, token lengths, throughput and actual cost. Do not automatically launch the remaining 900.

The next manifests, after review, are 597 development-core combinations and 16 remaining diagnostic combinations. The final 287 combinations belong to 40 locked sessions and require a frozen predictor/analysis protocol. Exclude all historical outcomes from those sessions from predictor training/retrieval. Keep evaluator access to answer keys/results separate from predictor-agent access.

This study can test cross-session recipe/serving prediction on historically observed recipes. It does not make old recipes into an untouched future-training test, and its background audit documents are not a leakage-clean predictor corpus.
