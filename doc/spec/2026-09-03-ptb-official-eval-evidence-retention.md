# Prospective official-evaluation evidence retention

Status: **isolated archive/index/compact CPU prototype tested; runtime/launcher/harvest integration not implemented or enabled**. This is operator/harness provenance work following the [P5 adjudication](../exp_protocol_iterations/trace-reviews/p5-serving-audit/planner-decision.md), not an `exp_protocol` candidate. Current PTB pin `dcf5da031435c54e3680b6ec3f63e7e317efc13e`, scientist six-path trees, receipts, queue and active jobs remain unchanged.

## Problem and verified source behavior

The three P5 developer logs survive on the durable data volume but were excluded from git bundles by the2 MB per-file policy. Original full logs directly establish1319 samples, request hashes and settings. Official logs are different: `run_task.sh` invokes evaluation with working directory `source/src/eval/tasks/gsm8k`; default Inspect `./logs` therefore lives in the per-job frozen source inside scratch. `single_task.sbatch` normally removes that scratch on exit. Increasing the operator's file cap does not address this upstream lifetime boundary.

The actual pinned evaluation image at `/rmeng_data/robtang/ptb-containers/vllm_debug.sif` was inspected without `--nv`, without a model/evaluation call, using the site executable `/rmeng_data/robtang/tools/apt-root/usr/bin/apptainer` from the canonical PTB `.env` key. Its Inspect implementation is under `/usr/local/lib/python3.10/dist-packages/inspect_ai/`:

- `_eval/eval.py:586–592` resolves an explicit `log_dir`, otherwise `INSPECT_LOG_DIR`, otherwise `./logs`, and checks writability.
- The frozen GSM8K `evaluate.py` supplies `log_format='json'` and no `log_dir`; the environment hook is therefore applicable without adding evaluator CLI options.
- `log/_recorders/json.py` keeps a growing in-memory `EvalLog`; its default local buffer is10 samples, and `flush` rewrites the entire JSON. `log_finish` writes the final status/results. A interrupted write can be partial/malformed; an early kill can leave no file.

Consequently, retain **local-scratch logging during inference** and archive after attempts. Moving repeated writes directly to shared durable storage is not assumed behavior-neutral, especially while investigating serving timing/non-repeatability.

The [pinned-recorder inspection](../exp_protocol_iterations/trace-reviews/p5-serving-audit/pinned-recorder-inspection.json) records the actual package version and inspected source-file hashes. It is source-level compatibility evidence, not a synthetic recorder or full-container integration pass.

## Scope and invariant

Preserve all emitted official-attempt JSON bytes, including failed/cancelled/partial attempts, and a receipt-backed index sufficient to locate and validate them later. Provide compact per-item evidence in git bundles for paired analysis without committing every repeated prompt/event payload. Never fill `metrics.json`, reconstruct an official score from partial rows, or redefine scientific completion.

No changes to evaluator/task/scorer/template, model/decode/memory/concurrency/token limits, evaluator image, scientist budget, retry count/order or official metrics selection. Logging location and post-attempt evidence handling are the only prospective differences. Preservation failure is explicit metadata failure, not a fabricated evaluation failure or success. Node loss/SIGKILL before any flush cannot be made recoverable by an exit trap; retain that limitation.

## Frozen opt-in and generation boundary

Proposed future manifest contract: optional `official_log_retention: inspect-json-v1`, default absent/off. Launcher records it in the immutable receipt and propagates a narrowly named `POST_TRAIN_BENCH_OFFICIAL_LOG_RETENTION` capability to a PTB commit that actually implements it. Unknown modes or unsupported source versions must fail prelaunch validation, not be silently ignored. Initial support is GSM8K's inspected JSON evaluator; do not silently enable other tasks or queues.

All members of any comparison must use the same retention mode and PTB generation. Do **not** enable it opportunistically on the already-frozen E2/J/K manifests or on a replacement in an otherwise old-generation comparison. Prepare it for a future common harness generation with its own matched baseline. Never edit an old receipt, frozen scratch source, running process or active manifest. No shared wrapper/.env default is changed.

The implementation may be developed and CPU-tested in isolation. A candidate PTB commit must be preserved on the configured tangxiangru fork and audited before the superproject pin changes; official upstream is only `aisa-group/PostTrainBench`. Keep the operator's current PTB pin while current-generation work is in flight. This spec itself grants no queue release or running-job cancellation authority.

## Proposed lifecycle

1. Before each existing official evaluation attempt, create a unique directory under that job's local scratch, keyed by the existing `EVAL_COUNTER`, e.g. `JOB_TMP/official-eval/attempt-0001/`. Refuse unsafe/ambiguous paths; never reuse another attempt's files. Inject `INSPECT_LOG_DIR` only into the official evaluator container in opt-in mode, before Python imports Inspect. Existing local filesystem/buffer policy is retained.
2. Run the existing evaluator with its unchanged arguments and timeout. Preserve its exit status separately; do not replace it with the archive helper's return code or alter the existing `metrics.json`-based retry flow.
3. After return or timeout, stop writing and archive the emitted regular JSON files to `EVAL_DIR/official_eval/attempt-0001/`. Use a stdlib streaming gzip/sha256 copier and atomic publication. Preserve malformed/partial bytes as raw evidence. Never follow symlinks, traversal names or paths outside this exact attempt. Archive all logs in the attempt, not the newest globally named log.
4. Publish an additive attempt receipt/index: schema version, batch/cell/job, attempt number, PTB/evaluator-image identity from existing frozen provenance, observed exit status, source basename/size/sha256, archive path/size/sha256, raw-parse classification, and observed canonical metrics-file hash/size if present at attempt end. Missing log and retention errors are explicit. A metrics hash is provenance, not proof that a log is valid or that a score is clean.
5. Before ordinary wrapper cleanup, idempotently finalize any started but unindexed attempt as an interruption/unknown-exit attempt and preserve available bytes. This fallback must never perform a model call or trigger evaluation/retry. Do not overwrite a valid prior receipt or double-count a repeat; collision/hash mismatch is an explicit error. If preservation fails, expose the failure and keep that attempt's local log evidence when feasible, rather than silently deleting it. No broad scratch retention of model/cache trees is needed.
6. Once published, the existing cleanup can remove the original attempt directory. Hard node loss or SIGKILL before persistence remains an incomplete attempt with unavailable evidence, not a claimed successful backup.

## Durable versus git-bundle representation

Durable data volume: raw compressed JSON for every emitted attempt, plus the immutable per-attempt index. This keeps full responses/events available for deeper reviews and preserves unreadable bytes for diagnosis. Atomic writes and content hashes prevent half-written archives from looking complete. An interrupted `.tmp` is never indexed as a complete archive; resume/recovery is idempotent and bounded to that attempt.

Git bundle: index, bounded structural metadata and compressed per-sample rows, not full repeated prompts/events. Minimum sample evidence: actual id/epoch, recorded scorer value(s), normalized input/request and completion hashes, and available finish/usage fields; metadata includes observed total/completed/scored/unscored counts, task/dataset, logged model args and `model_generate_config`. Preserve null/unknowns. Do not infer missing counts, decode settings or server seed. Specify canonical hashing exactly so independent readers can reproduce it. The raw source archive remains authoritative for content-level trace review.

If JSON cannot be parsed or compacted within a declared CPU/memory limit, preserve the raw archive and record why compact evidence is absent. A compact artifact is not allowed to omit rows silently or label partial samples complete. Do not reuse K's comparator certificate or change the legacy `complete` validator field: scientific completion, provenance eligibility and evidence-retention quality are separate axes.

Operator `harvest` should recognize only this explicitly indexed official-evidence namespace, verify archive/compact hashes and copy allowed compact artifacts. Keep generic task-tree2 MB policy and existing statuses/score semantics unchanged. Record skipped raw archives as deliberately retained on the data volume with their exact hashes, not “lost.” Legacy results lacking the namespace remain valid under their existing contract. No automatic reharvest/rewrite of old bundles is part of this change.

## Verification required before freezing implementation

- **Pinned-image compatibility:** use a synthetic CPU-only recorder or mocked evaluation fixture, not GSM8K/model inference, to prove the env sink selects the intended local directory and JSON remains byte-preservable. Actual container integration is still untested; source inspection alone is not acceptance.
- **Real shell boundary:** execute extracted orchestration with fake Apptainer/timeout/helper processes. Assert identical model/evaluator arguments, returned evaluator status and retry decisions; environment changes are isolated to the log sink and opt-in. Child subshell sees the attempt identity and paths. Paths with spaces work.
- **Attempts:** success, evaluator failure with a valid error log, partial/malformed JSON, no log, timeout, normal cleanup fallback, repeated helper invocation, multiple retries/logs and a late/interrupted archive. Preserve all attempts without overwriting or confusing canonical metrics.
- **Safety:** reject symlinks/path traversal, mismatched job/attempt identity and archive collisions; injected storage/permission errors are visible. Confirm fallback operates before cleanup and does not delete unpublished evidence. No current source/receipt/scheduler mutation in any test.
- **Data fidelity:** decompressed archive SHA matches original bytes; compact sample counts/hashes/scorer records match valid source JSON; malformed/unknown data is never certified. Use synthetic cases plus read-only replay on P5's three original developer JSONs as format fixtures, explicitly not official-evaluation evidence.
- **Consumers/compatibility:** operator harvest/index round-trip, incomplete retention visible without changing score/completion semantics, legacy bundles unchanged, source-feature checks reject unsupported frozen PTB commits, and all existing PTB runtime/ops/results tests pass.
- **Resource behavior:** stream raw compression; bound optional JSON compaction. Record time/bytes on retained fixtures. Keep inference on local scratch; no GPU/model/network calls in acceptance tests.

## Decision and next action

The [isolated prototype record](../exp_protocol_iterations/2026-09-03-official-evidence-prototype.md) covers the first helper unit:43 helper tests, including a synthetic file test inside the pinned image, plus111 existing PTB and4 meta tests (158 passed). The helper refuses existing results without the future `experiment.official_log_retention` provenance marker; current provenance is untouched. Three original developer logs pass lossless archive/compact format replay. These checks do not prove the remaining orchestration or consumer integration.

The archive/index/compact helper is the completed isolated unit; this is still **not a completed retention implementation**. Next wire and test the real timeout/cleanup boundaries and actual Inspect local sink, then add the opt-in launcher/harvest contract. Place one canonical helper in the future PTB source rather than keeping diverging copies. Freeze and audit all parts together before any common-generation enablement. Current16-GPU subqueue operations,29 held jobs and the ownership/native-isolation blocker remain unchanged; the hourly monitor continues independently.
