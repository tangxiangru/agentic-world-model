# Official-evaluation evidence retention — isolated CPU prototype

Status: **archive/index/compact helper implemented and CPU-tested; not connected to PTB, launcher or harvest**. This is the first implementation unit of the [prospective design](../spec/2026-09-03-ptb-official-eval-evidence-retention.md), not a protocol variant, new experiment or promotion. Its introducing commit preserves the tested source; [verification evidence](trace-reviews/p5-serving-audit/retention-format-probe.json) also records file hashes.

## Implemented unit

`tools/ptb_official_eval_evidence.py` is a standalone stdlib helper. It requires a future GSM8K opt-in marker in `runtime_provenance.json` (`experiment.official_log_retention: inspect-json-v1`), matching requested job ID, batch/cell/run-purpose identity, frozen top/PTB commits and evaluation-image hash. Current/historical results have no marker and are refused before any destination is created. Neither a shared Unix user nor this helper grants scheduler ownership or release permission.

The caller supplies the exact numbered `official-eval/attempt-0001` source directory and either a post-attempt observed exit code or a cleanup phase with unknown exit. Cleanup cannot attach whatever later `metrics.json` happens to exist to an earlier attempt. Source and canonical metrics are never deleted, modified or rescored.

The helper streams raw bytes into deterministic gzip, hashes source and archive, checks the source did not change while copying, and atomically publishes without overwriting different content. A per-attempt nonblocking file lock prevents concurrent writers. The final receipt is the publication point; interrupted unindexed artifacts can be verified/reused on retry. Existing receipts/archives are revalidated, not rewritten with a new timestamp or exit observation. New logs or changed bytes after finalization require an explicit recovery decision rather than silent replacement.

Malformed JSON, partial/error logs and no-log attempts stay distinct from scientific completion. Parse/compaction failures preserve raw evidence. Compact generation runs in a disposable subprocess with default128 MB input cap,512 MB address-space limit,10 CPU seconds,20-second wall timeout and2 MB compressed-output cap. Exceeding any limit produces an explicit unavailable compact record, not a fake complete result. Raw archival is streamed independently of optional compaction.

Compact rows preserve every parsed sample's id/epoch and scorer value(s), available output usage/finish reasons, and canonical hashes. Missing observations remain null. Input/target and model-request role/content are UTF-8 JSON with sorted keys, compact separators, `ensure_ascii=False` and no NaN; request hashing includes the ordered list of model calls, each with ordered role/content messages. Completion hashes are SHA-256 of exact UTF-8 completion text. These per-row hashes are a different scope from P5's ordered-whole-evaluation hashes; do not equate the two schemas. Raw archives remain the content-level authority.

## Verified behavior

- **43 helper tests**, all passed. They cover exact gzip/sample round-trip, unknown fields and partial outcomes, malformed/no logs, retries/multiple logs, old/no-opt-in rejection, wrong job/attempt/exit data, source/destination symlinks, FIFO/traversal rejection, non-overwriting collisions, late source writes, archive/receipt corruption, concurrent writers, compactor time/signal/memory failure, receipt-write recovery, and CLI paths containing spaces.
- One of those tests executes the standalone archiver and its worker inside the actual pinned evaluation image using a synthetic file fixture. No `--nv`, evaluator, benchmark, model or model API is invoked. This proves Python/container compatibility of this helper, **not** the unimplemented Inspect sink or timeout/cleanup integration.
- **111 existing PTB experiment/results/ops/gate/runtime tests and4 meta-file tests** passed alongside the43: **158 passed,0 skipped**.
- Three real, previously completed **developer** logs were used read-only as format fixtures. Their raw57–58 MB bytes round-trip exactly; archived gzip is7.9–8.1 MB, compact gzip203,787–203,901 bytes, and all1319 sample rows/scorer values are retained (948/965/991 correct respectively). Per-file archive+compact+verification elapsed0.960–1.040 seconds on this host; not a runtime performance guarantee. Artifacts live only in the owned temporary directory recorded in the verification JSON, not in current result bundles.
- `git diff --check` passes. Current `awm/`, scientist six shipped paths, PTB submodule and experiment manifests/queue are unchanged.

## Not yet implemented or proven

No current job invokes the helper. The native runtime does not yet emit the opt-in provenance marker, create per-attempt local log sinks, preserve the evaluator exit observation through timeout, run fallback before cleanup, or retain unindexed source evidence after a preservation failure. The launcher/checker does not yet support the proposed manifest field, and the operator does not yet harvest this namespace. These are required remaining units, not waived gates.

No old official log has been recovered, no PTB result made newly complete, and no new clean cell counted. This prototype must not be pointed at existing results by manually adding a marker. Before deployment, place the canonical helper in the future PTB fork source, avoid diverging copies, wire all callers/consumers, test their boundaries, and freeze a common new generation with its matched baseline. Keep current PTB `dcf5da0` and Round02 receipts intact.

## Operational handoff

The hourly monitor2086813 remains live. Last completed tick23:00:54 UTC had0/17 terminal. The last full queue check22:54 showed29 JobHeldUser and owned allocation16/16, with90820 outside its frozen nodes and registered demand17/16; native reservation still11 nodes. No submit/release/cancel or fresh Slurm poll was caused by this CPU prototype. Use the [operator dependency view](operator-state.md) for the existing waves and held-floor arithmetic.
