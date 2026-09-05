# P5 supplemental review — planner adjudication, 2026-09-03

**Decision: no new protocol candidate, no GPU repeat, no queue mutation.** Keep direction #24 as an observation/evidence-retention follow-up; do not declare concurrency causal or permanently rule it out. This closes a supplemental review of three already-counted cells, not another eight-new-clean window.

Local Opus5[1m] max session `145e42ee-b904-4829-9380-e4534ccbc7bf` delivered its [unaltered final report](report.raw.md). The planner read it fully, checked frozen code and final-artifact trace ranges, recovered original developer logs, and makes the corrections below. The helper's draft Write and ExitPlanMode attempts were rejected by the read-only tool set; no extra permission was granted. Its final output was saved by the planner, and the now-idle session was stopped. Its delivery claim that this directory was untracked was stale; the brief/launch already existed in commit `3daa6bb`.

## Accepted facts, with stronger source evidence

All three `task/evaluate.py` files match frozen PTB `dcf5da031435c54e3680b6ec3f63e7e317efc13e`, SHA-256 `02b97287d5cd2179dc26c4328a35386a5a2908919a360fd97418b32003b15529`. Its defaults are mc2, max_tokens4000 and gpu_memory_utilization0.3. `run_task.sh:637–642` adds no memory/concurrency override; each official log displays mc2/max_tokens4000. The0.3 official value is a frozen-command/default inference, not a retained official engine-runtime dump. Provenance fixes different agent/evaluation images; identical evaluator code does not establish identical serving libraries or resolved engine settings.

The original developer logs still exist at the receipt-backed raw result paths despite being absent from git bundles. Their57–58 MB size exceeds `awm.ptb_ops.PER_FILE_CAP` (2 MB); g01r01's status explicitly lists `analysis/samples_FINAL_full.json` at57,031,743 bytes as skipped. The [derived metadata](developer-metadata.json) preserves exact original paths and byte hashes, actual completion/scoring counts, logged generation settings and prompt fingerprints. The [read-only extractor](extract_developer_metadata.py) replays the derivation without GPU/model calls.

| final artifact | developer actual n/correct | logged mc / memory | official correct/1319 | official−developer |
|---|---|---|---:|---:|
| g01r01 soup27→final_model |1319/948|32 /0.85|937|−11 items,−0.834 pp|
| g01s02 exp10→final_model |1319/965|32 /0.85|956|−9 items,−0.682 pp|
| g01s07 soup0205→final_model |1319/991|16 /0.3|970|−21 items,−1.592 pp|

All three original developer reports are `status: success`, have1319 retained/total/completed/scored samples, zero unscored, and reproduce the reported correct counts directly from `samples[].scores.match.value`. Their per-model-call configs agree with `eval.model_generate_config`: mc32/32/16, max_tokens4000. Those settings are not in `eval.config`; querying the wrong schema branch would falsely report them missing. Inspect version matches across these developer logs. No resolved server dtype/seed is manufactured from absent fields.

The ordered (id,epoch,input,target) SHA-256 is identical across these three developer reads (`0c49d81f…`), as is the ordered (id,epoch,request role/content) hash (`d1c54374…`). Each sample has exactly one model call. This verifies normalized input/request equality **among these developer reads**, not against the unavailable official per-item log.

## Corrections to the helper's interpretation

1. **Same-cell noise is not automatically same-artifact noise.** g01r01's final soup27 repeats span948–950, not the19-item range observed on another checkpoint. g01s02's final exp10 reads964/965; the924→911 pair and65 discordant outcomes are on earlier exp05 weights. These prove that some artifacts vary under apparently unchanged settings, but do not calibrate the final models' distributions. Reject “the final gaps are explained by their noise ranges.” Also,65 paired disagreements is a count, not a uncertainty interval for a different aggregate score.
2. **No cross-model dose-response or memory refutation.** Comparing11/9-item gaps atmc32 with a21-item gap atmc16 changes model/artifact and other conditions. It cannot establish or refute a dose-response, and the largest gap in the memory-matched cell does not disprove a memory effect elsewhere. For g01s07 only one requested knob differs; the environment and stochastic/runtime effects remain uncontrolled.
3. **Token totals are not prompt identity.** Equal2,910,587 input-token totals and equal code/template hashes are useful checks, not proof of every rendered request. The recovered hashes strengthen the developer comparison only; official input equivalence still lacks per-item evidence. “Everything held fixed” overstates the known engine/library/seed information. Equal scalar accuracy is not proof of equal correctness sets or output bytes.
4. **Actual n comes first.** The report's SE algebra is a cross-check under a known estimator, not the primary completion proof. Actual original logs now establish1319 evaluated/scored items directly. Dataset population or a requested limit would not do so alone.
5. **The proposed follow-up is still two-factor.** Repeating(mc2,memory0.3) versus(mc32,memory0.85) would estimate a joint configuration contrast, not isolate concurrency. If a future authorized investigation is warranted, hold artifact, image/libraries, memory, seed policy, dataset/order/template, max_tokens and process policy fixed; vary only concurrency with repeated/counterbalanced reads, or explicitly use a factorial design. This is a design boundary, not an approved run or frozen new experiment.
6. **No generic saturation claim.** C v2 was withdrawn because its specific final-n threshold was already met (7/8 versus3/4), not because every uncertainty intervention is saturated. These cells' repeat diagnostics do not prove that no future four-cell evaluation-quality screen could move a properly predeclared metric. There is no presently justified distinct concurrency rule; retain the observation without pretending the question is solved.

## Where the missing official evidence was written

The report's “just harvest the official JSON” recommendation misses a persistence boundary. In frozen `run_task.sh`, the evaluator's working directory is `source/src/eval/tasks/gsm8k`, and Inspect writes relative `logs/...json` there. `single_task.sbatch:129–136` puts that source inside job-local scratch; its exit trap normally deletes the scratch. By contrast, the developer task tree is copied into the durable result directory before official evaluation.

The90647 Slurm stdout names `/mnt/localssd/posttrainbench/robtang_google_com/90647/source`. Exact official-log existence checks were attempted on the receipt's nodes (ondem0 for90647, ondem1 for90792/90797), but strict SSH host-key verification failed before access; it was **not bypassed**. Thus remote absence is not asserted as an observed fact. Frozen cleanup explains why these files are expected to be ephemeral unless scratch retention was enabled.

No matching official file was found in the local checkout path, and no official per-item artifact has been recovered. A future preservation change would need to save the official log into durable results **before cleanup**, then have the operator include or compact it; merely raising the git-bundle cap cannot recover a file that never reached that directory. Scope and validate that as operator/harness provenance work with a new frozen source identity, not a silent edit to active jobs or the scoring contract. No such code, source, manifest or environment change is made here.

## Operational decision and follow-up

Existing D/B/H→A/E2 priorities and J/K independent screens are unchanged; this evidence makes no pending block scientifically unnecessary.29 held were last checked at22:54; hourly monitor2086813 ticked23:00:54 with0/17 terminal and remains live. Ownership/ReqNodeList/native-isolation gates remain independent and closed; no submit/release/cancel was attempted.

Persist the reusable comparison and evidence-location lessons in meta metrics. Next available no-GPU follow-up is a narrowly specified durable official-log preservation design; any implementation must keep current frozen attempts/results untouched. Do not dispatch another P5 reviewer or rerun these models merely because the causal question remains unresolved.
