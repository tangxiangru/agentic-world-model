# HumanEval node runner independent review — 2026-09-05

Read-only source review of `/tmp/exp-protocol-nextwave-ckg3t0gc/repo/third_party/PostTrainBench`, plus its AWM `ptb_environment.py` consumer. No cluster, model, provider, sandbox, or training execution. Source was in active development; line references below describe the reviewed snapshot, not a claim about a later revision. Parent is already repairing execute() exception cleanup, so that known issue is not duplicated here.

## Original findings (preserved; see follow-up status below)

### 1. The two images do not exercise the actual scientist/evaluator home and working-directory layouts (P1)

`src/commit_utils/slurm/humaneval_environment_acceptance.py:124–138` constructs the same command for both images: a bind to `/home/ben`, no explicit `--home` and no `--pwd`. In contrast production `src/run_task.sh:349–382` uses `--home JOB_DIR:/home/ben --pwd /home/ben/task` for the scientist; official evaluation at `run_task.sh:645–674` uses `--home eval_home:${HOME}`, binds repo/EVAL_DIR/HF overlay, and runs from the task source directory. The probe moreover hardcodes its writable evidence root `/home/ben` (`humaneval_environment_probe.py:41`), so it cannot currently test the official evaluator's actual root without adaptation.

Consequence: success establishes bubblewrap works in **a** contained home on each image, not that each production layout can register/materialize its runtime and execute isolated code. Wrong HOME/TMPDIR/source-relative assumptions or home mount behavior in official evaluation can remain untested. Use explicit per-image layout construction aligned with production, preserving invented-only execution, and report actual HOME/cwd/source/data paths. The private program should test absence of those actual evaluator paths, not only `/opt/ptb-humaneval/evaluate.py`. A shared argument builder or focused argument-shape checks should prevent the acceptance layout drifting from production.

Necessary validation: native five-case acceptance in both explicitly declared layouts, with evidence-root writes surviving and generated code unable to see the real bound probe/evaluator/data roots. No provider or model call is needed.

### 2. A normal timeout case can pass as an ordinary program failure (P2)

`humaneval_environment_probe.py:89–97` checks only scores C/I, lack of sample error, and execution count. `invented-timeout` therefore satisfies acceptance if it exits with a program error before sleeping, as long as the backend records an admitted clean `program_failure`. The shared official evidence validator intentionally accepts both program failures and timeouts as I; it does not know which synthetic ID was designed to prove a timeout. The consumer repeats this weakened predicate (`awm/ptb_environment.py:153–162`).

Consequence: the five-case pass does not prove the native wall-time timeout branch ran. Extract per-case execution outcome/error_category, require `invented-timeout` => `timeout` plus `wall_timeout`, good/private => success, wrong/import => program_failure, and propagate those assertions to readiness validation. Preserve actual exit/error evidence where useful. This is a distinct check from sample score mapping.

Necessary validation: a deliberately substituted admitted program_failure record for invented-timeout must be rejected, even though its score is I and generic native publication is valid. A real native timeout must still pass.

### 3. Outer-timeout admission does not prove an admitted program was active at the deadline (P2)

`humaneval_environment_acceptance.py:94–95,106–120` accepts the conjunction “at some time saw a command containing --unshare-all/--as-pid-1”, GNU timeout exit124/137, and no surviving captured handles. The timer starts before Apptainer/Python/runtime materialization. `probe-started.json` (`humaneval_environment_probe.py:77–78`) is before `inspect_eval` and only establishes backend registration; it is not generated-code admission, and the runner does not check it anyway.

Consequence: if a native child runs and exits, followed by a hang elsewhere in evaluation until the outer timer expires, the same predicate passes. Conversely slow startup may consume the30-second deadline before the intended admitted sleep, yielding a failure that says little about cleanup of running generated code. The intended test is specifically interruption of an admitted long-running sandbox computation. Bind a durable start/ack or equivalent trusted supervision event to the synthetic attempt, its exact process handle and deadline; demonstrate that handle was live when the outer signal fired, then terminal afterward. Record elapsed/start/signal ordering. Keep hard total startup/runtime bounds and exact owned cleanup.

Necessary validation: “sandbox briefly appears then exits before parent stalls” must fail outer-timeout admission; real admitted long sleep killed by the outer deadline must pass with exact terminal handles. Exit124/137 alone is insufficient.

## Positive checks and limits

- Dispatch branches before preflight/context probing or `run_task.sh`, so the inspected path does not invoke a scientist/provider. Probe uses native `verify()` with `mockllm/model` and invented programs; actual HumanEval parquet is read only for bytes/metadata, not evaluated.
- Both images are required by exact name and frozen SHA256. Dataset hash/size/row count and helper/bwrap source hashes are checked. Frozen PTB source is materialized by `git archive` before the runner; other task modules are therefore commit-bound even though the runner only separately hashes its three listed sources.
- AWM readiness validates receipt membership, job/cell/source/non-root identity and node coverage; launch placement checks frozen requested-node membership. Those bindings are meaningful within the supported launcher path. Direct execution outside that path is not an admission mechanism.
- GNU timeout is a new process session. Emergency signalling in the reviewed normal path uses pidfds for captured descendants, and a survivor makes acceptance fail rather than silently repairing it into success. Parent's in-progress exception/finally cleanup still requires verification after implementation.
- Success/failure `acceptance.json` is atomically replaced/fsynced after entering the runner's try block, and logs are outside disposable scratch. SIGTERM/execute exception behavior and final raw-file coverage must be reassessed against the final implementation rather than inferred from the current happy path.

No source changes were made. The three findings above are tied directly to the requested native five-case, outer-timeout and production-layout acceptance claims.


## Follow-up review of first remediation

Status: **not ready yet**. Source-only reinspection, no tests or runtime executed by this reviewer.

1. **Original layout finding substantially addressed in producer.** `image_command` now constructs separate scientist `--home ...:/home/ben --pwd /home/ben/task` and evaluator `--home ...:hostHome --pwd frozen-source/.../humaneval` layouts. Probe records actual HOME/cwd and runner compares them. Public fixture binds are explicitly scoped; this establishes a native environment-layout test, not a full provider/model lifecycle test. The generated private program still checks `/opt` sentinel paths rather than the additional actual source/output paths; that is a narrower assertion, but the new layout construction resolves the original mismatched-home/cwd defect.

2. **Original normal-timeout finding addressed in producer, not yet in consumer.** `validate_rows` now requires `invented-timeout` execution `outcome=timeout,error_category=wall_timeout` and started/cleanup checks, followed by shared native publication validation. However `awm/ptb_environment.py:153–170` still checks the old C/I/count flags only. It does not require the newly exported outcome/started/cleanup fields, layout fields, or outer admission/live/alarm fields. Readiness validation should reject missing/regressed fields rather than merely trust a report-level `passed` flag. Update the positive fixture and negative tests accordingly.

3. **New P1 blocker: the admission observer is installed in the wrong process.** `humaneval_environment_probe.py:install_admission_observer` patches `_OutputCapture.feed` on the module loaded inside the evaluator process. Native `execute_python` at `ptb_python_sandbox.py:1141–1149` starts a fresh Python `-I -S <helper> --supervise` subprocess with a clean environment. `_OutputCapture` is created/used inside `_execute_python` in **that new subprocess** (helper lines813,951); the fresh interpreter never imports the probe's monkeypatch. Consequently the real marker is captured there, but the probe-installed observer never sees it and cannot create `outer-admitted.json`. The runner cannot arm its admission-based alarm and the intended real outer-timeout acceptance fails. Calling the patched capture directly in a CPU unit test would not exercise this process boundary.

The new alarm predicate itself now requires readiness, live captured sandbox, exact GNU-timeout pidfd SIGALRM, and terminal observed handles. That is materially stronger than the initial version, but remains unreachable through the actual native supervision path until the observer runs where stdout is really consumed or equivalent genuine admission evidence is obtained externally. Do not replace the real nested supervisor with a mock to claim this fixed.

Necessary validation of the next revision: native `register_backend` → Inspect verify → `execute_python` → fresh `--supervise` path must create admission evidence while the invented sleep remains live, then the exact outer alarm must interrupt it. Also test absence of marker/startup-only failure is rejected. This can use invented programs and no model/provider, but it must cross the real subprocess boundary.

Exception cleanup review: current execute now tracks `(pid,start_ticks)` identities, observes the created timeout root immediately, and has BaseException/finally cleanup with pidfd signalling and fd closure. This addresses the previous obvious missing-finally problem. Parent's in-progress real process tests remain the runtime evidence; this review does not claim they passed.
