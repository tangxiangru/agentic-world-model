# Round 02 E retention and E v2 preparation — 2026-09-03

Status: **E retained by the predeclared non-saturation bound; E v2 built and frozen, not registered or submitted.** This does not pass the separate full strict-guard safety gate and does not authorize any release under the active ownership/placement failure.

## Retention proof, without waiting for irrelevant reviews

The saturation rule is ≥7/8 of exact cohort g01s01–g01s08 below 0.15 h cumulative post-exit idle per cell. Two certain failures make that impossible even if every other cell passes.

| strict cell | conservative single-event bound | evidence |
|---|---:|---|
| g01s03 | latest plausible exit 15:21:37 → first returned observation 15:47:39 = **≥0.433 h** | D save traceback and checkpoint timestamp 15:21 in uncompressed trace L6436–6481; a `sleep 3300` call spans 14:52:39–15:47:39; retained system monitor bounds GPU release to 15:20:37–15:21:37; immediate follow-up 15:48:15 confirms zero trainer processes |
| g01s08 | latest plausible exit 10:34:51 → first returned observation 10:48:54 = **≥0.234 h** | OOM at step100 in `task/logs/exp-02_train.log` L47; uncompressed trace L4000–4072 shows `sleep 900`, its 10:48:54 result and immediate OOM inspection; monitor bounds release to 10:32:51–10:34:51 |

The planner checked these trace/exception excerpts against the local Opus interval reports. The reviewer excluded overlapping productive work; neither conclusion depends on counting evaluation waits, per-event-vs-total ambiguity, or borderline rounding. A single event already exceeds the per-cell threshold in each case.

Therefore at most **6/8** can pass. **Keep E; do not replace it with G or P1 on the saturation branch.** Remaining trace reviews still matter to guard safety and candidate design, but cannot restore the ≥7/8 condition. Sources: `doc/exp_protocol_iterations/trace-reviews/round01-strict-guard-addendum/cells/g01s03.md` and `g01s08.md`, harvested in 66ebd39. The final three cells were harvested in 5619442; their reports are now preserved in 9e1818e. All eight reports feed the full-cohort synthesis session `79d04a31-ab3f-4fd5-9a20-4b3500f0ab52`; final guard safety remains undecided.

## Why prepare E v2

The existing E commit 7832cb9 already adds producing-PID polling. It nevertheless states that an unchanged tail means the run is dead. That implication is not sound: a live process can be silent, while a file, expected log string or leftover GPU allocation can outlive the producing process. The window03 decision recorded this risk before release.

One intervention, still the E direction: **wait on the producing process and its exit result, not elapsed clock time or an artifact proxy**. Update only the coordinated waiting guidance in:
- `skills/exp_protocol/SKILL.md` rule 9;
- `skills/exp_protocol/hooks/stop_open_cards.py` reason text, not hook logic/counters;
- `skills/exp_protocol/pitfalls.yaml: run_dies_with_the_session` guidance/source.

Prefer launch/wait/exit-status handling and subsequent evaluation in one foreground script. If backgrounding is necessary, track the actual producer rather than a launcher or residual engine; check process state and exit evidence. An unchanged log is a diagnostic warning, not a death verdict. Apply the same principle to evaluation as to training/sampling. Failed runs remain recorded and closed; no new training recipe, schema field, model or budget is introduced.

## Screen contract

Four new independent cells, same task/model/scientist/effort/context/hours/PTB and same six shipped infrastructure paths as the guard drift baseline 2f64581. The current operator HEAD's six shipped paths were compared with that baseline and match; recheck at construction time.

Primary: cumulative non-overlapping post-exit idle <0.15 h/cell, with all relevant producing processes (including evaluation) and uncertainty bounds. Per-event maximum and number of proxy-only liveness decisions are secondary. Report false death classifications, premature card closure and lost work as guardrail failures, not time savings.

Score guardrail remains the predeclared protocol-baseline pool mean −0.03; a winner earns a separately frozen second four-cell block. Neither this screen nor its repetition promotes a baseline without the held-out task.

## Construction and queue boundary

The baseline-relative single-item commit is **`c6f11d803b5f563cd25cf5fb373f5f3078028493`**, protocol tree **`ceb685494c6a44e0f717d42724b91696a88acbd6`**. Its same-commit [prelaunch record](../exp_protocol_iterations/2026-09-03-round-02-e2-prelaunch.md) contains the source-card reads, test results and independent forward review. The immutable replacement [manifest](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-e-wait-on-process-x4-v2.yaml) names batch `exp-protocol-gsm8k-gemma4b-high-r02-e-wait-on-process-x4-v2`, cells `e02s01–04`, run_index 2. No submission receipt exists yet.

The four existing hook/skill-files/install/lock CPU suites pass all 34 tests. AST/YAML and SKILL prefix/suffix checks prove changes are confined to the three allowed text surfaces. The six-path comparison with `2f64581` differs only there; after freezing, the branch source was restored to guard tree `189319d63d301d64d96f8f41d051795404679f37` and all six shipped paths again match `2f64581`.

Validation on 2026-09-03: `awm ptb check` on the new manifest returns **0 issues**, both `--local-only` and full site mode. Parsed old/new manifests match after excluding batch description, spec pointer, batch/cell/run identities and the frozen candidate SHA/tree. These checks establish structural and cross-variant consistency, **not ownership/release authorization**; the independent OWNERSHIP FAIL and isolation gates remain closed.

Jobs 91064–91067 remain held and must not run with the stale-tail assertion. Register a valid replacement held receipt before withdrawing the old whole block through the queue/operator path; no RUNNING job is cancelled. **Current OWNERSHIP FAIL forbids new submissions, including this replacement.** Code/spec preparation may proceed, but registration and release wait for restored ownership plus the native-isolation or explicitly authorized per-receipt gate. The held buffer remains ≥8 without this replacement.
