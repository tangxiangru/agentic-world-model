# E2 retention proof: focused exit-evidence follow-up

Read-only audit of the two events used by `doc/spec/2026-09-03-exp-protocol-round02-e2-process-wait.md:7–16`. No candidate, spec, repository, runtime, git, evaluator, GPU, queue, or scheduler state was changed. This file is separate from the prior control audit. All event timestamps are 2026-09-03 UTC.

## Adjudication

**The frozen “two certain failures, therefore at most 6/8 pass” proof is conditional, not unconditional, under the requested ACTUAL-producing-process-exit metric.**

Both cells have convincing failed-training evidence and long clock-based waits. Neither cited event has a producer exit status or timestamped process-absence observation before its long sleep returns. The spec silently makes **GPU context gone ⇒ OS producer already exited** its “latest plausible exit” bound. The fatal traceback makes prompt exit a strong normal-execution inference; it does not turn that GPU observation into an unconditional latest-exit timestamp.

Consequently the **proof as written should be relabelled conditional; strict non-saturation remains unresolved from these two events alone**. This does not establish saturation, invalidate the E2 intervention, require a replacement candidate, or authorize submission/release. Other independently audited events could still establish non-saturation, but that would be a new supporting proof, not validation of these two claimed unconditional bounds.

## Exact source map

Repository root `R`:

`/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator`

Bundle root `B`:

`/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2`

- `T3 = B/g01s03/solve_parsed.txt.gz`; trace lines refer to original decompressed numbering.
- `T8 = B/g01s08/solve_parsed.txt.gz`.
- `M3 = /home/robtang_google_com/gangda_workspace/agentic-world-model-gangda-exp-protocol-evolve/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2_g01s03_formal_r2/gsm8k_google_gemma-3-4b-pt_90793/task/system_monitor.log`
- `M8 = /home/robtang_google_com/gangda_workspace/agentic-world-model-gangda-exp-protocol-evolve/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2_g01s08_formal_r2/gsm8k_google_gemma-3-4b-pt_90798/task/system_monitor.log`

The raw result directories above are the current paths in `B/g01s03/status.json:8` and `B/g01s08/status.json:8`. Older helper `launch.json` records use a historical checkout path; this audit read the current status-resolved paths.

## Event table

| Evidence | g01s03: exp-05 initial save failure | g01s08: exp-02 OOM |
|---|---|---|
| Detached producer launch | **14:49:12**, `setsid nohup python scripts/train_sft.py ... --save-steps 780 > logs/exp-05_train.log ... & echo launched; sleep 200; ...`, T3:6346–6351. No captured `$!`, `wait`, return-code file, or exit-time file. | **10:30:34**, `setsid nohup ... python train_sft.py ... --out ckpts/exp-02 ... & echo launched; sleep 120; ...`, T8:3729–3734. No producer PID wait or exit receipt. |
| What launch completion means | Tool returns **14:52:32**, while training is only step 57/1562, T3:6376–6386. It is **launcher/monitor-shell completion**, not producer exit. | Tool returns **10:32:34**, while training is step 15/4931, T8:3759–3775. Again not producer exit. |
| Last directly observed producing process | M3:4561–4565, **15:20:37**: Python PID **31551**, GPU utilization 76%, 42982 MiB process allocation. | M8:586–590, **10:33:51**: Python PID **2841**, GPU utilization 94%, 76792 MiB allocation. The original proof overlooks this snapshot when giving 10:32:51 as the early edge. |
| First no-CUDA-process observation | M3:4575–4578, **15:21:37**, 0 MiB and empty GPU-process list. | M8:600–603, **10:34:51**, 0 MiB and empty GPU-process list. Correct release bracket is **10:33:51–10:34:51**, not 10:32:51–10:34:51. |
| Fatal/output evidence | At **15:47:56**, T3:6446–6474 shows partial `checkpoint-780/` with only `config.json` timestamped **15:21** and failure through `Trainer._save_checkpoint` → `save_pretrained` → `GenerationConfig.save_pretrained` raising `ValueError`. Final bar at step780/1562 `[31:18]`, T3:6434. | `B/g01s08/task/logs/exp-02_train.log:14–48`: uncaught module→main→`trainer.train()` traceback, CUDA OOM at line47, then final tqdm refresh at step100 `[02:36]`. T8:4030–4038,4049–4060,4062–4257 show discovery at **10:48:54–10:49:03**. |
| Long wait with no productive overlap | `sleep 3300`, requested **14:52:39**, T3:6395–6400; shell-start event **14:52:42**, T3:6412; returned observation **15:47:39**, T3:6425–6437. No intervening useful tool action in this block. | `sleep 900`, requested **10:33:54**, T8:3997–4002; shell-start event **10:33:57**, T8:4014; returned observation **10:48:54**, T8:4027–4040. Prior script generation ends 10:33:50, T8:3993–3995, outside the claimed interval beginning 10:34:51. |
| Actual producer absence/status evidence | **15:48:15**, T3:6483–6493: `ps aux | grep -c "[t]rain_sft"` returns **0**. This is useful actual-process absence evidence, but it occurs **36 seconds after** the proof's first-returned-observation endpoint. No earlier `wait`/PID check/exit record is present. | The immediate **10:49:03** check is **only `nvidia-smi`**, T8:4062–4064,4252–4257; it does not query CPU processes. First later relevant name-query found is **13:06:59**, T8:5777–5784: only its own diagnostic shell matches `train_sft.py`. A targeted kill of a different `sft_fewshot` run precedes it, T8:5765–5775. No pre-10:48:54 producing-PID absence or exit status is recorded. |

Both monitors list only GPU processes. Their `[CPU]` section is a shared-node uptime/load average, not per-producer CPU state. A CUDA context may be freed during fatal unwinding before the interpreter, workers, or cleanup fully terminate. Neither trace proves that such a delay happened; equally neither provides a hard upper limit on it. With two DataLoader workers (`g01s03/task/scripts/train_sft.py:166`, `g01s08/task/train_sft.py:133`), the retained source does not establish a finite bound on runtime/interpreter teardown. No exception-recovery loop or productive continuation is visible; that is why prompt exit is plausible but still an assumption for a strict lower bound.

## Quantitative correction

Let `e` be actual producer exit, `r` the first returned informative observation, and `I = max(0, r − e)` for this isolated wait. The low-memory sample `m` does not by itself prove `e ≤ m`.

| Cell | Published inference | Arithmetic if exit-by-first-low-snapshot is assumed | What the retained direct evidence proves without that assumption |
|---|---|---:|---|
| g01s03 | “latest plausible exit 15:21:37” | `(15:47:39 − 15:21:37) = 1562 s = 0.433889 h`, hence the rounded-down **≥0.433 h** | For the selected event, **0 ≤ I ≤ 1622 s = 0.450556 h**, using last direct producer presence 15:20:37. Absence is only established by 15:48:15, after `r`; therefore it yields no positive lower bound before `r`. Partial config mtime 15:21 can tighten the ceiling to 1599 s (0.444167 h), but cannot raise the zero lower bound. |
| g01s08 | “latest plausible exit 10:34:51” | `(10:48:54 − 10:34:51) = 843 s = 0.234167 h`, hence rounded-down **≥0.234 h** | **0 ≤ I ≤ 903 s = 0.250833 h**, using actual producer presence 10:33:51. There is no actual-process absence check by `r`. This upper ceiling slightly precedes the sleep's requested start and intentionally overincludes at most a few seconds; it is conservative. |

These are selected-event bounds, **not whole-cell cumulative bounds**. The disputed labels are “certain,” “latest plausible exit,” and “conservative post-exit lower bound,” not the subtraction arithmetic or existence of failed compute and waiting.

Exact exit timestamps are not strictly necessary to restore a proof: timestamped actual-producer absence before a sufficient deadline, or another justified exit upper bound, would suffice. For these events individually to establish failure of `<0.15 h`:

- g01s03 would need `e ≤ 15:38:39` (nine minutes before 15:47:39).
- g01s08 would need `e ≤ 10:39:54` (nine minutes before 10:48:54).

Equivalently, a separately justified bound of at most **17 min 02 s** of post-CUDA-release teardown for g01s03, and at most **5 min 03 s** for g01s08, would preserve the threshold failures. There is no such measured/bounded teardown evidence in the cited records. Adopting an explicit ordinary-prompt-teardown assumption is a legitimate *conditional analysis*, not a recovered observation. It must not be presented as a new hidden definition of ACTUAL process exit.

## Provenance pitfalls

- g01s03's retained `task/logs/exp-05_train.log` is the **successful retry**, not the failed first attempt: T3:6611–6616 relaunches at 15:48:50 with `> logs/exp-05_train.log`, overwriting it. Its retained tail has `train_runtime: 3779.5775` and `[saved] .../final`. The initial failure's surviving evidence is the trace excerpt and relock record (T3:6446–6493,6598–6609), not the current log's ending.
- Raw retained task file mtimes are copy-time values, not original producer timestamps: g01s03 log is **17:54:57**, g01s08 exp02 log **17:46:50**. Do not use them as exit observations. The original g01s03 partial checkpoint timestamp is available only from the trace `ls` output at T3:6452–6455.
- The `task_notification status: completed` events in these intervals belong to finite launch/sleep/log-inspection shell calls; their commands explicitly detach the trainer. They do not report the trainer's wait status.
- Later negative process queries establish absence when queried; they do not retroactively date exit to the first empty GPU snapshot.

## Implication for the frozen non-saturation proof

The logical implication **two proven cell failures ⇒ at most 6/8 pass** is correct. The current premises are not unconditional: these two event intervals both straddle 0.15 h under strict exit uncertainty. From these two events alone, the number of proven failing cells is **0**, with **2 strongly suspected/conditional failures**. That does **not** mean either cell passes: their complete post-exit sums have not been re-audited here, and other events may independently force failure.

Recommended main-agent adjudication, without changing the frozen intervention:

1. Preserve the old helper reports and proof as historical evidence; attach this correction.
2. Relabel the cited 0.433/0.234 h figures as **conditional lower bounds under prompt-exit-by-GPU-release assumptions**. Mark the unconditional “at most6/8” conclusion **not established by the cited events**.
3. Keep the strict non-saturation question unresolved pending stronger in-scope evidence or explicit main/user adjudication of an acceptable bounded-teardown inference. Do not automatically claim saturation or swap to G/P1, and do not automatically release E2.

Places propagating the premise (read-only references; no edits made): `R/doc/spec/2026-09-03-exp-protocol-round02-e2-process-wait.md:3,7–16`; `R/doc/exp_protocol_iterations/2026-09-03-round-02-e2-prelaunch.md:31,41`; `R/doc/exp_protocol_iterations/directions-ledger.md:25,65`; the source reports' E tables in `R/doc/exp_protocol_iterations/trace-reviews/round01-strict-guard-addendum/cells/g01s03.md` and `g01s08.md`. Their separate runtime/test/ownership claims are outside this correction and are neither affirmed nor changed.
