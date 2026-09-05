# Window04 focused control timing audit (read-only supplement)

Scope: c01s01/c01s02/c01s03 only. The original helper reports are unchanged. No models, evaluators, network, scheduler, or git mutation were used. All times below are UTC, September 3–4, 2026. This audit makes no score-return inference and no E screen pass/fail decision.

## Result

1. The reported low-memory numbers reproduce, but their formula is **sum(last low-memory sample timestamp − first low-memory sample timestamp) over low-memory runs**. It discards every singleton sample. They are endpoint-span statistics, not measured cumulative low-memory duration, not post-exit idle, and not automatically lower/upper bounds on either duration.
2. c01s01's alleged 0.13 h post-exit RS idle is not directly established: generation is followed by CPU filtering and output writing, and the producing Python PID has no contemporaneous exit record. The selected transition has a conservative **0–0.144 h** post-exit ceiling, not a proven 0.13 h idle cost.
3. c01s02's pgrep self-match is established. The long third wait is **1.99917 h of uninterrupted tool waiting after an uncaught OOM had been logged**. Approximately two hours of post-failure idle is a strong inference; **2.00 h is not an unconditional lower bound on the stricter ACTUAL-process-exit metric**, because the failure log's mtime and CUDA-context disappearance do not timestamp OS exit. The three published 0.50/0.30/2.07 h events are not three clean post-exit terms that may be summed to 2.87 h.
4. c01s03's artifact polls have short observed completion-to-next-action transitions. A conservative ceiling for its **five audited long-producer transitions only** is **0.099 h**. That is not a whole-cell sum; neither “near-zero whole-cell idle” nor “comfortably passes E” follows from the 0.10 h monitor endpoint statistic.

## Exact source map and notation

`R = /home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator`

`B = R/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1`

`T1 = B/c01s01/solve_parsed.txt.gz`, `T2 = B/c01s02/solve_parsed.txt.gz`, `T3 = B/c01s03/solve_parsed.txt.gz`. A trace citation uses decompressed original line numbers, without renumbering after signature filtering.

The exact raw result directories, read from each `B/<cell>/status.json:8`, are:

- c01s01: `/home/robtang_google_com/gangda_workspace/agentic-world-model-gangda-exp-protocol-evolve/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1_c01s01_formal_r4/gsm8k_google_gemma-3-4b-pt_90813`
- c01s02: `/home/robtang_google_com/gangda_workspace/agentic-world-model-gangda-exp-protocol-evolve/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1_c01s02_formal_r4/gsm8k_google_gemma-3-4b-pt_90814`
- c01s03: `/home/robtang_google_com/gangda_workspace/agentic-world-model-gangda-exp-protocol-evolve/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1_c01s03_formal_r4/gsm8k_google_gemma-3-4b-pt_90815`

`M1`, `M2`, `M3` mean `<corresponding exact raw result directory>/task/system_monitor.log`. These contain GPU-process listings, **not an all-process CPU/PID census**; their CPU load averages are node-wide and cannot establish a given producer's liveness.

Important provenance limitation: retained raw `task` file mtimes were replaced during copying. For example c01s01 `task/logs/rs1.log` is dated 03:17:23, c01s02 `task/logs/sft2.log` 03:09:21, and c01s03 `task/logs/gen_rft.log` 02:56:11, all after their respective sessions. Those mtimes are unusable as producer end times. In-task `ls` results preserved in traces supply original minute-resolution mtimes instead.

## Monitor arithmetic, separately from elapsed time

Threshold reproduced: GPU `memory.used < 2048 MiB`. The `task/system_monitor.log` records nominal 60-second snapshots; actual gaps sometimes span 61 seconds.

| Cell | Low runs / low samples | Reproduced sum of endpoint spans | Merely `60 s × sample count` | What is supported |
|---|---:|---:|---:|---|
| c01s01 | 24 / 43 | 0.31722 h | 0.71667 h | Report's rounded 0.32 is the first formula, not cumulative GPU-free time. |
| c01s02 | 20 / 203 | 3.05722 h | 3.38333 h | Report's rounded 3.06 is the first formula. |
| c01s03 | 15 / 21 | 0.10000 h | 0.35000 h | Report's 0.10 is the first formula; it is not an upper bound on post-exit idle. |

Neither column estimates actual continuous duration without an interpolation convention; neither proves post-exit time. Unobserved sub-minute transitions and CPU work remain unresolved. The count-derived column is shown only to expose the lost singleton intervals, **not as a replacement metric**.

The claimed large events reproduce exactly as low-snapshot endpoint spans:

| Cell/event | Monitor lines | First/last low snapshot | Endpoint span |
|---|---|---|---:|
| c01s01 RS | M1:1696–1800 | 20:33:33–20:41:34 | 481 s = 0.13361 h |
| c01s02 SFT1 | M2:1172–1562 | 19:55:27–20:25:31 | 1804 s = 0.50111 h |
| c01s02 RFT | M2:2162–2396 | 21:08:37–21:26:40 | 1083 s = 0.30083 h |
| c01s02 failed SFT2 | M2:2451–4063 | 21:30:41–23:35:00 | 7459 s = 2.07194 h |

These are samples of low allocation. Notably several endpoints are **after productive actions have resumed**.

## Evidence and corrected bounds: c01s01

| Event boundary | Evidence |
|---|---|
| Launch | T1:3956–3991: `nohup python gen_rs.py ...`, requested 20:15:44; producing PID **7574**. Engine PID **7629** is a different process. |
| Generation, not exit | `B/c01s01/task/logs/rs1.log:49–51`: supported-tasks line 20:17:03, then request addition and generation `[15:31]`. Therefore completed generation cannot precede **20:32:34** (this intentionally ignores request-addition duration, giving an early/safe ceiling origin). T1:4037–4045 shows the completed generation and `solved ... kept ...` output at 20:40:53. |
| CPU tail/output write | `B/c01s01/task/gen_rs.py:65–94`: `llm.generate`, then CPU answer extraction/deduplication, JSONL write, and final summary print. Output must exist by **20:40:53**. Its exact original write time is not preserved. |
| GPU disappearance | M1:1682–1708: engine 7629 has 66% GPU utilization and 68434 MiB at 20:32:33; no GPU process at 20:33:33. This is not PID 7574 exit evidence. |
| Wait and next useful work | T1:4004–4045: fixed `sleep 1400` at 20:17:33; completion read 20:40:53. T1:4054–4105: dataset-building action **20:41:09–20:41:14**. T1:4107–4145: contamination check 20:41:18–20:41:22. Next train launch 20:41:26, T1:4148–4153. |

Strict interval for post-exit idle in this **selected RS-to-data transition**: **0 to (20:41:09 − 20:32:34) = 515 s = 0.14306 h**, rounded upward to 0.144 h. If the successful completion-log read itself counts as the next useful action, the ceiling is 499 s = 0.13861 h. Both are conservative ceilings, not measured idle.

The zero lower bound expresses missing producer-exit timing, not evidence that the wait was efficient. The likely operational story is that the sleep overshot a completed sampler by several minutes; the exact 0.13 h and a positive strict lower bound are not demonstrated. Do **not** include 20:41:09 onward: dataset construction/checking is useful CPU work even though the monitor stays low through 20:41:34. The session's other small transitions were not cumulatively adjudicated here; no whole-cell E conclusion follows.

## Evidence and corrected bounds: c01s02

| Event | Launch/end/output/next-action evidence | Correct quantitative conclusion |
|---|---|---|
| SFT1 pgrep wait | Launch 18:46:25, T2:3202. `B/c01s02/task/logs/sft1.log` and T2:3768–3772 show 3986.1093 s **training** followed by saving; `train_sft.py:190–199` explicitly saves after `trainer.train()`. Original files were written during **19:54**, T2:3787–3805. Actual producing Python PID **2777** still exists in M2:1158–1171 at **19:54:26**, 0% GPU with high allocation; that can be saving/cleanup, not idle. M2:1172 onward shows no GPU process starting 19:55:27. Wait T2:3702–3751 runs 18:54:56–20:24:57, then backgrounds. Useful completion inspection **20:25:02**, packaging **20:25:08**, T2:3760–3810; eval launches 20:25:34. | Exact OS exit unrecorded. For transition to packaging, **0–1842 s = 0.51167 h** (19:54:26→20:25:08) is a safe ceiling; to completion inspection, 0–1836 s = 0.51000 h. The report's 0.50 h approximates the sampled low-allocation gap, not a proven strict lower bound. Exclude packaging from 20:25:08 onward. |
| RFT pgrep waits | Launch **20:28:05**, T2:3978. `B/c01s02/task/logs/rft1.log:50–55`: 20:29:32 initialization marker, generation `[38:08]`, CPU-derived kept/pass-rate summaries, final `wrote`. Thus generation cannot finish before **21:07:40**, and trace `ls` establishes output + stats writes in **21:07**, T2:4304–4317. `gen_rft.py:117–166` has real CPU filtering/sorting/output work after generation. M2:2148–2174: engine 5595 active 21:07:37, absent 21:08:37; not the producing frontend's exit. Waits timeout 21:14:12 and 21:24:22, T2:4174–4191,4256–4273. T2:4275–4317 contains **sleep 30**, then useful output inspection at **21:24:56**. Data analysis **21:25:21**, mix creation **21:25:48–21:26**, T2:4326–4490. Train **21:26:21**, T2:4644–4649. | Strict producer exit unrecorded. **0–1036 s = 0.28778 h** to completion inspection, or **0–1061 s = 0.29472 h** to data analysis, are conservative ceilings. The 0.30 h low-memory span extends through actual data work and beyond the next train launch; it is not a clean idle term. Intermediate polling at 21:14:16/19 is not counted as productive data work, making these ceilings conservative. |
| Failed SFT2 and long pgrep wait | Launch **21:26:21**, T2:4644–4649. PID **7590** performs GPU compute at **21:29:41**, M2:2437–2450. T2:4722–4725 still sees step 40 at **21:30:01**. Failure at step 50 is an **uncaught** OOM through module→main→trainer, `B/c01s02/task/logs/sft2.log` (CR-normalized lines 106–147), T2:5174–5215; final tqdm refresh follows traceback. Original log mtime **21:30**, T2:5157. No CUDA allocation from 21:30:41. No contemporaneous all-process exit status. At **23:34:39**, `pgrep -af train_sft.py | head -3` lists wrapper PIDs **3101,7962,11093**, not producer 7590 (T2:5148–5161); fatal traceback then read at **23:34:44**, T2:5170–5215. Safety artifact copied **23:35:02–23:35:07**, restart **23:35:11**, T2:5230–5281. | Failure and self-match are real; exact exit time remains interval-censored. The uninterrupted wait **21:34:34→23:34:31 = 7197 s = 1.99917 h** is confirmed tool waiting (T2:5082,5112,5127–5129). Its post-exit component has strict bounds **0–1.99917 h**, unless an explicit normal-teardown assumption is adopted. Approximately 2 h post-exit idle is a strong inference, **not a proved unconditional 2.00 h lower bound**. |

Additional non-overlap correction for failed SFT2:

- The original report mentions only two Write calls, but there are **three**: `train_grpo.py` 21:32:03 (T2:4734–4894), `merge_lora.py` 21:34:14 (T2:4938–5005), and `run_eval.sh` 21:34:28 (T2:5041–5063). There is also useful retained-eval analysis at **21:34:17** (T2:5007–5032). Scientist generation of those scripts is productive work, not merely their instantaneous Write tool time.
- A separate pure sleep **21:32:09→21:34:01 = 112 s**, T2:4909–4929, also follows the timestamped failure. Together the two clearly isolated wait segments total **7309 s = 2.03028 h**, with strict post-exit component **0–2.03028 h**; calling that an observed idle lower bound would still require exit timing.
- A broad ceiling for the failed-producer gap up to the first informative process check is **21:30:00→23:34:39 = 7479 s = 2.07750 h** (failure log written during 21:30, so exit cannot precede 21:30:00). That ceiling intentionally overincludes productive intervals; use the isolated wait segments when excluding overlap. It is not an estimate of lost time.
- Under the explicitly stated, plausible assumption that the uncaught OOM terminated PID 7590 before 21:32:09, the two isolated waits yield a **conditional** 2.03028 h non-overlapping idle lower bound. That assumption is stronger than the retained process observations prove.

Self-match evidence is not speculative: T2:5154–5156 shows the literal pgrep pattern in each shell command line; T2:7124–7128 at 03:06:59 shows all four stale wrappers (including the two RFT wrappers 6465/7225). The later repair tracks actual PID 11137 using `echo $!` and `kill -0`, T2:5276–5326. These prove the waiting defect, not the exact early process exit second. No scientifically meaningful 2.87 h per-cell sum is established by adding the three monitor spans. A coarse upper ceiling for just these three selected gaps is 0.51167 + 0.29472 + 2.07750 = **2.88389 h**, before removing productive overlap; it is **not a whole-cell total or measured loss**.

## Evidence and corrected bounds: c01s03

`B/c01s03/task/train_sft.py:179–190` saves model/index and then tokenizer/template metadata. The index may appear **before the producer exits**, so file existence is neither an exit status nor proof of complete serialization. `gen_rft.py:92–127` likewise performs CPU answer filtering, sorting, and JSONL output after sampling. All five retained waits completed successfully; that does not make their primitive a PID check.

| Long producer | Launch / productive end and output evidence | Poll return and next useful action | Conservative selected-transition post-exit bound |
|---|---|---|---:|
| SFT1 | Launch **18:43:39**, T3:2642–2645. 4270.4379 s trainer runtime, T3:4616. Original final/index/tokenizer mtimes all **19:55**, T3:4619–4631. Output and process exit cannot precede 19:55:00. | Poll 60 s + extra 30 s, T3:4348–4353. Full completion output **19:56:38**, T3:4607–4631. Packaging/eval action **19:56:44**, T3:4636–4639. | **0–104 s** |
| RFT | Launch **20:00:42**, T3:4732–4734. `logs/gen_rft.log:44–49`: supported-tasks **20:02:01**, generation **2:16:52**, then stats/write. Earliest generation completion **22:18:53**; output complete by **22:19:40**, T3:5155–5164. Exact write/exit second absent. | Poll 120 s, T3:5122–5127. Return **22:19:40** (not 22:20:01), T3:5152–5166. Next data analysis **22:20:01**, T3:5175–5177. | **0–68 s** |
| SFT2 | Launch **22:21:08**, T3:5328–5331. 9577.8556 s trainer runtime and `saved`, T3:5516–5517. Actual producer **9704** still computing at **01:01:13**, M3:5438–5451. | Poll 120 s +20 s, T3:5481–5486. Output returned **01:02:05**, T3:5511–5519; index therefore present by approximately 01:01:45 (bounded only by sleep, not exact write timestamp). Packaging/eval **01:02:11**, T3:5521–5524. | **0–58 s** |
| Polish | Composite command begins **01:04:28** but packages incumbent before `nohup`, T3:5567–5571, so not an exact training launch second. 2155.7784 s trainer runtime, T3:5639. Actual producer **13312** still computing **01:40:19**, M3:5982–5995. | Poll 60 s +20 s, T3:5604–5609. Output returned **01:41:37**, T3:5634–5642; index present by approximately 01:41:17. Packaging/eval **01:41:51**, T3:5644–5647. | **0–92 s** |
| Broad pass / ep2 | Composite data-build + launch **01:56:56**, T3:5977–5989, not exact producer launch second. 2760.636 s trainer runtime, T3:6057. Actual producer **17214** still present at **02:43:29**, M3:6861–6874, even though GPU utilization is 0%. Artifact has original **02:43** mtimes, T3:6289–6301. | Poll 60 s +20 s, T3:6022–6027. Output returned **02:43:53**, T3:6052–6061; index present by approximately 02:43:33. Packaging/eval **02:44:03**, T3:6063–6066. | **0–34 s** |

These non-overlapping selected transition ceilings sum to **356 s = 0.09889 h**. They derive from producing-process presence or necessary output/generation chronology, not low-memory interpolation. Each zero lower bound acknowledges that the actual exit could have followed file appearance and even overlapped the next action. With next-useful-action defined as the informative completed log read, the ceilings are smaller; the table conservatively uses the next downstream action.

The 18:55:43–19:08:07 wait-primitive churn is correctly excluded from idle: production SFT1 was still running. Its background sleep completions are not training-exit notifications (T3:4418,4469,4515,4556,4591). In the 22:19:48–22:20:48 low-memory span, data analysis/rebuild from 22:20:01 onward is useful CPU work, not sampling idle.

The five transitions are **not every cell interval**: shorter smoke/evaluation/artifact/failed-compute transitions and post-evaluation waits would have to be separately timed and summed for an E primary-metric decision. Thus the full-cell near-zero/pass assertion should be withheld. The narrower statement “the five successful long-producer artifact waits show no large overshoot; combined post-exit contribution is at most about six minutes under the stated chronology” is supported.

## Bottom line for synthesis

Retain c01s02 as strong mechanistic pgrep-self-match evidence. Relabel its two-hour observation as an exact pure-wait duration and an explicitly inferred post-exit loss unless an actual exit record is recovered. Correct monitor numbers to their stated endpoint-span formula, preserve low-sample counts, and do not substitute a one-minute-per-sample estimate. Treat c01s01 and c01s03 as partially interval-censored timing evidence, not E passes inferred from GPU occupancy. No score improvement is inferred from any returned time.
