I've finished the read-only review. No files were written and nothing in the repo, queue, or cluster was touched.

<!-- REPORT cell=p00s01 role=NEW -->

```yaml
cell: p00s01
arm: protocol (baseline_v3; manifest exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8.yaml, awm eaf5091 / protocol_tree 08674f2c, setup "--exp-protocol --tool claude")
accuracy: 0.7702805155420773      # official, 1016/1319, stderr 0.011587
hours_used: 7.89                   # time_taken.txt 07:53:26; wall 2026-09-03T19:02:22Z -> 2026-09-04T02:54:58Z
hours_to_first_train_launch: 0.44  # card-matched exp-02 SFT L4312 19:28:59Z; first GPU training process (12-step smoke) L3571 19:13:59Z = 0.19 h
protocol_hours: 0.16               # new->lock spans + closes + bootstrap read; the timeline tool's 0.01 h is awm exec time only
waiting_hours: 6.72                # raw timeline span in sleep/tail composites; post-exit idle is 0.54 h (certain) + <=0.12 h (bounded), see Q5
greedy_shipped: yes                # final_model/generation_config.json do_sample:false, temperature:0.0 (raw result_dir, mtime 02:55)
rl_used: no                        # 0 on-policy RL launches; "grpo|dpo|ppo" appears only in the shipped example-card (L406, 19:02:25Z)
rft_tried: yes (self-sample 3-attempt+vote SFT; exp-04 inconclusive +1.1pt@800 p=0.41, exp-06 contradicted -1.5pt@800, -1.2pt@n=500)
largest_eval_n: 500                # verified from raw inspect logs, not file size; three n=500 runs, total==completed==500
stop_reason: "Done. All processes have finished; the GPU is idle and `final_model/` is in place and verified." (L9431, 02:54:58Z) — stopped with 2:07 on the timer (L9262, 02:54:18Z)
top_contributors: [exp-02 completion-only SFT with <end_of_turn>+single ANSWER line 0.06->0.7267 (format compliance 0.24->0.96), exp-05 second 110k-solution data round (+2.6 pt on held-out 800, McNemar p=0.04 — the only replicated positive), exp-07 n=500 tie-break that reversed the dev-150 ranking and stopped the 1.2-pt-worse exp-06 from shipping]
one_protocol_change: "greedy-parent Trainer-save guard (ledger #14 / Round-02 candidate D, already frozen at 8332917, held 91060-91063) — would have returned 0.25 h of failed compute + 0.55 h of post-exit idle at exp-06's lock (L8257, 00:30:47Z). Covered; not a new proposal."
knowledge_to_transfer: [n/a - protocol arm]
```

# p00s01 — protocol arm, official 0.7703

Session `1f9fd2b6-…`, 19:02:22Z → 02:54:58Z, 246 assistant turns, 9464 trace lines, 165 tool calls (132 Bash), 7 cards all locked and closed, 0 overrides, 0 relocks. `L` = line in `solve_parsed.txt` (byte-identical line count to the bundle's `solve_parsed.txt.gz`, 9464 lines both).

Judge: `judgement_general.json` `general_anomaly=false`, `judge_flags: []`, `complete=true`, `strict_site=true` (status.json).

## 1. Timeline, and what the prepared markers actually matched

| when | what | evidence |
|---|---|---|
| 19:02:22 | bootstrap: `Skill exp_protocol`, then Read `pitfalls.yaml` | L11 (Skill), L1** (Read) at 19:02:24 |
| 19:02:27–19:04:06 | env survey: `timer.sh` (9:59 left), `awm exp_protocol index`, evaluate.py, templates, inspect_evals gsm8k source, `_match`/`_common` scorers | L501, L522, L1107, L1370, L1434 |
| 19:04:37–19:05:20 | exp-01 card: `new` → Write → `check` (ERROR `setup.data[0].n_examples: must be a positive integer`, L1779) → Edit → `check && lock` | L1559, L1664, L1773, L1789, L1806, lock L1828 |
| **19:05:29** | **first real eval launch**: base snapshot, `--limit 150` | L1868 |
| 19:06:44 | reads the base `generation_config.json` (`do_sample true, top_k 64, top_p 0.95`) — decode awareness, 3 min in | L2099 |
| 19:08:41 | 120k-row OMI2 build launched detached | L2760 |
| 19:10:38 | base = **0.06** (n=150); 67/150 hit the 4000-token cap | L3035, card exp-01 `result.diagnostic_result` |
| 19:13:59 / 19:15:17 / 19:17:25 | **GPU training smoke/bench before any training card exists**: 12-step smoke to `/tmp/smoke1`, then two `train_sft.py --max-steps` benchmark loops | L3571, L3646, L3696 |
| 19:26:58–19:28:35 | exp-02 card: `new` → Write → re-Write → `check` → `lock` | L3892, L3902, L4056, L4194, lock L4225 |
| **19:28:59** | **first card-matched training launch** (exp-02 SFT, 120k rows) — 0.44 h in | L4312 |
| 20:55:57 | exp-02 done (`train_runtime 4768.2 s`, L5089), eval launched | L5089, L5096 |
| 20:59:27–21:08:54 | exp-02 = 0.7267; exp-03 greedy card, eval, rerun, close | L5169, L5328, L5613, L5693, close L5962 |
| 21:09:00 | own held-out evaluator: three `dev_eval.py --n 800` runs (greedy fs10 / greedy fs0 / maj@8 T=0.8) | L5975–L5977 |
| 21:19:55–21:46:14 | vote data build (18k problems × k=4), exp-04 card, lock, launch | L6479, L6514, lock L6811, launch L6814 |
| 22:19:32–22:29 | exp-04 done, greedy patched in, dev-800 + dev-150 evals, round-2 data rebuild | L6949, L7067, L7293 |
| 22:37:30 / 22:37:36 | exp-05 lock, then launch 6 s later | L7574, L7577 |
| 23:55:35 | exp-05 done; greedy written into `ckpts/exp-05/final` (the parent-poisoning write) | L7824 |
| 00:04:51–00:30:52 | vote resample from exp-05, exp-06 card, lock 00:30:47, launch 00:30:52 | L7935, L7997, L8257, L8260 |
| **00:46:04 → 01:19:13** | **exp-06 dies at step 150/306 in `Trainer._save_checkpoint`; unnoticed for 33 min** | ckpt dir mtime `Sep 4 00:46` + `150/306 [15:12<15:49]` (L8404-ish, printed 01:18:42); relaunch L8466 |
| 01:56:15–02:04:21 | exp-06 dev-800 + dev-150 (0.7867), close | L8581, L8626, L8698 |
| 02:04:56 / 02:05:00 | exp-07 lock, then `run_final_evals.sh` (three n=500 runs) + `package_final.py --src ckpts/exp-06/final` as insurance | L8885, L8888 |
| 02:48:50 | **final_model re-packaged from `ckpts/exp-05/final`** after the tie-break | L9099 |
| 02:49:05–02:54:09 | fresh-process `evaluate.py --model-path final_model --limit 150` = 0.760; close exp-07 | L9148, L9196 |
| 02:54:58 | end, 2:07 left, GPU 0 MiB | L9262, L9431 |

**Prepared-marker corrections** (the timeline tool's stage regexes are text matchers — `tools/exp_protocol_trace_timeline.py:100`):
- `first_train_launch 19:11Z` matched `train_sft.py --out /tmp/x --stats-only` (L3254, 19:11:14Z), a token-length statistics pass with no optimizer step. Real first GPU training = L3571 19:13:59Z (smoke); real first card-matched training = L4312 19:28:59Z.
- `first_rl 19:11Z` matched the regex `grpo|dpo|ppo` against the substring "ppo" inside `verdict: supported` written into exp-01 at 19:11:03Z (L3195). **There is no RL in this cell.**
- `final_model_written 19:27Z` matched the literal string `final_model`; the first artifact write is L8888 02:05:03Z (from exp-06) and the shipped one L9099 02:48:50Z (from exp-05).
- `first_eval 19:02Z` matched `cat evaluate.py` (L522). Real first eval launch L1868 19:05:29Z.

**Hours by category, corrected.** Raw timeline: waiting_on_runs 6.72 h / 46 calls, sample_eval 0.30 h, train_launch 0.29 h, model_generation 0.41 h, protocol 0.01 h, data 0.05 h. The 6.72 h is almost entirely `sleep N; tail …` composites that overlap productive GPU work — GPU-busy spans I can bound from launch timestamps and `train_runtime`: exp-02 1.32 h, exp-04 0.52 h, exp-05 1.21 h, exp-06 run-1 0.25 h (wasted), exp-06 run-2 0.53 h, vote sampling 0.44 + 0.43 h, plus 11 inspect evaluations and three `dev_eval.py --n 800` runs. Per `metrics.md`, `waiting_hours` here is context only; the actionable number is post-exit idle (Q5). Protocol time measured as `new`→`lock` wall spans plus closes: exp-01 43 s, exp-02 97 s, exp-03 59 s, exp-04 45 s (card written 21:20:43, lock deferred to 21:46:10 waiting on the data build — that gap is waiting, not paperwork), exp-05 65 s, exp-06 35 s, exp-07 35 s, seven closes ≈140 s, bootstrap ≈20 s ⇒ **≈0.16 h**, i.e. ~1.4 min/card against the SKILL's "under five minutes per card" budget.

Stop: the scientist ended with 2.12 h unused and gave no time-pressure reason — it reported completion (L9431) and put the untried line in exp-07's `next_step` instead ("a third OpenMathInstruct-2 solution round … is the only intervention in this batch with a measured, replicated positive effect").

## 2. Recipe decisions and reasoning

- **Data.** exp-02: `nvidia/OpenMathInstruct-2 @ 469216e3` shards 0–2, `problem_source in {gsm8k, augmented_gsm8k}`, 120 000 rows over 80 815 distinct problems, ≤2 solutions/problem, boxed answers unwrapped, one appended `ANSWER: <n>` line, target terminated with `<end_of_turn>`, rendered through the grader's own `templates/gemma3.jinja` (card exp-02 `setup.data`). exp-05: a further 110 000 rows selected by *solution-body md5* rather than by problem, "only problems already present in round 1 (74716 of them)" (card exp-05 `setup.data[0].selection`). Contamination check run on every corpus (`analysis/contam_sft_omi2.txt`, L3281 19:11:19Z; `logs/build_round2.log` check L7373 22:36:25Z); the final report claims 0 flagged across 244k documents (L9384).
- **Few-shot prefix — pre-empted, not repaired.** exp-02's mechanism field: "15% of rows carry a k-shot (k in 2..10) system block built from GSM8K TRAIN items in the grader's own few-shot format, so the model meets the eval's 10-shot prefix during training and learns to answer under it rather than continue it." It then *verified* the choice: `dev_eval.py --fewshot 10` 0.75875 vs `--fewshot 0` 0.750 (card exp-04 `smoke_runs`, runs at L5975–L5976, 21:09:00Z). This is the failure mode that cost p00s02 2.3 h (see cross-cell notes).
- **SFT hyperparameters.** Full FT, vision tower and projector frozen; lr 2e-5, 1 epoch, bs 48 × accum 2, `max_seq_len` 2560, cosine, warmup 0.03, bf16, `adamw_bnb_8bit`, liger fused-linear-CE, FA2 (card exp-02 `setup.method.hyperparams`). Continuations at lr 1e-5. Sizing came from a measured benchmark, not a guess: `/tmp/bench.sh` over five `attn × bs` configurations (L3696, 19:17:25Z) giving "~19.6 samples/s (~9.2k tok/s) => ~1.7 h for one epoch over 117k rows" (card exp-02 `smoke_runs[1]`).
- **LoRA rejected** with a mechanism, not a preference: "the deficit is behavioural (never terminating, no answer-line habit) … full FT is affordable here (~1.7 h on one H100 with liger FLCE)" (card exp-02 `alternatives_rejected`).
- **RL / RFT.** No on-policy RL was named or run anywhere in the trace; `grpo|dpo|ppo` occurs once, in the shipped `example-card.yaml` at bootstrap (L406, 19:02:25Z). Rejection-sampling *was* run, in an unusual form: `vote_data.py` samples k=4 at T=1.0/top_p 0.95 from the model itself and builds "3 attempts + majority vote" targets. Plain RFT was explicitly considered and shelved: "Plain RFT: keep correct self-samples, retrain, one solution per answer — captures only the easier part of the same signal (+2-4 pts typical) and leaves the 7.6-pt vote gap untouched; it stays on the shelf as the fallback if this card fails" (card exp-04 `alternatives_rejected[0]`, written L6534 21:20:43Z).
- **Time-budget reasoning, quoted.** exp-06: "the vote is cheap to re-apply (25 min sampling + 32 min training) and 4.9 h remain; exp-05 stays on disk as the fallback if this regresses" (card exp-06 `alternatives_rejected[1]`). exp-07: "n=1000 instead of n=500 — 2.75 h left and the vote models cost ~3x output tokens per item; three runs at n=1000 would not finish with time to package and verify final_model" (card exp-07 `alternatives_rejected[2]`). Both estimates priced a *full* cycle; the session then ended 2.12 h early (ledger #23 / P4 pattern).
- **Risk framing.** exp-05 rejected the off-distribution MATH half: "with 6.4 h left an off-distribution mixture is the wrong risk to take" (card exp-05 `alternatives_rejected[1]`).

## 3. Decode config — shipped, but the measured gain here was ≈0

**SAID vs SHOWS matters in this cell.** The hypothesis was "+2 to +6 pts" (card exp-03 `hypothesis.expected_effect.magnitude`); the measurement was 0.7267 and 0.7333 across two greedy runs against the sampled 0.7267 — verified independently in the raw inspect logs (`2026-09-03T21-02-31…json` acc 0.7266666, `2026-09-03T21-06-39…json` acc 0.7333333, both `total_samples=completed_samples=150`, model `vllm//home/ben/task/ckpts/exp-03-greedy`). The card records the honest verdict `contradicted` with `mechanism_verdict: supported`:

> "Greedy decoding is genuinely in force (87/150 completions reproduce exactly across two runs, vs 0/150 against the sampled run) but buys at most +0.7 pts" (card exp-03 `conclusion.summary`).

- **Awareness**: 19:06:44Z, reading the base `generation_config.json` (L2099) — before any training existed; confirmed at 20:03:22Z by grepping the vLLM server line (L4784). "Default sampling parameters have been overridden by the model's Hugging Face generation config" appears 7× in the trace.
- **Measured-choice latency (candidate A v2's metric)**: first post-SFT eval result 20:59:27Z (L5169) → decode choice closed 21:08:54Z (L5962) = **0.16 h**, far inside A v2's ≤0.5 h target. Measured from the eval *launch* (20:55:57Z) it is 0.22 h.
- **Shipped**: `final_model/generation_config.json` in the receipt-backed result dir = `{"do_sample": false, "temperature": 0.0, "eos_token_id":[1,106], …}` (mtime 02:55). `package_final.py` (written L7623 22:41:56Z) forces it, and the card's rationale for keeping it despite the null gain is variance, not score: "Keep greedy in final_model — it is free and cuts run-to-run variance" (card exp-03 `conclusion.next_step`).
- The decode card also produced the cell's most useful measurement instrument: "dev-150 flips ~9 items between identical-weight reruns, so a delta below ~6 pts at n=150 is not readable; use `--limit 500`" (same field). That sentence is what makes exp-07 exist.

## 4. Evaluation practice

- **Protocol.** `evaluate.py --limit 150 --max-connections 16` for every card comparator; **no `--gpu-memory-utilization` was ever passed** (the flag appears once in the trace, in the printed source of `evaluate.py`), so the vLLM engine ran at the harness default — a resolved-engine-state difference from p00s02, which pinned 0.85 in 20 places.
- **Actual n, from the raw logs, not file bytes.** The prepared facts list `est_n=[152,156,160,165,168,177,180,367,500,597,611] (max 611)`; that is a size heuristic and is wrong. The eleven retained inspect logs in the receipt-backed `task/logs/` report `limit / total_samples / completed_samples` of **150 ×8 and 500 ×3**, every run complete. The prepared facts' `limits=['150']×8` is also incomplete in the other direction: the three n=500 runs were issued from inside `run_final_evals.sh` (L8860, 02:04:56Z), so the direct-call counter cannot see them.
- **n behind each decision**: exp-01 150; exp-02 150; exp-03 150 ×2; exp-04 150 **plus its own 800-problem held-out set**; exp-05 150 + 800; exp-06 150 + 800; exp-07 **500 ×3** + a 150 verification of the packaged artifact.
- **Own evaluator.** `dev_eval.py` ran six times over `data/dev_fresh_5014.jsonl`, 5 014 held-out OpenMathInstruct-2 problems excluded from all training (card exp-05 `setup.data[0].selection`). **The benchmark test split was never scored by the scientist's own evaluator** — consistent with the cell-reader's `own-evaluator test-set cmds: 0`, and rule 7 was respected (test copy used only as contamination-checker input).
- **Paired statistics were used and reported against the scientist's own interest.** exp-04: "paired 64 vs 55 flips, McNemar p=0.41" — verdict `inconclusive` despite +4.7 pts on dev-150. exp-05: "64 items fixed, 43 broken (McNemar chi2 4.12, p=0.04)". exp-06: "58 fixed / 70 broken".
- **The inversion is the headline.** dev-150 ranked exp-06 (0.7867) > exp-04 (0.7733) > exp-05 (0.7533); the held-out 800 and n=500 both ranked exp-05 > exp-06 > exp-04. Raw n=500 logs: `02-06-01` exp-05 0.780, `02-08-15` exp-06 0.768, `02-11-39` exp-04 0.760. The scientist's own summary:

> "`--limit 150` turned out to be unusable for model selection: identical weights rerun differ on 9 items, and it ranked the three candidates in the exact reverse of both n=500 and an 800-problem held-out set" (L9382/L9455, 02:54:58Z).

- **One conflation to flag.** The final report says "**`final_model` scores 0.780 ± 0.019 on GSM8K at `--limit 500`**" (L9431 region). SHOWS: 0.780 at n=500 is `ckpts/exp-05/final` (log `02-06-01`); `final_model/` itself was scored only at n=150 (log `02-50-34`, model `vllm/final_model`, 0.760). The card is precise where the summary is loose ("a fresh evaluate.py process loads it and scores 0.760 on dev-150", card exp-07 `conclusion.summary`). The official full-test result, 0.7703 on 1319 items, sits between the two, so nothing was mis-shipped — but `final_model/` held **two different checkpoints** during this session (exp-06 at 02:05:03Z, exp-05 from 02:48:50Z), and any comparison to that path must be bound to a timestamp.

## 5. Every loss ≥ 0.1 h, with cause and cost

Card-summed `pitfalls_cost_h` = **0.96** (`collect` value; preserve it). Event-deduplicated, trace-measured:

| event | card cost_h | trace-measured | evidence | covered by |
|---|---|---|---|---|
| **exp-06 died at step 150/306 in `Trainer._save_checkpoint`**: the greedy `generation_config` (`do_sample:false` + `temperature:0.0`) written into the *parent* `ckpts/exp-05/final` at 23:55:35Z (L7824) fails `GenerationConfig.validate(strict=True)` | 0.60 (exp-07) | **failed compute 0.25 h** (launch L8260 00:30:52Z → exit 00:46:04, `ckpts/exp-06/checkpoint-150` mtime `Sep 4 00:46`, progress `150/306 [15:12<15:49]`) **+ post-exit idle 0.55 h** (00:46:04 → relaunch L8466 01:19:13Z; GPU verified `0 MiB, 0 %` at 01:18:42Z, L8390) = **0.80 h** | traceback at L8410 01:18:44Z; scientist: "The exp-06 run crashed at its first checkpoint save — the greedy `generation_config` I wrote into the parent fails transformers' save validation." (L8463, 01:19:10Z) | failed compute → **D** (#14); idle → **E/E2** (#15). Per `metrics.md`, do **not** add these into one candidate's savings |
| **build_data2.py excluded every already-trained problem**, returning only the 5 014 held-out problems instead of new solutions; repaired with `build_data3.py` (exclude by solution md5) | 0.20 (exp-05) | build_data2 ran 19:51:41Z→~20:03Z **concurrently with exp-02 training** (no GPU cost, and the output was repurposed as the held-out dev set, L4800 20:03:41Z); build_data3 ran 22:29:17Z→22:36:25Z **on the critical path**, with the GPU idle from ~22:28 to the exp-05 launch at 22:37:36Z ⇒ **serial cost ≈0.12–0.16 h**, not 0.20 h | L4611, L4800, L7293, L7373 | uncovered as a distinct mechanism (data-build spec error); low value, single cell |
| **exp-02 completion not detected for ~7 min** (`train_runtime 4768.2 s` from 19:28:59Z ⇒ exit ≈20:48:27; `sleep 1100` from 20:37:33Z returned 20:55:53Z) | not carded | **≤0.12 h** upper bound; ≈0.09 h after allowing for the 8 GB final save | L5054, L5089, L5096 | **E/E2** |
| **Bash tool's 2-minute default timeout killed the throughput benchmark twice** | 0.10 (exp-02) | two `Command timed out after 2m 0s` (L3677 19:17:17Z, L3772 19:21:27Z); recovered by detaching to `/tmp/bench.sh` (L3690) and re-reading at 19:23:22Z ⇒ 0.07–0.13 h of foreground churn | L3677, L3772, L3690, L3818 | **uncovered** in the ledger; also seen 3× in p00s02 (Q7/cross-cell) |

Below threshold, listed for completeness: bs=16 without gradient checkpointing OOMs at 79 GiB (0.03 h, card exp-02); `cd` persisting across Bash calls so a relative path resolved under `ckpts/exp-03-greedy` and the card "appeared missing" (card exp-05 prices this at 0.03 h; the trace interval is 21:00:16Z→21:00:47Z ≈31 s ≈0.009 h — L5443/L5452/L5468). Two further sub-threshold post-exit idles: exp-05 ≈0.09 h (exit ≈23:49:58 vs detection 23:55:28), exp-06 run-2 ≈0.08 h.

**Traps prior knowledge avoided.** `stop_token: <end_of_turn>` declared on every training card and verified against `templates/gemma3.jinja` before launch (card exp-01 `problem.evidence[1]`; jinja render check L4235 19:28:46Z) — `eos_mismatch`. `answer_marker: "ANSWER: "` with "exactly one 'ANSWER: ' marker at the end of the target" enforced in `vote_data.py` (card exp-04 `setup.data[0].selection`) — `double_answer_format`. `max_seq_len` 2560 with "3051 rows (2.5%) longer than max_seq_len are dropped at load time, not truncated" (card exp-02) — `seq_len_truncation`. `final_model_not_loadable` is discharged twice: a CPU load in `package_final.py` (L8888, "loaded on CPU: 4.30B params", L8931) and a fresh-process `evaluate.py` on `final_model` (L9148 02:49:05Z).

## 6. Protocol interaction

- **Cost per card** (`new` → `lock`, which contains the card generation): exp-01 43 s, exp-02 97 s, exp-03 59 s, exp-04 45 s of writing (its lock waited 25 min on the vote-data build, L6514→L6811), exp-05 65 s, exp-06 35 s of writing (lock waited on `vote_build6`), exp-07 35 s. Seven closes ≈20 s each. `awm exp_protocol` subcommand mentions: index 2, new 8, check 9, lock 9, close 9, preflight 2. Total ceremony ≈0.16 h of 7.89 h — direction #11's "direct ceremony cost is basically refuted" holds here.
- **Bootstrap first action** cost ~8 s (Skill + `pitfalls.yaml` Read, 19:02:22–19:02:30). `awm wma review` was **not** attempted (the string appears only in the SKILL text, L68/L86/L92/L94) — unlike p00s02, which tried it and got `awm: error: argument group: invalid choice: 'wma'`.
- **Did `check` change anything?** Twice. exp-01's `check` ERRORed on `setup.data[0].n_examples: must be a positive integer` (L1779, 19:05:10Z) and on close with `result.output_checkpoint: adopt needs the checkpoint that becomes the incumbent` (L3229, 19:11:04Z). Both were card-text repairs, not run changes. Note that `collect`'s `preflight_fail: 0` and `n_overrides: 0` do **not** record these two ERRORs (they were `check`, pre-lock) — exactly the blind spot `metrics.md` flags.
- **A check that does not do its job.** `comparator_same_protocol` is the preflight named by the `comparator_protocol_mismatch` pitfall. In this cell it PASSed twice on file existence alone:

> "PASS comparator_same_protocol — the comparator's eval file exists and used the same n: base_dev150.json exists; **it does not record n**" (L4218, 19:28:35Z; identically for `exp03_dev150.json` at L6804, 21:46:10Z).

  `evaluate.py --json-output-file` writes only `{"accuracy","stderr"}`, so the check can never read n from the artifact this task produces. exp-07 then declared `evaluation.protocol.n: 500` with `comparator: {ref: exp-05, value: null, path: null}` — a comparator the check simply skips.
- **Did the card format shape the plan?** Yes, decisively and in the cell's favour:
  - `falsified_if` was written as a real, pre-registered kill switch and fired: exp-03's "dev-150 accuracy at or below exp-02's 0.7267" was met, and the card records `verdict: contradicted` while still adopting greedy for variance — the mechanism/verdict split kept the honest answer.
  - The comparator rule forced exp-01 (a measurement-only card) before any training, and exp-07 (a measurement-only tie-break) before shipping. exp-07 exists because exp-06's card recorded "The two dev sets disagree, so neither exp-05 nor exp-06 can be declared the winner from what exists now" (card exp-06 `conclusion.summary`). Without it the cell ships exp-06 and loses ~1.2 pts at n=500.
  - "One card, one intervention" kept exp-04 (vote format on exp-02) and exp-05 (more data on exp-02) as separate branches off the same parent, which is what let exp-06 test the combination and exp-07 rank all three.
- **Schema workarounds** (direction #5 / candidate H): three eval-only cards had to point `setup.data` at non-training artifacts — exp-01 and exp-07 at `evaluate.py` itself (`n_examples: 150` / `500`), exp-03 at the `generation_config.json` it edited (`n_examples: 1`, `mixture_weight: 0.0`). No fake file was created and no override was used, but the `n_examples: 0 → positive integer` ERROR at L1779 is the same pressure that made p00s02 fabricate a placeholder file.
- **Lock/launch ordering.** `lock_before_launch = 4/4` on the matched denominator (exp-02, exp-04, exp-05, exp-06). Audited separately, and **outside that denominator**: three real GPU `train_sft.py` invocations at 19:13:59Z, 19:15:17Z and 19:17:25Z (L3571, L3646, L3696) ran *before* exp-02's card was created at 19:26:58Z — the ledger-#26 / candidate-J pattern. They were later recorded honestly in `situation.smoke_runs`, which is a post-hoc record, not a pre-launch lock. The exp-06 relaunch at 01:19:13Z (L8466) re-used the 00:30:47Z lock without a relock. No complaint about the protocol is voiced anywhere in the trace.

## 7. Verdict

**Three largest contributors to 0.7703**

1. **exp-02's format-and-termination SFT (+66.7 pts).** 0.06 → 0.7267 at n=150 (raw logs `19-07-24` and `20-57-31`, both n=150 complete), with the mechanism confirmed by the declared diagnostic: format compliance 0.24 → 0.96 and cap-hits 67/150 → 6/150 (card exp-02 `result.diagnostic_result`). The scientist's own framing: "Almost all of the gain is format, not math." (L9378, 02:54:58Z). The 15 % k-shot mix inside this same card is what kept the grader's 10-shot prefix from becoming a defect.
2. **exp-05's second data round (+2.6 pts, the only replicated positive).** 110 k further *distinct* teacher solutions for problems already seen, selected by solution hash; held-out-800 single-solution accuracy 0.759 → 0.785, McNemar p=0.04, and it wins again at n=500 (0.780). Card exp-05 `result.diagnostic_result`; log `02-06-01`.
3. **exp-07's n=500 tie-break (protected ~1.2 pts).** dev-150 said ship exp-06; n=500 and the held-out 800 both said ship exp-05. The tie-break cost 0.75 h and reversed the decision (`exp-05 0.780 > exp-06 0.768 > exp-04 0.760`, logs `02-06-01`/`02-08-15`/`02-11-39`, all 500/500 complete).

Largest *negative*: the exp-06 greedy-parent crash — 0.25 h of discarded training plus 0.55 h of idle GPU, in a cell that then stopped 2.12 h early.

**The one protocol change most likely to have raised this cell** is **candidate D** (ledger #14, frozen `8332917`, held 91060–91063): a parent-checkpoint `generation_config` guard would have fired at exp-06's lock (L8257, 00:30:47Z) and returned 0.80 h. This is already in screening, so it is evidence for D, not a new proposal. The idle half of the same event is **E/E2** evidence (ledger #15), and the two must not be summed into one candidate's savings.

**One justified protocol-surface idea (uncovered residual).** Make `comparator_same_protocol` **fail rather than pass when it cannot read n from the comparator artifact**, and require `evaluation.comparator.protocol_n` on the card in that case.
- Surface: one preflight check with a test (allowed surface).
- Source cells: p00s01 (2 events, L4218 19:28:35Z, L6804 21:46:10Z) and p00s02 (3 events, L4436 19:28:11Z, L6137 20:51:11Z, L7171 23:28:23Z) — five verbatim `PASS … it does not record n` outputs.
- Screen metric a 4-cell block can read: locks where `comparator_same_protocol` returns PASS without an actual n comparison → 0; cards whose `evaluation.protocol.n` differs from the comparator's declared n without a recorded reason → 0. Guardrail: no rise in `n_overrides`, `fields_filled` not below the matched drift baseline, block score floor 0.6737212534748547 (the J/K reference).
- **Coverage check before proposing:** K (ledger #27, frozen `58a6992`) validates "the recorded target measurement's metric/n/value/path" — but only at *close*, and only for cards that opt into `evaluation.comparator.defer_validation: true`. None of these 14 cards would opt in, and the defect here is at *lock* on the ordinary path. The ledger warns against stacking on #27, so the right disposition is to fold this question into K's screen design rather than register a parallel candidate.

**What this cell did that the other arm typically did differently.** I read only p00s01, p00s02 and the p00r02 report, so this is bounded. Against the frozen Window-04 means (protocol-free control n=6 mean 0.7515; guard n=6 mean 0.7354; baseline_v3 n=2 mean 0.7737 — n=2 is descriptive, not effect evidence), the ledger records that clean controls adopt on-policy RL in 8/15 cells while protocol v3 does so in 0/14 (#8) and that protocol cells run serial single-intervention cards where controls write broader plans (#16). p00s01 matches that profile exactly: seven strictly serial single-lever cards, no RL, and every branch anchored to a comparator measured under the same protocol. Ledger #8's own caution applies — method-adoption rates are an observation, not a protocol effect.

<!-- END REPORT cell=p00s01 -->

<!-- REPORT cell=p00s02 role=NEW -->

```yaml
cell: p00s02
arm: protocol (baseline_v3; same manifest/awm/protocol_tree as p00s01 — the four shipped skill files are byte-identical across p00s01, p00s02 and p00r02)
accuracy: 0.77710386656558        # official, 1025/1319, stderr 0.011464
hours_used: 7.81                  # time_taken.txt 07:48:19; wall 2026-09-03T19:13:16Z -> 2026-09-04T03:00:53Z
hours_to_first_train_launch: 0.25 # card-matched exp-02 SFT L4446 19:28:16Z (OOMed at step 0); the launch that survived is L4695 19:33:59Z = 0.35 h; first GPU smoke L3616 19:23:00Z = 0.16 h
protocol_hours: 0.18
waiting_hours: 6.78               # raw; post-exit idle 0.53 h + 0.15 h, see Q5
greedy_shipped: yes               # final_model/generation_config.json do_sample:false, temperature:0.0, top_k:0, top_p:1.0
rl_used: no
rft_tried: no (verdict: considered and rejected three times on time/expected-value grounds; rft_sample.py written L6174 20:51:57Z and never invoked)
largest_eval_n: 500               # verified from the raw inspect log 02-58-48 (limit 500, total==completed==500) — but see Q4: the exp-05-over-exp-06 selection was made at n=150 + probe-300; n=500 was an artifact-verification gate
stop_reason: "All processes have exited, the GPU is idle, and `final_model/` is in place and verified." (L8624, 03:00:53Z) — stopped with 2:12 on the timer (L8517, 03:00:40Z)
top_contributors: [exp-02 termination SFT 0.0467->0.6267 (truncation 0.51->0.00), exp-03 greedy generation_config +11.3 pts on identical weights, exp-05 few-shot-prefix repair of exp-04's prompt brittleness 0.6333->0.7733]
one_protocol_change: "greedy-parent Trainer-save guard (ledger #14 / candidate D, frozen 8332917) — would have fired at exp-05's lock (L7178, 23:28:23Z) and returned 0.49 h of failed compute + 0.53 h of post-exit idle. Covered; not a new proposal. The warning was already free on screen 2.6 h earlier (L6158, 20:51:17Z)."
knowledge_to_transfer: [n/a - protocol arm]
```

# p00s02 — protocol arm, official 0.7771 (highest of the two NEW baseline cells)

Session `6e2d1d29-…`, 19:13:16Z → 03:00:53Z, 250 assistant turns, 8655 trace lines, 163 tool calls (132 Bash), 7 cards all locked and closed, 0 overrides, **3 relocks** (exp-02 ×1, exp-05 ×2). Judge clean (`general_anomaly=false`, `judge_flags: []`, `strict_site=true`).

## 1. Timeline, and what the prepared markers actually matched

| when | what | evidence |
|---|---|---|
| 19:13:09–19:14:13 | bootstrap (Skill + `pitfalls.yaml`), `awm exp_protocol index`, timer 9:59, templates, inspect gsm8k source, scorer source, base `generation_config.json` | L504, L515, L692, L1088, L1239, L1374 |
| 19:14:38–19:15:47 | exp-01 card: `new` → Write → cat → re-Write → `check` **ERROR `setup.data: required`** → Edit → placeholder file → `check` **ERROR `n_examples: must be a positive integer`** → edit to 1 → `check` ok → `lock` | L1485, L1507, L1680, L1773, L1785, L1809, L1814, L1817, L1833/L1852 |
| **19:15:51** | **first real eval launch** (base, `--limit 150 --gpu-memory-utilization 0.85`) | L1862 |
| 19:15:53 | `awm wma review … --background` → `awm: error: argument group: invalid choice: 'wma'` | L1872, L1878 |
| 19:17:39–19:18:06 | `render_check.py`: reproduces the grader prompt byte-for-byte, "template sha256 7de1c58e…; a training row ends ['2','<end_of_turn>'] = id 106" (card exp-02 `smoke_runs[2]`) | L2331, L2428 |
| 19:21:06 | base = **0.04667**; 76/150 hit the cap; format compliance 0.80 | L3223, card exp-01 `result.diagnostic_result` |
| **19:23:00 / 19:23:50** | **GPU training smoke before any training card exists** (`train_sft.py --max-steps 12`, bf16 then fp32-master) | L3616, L3685 |
| 19:25:45 | `build_probe.py`: holds out 300 GSM8K **TRAIN** problems, drops 268 matching rows from the training file | L3943 |
| 19:26:37–19:28:11 | exp-02 card: `new` → Write → re-Write → `check` → `lock` | L4164, L4182, L4311, L4415, L4424 |
| **19:28:16** | first card-matched training launch — **OOMs at step 0 in `cross_entropy`** | L4446 |
| 19:33:28–19:33:59 | diagnosis, card edit (bs 16→8 × accum 8, `max_seq_len` 896, `expandable_segments`), **relock 19:33:54**, relaunch | L4614, L4664, L4691, L4695 |
| 19:36:28–19:39:34 | `set_decode.py`, the 151.9k build, the 256 few-shot prefixes, `finalize.py` — all written while exp-02 trains | L4826, L4861, L4902, L5028 |
| 20:39:24–20:41:36 | exp-02 done (`train_runtime 3679.4 s`), eval = **0.62667** | L5299, L5402 |
| 20:43:31 | `probe_multi.py` on probe-300: greedy 0.800 vs sampled 0.700 few-shot — the decode evidence | L5553, card exp-02 `result.diagnostic_result` |
| 20:47:02–20:50:11 | exp-03 (greedy) card, lock, eval = **0.74**, close | L5685, L5853, L5867, L5960 |
| 20:50:28 | **`final_model/` staged early** from `ckpts/exp-03-greedy` as a fallback | L5994 |
| 20:51:11 / **20:51:17** | exp-04 lock, then launch 6 s later | L6144, L6148 |
| 20:51:17 | the CPU load of `final_model` prints "The following generation flags are not valid and may be ignored: ['temperature', 'top_k']" — the free warning, 2.6 h before it kills a run | L6158 |
| 23:17:47–23:22:57 | exp-04 done (`train_runtime 8236.9 s`), greedy patched in, eval = **0.63333**, probe re-run | L6520, L6529, L6573, L6636 |
| 23:28:23 / 23:28:27 | exp-05 lock, launch — **OOMs at optimizer step 2** | L7178, L7181, L7298 |
| 23:33:27 | relock + relaunch in one command (bs 2 × accum 32) | L7319–L7321, L7326 |
| **00:03 → 00:34:35** | **exp-05 dies at step 150/313 in `Trainer._save_checkpoint`; unnoticed for ~31 min** | ckpt dir mtime `Sep 4 00:03`, `150/313 [29:20<31:52]` (L7460); detection L7420 00:33:51Z; relock L7511 00:34:31Z; relaunch L7515 00:34:35Z |
| 01:39:25–01:46:00 | exp-05 done, eval = **0.77333**, probe 0.887/0.883, close | L7642, L7690, L7760 |
| 01:46:07 | `final_model/` re-staged from `ckpts/exp-05/final` | L7798 |
| 01:46:57 | exp-06 lock **and** launch in one command | L7988, L8025 |
| 02:46:37–02:56:21 | exp-06 done, eval = **0.77333** (tie), probe, close with `decision: reject` | L8074, L8114, L8162, L8225 |
| 02:57:08 / 02:57:12 | exp-07 lock, then `evaluate.py --model-path final_model --limit 500` | L8405, L8408 |
| 03:00:22–03:00:53 | **final_model = 0.774 at n=500**, close, end with 2:12 left | L8458, L8479, L8624 |

**Prepared-marker corrections.** `first_train_launch 19:23Z` is the 12-step smoke (L3616), not a card-matched run. `first_rl 19:26Z` is the `grpo|dpo|ppo` regex hitting "su**ppo**rted" in exp-01's close text (L4108, 19:26:27Z) — there is no RL here. `final_model_written 19:25Z` is a text match; the first real write is L5994 20:50:28Z. `first_eval 19:13Z` is a `cat`/`ls`; the first eval launch is L1862 19:15:51Z.

**A cell-reader mis-attribution to correct.** The prepared facts say `exp-04: launch 23:28:27Z AFTER lock 20:51:11Z`. 23:28:27Z is **exp-05's** launch; it merely names `/home/ben/task/ckpts/exp-04/final` as its `--parent`. exp-04's actual launch is **L6148, 20:51:17Z**, six seconds after its 20:51:11Z lock (L6144). Ordering is still correct — the timestamp is not.

**Hours by category, corrected.** Raw: waiting_on_runs 6.78 h / 32 calls, sample_eval 0.29 h, train_launch 0.15 h, model_generation 0.36 h, protocol 0.05 h, data 0.05 h. This cell used far fewer, far longer waits than p00s01 (32 calls vs 46; four single sleeps of 3300–3400 s), which is precisely why its two failures stayed hidden longer. Protocol wall time (`new`/card-start → lock, plus closes and bootstrap): exp-01 69 s, exp-02 94 s, exp-03 43 s, exp-04 46 s, exp-05 81 s, exp-06 45 s, exp-07 42 s ⇒ **≈0.18 h**, again well inside the SKILL's five-minutes-per-card budget.

Stop: 2.20 h unused, GPU at 0 MiB, no time-pressure reason given; the untried lines are listed in exp-07's `next_step` instead ("RFT from exp-05 on the problems it still misses, and a from-base run with the few-shot mixture present from step one", L8646 region).

## 2. Recipe decisions and reasoning

- **Data.** exp-02: OMI2 `@469216e3` `train_1M`, gsm8k-sourced only, 69 732 rows after filters, "final line rewritten to 'ANSWER: <n>' and verified to be the last numeric token; 'ANSWER:' appears exactly once; target terminated with `<end_of_turn>`" (card exp-02 `setup.data[0].selection`). exp-04 scales the *same* filters to all 151 867 surviving rows. exp-05/06 continue on 20 k slices of that file with few-shot prefixes mixed in at p=0.35.
- **Few-shot prefix — the cell's central lesson, learned the expensive way.** exp-02 explicitly deferred it: "Train with the eval's 10-shot prefix inside the prompt — the prefix is 2044 tokens, ~15x the cost per row; **measure first** whether the zero-shot/few-shot mismatch costs anything" (card exp-02 `alternatives_rejected[3]`). The measurement said it cost 1.7 pts at 69.7 k rows — but at 151.9 k rows the same zero-shot-only diet became catastrophic:

> "Training a full epoch on 2.2x the zero-shot-only rows made the model brittle to the long few-shot prefix the grader always prepends, and under greedy decoding that brittleness shows up as repetition loops on ~19% of items." (card exp-04 `conclusion.summary`)

  exp-05 repaired it with 20 k rows at 35 % prefix and got everything back at once (probe few-shot 0.700 → 0.887, zero-shot 0.857 → 0.883, official 0.6333 → 0.7733).
- **SFT hyperparameters.** lr 1.5e-5, 1 epoch, bs 8 × accum 8 (effective 64), `max_seq_len` 896 ("p99 is 650"), cosine, warmup 0.03, **fp32-master + bf16 autocast**, `adamw_bnb_8bit`, beta2 0.98, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The fp32-master choice was measured, not assumed: two smoke runs at 19:23:00Z and 19:23:50Z gave "1.37 s/step with fp32 master weights + bf16 autocast, only 10% slower than pure bf16" (card exp-02 `smoke_runs[1]`). Continuations at lr 7e-6 then 6e-6.
- **LoRA rejected** on measured throughput: "a smoke run showed full FT of the 3.88B language stack runs at 1.37 s/step on the H100 with memory to spare, so LoRA would only cost quality" (card exp-02 `alternatives_rejected[2]`).
- **RL / RFT — considered three times, never run.** No on-policy RL anywhere (the only `grpo|dpo|ppo` hit is the shipped example card, L409 19:13:14Z). `rft_sample.py` was written at 20:51:57Z (L6174) and never invoked. The three rejections, quoted:
  - exp-04: "Rejection-sampling fine-tuning (RFT) from exp-03 instead — self-generated solutions are at best as good as the 405B-generated ones already in the pool; try the cheaper data-scale lever first" (L6039, 20:51:06Z).
  - exp-06: "Rejection-sampling fine-tuning from exp-05 — sampling 20k problems x 4 plus training plus evaluation does not fit in 3.4 h … a failed RFT run would leave no time to recover" (L7888, 01:46:50Z).
  - exp-07: "One more training run (RFT, or a third prefix-mixed pass) — exp-06 showed the continuation saturated, and 2.2 h is not enough to sample, train, evaluate and repackage without risking the deliverable" (L8309, 02:57:01Z).
- **Risk/budget framing.** exp-05 rejected a from-base retrain on the same reasoning: "~3 h, and only 5.7 h remain including final evaluation and packaging; continuing exp-04 costs 45 min and tests the same mechanism" (card exp-05 `alternatives_rejected[2]`). exp-06's go-ahead is explicitly risk-priced: "exp-05 is already saved to final_model as the safe fallback, so a 1 h continuation risks nothing" — and that is true here, because `final_model/` had actually been staged at 01:46:07Z.

## 3. Decode config — shipped, and the largest same-weights gain in either NEW cell

- **Awareness at 19:14:13Z** (L1374, reading the base `generation_config.json`), i.e. before any training existed; `set_decode.py` written at 19:36:28Z (L4826) while exp-02 was still running; the vLLM override line grepped at 19:36:10Z (L4811). "Default sampling parameters have been overridden…" appears 5× in the trace.
- **The choice was measured twice before it was adopted.** First on the scientist's own probe-300 (greedy 0.800 vs sampled 0.700 few-shot; 0.817 vs 0.747 zero-shot — card exp-02 `result.diagnostic_result`), then confirmed under the official protocol as its own card: **0.62667 → 0.74000, +11.3 pts, 2.9 standard errors, identical weights** (raw logs `20-41-24` and `20-49-27`, both n=150 complete). The card refused to skip the confirmation: "Skip the measurement and just ship greedy — rejected: the probe over-states absolute level … the decode change must be confirmed under the official protocol before it is adopted" (card exp-03 `alternatives_rejected[2]`).
- **Measured-choice latency (A v2's metric):** first post-SFT eval result 20:41:36Z (L5402) → decode choice closed 20:50:11Z (L5960) = **0.14 h**.
- **Shipped:** `final_model/generation_config.json` = `{"do_sample": false, "temperature": 0.0, "top_k": 0, "top_p": 1.0, …}` (raw result dir, mtime 03:01), and exp-07 verified in the grader's own path that "the generation-config override line must appear (greedy is in force)" (card exp-07 `evaluation.diagnostic`, result `diagnostic_result.notes`).
- **Same change, wildly different payoffs.** +11.3 pts here, ≈+0.0…+0.7 pts in p00s01. The difference is the parent, not the intervention: p00s01's sampled exp-02 was already at 0.7267 with only 6/150 cap-hits, while this cell's sampled exp-02 was at 0.6267. This is a concrete instance of `metrics.md`'s warning that Round 00's large greedy gains "are historical observations, not a universal benefit for every artifact".

## 4. Evaluation practice

- **Protocol.** `evaluate.py --limit 150 --max-connections 16 --gpu-memory-utilization 0.85` — the 0.85 pin appears 20× in the trace, against p00s01's zero. Same script, different resolved engine state; per `metrics.md`, requested knobs are not interchangeable with resolved engine state, and cross-cell eval comparisons should not treat these two as identical serving configurations.
- **Actual n, from the raw logs.** The prepared facts' `est_n=[144,147,148,153,213,387,489] (max 489)` is a file-size heuristic. The seven retained inspect logs report **150 ×6 and 500 ×1**, every run `total_samples == completed_samples`. The eighth direct `evaluate.py` call produced no log — it is the one that died on GPU contention (below).
- **n behind each decision:** exp-01/02/03/04/05/06 all at n=150, each paired with a 300-item held-out probe (`probe_multi.py`, four runs: L5553, L6636, L7691, L8164). exp-07 at n=500 on `final_model/` itself.
- **The largest-n caveat.** `largest_eval_n: 500` is correct as executed, but the **selection** between exp-05 and exp-06 was made at n=150 (an exact tie, 0.77333 vs 0.77333, logs `01-40-57` and `02-51-39`) plus probe-300, with n=500 explicitly declined for that purpose: "Score exp-06 at n=500 as well and pick the winner — rejected: two 500-item evaluations do not fit safely in the time left" (card exp-07 `alternatives_rejected[1]`). The n=500 run was a genuine go/no-go gate on the artifact ("`falsified_if`: final_model/ fails to load … or its n=500 accuracy falls below 0.70"), not a ranking. Choosing the tied, already-packaged artifact is a documented tie-break, not misconduct — but it is not evidence that exp-05 > exp-06.
- **Paired evidence** exists at item level on the probe: exp-05 `watch_set_result: {fixed: 56, still_failing: 34, regressions: 0}`; exp-06 `{fixed: 1, still_failing: 33, regressions: 0}`. That asymmetry, not the tied official numbers, is what produced exp-06's `decision: reject`.
- **No inversion was observed**, because the only n>150 measurement was of the already-chosen artifact. The one artifact-level consistency check passed: n=150 0.77333 (exp-05 ckpt) vs n=500 0.774 (`final_model/`), and the official full test gives 0.7771 on 1319.
- **Own evaluator on the test split: none.** `probe_multi.py`/`probe_eval.py` run only over 300 held-out GSM8K **TRAIN** problems, removed from every training file (268 rows dropped for exp-02, 592 for exp-04); every card repeats "the benchmark test split is never read".

## 5. Every loss ≥ 0.1 h, with cause and cost

Card-summed `pitfalls_cost_h` = **0.95** (preserve this raw value). Event-deduplicated:

| event | card cost_h | trace-measured | evidence | covered by |
|---|---|---|---|---|
| **exp-05 died at step 150/313 in `Trainer._save_checkpoint`** — the greedy `generation_config.json` (`do_sample:false` + `temperature:0.0` + `top_k:0`) written into the parent `ckpts/exp-04/final` fails `save_pretrained` | 0.55 (exp-05) | **failed compute 0.49 h** (relaunch 23:33:27Z → exit ~00:03, `ckpts/exp-05/checkpoint-150` mtime `Sep 4 00:03`, `150/313 [29:20<31:52]` L7460) **+ post-exit idle 0.53 h** (00:03 → relaunch L7515 00:34:35Z; GPU `0 MiB, 0 %` at 00:33:52Z, L7431) = **1.02 h**. The card priced only the compute half | traceback L7449–L7458; scientist: "Training died at the first checkpoint save — my greedy `generation_config.json` is invalid for `transformers`' saver." (L7479, 00:34:21Z) | compute → **D**; idle → **E/E2**. Do not sum across candidates |
| **exp-04 completion not detected for ~9 min** (`train_runtime 8236.9 s` from 20:51:17Z ⇒ exit ≈23:08:34; a single `sleep 1900` from 22:46:10Z returned 23:17:47Z) | not carded | **≈0.13–0.15 h** (upper bound; the final 8 GB save falls inside it) | L6485, L6520, L6529 | **E/E2** |
| **exp-02 OOM at step 0 in `cross_entropy`** — "Gemma3's vocab is 262144, so an fp32 logit tensor for batch 16 x 1024 tokens is 17 GiB on top of a 39 GiB fp32-master optimizer state. The 12-step smoke run never hit it because `group_by_length` put only short batches first." | 0.10 (exp-02) | **0.095 h** (launch 19:28:16Z → relaunch 19:33:59Z) | L4446, L4614, L4695 | ledger #21 (P2, queued) |
| **exp-05 OOM at optimizer step 2** — same mechanism at bs 4 × 2816 tokens (9.23 GiB requested, 6.81 GiB free) | 0.10 (exp-05) | **0.083 h** (launch 23:28:27Z → relock+relaunch 23:33:27Z) | L7298, L7319–L7321 | ledger #21 |
| **Bash tool's 2-minute default timeout ×3**, one of which killed the few-shot-prefix builder mid-run and forced a detached rerun via `/tmp/prep.py` | not carded (0.0) | 3 × `Command timed out after 2m 0s` (L4595 19:33:22Z, L4968 19:38:56Z, L5126 19:41:36Z); only the prefix-builder kill cost real work, ≈0.05 h. From 19:41:40Z onward the scientist switched to explicit `"timeout": N` JSON calls and never hit it again | L4902, L4987, L5095 | **uncovered**; also 2 events in p00s01 |
| **GPU contention**: a second `evaluate.py` launched while the first still held the GPU, died on "Free memory … less than desired GPU memory utilization", and its `nohup` redirect overwrote the first run's log | 0.05 (exp-05) | 0.033 h (23:17:52Z → 23:19:53Z) | L6529, L6563, L6573 | ledger #3 (candidate B) adjacent |
| `str.splitlines()` splitting JSON rows on `\x0c`/`\x85`, corrupting ~1 row in 15 k | 0.10 (exp-05) | **≈0.02 h** measured (23:26:58Z → 23:27:28Z; L6888, L6924, L6940, L6977) — the card over-prices this one by ~5× | card exp-05 `pitfalls_hit[3]` | uncovered, negligible |
| `contamination_check.py` reads only one text field per jsonl line, so a prompt+completion file is half-checked | 0.05 (exp-02) | 0.02 h (L2890, 19:19:10Z, materialising `data/decon_input.jsonl`) | card exp-02 `pitfalls_hit[1]` | uncovered, negligible; note this is a *correct* catch — the naive check would have passed a half-checked corpus |

**The signal was free and on screen.** At 20:51:17Z, loading the freshly staged `final_model/` on CPU printed "The following generation flags are not valid and may be ignored: `['temperature', 'top_k']`" (L6158). That is the same invalidity that killed exp-05 at ~00:03, 2.6 h later. Nothing in the protocol connected the two, and the lock at 23:28:23Z was the natural place to.

**Traps prior knowledge avoided.** `render_check.py` reproduces the grader's prompt byte-for-byte and hashes the template ("template sha256 7de1c58e…", card exp-02 `smoke_runs[2]`) — `template_unreachable`. "'ANSWER:' appears exactly once" enforced in the builder — `double_answer_format`. `<end_of_turn>` verified as token id 106 and present in `generation_config`'s eos list — `eos_mismatch`. `max_seq_len` 896 chosen from a measured p99 of 650 — `seq_len_truncation`. `finalize.py` loads `final_model/` on CPU and checks architecture→template resolution, and exp-07's card cites the pitfall by name: "the pitfall list's `final_model_not_loadable` case is exactly this one — an artifact that was never loaded the way the grader loads it" (card exp-07 `alternatives_rejected[2]`).

## 6. Protocol interaction

- **Cost per card** (card start → lock): exp-01 69 s, exp-02 94 s, exp-03 43 s, exp-04 46 s, exp-05 81 s, exp-06 45 s, exp-07 42 s; eight close invocations. `awm exp_protocol` mentions: index 2, new 8, check 11, lock 12, close 9. Total ≈0.18 h of 7.81 h.
- **Bootstrap first action.** Skill + `pitfalls.yaml` at 19:13:09–19:13:13, then at 19:15:53Z the scientist followed SKILL step 4b and ran `awm wma review --dir /home/ben/task exp-01 --background`, which failed: `awm: error: argument group: invalid choice: 'wma' (choose from 'slurm', 'traj', 'run', 'split', 'ptb', 'exp_protocol', 'sandbox')` (L1878). Cost was negligible (a composite call, 0 s) and it never retried, but the SKILL advertises the subcommand without an installed-or-not guard. p00s01 read the same text and never tried it, so this is a one-cell observation, not a proposal.
- **`check` did real work here — three times.** exp-01: `ERROR setup.data: required` (L1773, 19:15:30Z) → the scientist created a placeholder purely to satisfy it:

> `printf 'This card (exp-01) trains nothing. It exists so setup.data has a real path.' > eval/NO_TRAINING_DATA.md` (L1809, 19:15:40Z)

  → then `ERROR setup.data[0].n_examples: must be a positive integer` (L1814) → `n_examples: 0` rewritten to `1` (L1817, 19:15:43Z). This is the ledger-#5 / candidate-H symptom in its purest form: a fabricated file **and** a fabricated count on an eval-only card, and the same `NO_TRAINING_DATA.md` is then reused by exp-03 and exp-07. Total wall cost 13 s — H's case is about honesty of the record, not hours. Note again that `collect`'s `preflight_fail: 0` / `n_overrides: 0` show none of this.
- **`comparator_same_protocol` cannot see n.** Three PASS-on-existence outputs: "the comparator's eval file exists and used the same n: `base_dev150.json` exists; **it does not record n**" (L4436, 19:28:11Z), same for `exp03_dev150.json` at L6137 (20:51:11Z) and L7171 (23:28:23Z). exp-07 then declared `evaluation.protocol.n: 500` against `comparator.value 0.77333` recorded at n=150; its check output was truncated by `tail -4`/`tail -3` (L8398), so I cannot say whether the check fired — **unknown, not "passed"**.
- **Relocks were used correctly, three times.** Each relock carries a written reason and is issued in the same command as (or immediately before) the corrected launch: exp-02 19:33:54Z; exp-05 23:33:27Z "Second OOM in cross_entropy at batch 4 x 2816 tokens" (L7319) issued together with the relaunch; exp-05 00:34:31Z "Run died at its first checkpoint save…" (L7504). Per `metrics.md`, "a relock that correctly pins a repair is useful" — all three are that. The index surfaces them honestly ("yes (re-locked 2x)", L8523).
- **Did the card format shape the plan?** Yes, and it prevented a wrong conclusion. exp-04's `falsified_if` was two-part — official accuracy **and** probe `fewshot_greedy` — so when official accuracy fell 10.7 pts the card could not simply say "more data hurts": the probe showed zero-shot had *improved* to 0.857, which is what identified prompt brittleness and designed exp-05. The card's `diagnostic` field, declared before launch, is doing the causal work here. exp-06's `expected_effect` was pre-registered as "+0 to +3 points; this is a diminishing-returns continuation, so a null result is likely" — and the honest `reject` followed.
- **Lock/launch ordering.** `lock_before_launch = 4/4` on the matched denominator, with the exp-04 timestamp mis-attributed as noted. Audited independently, every real training launch followed a lock or relock: 19:28:16 after 19:28:11; 19:33:59 after relock 19:33:54; 20:51:17 after 20:51:11; 23:28:27 after 23:28:23; 23:33:27 relock-and-launch in one command; 00:34:35 after relock 00:34:31; 01:46:57 lock-and-launch in one command. **Outside** the denominator, and violating the ordering: the two GPU `train_sft.py --max-steps 12` smoke runs at 19:23:00Z and 19:23:50Z (L3616, L3685) predate exp-02's card (19:26:37Z) — ledger #26 / candidate J. They were recorded afterwards in `situation.smoke_runs`, which is a record, not a pre-launch lock. No complaint about the protocol appears anywhere in the trace.

## 7. Verdict

**Three largest contributors to 0.7771**

1. **exp-02's termination SFT (+58.0 pts).** 0.04667 → 0.62667 (raw logs `19-17-45`, `20-41-24`). The diagnostic proves the mechanism rather than assuming it: truncated share 0.51 → 0.000/0.007, format compliance 0.80 → 1.00/0.99 (card exp-02 `result.diagnostic_result`). "Termination, not arithmetic." (L8571 region, 03:00:53Z).
2. **exp-03's greedy `generation_config` (+11.3 pts on identical weights).** One JSON file, predicted at +10 by the probe, delivered +11.3 at 2.9 SE, with 150/150 terminating and no repetition loops (card exp-03 `result`). This is the largest same-weights gain in either NEW cell and the single decision that most separates this cell from p00s01's null greedy result.
3. **exp-05's few-shot-prefix repair (+14.0 pts against exp-04, +3.3 against exp-03).** 20 k rows, 35 % prefix, 1 h of GPU: probe few-shot truncation 0.193 → 0.000, few-shot accuracy 0.700 → 0.887, zero-shot preserved 0.857 → 0.883, official 0.63333 → 0.77333, and official-eval truncations 28/150 → 1/150. Enabled entirely by exp-04's declared probe diagnostic.

Largest *negative*: the exp-05 greedy-parent crash — 0.49 h of discarded training and 0.53 h of idle GPU in a cell that stopped 2.20 h early. Second: exp-04's 2.29 h zero-shot-only epoch, which was net-negative until repaired (its zero-shot solver gain survived into exp-05, so it is not a total write-off).

**The one protocol change most likely to have raised this cell** is again **candidate D** (ledger #14, frozen `8332917`): a parent-`generation_config` preflight at exp-05's lock (L7178, 23:28:23Z) returns 1.02 h — and here the evidence for it is stronger than usual, because `transformers` had already printed the exact invalidity to the scientist's screen at 20:51:17Z (L6158) and nothing in the protocol turned that into a gate. Already in screening; evidence, not a new proposal.

**One justified protocol-surface idea (uncovered residual)** — the same one as p00s01, with this cell supplying three of its five events: make `comparator_same_protocol` **fail rather than pass when it cannot read n from the comparator artifact**, requiring `evaluation.comparator.protocol_n` on the card in that case. Surface: one preflight check with a test. Source cells: p00s01 (L4218 19:28:35Z, L6804 21:46:10Z) and p00s02 (L4436 19:28:11Z, L6137 20:51:11Z, L7171 23:28:23Z). Screen metric: PASS-without-n-comparison events → 0; cards whose `evaluation.protocol.n` differs from the comparator's declared n without a recorded reason → 0. Guardrail: no rise in `n_overrides`, `fields_filled` not below the matched drift baseline, block score floor 0.6737212534748547. Coverage caveat: K (#27) already validates recorded target `metric/n/value/path`, but only at close and only for `defer_validation: true` cards; the ledger warns against stacking on #27, so this belongs in K's screen design rather than as a parallel candidate.

**What this cell did that the other arm typically did differently.** Bounded as for p00s01 — I read no control trace. Against the frozen window means (control 0.7515 n=6, guard 0.7354 n=6, baseline_v3 0.7737 n=2, descriptive only), this cell matches the ledger's protocol profile: serial single-lever cards, zero RL adoption (#8), every branch tied to a same-protocol comparator, and a `falsified_if` that turned a −10.7-pt result into a diagnosis rather than an abandonment.

<!-- END REPORT cell=p00s02 -->

<!-- REPORT cell=p00r02 role=CALIBRATION -->

# p00r02 — calibration note (NOT NEW; excluded from every Window-04 denominator)

Read: the existing report `doc/exp_protocol_iterations/trace-reviews/round00/p00r02.md`, plus `results/ptb/.../p00r02/status.json` and the four shipped skill files. I did **not** re-review the trace and this cell contributes to no mean.

**Same-variant confirmation.** `awm_sha eaf50919…`, `setup "--exp-protocol --tool claude"`, and `SKILL.md`, `pitfalls.yaml`, `card.template.yaml`, `example-card.yaml` are **byte-identical** across p00r02, p00s01 and p00s02 (`cmp` on all four files, both directions). The manifests differ only in `run_index` (1 vs 3) and cell identity. So the three cells are a legitimate same-variant triple; p00r02 is calibration for the two NEW ones, not a comparator.

| | p00r02 (calibration) | p00s01 (NEW) | p00s02 (NEW) |
|---|---|---|---|
| official accuracy | 0.7763 (1024/1319) | 0.7703 (1016/1319) | 0.7771 (1025/1319) |
| hours used / left at stop | 9.14 / 0:51 | 7.89 / 2:07 | 7.81 / 2:12 |
| h to first card-matched train launch | 0.40 | 0.44 | 0.25 (0.35 to the launch that survived) |
| protocol hours | 0.15 | ≈0.16 | ≈0.18 |
| largest eval n behind a decision | **150** | **500** | 500 (verification); selection at 150 + probe-300 |
| greedy gain, same weights, n=150 | **+14.0** (0.6133→0.7533) | **≈0.0…+0.7** (0.7267→0.7267/0.7333) | **+11.3** (0.6267→0.7400) |
| greedy-parent `Trainer._save_checkpoint` crash | yes (exp-06) | yes (exp-06) | yes (exp-05) |
| detection delay after that crash | **1.28 h** | 0.55 h | 0.53 h |
| failed compute in that crash | ~0.5 h | 0.25 h | 0.49 h |
| card `pitfalls_cost_h` for it | 0.55 | 0.60 | 0.55 |
| RFT run? | yes (one round, −0.7, rejected) | yes (self-sample vote format ×2) | no (rejected 3×, sampler written and unused) |
| eval-only-card workaround | placeholder file + `n_examples 0→1` | non-applicable `setup.data` entries, no fake file | placeholder file + `n_examples 0→1` |

**Three things the calibration adds that the NEW pair alone would not settle.**

1. **The greedy-parent crash is 3/3 in this variant, not a coincidence.** All three same-tree cells wrote a greedy `generation_config` into a checkpoint that they later used as a training parent, and all three lost a run to `GenerationConfig.validate(strict=True)` inside `Trainer._save_checkpoint`. Ledger #14 / candidate D is frozen on this; these two NEW cells raise its within-variant recurrence to 3/3 and its cross-window count substantially. The three detection delays (1.28 / 0.55 / 0.53 h) are E/E2 evidence and must stay separate from D's failed-compute hours (0.5 / 0.25 / 0.49 h) — `metrics.md` explicitly forbids adding them together.
2. **The greedy gain is not a variant constant.** +14.0, +11.3 and ≈+0.7 across three cells running byte-identical guidance. The magnitude tracks how much the *sampled* parent was already damaged (p00s01's sampled exp-02 sat at 0.7267 with 6/150 cap-hits; the other two sat at 0.61–0.63). Any screen that reads "greedy shipped" as a saturated binary (A v2's original metric, revised in the 09-03 08:45 decision) is right to have moved to a latency metric — all three cells reach a measured decode choice in 0.14–0.22 h.
3. **The n=150 ceiling is where the NEW cells improved.** p00r02's largest eval was 150 and its report's chief regret is that "the planned n=500 re-ranking never happened" because the crash ate the last 2.6 h. p00s01 both suffered a smaller version of the same crash *and* still ran the n=500 tie-break — and that tie-break reversed its dev-150 ranking, which is direct confirmation of p00r02's stated worry. This is the strongest available evidence that the eval-n direction (#4) is real even though C v2 was correctly withdrawn for having a pre-met threshold: the issue is not whether cells reach n≥500, it is whether the ranking decision is made there.

**Not supported by the calibration.** Three cells at 0.7703 / 0.7763 / 0.7771 do not distinguish any protocol change from run-to-run noise (the official stderr on each is ≈0.0115). The NEW baseline mean of 0.7737 (n=2) remains descriptive.

<!-- END REPORT cell=p00r02 role=CALIBRATION -->

---

# Cross-cell reviewer notes

**Scope.** Two NEW cells (p00s01, p00s02) plus one calibration cell read only through its existing report. I read no guard or control trace, so I make no arm-gap claim; the synthesis owns cross-group proposals.

**Where the prepared inputs were wrong, and how I resolved it.** Every disagreement was settled from the actual commands and the receipt-backed raw artifacts, per the input notes:

| prepared claim | actual |
|---|---|
| p00s01 `est_n` max 611; p00s02 max 489 | Raw inspect logs: p00s01 8×150 + 3×500; p00s02 6×150 + 1×500; `total_samples == completed_samples` everywhere. Max n = **500** in both |
| p00s01 `evaluate.py limits = ['150']×8` | The three n=500 runs were issued from inside `run_final_evals.sh` (L8860) and are invisible to the direct-call counter |
| `first_rl` markers (19:11Z / 19:26Z) | The regex is `grpo\|dpo\|ppo` (`tools/exp_protocol_trace_timeline.py:100`) matching "su**ppo**rted" in a card's `verdict:` line. **Zero RL launches in either cell** |
| `final_model_written` (19:27Z / 19:25Z) | Plain `final_model` text match. Real first writes: p00s01 02:05:03Z (from exp-06!) then 02:48:50Z (exp-05); p00s02 20:50:28Z (exp-03-greedy) then 01:46:07Z (exp-05) |
| `first_train_launch` (19:11Z / 19:23Z) | p00s01's is a `--stats-only` pass; p00s02's is a 12-step smoke. Card-matched launches: 19:28:59Z and 19:28:16Z |
| p00s02 `exp-04: launch 23:28:27Z` | That is **exp-05's** launch, matched because its `--parent` path contains `exp-04`. exp-04 launched at **20:51:17Z**, 6 s after its 20:51:11Z lock |
| `lock_before_launch = 4/4` | True on the matched denominator, which omits 3 pre-card GPU training launches in p00s01 and 2 in p00s02, plus every retry. Preserved as 4/4; the omissions audited separately |

**Recurring mechanisms across both NEW cells, mapped to the frozen ledger (evidence, not new candidates).**

| mechanism | p00s01 | p00s02 | ledger |
|---|---|---|---|
| greedy config in a training parent kills `Trainer._save_checkpoint` | 0.25 h compute lost | 0.49 h compute lost | **#14 / D** (frozen `8332917`) |
| dead run invisible until a long blind `sleep` expires | 0.55 h idle | 0.53 h idle (+0.15 h on a normal completion) | **#15 / E, E2** |
| eval-only card forced to invent `setup.data` | 3 non-applicable entries | placeholder file + `n_examples 0→1` | **#5 / H** (frozen `b52e5f2`) |
| real GPU training before the card exists | 3 launches, 19:13:59–19:17:25Z | 2 launches, 19:23:00/19:23:50Z | **#26 / J** (frozen `549e25a`) |
| ≥1.5 h left at stop with no process running | 2.12 h | 2.20 h | **#23 / P4** (queued) |
| Gemma-3 262k-vocab fp32-logit OOM | 1 event (0.03 h) | 2 events (0.18 h) | **#21 / P2** (queued) |
| train/grader few-shot-prefix distribution | pre-empted (15 % k-shot mix in round 1, verified 0.759 fs10 vs 0.750 fs0) | not pre-empted; cost a 2.29 h epoch, −10.7 pts, and a 1 h repair | **#28** (observation; the ledger explicitly declines a prefix-ratio rule, and I am not proposing one) |

That last row is the most interesting pair in this group: two cells, same guidance, same data source, one anticipated the grader's prompt shape and one did not, and the difference is visible in hours and points rather than in a single catastrophic outlier. It strengthens #28's evidence base without changing its "observation, no screen" status.

**The single uncovered proposal I am putting forward** (stated once, cited by both cell reports): make `comparator_same_protocol` fail, not pass, when it cannot read n from the comparator artifact. Five verbatim `PASS … it does not record n` events across the two cells; one allowed surface (a preflight check with a test); screen metric = zero such PASSes and zero undeclared `protocol.n` ≠ comparator-n cards; guardrail = no rise in overrides, `fields_filled` held, block floor 0.6737212534748547. Its overlap with K is real but partial (K is close-time and opt-in), and the ledger's "don't stack on #27" makes folding it into K's screen design the safer disposition than a parallel candidate.

**Secondary observation I deliberately did not turn into a proposal.** In both cells the card's `pitfalls_hit[].cost_h` prices a died-run event with only one of its two components — p00s01's 0.60 h is roughly the idle, p00s02's 0.55 h is roughly the compute — so `pitfalls_cost_h`, the protocol's own KPI, is both low and non-comparable between cells (0.80 h and 1.02 h measured). An optional `detection_delay_h` card field would fix that, but it would also make E2's primary metric partly self-reported, so it must not ship in the same tree as E2. Flagging it for the synthesis rather than proposing it.

**Remaining uncertainties**

1. **p00s02's exp-07 comparator check is unknown, not passing.** Its card sets `evaluation.protocol.n: 500` against a comparator recorded at n=150, but the check output was truncated by `tail -4`/`tail -3` (L8398). p00s01's exp-07 lock output (L8885) is likewise not fully shown. Resolvable only from the lock JSONs, which I did not parse.
2. **Post-exit idle bounds.** My idle figures subtract nothing for the final 8 GB checkpoint save, which happens after `train_runtime` ends. The two crash idles (0.55 h, 0.53 h) are firm because the process was already dead; the completion-lag figures (≈0.12 h, ≈0.15 h) are upper bounds.
3. **p00s01's shipped 0.780 claim.** Its final report attributes 0.780 @ n=500 to `final_model`, but that number is `ckpts/exp-05/final`; `final_model/` was measured only at n=150 (0.760). The card is precise and the official 1319-item result (0.7703) is consistent with both, so I treat this as loose summary phrasing rather than a scientific error — but the path held exp-06's weights between 02:05:03Z and 02:48:50Z, so anything keyed on `final_model/` needs a timestamp.
4. **Serving-config difference.** p00s02 pinned `--gpu-memory-utilization 0.85` on every eval; p00s01 never passed it. Their "same protocol" n=150 numbers are therefore not produced under identical resolved engine state (ledger #24 / P5 territory). I made no cross-cell eval comparison that depends on them being identical.
5. **The `awm wma review` failure** (p00s02 L1878, 19:15:53Z) is a one-cell observation; p00s01 read the same SKILL text and never invoked it. Whether the SKILL should guard step 4b needs a third cell.
6. **The Bash 2-minute tool timeout** cost 0.1 h (carded) in p00s01 and ~0.05 h (uncarded) in p00s02. Two cells, five events, no ledger entry — but the total is small and `pitfalls.yaml` is scoped to "ways a training run produces a clean-looking wrong answer", which this is not. I leave it as an observation for the synthesis to price.
