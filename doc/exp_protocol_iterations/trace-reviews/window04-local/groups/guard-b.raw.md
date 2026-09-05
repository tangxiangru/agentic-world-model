I have completed the read-only review. Here are the three reports.

<!-- REPORT cell=g01r06 role=NEW -->

```yaml
cell: g01r06
arm: protocol (guard; batch exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1, awm 4ae3d87, setup --exp-protocol --tool claude --stop-hook)
accuracy: 0.7081122062168309        # = 934/1319; guard6 mean −0.0273, control6 mean −0.0433
hours_used: 7.74                     # time_taken.txt 07:44:26; trace 18:31:04Z → 02:14:35Z
hours_to_first_train_launch: 0.38    # card-matched exp-02 18:53:42Z; first GPU training command of any kind (smoke) 18:42:12Z = 0.19 h
protocol_hours: 0.03 (tool time, 53 calls) / 0.125 (card create→first-lock wall, 7 cards, 1.07 min mean)
waiting_hours: 5.84 raw (27 calls); post-exit idle bounded ≤ ~0.02 h per event by artifact polling
greedy_shipped: yes (from the first checkpoint the script ever saved, 18:48:00Z) — but never measured against sampling on the same weights
rl_used: no
rft_tried: yes (exp-03) — verdict inconclusive, mechanism_verdict contradicted; exp-04 re-tried the same corpus as one-stage SFT, contradicted
largest_eval_n: 1319 (8 full-test runs)
stop_reason: "Nothing further." — exp-07 next_step L8129 02:14:12Z, with 2:16 (2.27 h) unused
top_contributors: [output-contract SFT +0.61 at n=200, two-way weight soup +0.9 pt on 1319, 0.85 h of dead GPU from the save_model GenerationConfig crash]
one_protocol_change: candidate D (#14, greedy parent → Trainer save validation, frozen 8332917, held 91060–91063) — it targets this cell's single largest loss exactly
knowledge_to_transfer: n/a (protocol arm)
```

## 1. Timeline

Session `4ca573f1`, 18:31:04Z → 02:14:35Z (7.73 h of trace span; `time_taken.txt` 07:44:26). Budget 10 h; 27 `timer.sh` readings, first 9:59, last 2:16.

Stage sequence (all from the trace, not the prepared markers):

| time | event | evidence |
|---|---|---|
| 18:31:04 | `Skill` tool loads `exp_protocol`; SKILL.md, pitfalls.yaml, card.template.yaml read by 18:31:10 | L10, L178/L252/L407 |
| 18:31:14 | `awm exp_protocol index --dir /home/ben/task` (bootstrap first action complete in ~10 s) | L570 |
| 18:33:21 | exp-01 created | card `created_at` |
| 18:34:50 | exp-01 locked (first lock) | exp-01.lock.json |
| 18:35:00 | base n=200 eval launched (after its lock) | L1825 |
| **18:42:12** | **first real GPU training command: `timeout 1500 python scripts/train_sft.py … --out ckpts/smoke --limit 3000`** — OOM 14 s later | L3005; traceback L3040 18:42:26 |
| 18:42:53 | smoke relaunched with `expandable_segments` | L3139 |
| 18:45:53 | n=50 eval of `ckpts/smoke/final` | L3306 |
| 18:51:53 | exp-02 created | card |
| 18:53:37 → **18:53:42** | exp-02 locked → **first card-matched training launch** | L4066 / L4072 |
| 20:33:17 | exp-02 closed **and `cp -r ckpts/exp-02/final final_model`** — first real deliverable staged at +2.04 h | L4608, L4638 |
| 20:34:00 | `gen_rft.py` graded self-sampling launched (k=4, T=0.9) | L4651 |
| 21:21:50/55 → 22:20:57 | exp-03: lock, three launches, two deaths, three relocks (§5) | L5120, L5540, L5845 |
| 23:20:04/10 | exp-04 locked → launched | card |
| 01:31:16 → 01:42:36 | **three full 1319-item evals in one command** (exp-02/03/04) | L6643; results L6677–6687 |
| 01:43:32 → 01:51:15 | exp-05 soup built, scored full, **`rm -rf final_model && cp -r ckpts/exp-05/final final_model`** | L7220 |
| 01:51:46 / 01:59:33 | exp-06 (3-way soup, rejected) / exp-07 (verification) | cards |
| 02:00:48, 02:05:19, 02:10:11 | final_model full, final_model full rep2, exp-03 full rep2 | L7904, L7960, L8056 |
| 02:14:35 | last assistant turn | L8184 |

**Corrections to the prepared timeline.** `first_train_launch 18:42Z` is the *smoke*, not the card's run; the real card-matched launch is 18:53:42Z. `first_rl 18:42Z` is a false positive: the tool's pattern is `grpo|dpo|ppo` case-insensitively matched against tool text (`tools/exp_protocol_trace_timeline.py:100`), so it fires on substrings such as "su**ppo**rted" in card YAML. There is no GRPO/PPO/DPO invocation anywhere in the trace; the only RL-family strings are `trl.SFTTrainer` in the bootstrap example card (L470). `final_model_written 20:33Z` *is* a real write, but of the **exp-02 SFT checkpoint**, not the shipped artifact — the shipped soup was written to the same path at 01:51:15Z (+7.34 h).

Hours by category (tool execution time, from the timeline, corrected by reading): waiting_on_runs 5.84 h (27 calls), sample_eval 0.91 h (12), train_launch 0.39 h (7), bash_other 0.05, data 0.04, protocol 0.03 (53 calls), model_generation 0.39 h. I found no misclassified composite in this cell: the 01:31Z 0.19 h `sample_eval` really is three full-test evals (11 min 20 s wall, 01:31:16→01:42:36), which is fast but consistent with `--max-connections 16` and ~225 output tokens/item.

**Stop reason (SAID and SHOWS agree).** exp-07 `next_step`: *"Nothing further."* (L8129, 02:14:12Z), and the last turn is preceded by a liveness check returning `no runs alive / 0 MiB` (L8161–8166, 02:14:19Z). 2.27 h were left. The earlier justification was accurate about the clock: *"2.5 h remain and nothing else is queued"* (L7738, 02:00:26Z).

## 2. Recipe decisions and their reasoning

- **exp-02 (SFT, base):** 129,223 rows = `nvidia/OpenMathInstruct-2` (train_1M) + `openai/gsm8k` train; lr 1e-5, 1 epoch, bs 32×1, max_seq_len 1536, cosine, bf16, seed 0; targets end `ANSWER: <n><end_of_turn>`. Budget reasoning, quoted: *"2 epochs over the 129k corpus … 3.4 h of a 9.5 h budget on one card; run 1 epoch (1.7 h), measure, and spend the next block on whatever the measurement says is wrong."*
- **Few-shot prefix:** handled as a *diagnostic*, not a training knob. exp-03's `diagnostic_result` notes the same checkpoint scores 0.780 with the grader's 10-shot block prepended vs 0.736 zero-shot (`analysis/probe_exp03_fs10.json`), i.e. the prefix **helps** this model by ~4 points. No prefix share was engineered into the corpus.
- **RFT (exp-03):** on-policy rejection sampling from exp-02, k=4 @ T=0.9 over 32,223 problems, exact-match filter, ≤2 kept per problem → 52,233 rows + 25,000-row replay slice; lr 7e-6. Explicit risk/budget reasoning against RL, quoted: *"DPO on correct/incorrect sampled pairs … preference training on a model this fresh off SFT is a larger risk per GPU-hour than RFT, and only ~4 h of usable budget remain after this card."* And against more sampling: *"k=8 … the k=4 run already took 42 min and reached 86.9% coverage; doubling it would buy the tail of hard problems at the cost of the training block."*
- **On-policy RL: never attempted.** No GRPO/PPO command exists (§1). The only RL-adjacent consideration is the DPO rejection above.
- **exp-04:** one-stage SFT from base on the union of the two corpora (181,452 rows). Designed under the one-intervention rule: *"keeping the teacher corpus byte-identical to exp-02's makes the union the only difference between exp-02 and this card, so the comparison actually attributes the change."*
- **exp-05/06 (merge):** uniform weight soup. exp-05 (exp-03+exp-04) adopted; exp-06 (3-way with exp-02) rejected with a mechanism reason: *"exp-03 is exp-02's own continuation, so the uniform three-way weights that trajectory two-thirds."*

## 3. Decode config

**Shipped greedy, decided a priori, never measured on the same weights.** The base snapshot's `generation_config.json` (`do_sample: true`) was read at 18:32:18 (L1188–1192, +0.02 h). The scientist wrote the greedy config into `scripts/train_sft.py` itself, so every checkpoint it saves carries it — verified in the *smoke* checkpoint at 18:48:00: `"do_sample": false, "temperature": 0.0, "top_k": 1, "top_p": 1.0` (L3341–3344), together with `preprocessor_config.json` and `processor_config.json` (L3357–3358). Recorded as a zero-cost pitfall on exp-02: *"The harness sends no temperature, so vLLM falls back to the checkpoint's generation_config.json"* (L3788, 18:52:45Z).

The shipped `final_model/generation_config.json` in the receipt-backed result dir confirms `do_sample:false, temperature:0.0, top_k:1, top_p:1.0, eos [1,106]`. **Gain measured: none** — there is no sampled-vs-greedy A/B in this cell, because the sampling config never touched a trained checkpoint. Under A v2's reading (`同权重对照` decode choice within 0.5 h of the first post-SFT eval), this cell made the choice ~1.7 h *before* the first post-SFT eval but supplied no same-weights evidence for it. The choice also had a cost: it is the direct cause of the 0.85 h save crash (§5).

## 4. Evaluation practice

- Screening n = 200 for exp-01…exp-04 (`--limit 200`, `--max-connections 16`, `--gpu-memory-utilization 0.85`), with an explicit justification for going above the harness default: *"n=200 costs ~10 min more but lowers stderr on every comparison this batch; a 150-item subset of a 200-item run is not the same protocol"* (exp-01 alternatives_rejected).
- **Shipping decisions were all made on the full 1319-item test set** — 8 full-test runs. `evaluate.py` (the official harness) only; `own-evaluator test-set cmds: 0`. The private `scripts/eval_probe.py` probe-250 set is built from held-out GSM8K **train** items and is used as a secondary signal, with the limitation stated honestly in the final message: *"an `augmented_gsm8k` rewrite of a held-out problem can survive that filter — it's a ranking signal, not a clean generalization measurement"* (L8217).
- **Ranking inversion at larger n, realised in this cell.** At n=200 exp-04 (0.710) > exp-03 (0.690); at n=1319 exp-04 (0.7028) < exp-03 (0.7058). The scientist had predicted the need: *"the full 1319-item test set (stderr ~1.3 points) will settle the three-way"* (exp-04 alternatives_rejected), and exp-04's `falsified_if` named the full-test comparison.
- **Repeated reads on the same artifact.** exp-05/final scored 0.71494; the byte-identical `final_model/` scored 0.70811 and 0.70887 on two further servers; exp-03/final scored 0.70584 then 0.70887. Same weights → 943 / 934 / 935 correct on 1319 (0.68 pp spread). Byte-identity was checked, but only for shard 1: `md5sum` of `model-00001-of-00002.safetensors` matched (`d4de0dc9860ed944f2230b95058a4e5c`, L7697–7699, 01:59:51Z) — the card SAYS "the files are byte-identical", the trace SHOWS one of two shards hashed. The scientist launched the repeats deliberately ("Replicate exp-03 full eval for a fair mean comparison", L8057, 02:10:11Z) and corrected its own record: *"greedy decoding under vLLM is not reproducible across server instances to better than ~0.7 points on 1319 items, so exp-05's 0.7149 was the high end of its own range"* (L8129).
- **No paired/McNemar statistic.** Comparisons are point estimates plus a private watch-set fixed/regressed count (e.g. exp-03: 17 fixed / 15 broken; exp-04: 15 / 16).
- **Official run.** `final_eval_1.txt`: 1319/1319 samples, `max_connections: 2`, total time 14:42, accuracy 0.708. The official 934/1319 equals the scientist's first `final_model` full run exactly, on a different server and a different `max_connections`; equal scalars, and no official per-item log is retained here, so item-set identity is unverifiable (the P5 boundary, ledger #24).
- **Sample-count caution, demonstrated.** The cell-reader's `est_n` (bytes/44 KB) reports 497 for the base eval; the retained developer log header shows `"limit": 200` on a 1319-sample dataset (`logs/2026-09-03T18-37-00…json`, 21.8 MB, because the base model emitted ~2231 tokens/item). Also `evaluate.py direct calls: 11` vs **13** inspect logs in the receipt-backed `task/logs/` — the direct-call counter is not a complete evaluation audit.

## 5. Every loss ≥ 0.1 h, with cause and cost

Raw `pitfalls_cost_h = 2.30`. **This double-counts**: exp-04's single 1.05 h entry is a verbatim carry-forward of exp-03's overlay (0.15) + save-crash (0.90) entries ("*Carried from exp-03: …*"). Event-deduplicated card cost = **1.25 h**; trace-bounded ≈ 1.0 h.

| # | event | card cost | trace-bounded interval | covered by |
|---|---|---|---|---|
| 1 | First trainer launch OOM — Gemma-3's 262k-vocab `logits.float()` asks ~48 GiB at 32×1536 | 0.10 h | 18:42:12 → 18:42:53 = **0.011 h** (L3005→L3139) | #21 P2 (queued) |
| 2 | 64 MB root overlay filled with `__pycache__`; `liger_kernel/…/gemma3.py` became 13,389 null bytes; exp-03 launch #1 died `ValueError: source code string cannot contain null bytes` | 0.15 h | 21:21:55 → 21:30:01 = **0.135 h** (L5120 → L5190 21:27:19 → L5540) | #21 P2 (queued) |
| 3 | **48-minute training run completed then died in `trainer.save_model`**: `transformers` validates `model.generation_config` strictly on save and the parent (exp-02/final) carries the greedy config this script writes. Only `config.json` reached disk; weights lost | 0.90 h | 21:30:01 → 22:20:57 = **0.849 h**, of which ~0.83 h is dead GPU compute (L5540 → traceback L5702–5726 at 22:19:58 → relaunch L5845) | **#14 candidate D, frozen `8332917`** |
| 4 | 3 rejection-sampled rows wrote `ANSWER: 20 $10 bills`; failed the last-number preflight | 0.05 h | not separately timed (CPU) | pitfalls `double_answer_format` already |
| 5 | `scripts/soup.py` hit the *same* `save_pretrained` GenerationConfig validation | 0.05 h | caught within the exp-05 relock window 01:44:24 → 01:44:56 | **#14 D** |

Two further observations on #3: the scientist's first repair was itself wrong and it caught that in 15 s — relock at 22:20:37 ("*mid-run checkpointing added*") followed at 22:20:52 by "*Same save-validation fix as the previous relock, **corrected**: the replacement GenerationConfig must be assigned before `trainer.train()`, otherwise the mid-run checkpoint saves hit the identical validator error*". Second, the artifact-existence wait it used (`for i in $(seq 1 40); do if [ -f ckpts/exp-03/final/config.json ]; …`, L5624, 22:05:14Z) broke early on a `config.json` written by the *failed* save — the exact "artifact proxy outlives/mispredicts the producer" case E v2 names. Here it was benign because the same command tailed the log, so the crash surfaced at 22:19:58 with no idle.

**Traps prior knowledge avoided.** The `double_answer_format` pitfall changed a *data-source* decision: *"MetaMathQA … its bodies carry gsm8k's own '#### N' line alongside an 'answer is' phrase - the double_answer_format pitfall"* (exp-02 alternatives_rejected). Rule 2 changed the baseline decision: *"published numbers use a different prompt and scorer; rule 2 requires a comparator measured under this protocol"* (exp-01). And the missing-processor-config trap that cost g01r08 0.2 h never fired here, because the training script writes `preprocessor_config.json`/`processor_config.json` into every checkpoint (L3357–3358).

**Post-exit idle.** Not a material cost in this cell. Every long wait was a blind `sleep` *followed by* an artifact poll with 30–40 s granularity and a short trailing sleep, so detection lag is bounded at ~1–2 min per run (exp-02 done ~20:27, seen 20:29; exp-03 done ~23:11, seen 23:12; exp-04 done ~01:24, seen ~01:27). The largest gap is ~4 min after `gen_rft` (finished ~21:16, poll returned 21:20). The scientist also *upgraded* its own poll predicate after the save crash, from `config.json` to `generation_config.json` — the last file written (L5987, 22:50:33Z).

## 6. Protocol interaction

- **Cost.** `protocol` tool time 0.03 h over 53 calls. Card authoring (create → first lock) totals **7.5 min over 7 cards, mean 1.07 min** (max 1.73 min), well inside SKILL's "under five minutes per card". Bootstrap first action (Skill load → `awm exp_protocol index` returned) ≈ 10 s, 18:31:04 → 18:31:14.
- **Locks/overrides.** 7 cards, 7 closed, 0 open, **0 overrides**, 4 relocks (3 on exp-03, 1 on exp-05), every `lock` reporting `preflight_fail: 0`. That last figure hides two pre-lock `check` rejections visible only in the trace: `ERROR problem.watch_set: needs path and n` / `ERROR setup.data: required` (L1686–1687, 18:34:18Z) and `ERROR result.output_checkpoint: adopt needs the checkpoint that becomes the incumbent` (L2984, 18:42:03Z).
- **Ordering.** `lock_before_launch = 3/3` matched, and I confirmed all four exp-03 (re)launches followed their (re)lock by 5 s (21:21:50→55, 21:29:56→30:01, 22:20:52→57). **Outside that denominator**, three GPU executions preceded any covering card: the smoke training at 18:42:12/18:42:53 and the n=50 smoke eval at 18:45:53 (exp-02 created 18:51:53), and the 42-minute answer-verified `gen_rft` sampling at 20:34:00 (exp-03 created 21:20:57). All were recorded retrospectively in `situation.smoke_runs`. This is exactly the pattern candidate **J** (#26, frozen `549e25a`) extends rule 1 to cover — including its forward-review finding that *graded* generation is evaluation and cannot ride an unlocked `build_command`.
- **Card format shaped the plan — with evidence.** The one-intervention rule is cited to reject a confound (exp-04: "*keeping the teacher corpus byte-identical … makes the union the only difference*"). A `falsified_if` closed a direction: exp-03's predicate ("*n=200 accuracy at or below 0.685, or fewer than 10 of 68 watch items fixed, or regressions exceed fixes*") produced 17 fixed / 15 broken → mechanism contradicted → *"Rejection sampling on top of an already-converged SFT model is not the lever."* The comparator rule forced the exp-07 verification card and the choice to run it at 1319 rather than 200: *"the full 1319 items make the verification a same-protocol reproduction of the shipped number."*
- **Workaround (H, #5).** Two non-training cards carry a `setup.data` entry they never read: exp-05 and exp-06 (`data/sft_v4.jsonl`, n=181452, `selection: "no training happens in this card"`). exp-01/exp-07 name `data/eval_manifest.jsonl`, a real 5 KB id-only manifest the eval genuinely indexes — under H's frozen wording that is a permitted applicable reference, not a fake entry. So: **2 non-applicable entries, 0 fabricated files, 0 overrides.**
- **Stop hook: never fired.** No `.stop_hook.json` exists in `task/memory/` of the receipt-backed original result dir (only `cards/` and `index.md`), and 0 cards were left open. Rule 9 was nonetheless in front of the scientist at every lock — the `run_dies_with_the_session` reminder is printed by `awm exp_protocol lock` (e.g. L4066, 18:53:37Z) — and its `setsid nohup … < /dev/null &` form was used for every long run (L4072, L5120, L5540, L5845, L6416). `background tasks killed/stopped at exit: 0`.

## 7. Verdict

**Three largest contributors to 0.7081.**

1. **The output-contract SFT (exp-02): 0.075 → 0.685 at n=200**, i.e. 61 of the ~63 points the cell gained. Mechanism verified, not assumed: format/termination fell from 33.5 % of base failures to 3.2 %, mean output from the 1024-token cap to 151 tokens (`analysis/exp02_n200_failuretags.json`, `probe_exp02.json`).
2. **The two-way weight soup (exp-05): +0.9 pt on 1319 over exp-03** (0.71494 vs 0.70584), for ~8 min of CPU — and it was correctly *not* extended: the 3-way soup lost 1.0 pt and was rejected on the same 1319 items.
3. **Losing ~0.85 h of GPU to the save_model crash, and ~0.7 pt to server-to-server variance.** The shipped artifact's own three reads span 0.7149 / 0.7081 / 0.7089; the official harness landed at the low end (934/1319). The 0.85 h did not obviously cost score here (2.27 h went unused), but it is the cell's clearest avoidable waste.

**The one protocol change most likely to have raised this cell: candidate D (#14).** The failure is literally D's predicate — a *greedy parent* checkpoint makes `save_pretrained`'s strict `GenerationConfig` validation kill a finished 48-minute run — and this cell paid for it twice (exp-03, and again in `soup.py`). D is already frozen (`8332917`, held 91060–91063, first releasable wave); g01r06 adds a fourth source cell with an exact 0.849 h interval (21:30:01Z → 22:20:57Z, L5540→L5845) and the extra detail that the *repair* also has a failure mode (the config must be assigned before `trainer.train()`, or mid-run checkpoints hit the same validator). I am deliberately **not** proposing anything new for this cell: its other losses map onto queued P2 (#21, overlay/liger + logits OOM) and its ordering gaps onto frozen J (#26).

**What this cell did that the other arm typically did differently.** No on-policy RL (0/1 here; ledger #8 records protocol v3 0/14 and clean controls 8/15) — but ledger #8 also says method-adoption rate is an observation, not a protocol effect, and this cell's RFT rejection is argued from measured GPU-hour risk, not from the protocol. More distinctively, this cell moved its *entire* selection to n=1319 with repeats, which is stronger than either arm's typical n=150–500 practice and is the behaviour a redesigned C (#4) would want to induce; here it cost only ~4 min per full-test run.

<!-- END REPORT cell=g01r06 -->

<!-- REPORT cell=g01r07 role=NEW -->

```yaml
cell: g01r07
arm: protocol (guard; batch exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1, awm 4ae3d87, setup --exp-protocol --tool claude --stop-hook)
accuracy: 0.7217589082638363        # = 952/1319; guard6 mean −0.0136, control6 mean −0.0297
hours_used: 8.37                     # time_taken.txt 08:22:25; trace 18:30:58Z → 02:52:34Z
hours_to_first_train_launch: 0.29    # card-matched exp-02 18:48:18Z; first GPU training command (smoke) 18:42:08Z = 0.19 h
protocol_hours: 0.01 (tool time, 49 calls) / 0.217 (card create→first-lock wall, 8 cards, 1.63 min mean)
waiting_hours: 5.21 corrected (6.21 raw, minus 1.00 h of composite 400-item GPU probes misclassified as waiting); post-exit idle ≈ 0.05 h total
greedy_shipped: yes — but only at 02:39:46Z, 8.15 h in; +5.8 pt at n=500, McNemar p=0.0076
rl_used: no
rft_tried: yes, twice — exp-04 supported/adopt (+6.0 pt at n=150), exp-05 contradicted/reject
largest_eval_n: 500 (7 runs; confirmed `"limit": 500` in the retained developer log headers)
stop_reason: "Nothing further is launched; the remaining time is under an hour and any new run would be unverifiable." — exp-08 next_step L9299 02:51:28Z (timer.sh showed 1:38, L9474)
top_contributors: [format SFT 0.087→0.613, on-policy RFT +6.0 pt, greedy decode +5.8 pt discovered with 14 min to spare]
one_protocol_change: A v2 (#2) is the frozen fit; the genuinely uncovered gap this cell exposes is that decode config is not part of `evaluation.protocol`, so `comparator_same_protocol` cannot see a decode mismatch (proposal below, sourced from g01r07 + g01r08)
knowledge_to_transfer: n/a (protocol arm)
```

## 1. Timeline

Session `ec5f0240`, 18:30:58Z → 02:52:34Z (8.36 h span; `time_taken.txt` 08:22:25). 30 timer readings, 9:59 → 1:38.

| time | event | evidence |
|---|---|---|
| 18:30:58 | `Skill` loads `exp_protocol`; SKILL/pitfalls/template read by 18:31:03 | L10, L189/L263/L411 |
| 18:31:08 | `awm exp_protocol index --dir .` | L579 |
| 18:33:22 | base snapshot `generation_config.json` read — `"do_sample": true` | L1698 → L1741 |
| 18:34:12 / 18:35:54 | exp-01 created / locked | cards |
| 18:36:01 | base n=150 eval launched (after lock) | L2519 |
| **18:42:08** | **first GPU training command: `timeout 1500 python scripts/train_sft.py --out ckpts/smoke --limit 1024 --bs 8`** — OOM | L3923 |
| 18:43:48 | smoke relaunched at bs 16 with `expandable_segments` | L4247 |
| 18:46:27 → 18:48:14 → **18:48:18** | exp-02 created → locked → **first card-matched launch** | L4478 / L4785 / L4814 |
| 18:51:42 | `scripts/finalize.py` written (copy-into-`final_model` + CPU-load verifier) | L5158–5197 |
| 20:26 → 20:31:36 | exp-02 done; **first post-SFT eval results** 0.5733 (e2) then 0.6133 (e1) | L5490, L5650 |
| 20:32:15 | 400-item held-out 10-shot/0-shot probe, 30 min foreground | L5747 |
| 21:04:00 → 21:09:00/04 | exp-03 created → locked → launched | L5989 / L6204 / L6224 |
| 23:52:58 | `rft_sample.py` (k=4 @ T=1.0, gold-filtered) launched | L6599 |
| 00:15:46 → 00:17:35/40 | exp-04 created → locked → launched | L6784 / L7130 / L7142 |
| **01:22:12** | **first `final_model` write** (`finalize.py --src ckpts/exp-04/final`) | L7363 |
| 01:49:24 → 01:49:25/29 | exp-05 created → locked → launched | L7571 / L7708 / L7720 |
| 02:31:43 → 02:36:13 | exp-06 soup built, 0.664 @ n=500, staged into `final_model` | L8107 / L8390 |
| **02:38:00 / 02:38:16** | soup re-check at n=150 returns **0.58** vs 0.664 at n=500 on the *same weights*; the vLLM log line `Default sampling parameters have been overridden by the model's Hugging Face generation config` is found | L8469 / **L8491** |
| 02:39:46 | greedy written into `final_model/generation_config.json` | L8720 |
| 02:43:22 → 02:48:20 | exp-08 locked; two greedy n=500 evals reverse the ranking; `final_model` re-staged **from exp-04/final** | L9015 / L9163 |
| 02:51:07 | shipped path verified through the harness: 0.728 @ n=500 | L9253 |
| 02:52:34 | last assistant turn | L9498 |

**Corrections to the prepared timeline.** `final_model_written 18:51Z` is a false positive — 18:51:42 is the *creation of `scripts/finalize.py`*, whose docstring contains the string `final_model` (L5161); the first actual write is 01:22:12Z (+6.85 h) and the shipped write is 02:48:20Z (+8.29 h). Same path, three different checkpoints (exp-04 → soup → exp-04). `first_rl 18:41Z` is the `grpo|dpo|ppo` substring false positive described in the g01r06 report; RL launches = 0. `first_train_launch 18:42Z` is the smoke, not the card's run.

**Category correction.** The timeline's `bash_other 0.52 h` is almost entirely one item: the 30-min 400-item GPU probe at 20:32:15 (L5744, `timeout 1800 python scripts/probe.py … --n 400`). Two further 0.50 h calls classified as `waiting_on_runs` (23:21:41 L6464, 00:51:36 L7280) are composites of `analyze_eval.py` plus the same 30-min probe. Corrected: **productive sampling/eval ≈ 2.34 h**, **waiting ≈ 5.21 h**, `bash_other ≈ 0.02 h`.

**Stop reason — SAID ≠ SHOWS.** exp-08's `next_step` (L9299, 02:51:28Z): *"Nothing further is launched; the remaining time is under an hour and any new run would be unverifiable."* Fifty seconds later `bash timer.sh` printed **1:38** (L9474–9475). The stop was deliberate and safe, but the budget was under-priced by ~40 min — an instance of ledger #23 (P4).

## 2. Recipe decisions and their reasoning

- **exp-02 (SFT, base):** 58,217 rows from GSM8K train + OpenMathInstruct-2 train_1M + MetaMathQA; lr 1e-5, **2 epochs**, bs 16×2, max_seq_len 3328. Result: epoch 1 = 0.6133, epoch 2 = 0.5733 — the incumbent became `checkpoint-1820` (epoch 1). Pre-launch render check quoted: *"a training row's prompt is byte-identical to the string the grader builds for the same question, zero-shot and 10-shot."*
- **Few-shot prefix — measured, not assumed.** 6 % of rows carry the eval prefix, with the cost stated: *"the prefix is 2044 tokens against a 155-token target, ~5x the compute for the same supervised tokens."* Raising it was rejected on evidence: *"measured, not assumed: the probe on 400 held-out augmented_gsm8k problems scores the incumbent HIGHER with the 10-shot prefix (0.730) than without it (0.688), so there is no train/[grade mismatch]"* (exp-03 alternatives_rejected). This is directly relevant to ledger #28 as a **counter-case**: this cell's prefix share is 6 % and its 10-shot behaviour is fine.
- **exp-03 (data scaling):** 148,352 unique rows (2.5×), same schedule. Deliberately one-factor: *"Lower the learning rate as well as scaling the data … two interventions in one card; lr stays at 1e-5 so the delta is attributable to unique data alone."* Result 0.6067 vs 0.6133 — rejected. Terminal loss 0.2891 vs exp-02's 0.2907 "*essentially identical despite 2.5x the rows*".
- **exp-04 (RFT round 1):** 25,249 self-generated gold-matching chains from exp-03/final + 15,000 real-GSM8K SFT rows, lr 5e-6 → 0.6733 at n=150 (+6.0 vs incumbent), watch-set 23 fixed / 13 regressed. Adopted; mechanism marked *contradicted* because pass@1 did not move.
- **exp-05 (RFT round 2):** same recipe, fresh pools → 0.6467, rejected. The card names the pre-launch warning sign honestly: *"pass@1 was identical before and after round 1, and a second round of the same recipe bought nothing."*
- **RL: considered and declined, twice, with budget reasoning.** *"DPO or GRPO on correct/incorrect pairs — the sampler produced 20491 wrong chains so pairs exist, but a preference/RL stage is the riskiest thing to debug with one shot left and no time to recover from a bad run"* (L6862, 00:16:44Z). No RL command was executed.
- **exp-06 (soup):** 0.5/0.5 of exp-04 and exp-05, justified by disagreement rather than by score: *"the pair being tied at 0.644/0.656 while disagreeing on 130 of 500 items is exactly the condition under which averaging two checkpoints from one trajectory helps."*

## 3. Decode config — the defining event of this cell

**Aware at 18:33:23, acted at 02:39:46 — a 8.1 h gap.** The base `generation_config.json` was read in the first three minutes and `"do_sample": true` is right there in the tool result (L1741). Nothing was done with it: the training script did not write a decode config, so every checkpoint inherited `do_sample:true / top_k:64 / top_p:0.95`. The pre-change state is preserved as an artifact: `task/data/exp07_generation_config_before.json`.

The discovery was accidental and late. At 02:38:00 the shipped soup scored **0.58 at n=150** against **0.664 at n=500** on the *same weights* (L8469) — too large a split for item difficulty — and grepping the vLLM log surfaced `Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator` (L8491, 02:38:16Z).

The fix was then measured properly: `do_sample:false, temperature:0.0`, top_k/top_p removed, weights untouched → **0.664 → 0.722 at n=500**, `analysis/greedy_mcnemar.json` = `{greedy_only: 70, sampled_only: 41, p: 0.0076}`. The scientist's own summary (L9523): *"greedy decoding | +5.8 pts at n=500, p=0.0076 — the only statistically significant result in the batch."*

Two consequences it recorded itself:

- exp-07's `pitfalls_hit` entry for this carries **`cost_h: 0.0`** — the largest cost in the cell is booked at zero. The prose is accurate (*"every card compared two checkpoints measured the same wrong way — a shared bias cancels in a delta and hides in plain sight"*), the number is not.
- **The selection had been made on noise.** exp-08 re-ran all three candidates greedily: exp-04 0.736 > soup 0.722 > exp-05 0.708, exactly reversing the sampled ordering 0.644 < 0.656 < 0.664 (`analysis/greedy_selection.json`). `final_model` was re-staged from exp-04/final at 02:48:20 and re-verified at 0.728 (n=500). Directly traceable re-work: the exp-06 soup card (0.42 h) plus exp-08's two re-measurements (0.42 h) ≈ **0.8 h**, none of it in `pitfalls_cost_h`.

A v2's metric applied to this cell: first post-SFT eval result 20:29:16Z → same-weights measured decode choice in hand by 02:42:22Z = **6.22 h** against A v2's ≤ 0.5 h target.

## 4. Evaluation practice

- n = 150 (the harness default) for exp-01 through exp-05, with the reason given up front: *"500 items costs ~3x the wall time; 150 is the harness default and every later card must reuse the same n."* n = 500 from exp-06 onward. **Largest n = 500**; the full 1319 set was never used by the scientist.
- `evaluate.py` only; `own-evaluator test-set cmds: 0`. `scripts/probe.py` runs on 400 held-out `augmented_gsm8k` problems, kept separate from harness numbers.
- **Inversion at larger n, twice.** (i) exp-04 vs exp-05 at n=150: 0.6733 > 0.6467; at n=500 (same sampled decode): 0.644 < 0.656 — the ordering flips. (ii) The soup at n=500 = 0.664 vs the *same weights* at n=150 = 0.58. The scientist drew the right operational conclusion: *"n=150 is exactly the resolution that produced the contradictory orderings in exp-04/exp-05; re-running the same underpowered comparison would answer nothing"* (exp-08 alternatives_rejected).
- **Paired statistics — the only cell in this group that used them.** `analysis/greedy_mcnemar.json` (p = 0.0076) and, for the candidate selection, *"the three are within noise of each other anyway (all McNemar p > 0.4)"* (exp-07 alternatives_rejected). `analysis/soup_mcnemar.json` records the three n=500 endpoints. This is precisely the observable C v2 was written to read.
- **Same-artifact repeat variance.** exp-04/final scored 0.736 from `ckpts/`, and the identical weights installed at `final_model/` scored 0.728 at the same n=500 — 4 items apart. Official at n=1319: 0.7218 (952/1319).
- **Counting caveats.** The cell-reader reports `evaluate.py direct calls: 10`, but the receipt-backed `task/logs/` holds **14** inspect logs (six at `limit: 150`, seven at `limit: 500`, one more at 150) — four evaluations were launched inside composite commands the direct-call regex missed. `est_n` maxes at 487 for runs that the log headers show were `"limit": 500`.
- Official `final_eval_1.txt`: 1319/1319 samples, `max_connections: 2`, 11:04 total, accuracy 0.722.

## 5. Every loss ≥ 0.1 h, with cause and cost

Raw `pitfalls_cost_h = 0.10` — the lowest of the three cells, and misleadingly so.

| # | event | card cost | trace-bounded interval | covered by |
|---|---|---|---|---|
| 1 | Smoke OOM — `Gemma3ForConditionalGeneration.forward` upcasts `[B,T,262208]` to fp32 and makes a second masked copy (~20 GB at batch 8) | 0.10 h | 18:42:08 → 18:43:48 = **0.028 h** (L3923 → L4247) | #21 P2 (queued) |
| 2 | **Eight hours of decisions taken under the inherited temperature-1.0 sampling config**; the exp-04/exp-05/soup selection was reversed once decode was fixed | **0.0 h (as booked)** | discovery 02:38:16Z (L8491); traceable re-work ≈ **0.8 h** (exp-06 0.42 h + exp-08 0.42 h) | **#2 candidate A v2** (frozen `f6cdccc`, held 91046–91049) |
| 3 | Card-schema churn on the decode card: `n_examples: 0` rejected → set to 1 → `close()` refused because the card *edits* the file it named as `setup.data` → a purpose-built copy `data/exp07_generation_config_before.json` created and the card re-locked | not booked | 02:39:37 → 02:42:38 = **0.050 h** (L8697 → L8873) | **#5 candidate H** (frozen `b52e5f2`, held 91068–91071) |

So: **no single trace-verified loss reaches 0.1 h**, and the cell's real cost is a mis-priced correctness failure that the card system booked at zero. Nothing here is `uncovered` in the loss sense.

**Traps prior knowledge avoided.** Rule 2 shaped exp-01 (*"a published number is a different protocol … rule 2 forbids it as a comparator"*). The immutability rule blocked the wrong fix for decode: *"Modify evaluate.py to pass temperature=0 — forbidden by the task rules; the fix belongs in my own model's generation_config.json, which is mine to set"* (exp-07). `finalize.py` (written at 18:51:42, before it was needed) load-verifies `final_model/` from a fresh CPU process every time it stages, so `final_model_not_loadable` never fired.

**Post-exit idle: essentially zero, and by design.** Every training run was waited on with a completion-marker loop — `while ! grep -q "^saved ckpts/exp-0N/final" logs/exp-0N.log; do sleep 60; done` (L5360, L6369, L7188, L7760) — and every sampling run with `while ! grep -q "^wrote " …` (L6680, L7471). Measured lag: exp-02 finished ~20:26, seen 20:27 (1 min); exp-03 ~23:17, seen 23:19 (2 min); exp-04 ~00:48, seen 00:49; exp-05 ~02:21, seen 02:22. Blind `sleep` was used only for mid-run progress checks. This is the behaviour E v2 (#15) is trying to induce, demonstrated at no cost.

## 6. Protocol interaction

- **Cost.** `protocol` tool time 0.01 h over 49 calls — the lowest of the three. Card authoring (create → first lock) 13.0 min over 8 cards, mean 1.63 min; the maximum is exp-03 at 5.0 min. exp-05 shows the working pattern: the whole card was composed in one assistant turn (A204, 01:49:24Z), with `awm exp_protocol new --dir . >/dev/null 2>&1` used only to allocate the id and its printed questions discarded (L7571) — four of eight cards do this. Bootstrap first action ≈ 10 s.
- **Locks/overrides.** 8 cards, 8 closed, 0 open, **0 overrides**, 1 relock. `lock_before_launch = 4/4`. Pre-lock `check` failures that `preflight_fail: 0` hides: L2309–2310 (18:35:15Z), L3896 (18:41:58Z), L8697 (02:39:37Z).
- **Ordering outside the matched denominator.** Four GPU executions preceded any covering card: the two smoke trainings (18:42:08, 18:43:48; exp-02 created 18:46:27); the 400-item **graded** probe at 20:32:15 (exp-03 created 21:04:00); `rft_sample.py` at 23:52:58 (exp-04 created 00:15:46); and the second sampling run, already writing `logs/rft_sample2.log` by 01:22:50 (exp-05 created 01:49:24). All appear retrospectively in `situation.smoke_runs`. J (#26) covers exactly this, including graded generation.
- **The clearest H (#5) instance in the window.** exp-07 is a decode-config card with no corpus. It first set `n_examples: 0` → `ERROR … must be a positive integer` (L8697); then named the live `final_model/generation_config.json` with `n_examples: 1` and the comment *"not a corpus; named so the lock hashes the exact file this card edits"* (L8705); `close()` then correctly refused: `ERROR setup.data[0].path: … changed after the lock; the run did not train on the data the card names` (L8842). The scientist's response is the honest one and it is on the record — *"The lock correctly caught that this card edits the file it named as input"* (L8854) — but the resolution was to **create a file whose only purpose is to satisfy the schema** (`data/exp07_generation_config_before.json`, 180 bytes, L8858–8860), reused by exp-08. Tally for H: **1 fabricated file used by 2 cards, plus 1 non-applicable entry** (exp-06 names `rft_mix2.jsonl` with `selection: "not re-read; see exp-05"`), **0 overrides**.
- **`falsified_if` did real work.** exp-03's predicate (`at or below 0.6133`) → 0.6067 → the data-scaling axis was closed. exp-05's (`at or below 0.6733`) → 0.6467 → the second RFT round rejected. exp-08's predicate is unusually well posed — *"both endpoints score within 1 pt of the soup's 0.722, in which case the three are indistinguishable"* — and it did not fire, which is what licensed the switch back to exp-04.
- **The comparator rule forced the decisive evals.** exp-08 exists only because the comparator basis changed under greedy; exp-07's `--limit 500` re-measurement of the same weights is what produced the p = 0.0076 result.
- **Stop hook: never fired.** No `.stop_hook.json` in the receipt-backed `task/memory/`; 0 open cards; `background tasks killed/stopped at exit: 0`; final `grep -l "decision: null"` → `none open` (L9478). The scientist also wrote a persistent memory note at 02:52:12 titled *"decode config is part of the deliverable"* (L9459) — the guard's own lesson, self-recorded.

## 7. Verdict

**Three largest contributors to 0.7218.**

1. **The format SFT (exp-02, epoch 1): 0.0867 → 0.6133 at n=150.** Mechanism verified exactly: `no_stop_maxtokens` 66/150 → 0/150, all 79 recovered base items, zero regressions. Also the correct call to *stop at epoch 1* — epoch 2 cost 4 points.
2. **On-policy RFT (exp-04): +6.0 pt**, the first lopsided watch-set diff in the cell (23 fixed / 13 regressed vs exp-03's 23/24), and the right decision to stop after round 2 failed.
3. **Greedy decode (exp-07): +5.8 pt at n=500, p = 0.0076 — found with 14 minutes of work left.** Had the 02:38 n=150 anomaly not appeared, this cell would very likely have shipped the soup under a sampling config; the greedy re-run also reversed the shipped checkpoint from the soup to exp-04. This is simultaneously the cell's largest gain and its largest near-miss.

**One protocol-surface idea (NEW, ≥2 source cells).** A v2 (#2) already targets the *timing* of the decode decision and is frozen; I am not re-proposing it. What A v2 does not cover, and what this cell shows sharply, is that **the decode configuration is not part of `evaluation.protocol`**, so rule 2 and the `comparator_same_protocol` preflight check are satisfied by matching `n`/dev-set/seed while the two sides of the comparison decode differently — or while the comparator becomes stale the moment the decode config changes.

- **Surface (one item, optional card field):** add an optional `evaluation.protocol.decode` field recording the effective `generation_config` of the scored artifact (e.g. `do_sample`, `temperature`), and have the existing `comparator_same_protocol` check compare it when both sides declare it.
- **Source cells:** g01r07 (comparators decode-mismatched for 8.1 h; ordering reversed once fixed — L8491, `analysis/greedy_selection.json`) and **g01r08**, the positive control, which had to enforce this by hand twice: `cp ckpts/exp-04-greedy/generation_config.json ckpts/exp-05/final/` before the comparison eval (L7766, 00:43:25Z, description *"Evaluate RFT model with greedy config"*) and again in exp-08's smoke (*"set greedy decode on both rival checkpoints"*). g01r06 sidesteps it a third way, by writing greedy in the training script.
- **Screen metric a 4-cell block can read:** number of cross-checkpoint comparisons whose two sides carried different effective `generation_config` (target 0/4 cells), plus the count of decision cards whose comparator was invalidated by a later decode change (target 0). Guardrail: the Round-02 protocol-pool score floor, and `fields_filled` must not fall (the field is optional).
- **Sequencing caveat, explicitly:** this is a second intervention in A v2's area. It should be stacked behind or folded into A v2 rather than launched in parallel — two simultaneous decode-area changes would make neither screen attributable. That call is the planner's, not mine.

**What this cell did that the other arm typically did differently.** It carried paired statistics into a shipping decision (McNemar, twice), which ledger #19/#4 records as the missing practice on both arms; and it used no on-policy RL, consistent with the protocol arm's pattern in ledger #8 — though here the decline is argued from time-to-recover, not from the protocol.

<!-- END REPORT cell=g01r07 -->

<!-- REPORT cell=g01r08 role=NEW -->

```yaml
cell: g01r08
arm: protocol (guard; batch exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1, awm 4ae3d87, setup --exp-protocol --tool claude --stop-hook)
accuracy: 0.7028051554207733        # = 927/1319; lowest of the six NEW guard cells; guard6 mean −0.0326, control6 mean −0.0486
hours_used: 7.45                     # time_taken.txt 07:27:17; trace 18:30:58Z → 01:58:01Z
hours_to_first_train_launch: 0.27    # card-matched exp-02 18:47:20Z; first GPU training command (smoke) 18:42:19Z = 0.19 h
protocol_hours: 0.00 (tool time, 56 calls) / 0.40 wall create→first-lock over 8 cards, of which 0.27 h overlapped a running n=500 eval → ~0.13 h non-overlapping
waiting_hours: 4.70 raw (33 calls), all blind clock sleeps; includes a demonstrated ≥0.20 h wait on an already-dead producer
greedy_shipped: yes — measured on identical weights as exp-04 at 22:57:34–23:04Z, +8.0 pt at n=150
rl_used: no
rft_tried: yes (exp-05) — contradicted / reject
largest_eval_n: 500 (4 runs)
stop_reason: "Done — no processes running, GPU free, all cards closed." (L8917, 01:58:01Z) with 2:32 (2.53 h) unused; the last rejected option was "Verify at n=1319 — ~60 min; leaves too little margin"
top_contributors: [few-shot-prefix continuation SFT 0.060→0.627, greedy decode +8.0 pt, ~0.9 h of mechanical bring-up loss and 2.53 h left on the clock]
one_protocol_change: queued P4 (#23, rule-8 wording on pricing the remaining time) is the most plausible score lever here; the uncovered mechanical item is tool-timeout kills of foreground model commands (proposal below, g01r08 + g01r07)
knowledge_to_transfer: n/a (protocol arm)
```

## 1. Timeline

Session `b88e048b`, 18:30:58Z → 01:58:01Z (7.45 h span; `time_taken.txt` 07:27:17). 12 timer readings, 9:59 → 2:32.

| time | event | evidence |
|---|---|---|
| 18:30:58 | `Skill` loads `exp_protocol`; SKILL/pitfalls/template read by 18:31:05 | L10, L178/L252/L407 |
| 18:31:09 | `awm exp_protocol index` | L570 |
| 18:33:30 | base `generation_config.json` read — `"do_sample": true` | L1850 → L1892 |
| 18:33:54 / 18:35:22 | exp-01 created / locked | cards |
| 18:35:28 | base n=150 eval launched (after lock) | L2525 |
| **18:42:19** | **first GPU training command: `timeout 900 python scripts/train_sft.py … --out ckpts/smoke --limit-rows 4000 --max-steps 12`** | L3679 |
| 18:43:21 | second smoke at bs 32 → CUDA OOM 21.65 GiB | L3747 → L3780 |
| 18:44:20 → 18:47:15 → **18:47:20** | exp-02 created → locked → **first card-matched launch** | cards / L4265 / L4269 |
| ~20:25:4x | exp-02 eval launched | L5394 (task completed 20:25:54) |
| **20:25:56** | **vLLM server dies: `Can't load image processor for ckpts/exp-02/final`** | L5519, L5530 |
| 20:25:57 → 20:38:04 | two blind sleeps (420 s, 300 s) then the log is finally tailed | L5422, L5459, L5495 |
| 20:38:17 | repair: `cp $SNAP/preprocessor_config.json $SNAP/processor_config.json ckpts/exp-02/final/` + relaunch | L5551 |
| ~20:46 | exp-02 = **0.060** at n=150 (below base) | card |
| **20:47:36 → 21:17:36** | **foreground python heredoc running vLLM killed: `Exit code 143 / Command timed out after 30m 0s`** | L5684 → L5738–5739 |
| 21:23:56 → 21:24:46/51 | exp-03 created → locked → launch #1 | L6200/L6215/L6219 |
| 21:28:55 | launch #1 OOM — **process 6893 (orphan vLLM probe) holding 64.46 GiB** | L6336 |
| 21:29:01–21:29:24 | orphan killed; GPU back to 0 MiB | L6348, L6422 |
| 21:29:28 → 21:32:28 | launch #2 (bs 8) → OOM 16.24 GiB | L6432 → L6466 |
| **21:32:37** | launch #3 (bs 4 × accum 8, `expandable_segments`) — succeeds | L6479 |
| **21:36:13** | exp-03 **re-locked**, 3.6 min *after* the launch that produced the shipped weights | exp-03.lock.json |
| 22:49:43 | processor configs copied into exp-03/final (preventive) + eval | L6764 |
| 22:56:57 → 22:57:34/39 | exp-04 (decode) created → locked → eval → **0.7067 (+8.0)** | cards / L7033 |
| **23:04:54** | RFT sampling launched **and** *"stage safety-net final_model"* (exp-03 weights + greedy config) | L7130–7132 |
| 23:51:17 → 23:52:09/14 | exp-05 (RFT) created → locked → launched | cards / L7641 |
| 00:43:25 | greedy config copied onto exp-05/final, then evaluated | L7766 |
| 00:50:51 | exp-06 created | card |
| **00:51:49** | **exp-06's first declared n=500 evaluation launched — 15.1 min before its lock** | L8106 |
| 01:06:55 / 01:06:58 | exp-06 locked / second n=500 eval | L8161 / L8167 |
| 01:23:12 → 01:23:54 | exp-07 soup → 0.666, rejected | cards / L8511 |
| 01:40:31 → 01:41:14 | exp-08 verification → `final_model` 0.706 @ n=500 | cards / L8780 |
| 01:58:01 | last assistant turn | L8916 |

**Corrections to the prepared timeline.** `final_model_written 22:57Z` is a false positive: 22:57:31 is card text inside exp-04's `alternatives_rejected` (*"…what generation_config final_model ships with"*, L6945). The first actual write is 23:04:54Z (+4.57 h). `first_rl 18:44Z` is the `grpo|dpo|ppo` substring artefact; RL launches = 0. `first_train_launch 18:42Z` is the smoke.

**Category correction.** The timeline's `data 0.57 h (21 calls)` is dominated by one item that is not data work: the 30-minute foreground vLLM probe at 20:47:36 that was killed by the tool timeout (L5681). Corrected: **failed compute ≈ 0.50 h**, `data ≈ 0.07 h`.

**Stop reason.** *"Done — no processes running, GPU free, all cards closed."* (L8917), preceded by a verified liveness check (`0 MiB`, 12 files in `final_model/`, `index ok`, `2:32` remaining, L8909–8914). With 2.53 h left the scientist had rejected a full-test verification 17 min earlier: *"Verify at n=1319 — ~60 min; leaves too little margin, and n=500 already gives a +/-2 point interval"* (L8689, 01:41:06Z, when ~2:49 remained). SAID "too little margin"; SHOWS 2.8× the required time.

## 2. Recipe decisions and their reasoning

- **exp-02 (SFT, base):** 127,173 rows from OpenMathInstruct-2 (`problem_source ∈ {gsm8k, augmented_gsm8k}`) + GSM8K train; lr 1e-5, 1 epoch, bs 16×2, max_seq_len 1024, `cosine_with_min_lr_0.1`. Result **0.060 at n=150 — below the base model's 0.073**, and the card is honestly marked `contradicted`.
- **The decisive diagnosis (exp-03).** The offline `diag_stop` probe showed the model at **0.76 zero-shot but 0.12 with the harness's 10-shot prefix**: it answered correctly, then continued the in-context Q/A pattern with an invented problem. The fix was a prompt-distribution one, argued explicitly against the two alternatives: *"Change the eval prompt / disable few-shot — evaluate.py and templates/ are immutable and the harness always uses fewshot=10"*, and *"Set eos/stop tokens differently … the model already emits `<end_of_turn>` correctly when the prompt is bare — the offline probe stops on it — so this is not a stop-token-id problem but a prompt-distribution problem."* exp-03 continued from exp-02 on 35,000 rows, ~20k with a `k~U{1,8}` harness-style few-shot prefix + 15k bare Orca-Math rows, keeping ~43 % zero-shot *"to protect the bare-prompt behaviour"*. Result **0.060 → 0.6267** and over-continuation 83 % → 0 %.
- **RFT (exp-05):** 23,206 self-generated correct chains over 13,006 problems (k=4, T=1.0), 15,000 re-rendered with a `k~U{1,10}` prefix, lr 5e-6 → 0.6733 vs exp-04's 0.7067 → rejected.
- **RL: declined with an explicit budget reason.** *"DPO on correct/incorrect sample pairs — pairs are available from the same sampling run, but DPO on a 4B model needs a reference-model forward pass and more tuning than the ~1 h of GPU left after data building; RFT is the [cheaper move]."* No RL command executed.
- **exp-07 (soup):** 0.5/0.5 of exp-03 and exp-05 → **0.666 at n=500, below both parents** (0.714, 0.690); rejected with a mechanism reading: *"the two checkpoints are evidently not close enough in weight space for uniform averaging to cancel error rather than blur a solution."* This is a useful negative for ledger #18 alongside g01r06's positive.
- Note the recipe difference against the other two cells: this cell is the only one whose gain came from **matching the training prompt distribution to the grader's 10-shot prefix**, which g01r07 measured as *unnecessary* for its own checkpoint (0.730 with prefix vs 0.688 without). Ledger #28 is right to hold this as an observation rather than a rule: the same intervention is decisive in one cell and unneeded in another.

## 3. Decode config

**Aware at 18:33:30, deliberately deferred, then measured cleanly at 22:57.** The base `do_sample: true` was read in the first three minutes (L1892). Unlike g01r07 the deferral was reasoned rather than accidental — exp-02 rejects *"Decode-config only (e.g. greedy generation_config.json) — cannot fix a base model that never stops after one answer"* (L3921), which was correct at that point.

exp-04 is then a textbook same-weights A/B: exp-03's weights (symlinked safetensors) with only `generation_config.json` changed → **0.6267 → 0.7067 at n=150** (`eval/exp-04_greedy_dev150.json`), 0/150 format failures both ways. The card states the mechanism plainly: *"Eight points were being lost to the base model's inherited temperature-1.0 sampling defaults, which the harness silently adopts from the checkpoint."* The correct fix location was identified too: *"Pass a temperature on the evaluate.py command line — evaluate.py is immutable and exposes no temperature flag; the generation config is the only supported route."*

Crucially, the scientist then **kept the decode variable matched across every later comparison**: it copied the greedy config onto `ckpts/exp-05/final` before evaluating it (L7766, 00:43:25Z, *"Evaluate RFT model with greedy config"*), and exp-08's smoke records *"ckpts/exp-04/final and ckpts/exp-05/final generation_config.json set to do_sample=false…"*. This is the discipline g01r07 lacked, and it is why this cell's n=150 and n=500 orderings agree.

Shipped `final_model/generation_config.json` in the receipt-backed dir: `do_sample:false, temperature:0.0, eos [1,106]`. A v2's clock: first post-SFT eval result ~20:46:50Z → measured decode choice ~23:04Z = **2.29 h**.

## 4. Evaluation practice

- n = 150 for exp-01…exp-05 (`--limit 150 --max-connections 16 --gpu-memory-utilization 0.6`), n = 500 for the three decisions that mattered (exp-06 selection, exp-07 soup, exp-08 verification). **Largest n = 500**, 4 runs; the full 1319 set was rejected for time (§1).
- Escalation was reasoned, not habitual: *"a 3.3-point gap at this n is within noise; two 500-item evals cost ~50 min of the 3.6 h left and turn the choice into a measurement"* (exp-06 alternatives_rejected), and the problem statement is unusually sharp — *"Shipping on that basis is a coin flip dressed as a decision"* (L7902, 00:51:16Z).
- **No inversion at larger n in this cell.** exp-04 > exp-05 at n=150 (0.7067 vs 0.6733) and at n=500 (0.714 vs 0.690), with the decode config matched on both sides. The scientist noted the consistency explicitly: *"the same ordering as at n=150 … now with a standard error half the size."*
- **Shipped-path verification** with a byte-identity check: `md5sum final_model/model-00001-of-00002.safetensors ckpts/exp-03/final/…` identical (`51626de5596dc93e77f872684bf68abe`), CPU load as `Gemma3ForConditionalGeneration`, 4.30 B params, then the official harness on `final_model/` → 0.706 ± 0.020 at n=500 vs 0.714 on the same weights via the symlinked dir — 0.8 pt of server-to-server variation, attributed correctly (*"vLLM batching nondeterminism under greedy decoding"*).
- `evaluate.py` only; `own-evaluator test-set cmds: 0`. No paired/McNemar statistic anywhere; the per-card `analysis/exp-NN_format_fail.json` files are failure-tag counts, not paired tests.
- **Counting caveats.** `evaluate.py direct calls: 10` but only **9** inspect logs exist in the receipt-backed `task/logs/` — the missing one is the eval that died at 20:25:56 before writing a log, so the asymmetry is itself the loss record. `est_n` reports 281 and 346 for two runs whose log headers show `"limit": 150` (the exp-02 and base checkpoints over-continued, inflating bytes/sample to ~83 KB and ~101 KB).
- Official `final_eval_1.txt`: 1319/1319 samples, `max_connections: 2`, 12:22, accuracy 0.703 (927/1319).

## 5. Every loss ≥ 0.1 h, with cause and cost

Raw `pitfalls_cost_h = 1.00`, with **no carry-forward duplication** in this cell (unlike g01r06), so raw ≈ event-deduplicated. Trace-bounded total ≈ 0.85–0.90 h.

| # | event | card cost | trace-bounded interval | covered by |
|---|---|---|---|---|
| 1 | Data cleaner rejected every solution whose answer was an inline `\boxed{}`: 515 of 153k rows survived | 0.10 h | detected 18:38:33 (L2951), `BOXED_RE` unwrap written 18:38:53 (L2972) — ~1 min visible, CPU only | not a protocol candidate (recipe) |
| 2 | `per_device_train_batch_size 32` OOM (262k-vocab logits ~22 GiB) | 0.05 h | 18:43:21 → 18:44:2x ≈ 0.02 h (L3747 → L3780) | #21 P2 (queued) |
| 3 | **`trainer.save_model()` does not copy `preprocessor_config.json`/`processor_config.json`; vLLM refuses the checkpoint (`Can't load image processor`) and the eval server exits — then two blind sleeps run on the dead producer** | 0.20 h | producer error logged **20:25:56**, first observation **20:38:04** = **≥0.202 h of idle on a dead run** (L5530 → L5495); repair 20:38:17 (L5551) | **#22 P3** (queued) for the cause; **#15 E v2** (frozen `c6f11d8`) for the idle |
| 4 | **A foreground python heredoc running vLLM hit the Bash tool's 30-minute limit and was killed** (`Exit code 143 / Command timed out after 30m 0s`); the probe had to be rewritten as `scripts/diag_stop.py` and re-run in the background | 0.50 h | 20:47:36 → 21:17:36 = **0.500 h exactly** (L5684 → L5739) | **uncovered** (see §7) |
| 5 | Orphaned background vLLM probe (pid 6893) still holding 64.46 GiB → exp-03 launch #1 OOM; then a genuine bs-8 OOM on launch #2 | 0.10 h | 21:24:51 → 21:29:28 = 0.077 h, plus 21:29:28 → 21:32:37 = 0.052 h; combined **0.13 h** (L6219 → L6336 → L6432 → L6466 → L6479) | #21 P2 (queued); the orphan-engine half is a `pitfalls.yaml` mechanism already listed under B (#3) |
| 6 | First `rft_mix` build drew both halves from the same file → 5,295 duplicated `(question,target)` rows | 0.05 h | detected 23:50:45, fixed 23:50:57 (L7407 → L7458) | not a protocol candidate (recipe) |

**A second, weaker post-exit idle.** exp-03 launch #3 (21:32:37) has `train_runtime 4233.93 s` and `[done] saved ckpts/exp-03/final` in the retained `task/logs/exp-03_train.log`; the next observation is at 22:49:33 after a 14-minute blind sleep. The log carries no timestamps and process start includes model load, so the producer exit is only bounded to roughly 22:45–22:47 → idle of **~0–6 min, not a demonstrated ≥0.1 h loss**. I report it as unresolved rather than counted.

**Traps prior knowledge avoided.** The pitfalls list changed the final decision: *"Ship without re-running the harness on final_model/ — the known pitfall is exactly this: a directory that loads for me but not for the grader (missing processor config cost 0.2 h earlier today)"* (exp-08 alternatives_rejected). Rule 2 shaped exp-01 (*"published numbers use a different harness and prompt; rule 2 requires a comparator measured under this protocol"*). And the scientist generalised its own 0.2 h loss preventively, copying the processor configs into exp-05/final at 00:43:25 before that checkpoint was ever served.

## 6. Protocol interaction

- **Cost.** `protocol` tool time 0.00 h over 56 calls. Card authoring (create → first lock) 24.0 min over 8 cards; **16.1 min of that is exp-06 alone (00:50:51 → 01:06:55) and it overlapped a running n=500 evaluation**, so non-overlapping authoring is ~8 min, mean ~1 min/card. Bootstrap first action (Skill → `index` returned) ≈ 11 s, 18:30:58 → 18:31:09.
- **Locks/overrides.** 8 cards, 8 closed, 0 open, **0 overrides**, 1 relock. Pre-lock `check` failures hidden by `preflight_fail: 0`: L2318–2319 (18:34:44Z), L3844 (18:44:08Z).
- **Ordering — two findings, both outside the matched denominator.** `lock_before_launch = 3/3` is correct for the first launch of each training card and I preserve it as stated. Separately:
  - **exp-03's shipped configuration was locked after it launched.** Launch #3 (bs 4 × accum 8) ran at **21:32:37**; the relock recording that change is stamped **21:36:13** — 3.6 min later. The cell-reader matches only the first launch (21:24:51, after the 21:24:46 lock), so this does not appear in the ratio. (g01r06 did the same repair sequence correctly, re-locking 5 s *before* each relaunch.)
  - **exp-06's first declared evaluation preceded its lock by 15.1 min.** exp-06 is an evaluation-only card whose whole intervention is two n=500 runs; the exp-05 run launched at **00:51:49** (L8106) and the lock landed at **01:06:55** (L8161), with the exp-04 run launched 3 s after the lock. J (#26) covers evaluation commands explicitly, and its frozen rule-1 text is the right home for both of these; nothing new is needed.
  - Pre-card GPU work, as in the other two cells: smoke trainings at 18:42:19/18:43:21 (exp-02 created 18:44:20) and 43 minutes of gold-filtered `vllm_gen.py --mode sample` at 23:04:54 (exp-05 created 23:51:17).
- **H (#5).** Five non-training cards (exp-01, exp-04, exp-06, exp-07, exp-08) all name `data/dev_train300.jsonl` in `setup.data`. For exp-01 that is genuine (it is the offline watch/dev set). The other four carry `mixture_weight 0` with `selection: "nothing is trained in this card"` — **4 non-applicable entries, 0 fabricated files, 0 overrides**.
- **`falsified_if` closed two directions.** exp-05 (`at or below 0.7067`) → 0.6733 → RFT rejected. exp-07 (`at or below 0.714`) → 0.666 → soup rejected. exp-02's predicate (`stays below 0.35`) fired correctly at 0.060 and produced an `iterate`, not a quiet retry — which is what turned the cell toward the prompt-distribution diagnosis.
- **The card format visibly forced the two n=500 evals** that decided the shipped model, and forced exp-08's end-to-end verification from the shipped path.
- **Stop hook: never fired.** No `.stop_hook.json` in the receipt-backed `task/memory/` (only `cards/` and `index.md`); 0 open cards; `background tasks killed/stopped at exit: 0`. Rule 9's reminder was printed at every lock (L2513, L4159, L4264, L6214) and `setsid nohup … < /dev/null &` was used for every training run, every eval and every sampling run — **but not for the 30-minute diagnostic that died** (§5 #4). One tool timeout, one loss.

## 7. Verdict

**Three largest contributors to 0.7028.**

1. **The few-shot-prefix continuation (exp-03): 0.060 → 0.6267 at n=150.** The whole cell turns on the `diag_stop` probe (0.76 bare vs 0.12 with the harness prefix) correctly identifying an in-context pattern-completion habit rather than a stop-token or maths problem — and on the fix keeping 43 % zero-shot rows so the bare-prompt ability survived.
2. **Greedy decode (exp-04): +8.0 pt at n=150 on identical weights**, then held constant across every subsequent comparison (L7766). This cell is the group's positive control for decode discipline.
3. **~0.9 h of mechanical bring-up loss and 2.53 h left unspent.** The 0.50 h tool-timeout kill, 0.20 h of blind waiting on a dead eval, and 0.13 h of OOM churn are all recoverable; more consequentially, the cell finished at 01:58 with 2:32 on the clock having declined a 60-minute full-test verification for "too little margin". It shipped the exp-03 weights, whose only competitor (exp-05) was 2.4 pt behind at n=500 — with 2.5 h it could have afforded a second training round or a 1319-item confirmation of the shipped artifact.

**The one protocol change most likely to have raised this cell: queued P4 (#23, SKILL rule-8 wording on pricing the remaining time).** The ledger already states P4's surface and target (*"余 ≥1.5 h 且无进程的 cell ≤1/4，且最后一次 `alternatives_rejected` 引用实测成本"*). This cell satisfies the failure condition twice over — 2.53 h idle at exit, and a rejected option priced at "~60 min / too little margin" against 2:49 actually remaining. All three cells in my group end ≥1.6 h early with no process alive (g01r06 2.27 h, g01r07 1.63 h, g01r08 2.53 h), so P4 gains three source cells from this window. I am not restating or extending its clauses.

**One protocol-surface idea (NEW, 2 source cells) — with an explicit collision caveat.** The 0.50 h loss here has a mechanism that no frozen candidate names: **the Bash tool's own timeout kills a foreground command that is running a model.** The frozen `run_dies_with_the_session` guidance says *"Launch **long runs** with `setsid nohup <cmd> > <log> 2>&1 < /dev/null &` so a tool timeout cannot take them down"* — the scientist applied that faithfully to every training, evaluation and sampling launch, and then lost a 30-minute *diagnostic probe* that it did not classify as a "long run".

- **Surface (one item, `pitfalls.yaml`):** extend the `run_dies_with_the_session` entry so its scope is *any command that loads or runs a model, including probes and diagnostics*, and so it names the two safe forms: background with `setsid nohup`, or foreground with an explicit inner `timeout` strictly below the Bash tool timeout.
- **Source cells:** g01r08 (realised loss: `Exit code 143 / Command timed out after 30m 0s`, L5738–5739, exactly 0.500 h) and g01r07 (same exposure, survived only because it set `timeout 1800` inside a Bash call with `"timeout": 1900000` — 100 s of headroom — three times: L5747, L6467, L7283).
- **Screen metric a 4-cell block can read:** number of model-executing commands terminated by a tool timeout (target 0/4 cells) and the hours attributed to them; secondary, the share of model-executing commands that are either backgrounded or carry an inner `timeout`.
- **Collision caveat, stated plainly:** E v2 (#15, frozen `c6f11d8`) already edits this exact `pitfalls.yaml` entry *and* nudges in the opposite direction — *"Prefer launch/wait/exit-status handling and subsequent evaluation in one foreground script."* Applied without a timeout-aware wrapper, that preference would reproduce g01r08's 0.5 h loss. My recommendation is therefore **not** a separate candidate but an addition to E v2's readout: count tool-timeout kills of model-executing commands as a guardrail failure alongside "false death classifications, premature card closure and lost work". Whether to reopen E v2's frozen text or carry this as a screen observable is the planner's decision, not mine.

**What this cell did that the other arm typically did differently.** No on-policy RL (ledger #8's protocol-arm pattern), and — unlike the broader-plan control style recorded in ledger #16 — a strictly serial single-lever chain of eight cards, each rejecting the next-largest alternative on measured cost. The distinctive positive is decode-matched comparison, which neither arm has been shown to do systematically.

<!-- END REPORT cell=g01r08 -->

---

# Cross-cell reviewer notes (guard-b group)

**Group position.** All three of my cells sit below the guard-6 mean of 0.7354056103108415 (−0.0136 to −0.0326) and below the protocol-free control-6 mean of 0.7514531210513015 (−0.0297 to −0.0486). g01r08 is the lowest of the six NEW guard cells. The baseline-v3 comparison (n=2) is descriptive only. Official counts: 934, 952 and 927 of 1319; all three official runs completed 1319/1319 with `max_connections: 2` and identical input-token totals (2,910,587), which is consistent with the same fixed prompt set but does not prove per-item identity.

**What is common to all three.**

1. **Zero Stop-hook activity.** No `.stop_hook.json` exists in any of the three receipt-backed original `task/memory/` directories, 0 cards were left open, 0 background tasks were killed at exit, and all three ended deliberately with a verified-idle GPU. Consistent with the strict cohort's result, and — as ledger #1 already says — *not* evidence that the block works, only that it was not needed. What the trace does show is that rule 9's guidance reached the scientist repeatedly: `awm exp_protocol lock` prints the `run_dies_with_the_session` reminder, so it appeared at every one of the 23 locks.
2. **Real GPU work before any covering card, in all three cells.** Smoke trainings at 18:42:08–18:42:19 (5–9 min before the exp-02 card existed), and 13–43-minute **gold-filtered self-sampling runs** 25–47 min before their RFT cards. g01r08 adds a 15-minute pre-lock *evaluation* on an evaluation-only card. All are recorded retrospectively in `situation.smoke_runs`. This is candidate J's exact predicate, including its forward-review point that graded generation is evaluation; the window adds three NEW source cells to #26.
3. **Prepared-marker false positives, now explained.** `first_rl` in all three cells is spurious: `tools/exp_protocol_trace_timeline.py:100` matches `grpo|dpo|ppo` case-insensitively against tool text, so it fires on substrings like "su**ppo**rted" in card YAML. `first_train_launch` matches the *smoke*, not the card's run. `final_model_written` matched script source in g01r07 and card prose in g01r08; it was a genuine write in g01r06 but of an interim checkpoint. In all three cells the shipped `final_model/` held a *different* checkpoint from the first thing written to that path — g01r07 wrote it three times with two different models.
4. **All three ended ≥1.6 h early with nothing alive** (2.27 / 1.63 / 2.53 h). Two of the three under-priced the remaining budget when justifying the stop. Three source cells for queued P4 (#23).
5. **Zero `--override` in all three cells, and three different workarounds for the same schema gap.** g01r07 created a placeholder file so a decode-config card could have a `setup.data`; g01r08 attached a `mixture_weight 0` corpus to four non-training cards; g01r06 to two. H's frozen framing (fake files, non-applicable entries and overrides counted separately; genuinely applicable references permitted) is the right lens, and the window adds 1 fabricated file + 6 non-applicable entries from this group.

**Where the three cells diverge, and what that is worth.** The group is close to a natural experiment on decode discipline. g01r06 wrote greedy into its training script from the first checkpoint (and paid 0.85 h for it, because a greedy parent breaks `save_pretrained`); g01r08 deferred deliberately, measured +8.0 pt on identical weights at 22:57, and then held the decode config constant across every later comparison; g01r07 knew at 18:33 and did nothing until 02:38, when an anomalous repeat exposed it — its candidate ranking then inverted and its shipped checkpoint changed 4 minutes before the session ended. A v2 (#2) reads the *timing* of that choice; nothing frozen reads whether the two sides of a comparison decoded the same way, which is the proposal I raised in the g01r07 report (sourced from g01r07 + g01r08, sequenced behind A v2).

**Same-artifact repeat variance, quantified.** g01r06 gives the cleanest measurement in the window: byte-identical weights (shard-1 md5 verified) scored 0.71494 / 0.70811 / 0.70887 on three vLLM server instances at n=1319 — 943 / 934 / 935 correct, a 9-item spread, with the official run landing at the low end. g01r07 shows 0.736 vs 0.728 on the same weights at n=500, and 0.664 vs 0.58 on the same weights under *sampling* at n=500 vs n=150. g01r08 shows 0.714 vs 0.706. These are all at or above the margins the cells used to choose a checkpoint. g01r06 also demonstrates the affordability of the fix: a full 1319-item run cost ~4 minutes here, and it ran eight of them. If C (#4) is redesigned, that is the movable criterion the ledger asks for; I am not proposing it, since C v2 was withdrawn and the redesign is the planner's.

**Metrics-hygiene items this group confirms.** (a) `pitfalls_cost_h` double-counts: g01r06's raw 2.30 h contains a verbatim 1.05 h carry-forward, event-deduplicated 1.25 h. (b) `est_n` from log bytes is unreliable in both directions here — 497 for an actual n=200 base eval, 281/346 for actual n=150 runs — while the retained log headers carry `"limit"` directly. (c) `evaluate.py direct calls` is not an evaluation audit: 11 vs 13 logs (r06), 10 vs 14 (r07), 10 vs 9 (r08, the deficit being the eval that died before writing a log). (d) `lock_before_launch = N/N` matches first launches only; g01r08's shipped exp-03 configuration was re-locked 3.6 min *after* the launch that produced it.

**Remaining uncertainties.**

- **g01r08's exp-03 post-completion idle is unresolved** (~0–6 min). `task/logs/exp-03_train.log` has no timestamps and `train_runtime` excludes model load, so the producer exit cannot be pinned. Only the 20:25:56 → 20:38:04 event (≥0.202 h) is demonstrated, because that one is timestamped in the log itself.
- **Item-set identity of the official runs is not verifiable** from what is retained. g01r06's official 934/1319 equals its own first `final_model` full-test run exactly, but no official per-item log survives in the receipt-backed directory — the same boundary P5 left open (#24). I make no claim beyond equal scalars.
- **The tool-timeout item has one realised loss and one near-miss** in my group, so it clears the two-cell bar only in the weak sense (exposure in both, realisation in one). If another group's cells show a second realised kill, it strengthens materially; if not, the planner may prefer to carry it purely as an E v2 guardrail observable rather than as anything more.
- **g01r06's greedy-from-the-start policy is a genuine trade-off I cannot resolve from three cells**: it guarantees a measured-decode deliverable but is the direct cause of the D-class save failure, and it removes the same-weights decode evidence A v2 wants to see. Whether A v2 should count "greedy written a priori, never A/B'd" as a pass, a fail, or a separate category is a screen-design question for the planner.
- I did not read the control or baseline traces, so every cross-arm statement above is grounded in the frozen ledger (#8, #16, #19) rather than in traces I inspected. Cross-group synthesis owns those comparisons.
