I have all the evidence I need. Here are the three reports.

<!-- REPORT cell=c01s01 role=NEW -->
```yaml
cell: c01s01
arm: control (protocol-free; manifest awm.paths excludes skills/exp_protocol, setup "--tool claude")
accuracy: 0.7892342683851402      # 1041/1319, official, stderr 0.0112
hours_used: 8.75                  # time_taken.txt 08:45:34; 18:31:20Z -> 03:16:24Z
hours_to_first_train_launch: 0.11 / 0.50   # 0.11 h = 400-row smoke 18:37:57 (died on ENOSPC); 0.50 h = first production SFT on the full 97k at 19:01:33
protocol_hours: 0                 # no protocol installed; self-structure = 5 TaskCreate + 6 TaskUpdate, 0.00 h tool time
waiting_hours: 7.19               # timeline tool-time in waiting_on_runs (31 calls). GPU-free by system_monitor: 0.32 h cumulative, largest single span 0.13 h
greedy_shipped: yes               # finalize.py written 18:58:55 with do_sample:false/temperature:0.0 as the default path; every artifact went through it
rl_used: yes                      # GRPO (TRL, Dr.GRPO, beta=0), 3 runs; shipped artifact is GRPO checkpoint-350
rft_tried: yes (adopted as the GRPO parent: rft1 0.788 vs sft1 0.778 @500; rft1 0.7415 @1319)
largest_eval_n: 1319              # six full-test evals; a full-test read cost 1 min 29 s at --max-connections 64
stop_reason: "Done. All processes finished, GPU idle, `final_model/` is in place." (L6595, 03:16:24Z) with 1:14 left on timer.sh (L6578) and 0 processes / 0 MiB confirmed (L6579-6580)
top_contributors: [GRPO stage worth +5.84 pts at n=1319 over its own RFT parent (0.7415 -> 0.7998, same protocol, same parent path), near-total GPU occupancy (0.32 h GPU-free in 8.75 h; only one idle span >= 0.1 h), catching TRL's truncation-mask/eos-id bug in 7 minutes so the RL stage produced a model at all]
one_protocol_change: "already-frozen G (#6, TRL EOS zero gradient) — a pitfalls entry whose symptom is `loss/grad_norm identically 0` with `completions/clipped_ratio: 1.0`; this cell is a fresh NEW-cell exposure and would have saved ~0.17 h and the risk of a silently null 4-hour RL run"
knowledge_to_transfer: [a full 1319-item greedy eval costs ~1.5 min at --max-connections 64, so n=1319 for the final selection is essentially free; write the shipped generation_config in the checkpoint-packaging script so no artifact can ever leave without it; verify the shipped final_model against the selected checkpoint by content hash, not by path]
```

# c01s01 — protocol-free control, official 0.7892 (highest of the six NEW controls)

Session `ca186d6e-…`, 18:31:20Z → 03:16:24Z, 211 assistant turns, 6691 trace lines, **0 cards** (no protocol installed). `L` = line in `solve_parsed.txt.gz` unless another file is named. Arm identity confirmed from the frozen manifest `experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8.yaml:61-64` (`paths` without `skills/exp_protocol`, `setup: "--tool claude"`), not from the AWM SHA.

## 1. Timeline

| when | what | evidence |
|---|---|---|
| 18:31:20 | start; `ls -la && bash timer.sh` → 9:59 | L10, L25 |
| 18:31:26–18:31:55 | reads `evaluate.py`, all four templates, `inspect_evals/gsm8k/gsm8k.py`, package versions | L83, L706, L814 |
| 18:32:47–18:32:51 | **4 `TaskCreate` calls** = its self-imposed plan (baseline / data / SFT / rejection sampling) | L922–L958 |
| 18:32:54 | base eval launched, `--limit 150 --max-connections 32 --gpu-memory-utilization 0.85` | L970 |
| 18:35:12–18:35:26 | **reads inspect's vLLM provider source and greps the baseline log** — finds `"Default sampling parameters have been overridden by the model's Hugging Face generation config"` | L1456, L1466, L1477, L1639 |
| 18:36:11–18:38:44 | liger into `/usr` → `OSError: [Errno 28] No space left on device`; `df` shows `fuse-overlayfs 64M 64M 0 100% /`; reinstalled with `--target /home/ben/task/pylibs` (0.04 h) | L1664, L2438, L2454, L2562 |
| 18:36:44–18:37:32 | `prep_data.py` → 97,473 rows; contamination check → 0 flagged (`work/contam_out.jsonl` 0 bytes) | L1897, L2136, L2228 |
| 18:37:01 | base = **8.0 %** @150 (`work/baseline.json`) | L2049, L2074 |
| 18:37:57 | first training process (400-row smoke) dies on the same ENOSPC | L2406–2438 |
| 18:40:39–18:50:41 | capacity sweep over dtype/bs/accum/gc/attn (9 configs, 3 OOMs) → bs 64 fp32-master + bf16 autocast | L2730, L2778, L2828, L2878; OOMs L2768, L2866, L2868 |
| 18:50:41 | 6k-row pilot launched | L2972 |
| **18:58:55** | **`finalize.py` written — greedy `generation_config` is the default, `--keep-sampling` is the opt-out** | L3247–3284 |
| 19:01:21 | pilot = **67.3 %** @150 | L3402 |
| **19:01:33** | **first production run: full 97k SFT**, bs 32, lr 1.5e-5, 1 epoch (train_runtime 4052.9 s) | L3426; `task/logs/sft1.log` |
| 20:13:14 | sft1 → **0.785** @200 | L3908, `logs/sft1_eval.log` |
| 20:15:44–20:41:09 | `gen_rs.py` k=8 @T=1.0 on GSM8K train → **solved 7199/7473 = 0.963, kept 41973** | L3957; `logs/rs1.log` |
| 20:41:26 | RFT run from the SFT checkpoint (20.5k RS + 12k replay, lr 7e-6, 1332 s) | L4149 |
| 21:06:28–21:08:58 | rft1 **0.7767** @300; then sft1 **0.778** / rft1 **0.788** @500 | L4596, L4644 |
| 21:13:19 | GRPO 3-step smoke | L4796 |
| **21:17:04** | GRPO run 1 launched (400 steps, save every 50) | L4880–4888 |
| 21:23:47 | **`'loss': 0.0, 'grad_norm': 0.0 … 'clipped_ratio': 1.0`** at steps 5–9 | L4925–4929 |
| 21:24:19–21:26:23 | kills it, probes vLLM stop behaviour with two throwaway scripts (`work/chk.py`, `work/chk2.py`) | L4975, L5045, L5130 |
| **21:27:27** | relaunch with `trainer.eos_token_id = 106`; by 21:34:30 loss ≈ −0.004, grad_norm 0.06, reward 0.70–0.82 | L5200–5211, L5249–5268 |
| 21:34:42 | **safety net: `final_model` = `work/rft1_bf16`** | L5279 |
| 01:15:51 | GRPO 1 killed near step 350–378 | L5622 |
| 01:16:01–01:24:01 | ckpt 250 / 300 / 350 @500 → **0.792 / 0.788 / 0.812** | L5687–5691 |
| 01:24:29–01:40:52 | GRPO 2 (resume, max-steps 460) → last-10 reward 0.7465, killed without evaluation | L5738, L5822, L5836 |
| 01:41:01–01:48:48 | **g350 0.7998 / g250 0.7809 / rft1 0.7415 @1319** | L5895–5898, L5942 |
| 01:49:02 | GRPO 3 from `work/g350`, new data seed | L5962 |
| 01:54:07 | **`final_model` replaced with `work/g350`** | L6012 |
| 02:39:23–03:05:55 | GRPO 3 killed; ckpt 100 / 75 @1319 → **0.7877 / 0.7801** (both below g350) | L6125, L6208–6211 |
| 03:06:21–03:11 | 4-checkpoint soup → **0.7854** @1319 (rejected) | L6280, `work/soup.json` |
| 03:11:39 | **default `python evaluate.py` on `final_model` → 0.7667 @150** | L6384, `work/final_default.json` |
| 03:15:41 | **`md5sum` of `final_model` vs `work/g350` shard 1 → `2 b05b330928a7a9dccaf3829511806462`** (identical) | L6478–6480 |
| 03:16:10–03:16:24 | timer 1:14, 0 processes, 0 MiB, `RESULTS.md`, final turn | L6577–6595 |

Hours by category (tool execution time, `exp_protocol_trace_timeline.py`): waiting 7.19 h / sample_eval 0.72 h / train_launch 0.44 h / bash_other 0.06 h / data 0.02 h; model generation 0.28 h. **Correction:** those are tool-occupancy figures, not idle. `task/system_monitor.log` (60 s sampling) shows GPU memory <2 GiB for only **0.32 h cumulative** across the whole session, in 24 spans of which exactly one reaches 0.1 h.

## 2. Recipe decisions and their reasoning

The thinking blocks in this trace are redacted (`"thinking": ""`, e.g. L2443), and the cell emitted only **two** assistant text turns (L6 and the closing summary L6594). Reasoning must therefore be read from the `TaskUpdate` bodies, the code it wrote, and the order of its commands — I mark SAID vs SHOWS accordingly.

- **Data.** 90k from OpenMathInstruct-2 `gsm8k`/`augmented_gsm8k` + the 7,473 GSM8K-*train* references = 97,473 rows, targets rewritten to `…\n\nANSWER: N`. SAID (TaskUpdate L2247, 18:37:35): *"work/sft_v1.jsonl: 97473 examples … 15% carry a random 1-10 shot prefix. Contamination check: 0/97473 flagged."* SHOWS: `prep_data.py` written 18:36:33 (L1676), run 18:36:44 (L1897), checker run 18:37:14 (L2136) with an empty `work/contam_out.jsonl`.
- **Few-shot prefix.** 15 % of rows carry a random 1–10-shot prefix, "so the model ignores the eval's 10-shot system message style" (`task/RESULTS.md`). This is the same lever ledger #28 flags; here it is a design choice with no measured ablation.
- **SFT hyperparameters.** 1 epoch, bs 32, lr 1.5e-5 cosine, fp32 master weights + bf16 autocast, 8-bit AdamW, liger; chosen by the 18:40–18:50 sweep, not asserted.
- **RFT.** k=8 at T=1.0, 96.3 % of train questions solved at least once, 41,973 kept, retrained from the SFT checkpoint at lr 7e-6 on 20.5k + 12k replay. Verdict on the card-equivalent (TaskUpdate L5282, 21:34:43): *"RFT (20k RS + 12k replay, lr 7e-6 from sft1): 78.8% @500 vs sft1 77.8% @500."* — a +1.0 pt read at n=500 that it accepted without a paired test.
- **On-policy RL.** Adopted and pursued hardest of my three cells: Dr.GRPO, β=0 (no reference model), 8 generations × 32 prompts, lr 1e-6, vLLM colocate, exact-match reward. SAID (TaskCreate L5327, 21:34:51): *"Run GRPO from rft1 … eval checkpoints, pick best, package as final_model."* SHOWS three GRPO processes (21:17:04, 21:24:29 resume, 01:49:02 continuation) and the shipped artifact is checkpoint-350.
- **Risk/budget reasoning.** Almost none is verbalised. It read the timer only 4 times in 8.75 h (9:59, 9:40, 9:12, 1:14 — L25, L2979, L3617, L6578) and computed a wall-clock deadline once (`python -c "…1788460261+10*3600…"`, L3659, 19:18:56). The visible risk management is behavioural: an early `final_model` safety net at 21:34:42 while the RL run was still unproven, replaced only after g350 had been measured at n=1319.

## 3. Decode config

**Shipped: yes, from the first packaged artifact onward.** `finalize.py` (L3247–3284, 18:58:55) writes `{"do_sample": false, "temperature": 0.0, "top_p": 1.0, "top_k": -1}` by default and only writes the stock sampling config behind an explicit `--keep-sampling`; `grep` across the trace shows that flag is defined (L3261) and **never used**. The final `final_model/generation_config.json` was re-read at 01:54:07 (L6012) and the artifact matched `work/g350` by md5 at 03:15:41.

**When and on what basis.** The decision was made *before* any post-SFT evaluation, from two mechanical observations at 18:35:12–18:35:26: the inspect vLLM provider source (L1456–1478) and the baseline log line *"Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator"* (L1639).

**Gain measured: none on its own trained model.** The only sampling-vs-greedy contrast this cell has is base-model 8.0 % under the stock config (L2049) — i.e. it never ran the A/B that c01s03 ran (+5.7 pts at n=300). Under candidate **A v2's** metric ("first post-SFT eval → *measured* decode choice ≤ 0.5 h"), this cell had already shipped greedy 0.36 h *before* its first post-SFT eval, with source/log verification rather than a measurement. That is a definitional edge case A v2's screen should resolve before launch: the top-scoring control in this group would otherwise be scored as "unmeasured".

## 4. Evaluation practice

Only `evaluate.py` (md5 `2490079a39a1e242ab1b5286a0087137`, identical in all three of my cells); **0 own-evaluator test-set commands**.

| decision | n behind it | evidence |
|---|---|---|
| pilot recipe accepted | 150 | `work/pilot.json` 0.6733 |
| SFT accepted | 200 | `logs/sft1_eval.log` 0.785 |
| RFT accepted over SFT | 300, then 500 | 0.7767; 0.778 vs 0.788 @500 |
| GRPO checkpoint chosen | **500 then 1319** | 0.792/0.788/**0.812** @500 → **0.7998**/0.7809 @1319 |
| grpo3 / soup rejected | 1319 | 0.7877, 0.7801, 0.7854 |
| shipped artifact verified | 150 (grader defaults) | `work/final_default.json` 0.7667 |

**Inversions and n-dependence.** The @500 slice reads ~1.1–1.2 pts *higher* than the full test for the same artifact (g350 0.812→0.7998; g250 0.792→0.7809), and the grader's default 150 reads ~3.3 pts *lower* (0.7667). The scientist stated the mechanism (`RESULTS.md`: *"the first 500 test items are noticeably easier than the rest"*). The g250-vs-g300 ordering at n=500 (0.792 > 0.788) is inside noise; the top-1 choice did not invert.

**Paired statistics: none.** Every comparison is a scalar difference. This is the one evaluation practice c01s02 and c01s03 both did and this cell did not.

**Cost of full-test evaluation.** `logs/g350_full.log` records `total time: 0:01:29` for 1,319 samples at `--max-connections 64`. Six full-test evaluations cost this cell roughly 10 minutes in total.

**Dev vs official on the identical artifact.** `final_model` is byte-identical to `work/g350` (md5, L6478–6480). The scientist measured **0.7998** (1055/1319); the official run measured **0.7892** (1041/1319) — a 14-item, 1.06 pt gap on the same weights. Three things differ and cannot be separated: `--max-connections` 64 vs 2, `--gpu-memory-utilization` 0.85 vs 0.3 (`final_eval_1.txt` header; `evaluate.py:41-55` defaults), and the image (`opus_5.sif` for the scientist vs `vllm_debug.sif` for the official eval, `runtime_provenance.json`). This is ledger #24/P5 territory, where the planner has already ruled out new concurrency rules; I record the gap and explicitly do **not** attribute it. Note also that the sign is not systematic: c01s03's dev full-test read is 0.60 pt *below* its official one.

## 5. Pitfalls — every loss ≥ 0.1 h

| # | window | cost | cause | coverage |
|---|---|---|---|---|
| 1 | 18:40:39–18:50:41 | **0.17 h** | capacity sweep for Gemma-3 4B; 3 of 9 configs OOM in CE/logits (L2768, L2866, L2868) | **covered — #21 / P2** (queued). Partly deliberate search; the avoidable part is the three OOM configs, which I cannot separate cleanly from the productive part of the sweep |
| 2 | 21:17:04–21:27:27 | **0.17 h** (0.12 h of it GPU) | TRL compares the last token against `tokenizer.eos_token_id` (1) while Gemma-3 turns end with `<end_of_turn>` (106); with `mask_truncated_completions=True` every completion is masked → `'loss': 0.0, 'grad_norm': 0.0, 'clipped_ratio': 1.0` (L4925) | **covered — #6 / G** (queued, `f28dd88` built, not registered). Ledger says the strict guard cohort showed no GRPO exposure; **this NEW control cell is a fresh exposure of the same mechanism** |
| 3 | 20:33–20:41 | **0.13 h** | GPU sat empty after `gen_rs.py` exited because the wait was a fixed `sleep 1400` issued at 20:17:33 (L4005) rather than a producer check | **covered — #15 / E, E v2.** Worth calibrating on: this is the cell's *only* ≥0.1 h idle event and its 0.32 h cumulative GPU-free time is dominated by sub-0.05 h stage transitions, so under E's `<0.15 h/cell` primary metric this cell is close to the bar from the passing side |
| 4 | 01:24:29–01:40:52 | 0.27 h | GRPO 2 (resume to step 460) killed on a reward trend, never evaluated | exploration with a negative read, not a defect. Adjacent to **#23 / P4** (late-session pricing) |
| 5 | 01:49:02–03:05:55 | 1.28 h | GRPO 3 from g350 + two full-test evals of its checkpoints; both below g350 | same class as #4; `final_model` was already safe from 01:54:07, so the downside was bounded |

Below the threshold but diagnostic:

- **0.04 h, 18:36:11–18:38:44** — `uv pip install --system liger-kernel` into the 64 MB `/` overlay → ENOSPC (L2438, L2454), repaired by installing to `pylibs` and `PYTHONPATH`. Same root cause as c01s02's silent corruption, different symptom. **#21 / P2.**
- **0.02 h, 03:06:21–03:07:40** — `soup.py`'s `save_pretrained` refused the config: *"`top_k` is set to `-1` — this flag is only used in sample-based generation modes … Fix these issues to save the configuration"* (L6312–6315). This is **D's mechanism (#14)** appearing in a NEW cell. It cost 0.02 h here only because the save was in the foreground and the output was read immediately — the same fault inside a background `Trainer.save` is what D prices at 5.4 h over five cells.

**Traps prior knowledge avoided**, with the line that shows the knowledge:

- `add_special_tokens` audit of TRL before removing the literal `<bos>` from the GRPO prompt (L4262–4266 at 20:44:08, patch at L4531–4534 at 20:44:36) — the double-`<bos>` trap in #3/B.
- `finalize.py` copies `preprocessor_config.json` / `processor_config.json` from the base snapshot **and then writes the generation config** (L3272–3284), i.e. it takes the P3 (#22) repair without letting it overwrite the measured decode config — exactly the constraint the ledger records for P3.
- The vLLM stop-token probes at 21:24:29 and 21:26:19 (`work/chk.py`, `work/chk2.py`, results at L5176–5182) tested `greedy` vs `greedy+stop106` before changing the trainer, rather than guessing.

## 6. Control-arm structure, waiting, and ending

**Structure it imposed on itself.** There are no cards, but the cell built a five-item ledger out of the CLI's own task tool: 5 `TaskCreate` + 6 `TaskUpdate` calls (L922, L934, L946, L958, L5327; L2074, L2247, L3414, L5282, L6549, L6560). The updates carry exactly the fields a card's `result`/`conclusion` would: dataset size and contamination status (L2247), a comparator with its n (L5282: *"78.8% @500 vs sft1 77.8% @500"*), and a final verdict with the alternative it beat (L6560: *"GRPO step 350 is best: 79.98% full test. Steps 400-450 and a 4-ckpt soup were worse (78.0-78.8%)."*). Total tool time for all eleven calls: **0.00 h** (`tool_taskcreate` + `tool_taskupdate` rows in the timeline). It also wrote a 3.2 KB `RESULTS.md` at the end (L6484) containing the pipeline, the results table and three "Notes" that read like pitfall entries. What it did **not** produce is anything pre-registered: no hypothesis, no `falsified_if`, no comparator declared before the run.

**How it waited.** Fixed-duration foreground sleeps sized to the run (`sleep 400/420/1300/1500/1700/2900/3300/3500`), each combined with a log-extraction command, plus `ps aux | grep "[g]rpo_train"` with the bracket trick that avoids self-matching. Result: 0.32 h GPU-free in 8.75 h. The one place the clock-based estimate overshot the producer cost 0.13 h (§5 #3).

**How it ended.** Verified state before claiming completion: `bash timer.sh; ps aux | grep -E "[t]rain_sft|[g]rpo_train|[e]valuate.py|[g]en_rs" | wc -l; nvidia-smi …; ls final_model` → `1:14`, `0`, `0 MiB`, twelve files (L6573–6592). 1.23 h of budget unused, no live process, nothing killed at exit.

## 7. Verdict

**Three largest contributors to 0.7892**

1. **The RL stage, run long enough to matter.** Same parent, same protocol, same n: `work/rft1_bf16` = **0.7415** @1319 (L5942) → `work/g350` = **0.7998** @1319 (L5895). **+5.84 pts from GRPO alone**, and it is the whole difference between this cell and a strong SFT+RFT cell. Train reward rose 0.75 → 0.90 over 350 steps (`RESULTS.md`, computed from `logs/grpo1.log` at L5391).
2. **GPU occupancy.** 0.32 h GPU-free in 8.75 h let three training stages, an 8-sample-per-question rejection-sampling pass and six full-test evaluations all fit. For contrast, c01s02 ran the same three-stage recipe but lost 2.87 h to idle and got 90 LoRA GRPO steps instead of 350 full-parameter ones.
3. **Seven-minute diagnosis of the TRL truncation-mask bug.** Without the 21:27:27 fix, contributor #1 does not exist: the run would have burned ~4 h at `grad_norm 0.0`.

**The one protocol change most likely to have raised this cell:** the already-frozen **G (#6)** — a pitfalls entry keyed on `loss` and `grad_norm` identically zero together with `completions/clipped_ratio: 1.0`, naming the `eos_token_id` vs `<end_of_turn>` mismatch. It would have returned ~0.17 h here, and more importantly it removes the tail risk of a silently null RL run, which is the largest single lever in this cell. I am not proposing anything new for it; the candidate exists and is queued.

**What this cell did that the protocol arm typically did differently:** it wrote *zero* pre-registered artifacts and still made every shipping decision at n≥500 with a full-test confirmation, and it kept a loadable `final_model` from 21:34:42 onward while continuing to explore. Its cheap self-ledger (0.00 h of tool time for eleven task updates) recorded comparator, n and rejected alternatives — the same information a card carries, at roughly zero cost.

<!-- END REPORT cell=c01s01 -->

<!-- REPORT cell=c01s02 role=NEW -->
```yaml
cell: c01s02
arm: control (protocol-free; same frozen manifest anchor as c01s01)
accuracy: 0.7816527672479151      # 1031/1319, official, stderr 0.0114
hours_used: 8.61                  # time_taken.txt 08:37:32; 18:31:18Z -> 03:07:39Z
hours_to_first_train_launch: 0.16 / 0.25   # 0.16 h = 3k-row smoke 18:41:01 (OOM); 0.25 h = first production SFT 18:46:25
protocol_hours: 0                 # no protocol installed; no self-ledger tool either (0 TaskCreate/TaskUpdate)
waiting_hours: 7.62               # timeline tool-time (40 calls). GPU-free by system_monitor: 3.06 h cumulative, of which 2.87 h in three post-exit events
greedy_shipped: yes               # make_final.py 18:50:17 writes do_sample=true/temperature=1e-6/top_k=1 (valid for both HF and vLLM)
rl_used: yes                      # GRPO, LoRA r=32, 90 steps, merged into final_model
rft_tried: yes (adopted: 28,772 verified traces at 2x weight inside the stage-2 SFT mixture)
largest_eval_n: 500               # never evaluated any artifact on the full 1319 test set
stop_reason: "Done. `final_model` is in place and verified with the default `python evaluate.py` invocation." (L7410, 03:07:39Z) with 1:24 left (L7089)
top_contributors: [203k format-matched stage-2 SFT mixture with RFT data at 2x weight, a 90-step LoRA GRPO round that added +1.6 pts at n=500, paired McNemar at n=500 for the final selection; against these, 2.87 h (33% of the session) of GPU idle from name-pattern process waits]
one_protocol_change: "already-frozen E v2 (#15) — 'track the actual producer rather than a launcher or residual engine; check process state and exit evidence'. This cell's three idle events all come from `while pgrep -f \"<script>.py\"` loops that matched their own bash wrapper; the repair it eventually adopted (`echo $! > /tmp/x.pid; while kill -0 $PID`) is exactly E v2's text, and it eliminated all further idle"
knowledge_to_transfer: [a name-pattern process wait can match the wait loop's own shell - keep the PID and use `kill -0`; a tqdm progress line can be flushed *after* a traceback, so `tail -1` of a training log is not a liveness signal; encoding greedy as do_sample=true/temperature=1e-6/top_k=1 passes HF GenerationConfig validation while do_sample=false/top_k=-1 does not]
```

# c01s02 — protocol-free control, official 0.7817 (second of the six NEW controls)

Session `dd7d6aa5-…`, 18:31:18Z → 03:07:39Z, 211 assistant turns, 7510 trace lines, **0 cards**. Thinking blocks are redacted; this cell did leave nine short assistant text turns, which I quote as SAID.

## 1. Timeline

| when | what | evidence |
|---|---|---|
| 18:31:18–18:32:33 | reads `evaluate.py`, templates, `gsm8k.py`, base `config.json` + `generation_config.json`, inspect's vLLM provider | L83, L231, L809, L927, L993 |
| 18:33:25 | SAID: *"Let me start the baseline eval in the background and prepare data in parallel."* | L1426 |
| 18:33:26 | base eval launched `--limit 150 --max-connections 16` | L1430 |
| 18:36:01–18:36:03 | **reads the scorer source** `inspect_ai/scorer/_match.py` and `_common.py` before shaping targets | L1866, L1927 |
| 18:37:13–18:37:35 | `prep_data.py` → **152,888** rows; decontamination input built and checked | L2092, L2262, L2410 |
| 18:40:30 | base = **7.3 %** @150 | `runs/baseline.json`, L2742 |
| 18:41:01 | first training process (3k smoke) → OOM in the unfused CE at the 262k vocab | L2853, L2926 |
| 18:41:35 | `uv pip install --system liger-kernel` → prints `ok`; **the `chunked_loss` submodule is silently written with null bytes** (discovered 6.6 h later) | L2939; L5870–5879 |
| 18:43:04 / 18:44:29 | capacity probes at bs 32/accum 4 and bs 64/accum 2 | L3081, L3123 |
| 18:46:15 | `df -h /home/ben/task /home/ben /tmp` — **`/` is not checked** | L3164–3171 |
| **18:46:25** | **production SFT-1 launched** (bs 64 × accum 2, lr 1e-5, fewshot-prob 0.15) | L3202 |
| 18:50:17 | `make_final.py` written: bf16 + greedy generation config + processor files | L3353 |
| 18:54:56 | `while pgrep -f "train_sft.py" …` wait started, 5400 s tool timeout | L3705 |
| ~19:53 / 19:55 | SFT-1 ends (`train_runtime` 3986.1 s → 66.5 min); monitor shows GPU at 0 from **19:55** | `logs/sft1.log`, L3770; `system_monitor.log` |
| 20:24:56 | the wait is **moved to the background on its timeout**, not because the run ended | L3751 |
| 20:25:02 | `memory.used 0 MiB`, `saved to runs/sft1` | L3768–3774 |
| 20:25:34 | sft1 → **0.765** @200 (greedy) | `runs/sft1_greedy.json` |
| 20:28:03 | SAID: *"76.5% (from 7.3% baseline). Now let me generate on-policy rejection-sampling data."* | L3974 |
| 20:28:05 | `gen_rft.py` launched, 16 samples @T=1.0 | L3978 |
| ~21:07 / 21:08 | RFT generation finishes (`Processed prompts: 100% … [38:08]`); GPU idle from **21:08** | L4211 |
| 21:14:12 / 21:24:22 | two more `while pgrep -f "gen_rft.py"` waits time out into the background | L4191, L4273 |
| 21:25:48–21:26:21 | `prep_mix.py` → 202,658 rows; **SFT-2 launched with `--max-len 1280`** | L4474, L4645 |
| ~21:29:41 / 21:30 | **SFT-2 OOMs at step 50** in Liger fused-linear CE; `logs/sft2.log` mtime 21:30; GPU idle from **21:30** | L5158, L5213, L5157 |
| 21:34:01 | `tail -1` of the log returns `3%| | 50/1633 [03:20<1:46:02, 4.02s/it]` — **a healthy-looking progress line written after the traceback** | L4929 |
| 21:34:31 | third `while pgrep` wait started (7200 s timeout) | L5068 |
| 23:34:31 | that wait times out; 23:34:39 `pgrep -af "train_sft.py"` returns **only bash wrappers** (pids 3101/7962/11093), `0 MiB, 0 %` | L5129, L5154–5161 |
| 23:35:01 | SAID: *"Training crashed at step 50 with OOM (I raised max-len to 1280). Let me secure the current best model first…"* | L5225 |
| 23:35:02 / 23:35:11 | `final_model` = `models/sft1` (safety net); SFT-2 relaunched at `--max-len 1024` **with `echo $! > /tmp/sft2.pid`** | L5230, L5278–5281 |
| 01:11:38–01:13:42 | SFT-2 ends (5829.4 s); sft2 → **0.790** @200 | L5505, `runs/sft2.json` |
| 01:16:01 | `final_model` = `models/sft2` | L5665 |
| 01:16:23–01:18 | repetition-penalty variant `models/sft2_rp` → **0.775** @200 (rejected) | L5742, `runs/sft2_rp.json` |
| 01:18:28 / 01:21:01 / 01:23:12 / 01:25:24 | four GRPO launches: liger null-byte `ImportError` → missing chat template → OOM → success (bs 8 × accum 16, 90 steps) | L5799/L5837, L5986/L6030, L6045/L6091, L6104 |
| 01:56:27–01:57:55 | GRPO exits (1797.1 s); `merge_lora.py` fails twice on the generation-config validation, then patched | L6282, L6321, L6389, L6427 |
| 01:58:46–02:03:16 | grpo1 **0.790** @200; sft2 **0.782** @500; grpo1 **0.798** @500 | `runs/*.json` |
| 02:01:36 | SAID: *"GRPO gave no gain (79.0%, same as SFT2). Let me run larger evaluations to pick the final model more reliably."* | L6543 |
| 02:04:56 | **paired McNemar at n=500: sft2-only 21, grpo1-only 29, both 370, two-sided p ≈ 0.322** | L6646–6669 |
| 02:05:12 | SAID: *"GRPO is directionally better (29 vs 21 flips). Promoting it…"* | L6679 |
| 02:05:13 / 02:05:24 | `final_model` = `models/grpo1`; default eval → **0.7933** @150 | L6684, `runs/final_default.json` |
| 02:08:50–03:00:54 | GRPO round 2 → merged, **0.788** @500, rejected (grpo1-only 40 vs grpo2-only 35) | L6780, L7008–7009 |
| 03:04:00 | chat template added **into `final_model` in place**, after the 02:05 verification | L6995–7002, L7015 |
| 03:04:06 | second default eval → **0.80** @150 | L7052, `runs/final_check.json` |
| 03:06:55–03:07:02 | kills the four stale wait loops; `0 MiB`; prints every `runs/*.json` | L7100, L7111–7123, L7139 |
| 03:07:26–03:07:39 | `README.md`; final turn, 1:24 left | L7326, L7089, L7410 |

Hours by category (tool time): waiting 7.62 h / train_launch 0.32 h / bash_other 0.16 h / sample_eval 0.16 h; model generation 0.28 h. **Correction:** `system_monitor.log` puts GPU memory below 2 GiB for **3.06 h cumulative**, concentrated in three events of 0.50 h, 0.30 h and 2.07 h.

## 2. Recipe decisions and their reasoning

- **Data.** Stage 1: the GSM8K-derived slice of OpenMathInstruct-2, 152,888 rows, `\boxed{}` stripped, reformatted so the last line is `ANSWER: <n>` — chosen *after* reading the scorer at 18:36:01 (L1866) and confirming it is `match(numeric=True, location="end")` (`README.md`). Stage 2: 202,658 rows = OMI-2 + RFT at 2× weight (3× for low-pass-rate problems).
- **Few-shot prefix.** 12–15 % of rows carry a random k-shot prefix (`--fewshot-prob 0.15` at L3204, 0.12 at L4645).
- **SFT hyperparameters.** Full-parameter, vision tower frozen, fp32 master + bf16 autocast, 8-bit Adam, gradient checkpointing, Liger fused linear CE, `group_by_length`, effective batch 128, lr 1e-5 cosine, 1 epoch. `--max-len` is the load-bearing knob: 1280 OOMs, 1024 works (`README.md`: *"Keep `--max-len 1024` at `--bs 64` — 1280 OOMs on an 80 GB H100."*).
- **RFT.** 16 samples @T=1.0 on the 7,473 train questions from the stage-1 model → 28,772 verified traces, 79.8 % sample pass rate, 98.3 % of problems solved at least once (`README.md`). Adopted into the stage-2 mixture rather than used as a separate stage.
- **On-policy RL.** Considered and run: LoRA r=32 GRPO, β=0, 8 samples/prompt, 90 steps, binary correctness reward with a −0.2 penalty for a missing `ANSWER:` (`task/train_grpo.py:39-49`), plus a `--filter-easy 16` prompt filter that drops problems the RFT stats show as always-solved. **Its GRPOConfig does not set `mask_truncated_completions`** (`task/train_grpo.py:106-134`), which is why the eos-106 trap that hit c01s01 did not bite here — a genuine difference in exposure, not a fix.
- **The only explicit budget/risk reasoning in text** is at 23:35:01 (L5225), where it secures a safety-net artifact before relaunching. Otherwise budget awareness shows up as behaviour: 20 timer reads (vs 4 in c01s01), and a second GRPO round started at 02:08:50 with 3 h left and abandoned on a n=500 read.

## 3. Decode config

**Shipped: yes.** `make_final.py` (L3353, 18:50:17) sets `do_sample: true, temperature: 1e-6, top_p: 1.0, top_k: 1` — the final `final_model/generation_config.json` is printed at L7027–7039. SAID (`README.md`): *"`final_model` ships a greedy one (`temperature=1e-6, top_k=1`, valid for both HF and vLLM)."*

That encoding is not cosmetic. c01s01 shipped the mathematically equivalent `do_sample: false, top_k: -1` and paid 0.02 h when `save_pretrained` refused it (c01s01 L6312–6315); c01s02's form passes HF validation and still forces greedy in vLLM. Three cells, three valid encodings of "greedy" for the same grader.

**When:** 18:50:17, i.e. **before** any post-SFT evaluation and 0.32 h after the production launch, on the strength of the base-model diagnosis. **Gain measured on its own model: none** — like c01s01, the only sampling contrast is the 7.3 % base floor. It *did* run one adjacent decode measurement: `repetition_penalty=1.05` at 01:16:23 → **0.775 vs 0.790 @200**, rejected (`README.md`: *"the penalty cost more than it recovered"*). Same A-v2 definitional issue as c01s01.

## 4. Evaluation practice

Only `evaluate.py`, via a `run_eval.sh` wrapper (`--max-connections 32 --gpu-memory-utilization 0.6`); **0 own-evaluator test-set commands**.

| decision | n | value |
|---|---|---|
| SFT-1 accepted | 200 | 0.765 |
| SFT-2 accepted | 200 | 0.790 |
| repetition penalty rejected | 200 | 0.775 |
| **grpo1 vs sft2 (shipping decision)** | **500 + paired** | 0.798 vs 0.782; McNemar p ≈ 0.322 |
| grpo2 rejected | 500 + paired | 0.788; grpo1-only 40 vs grpo2-only 35 |
| shipped artifact verified | 150 (grader defaults) | 0.7933, then 0.80 |

**Inversion at larger n — resolved correctly.** At n=200 grpo1 and sft2 were tied at 0.790 and the scientist SAID *"GRPO gave no gain"* (L6543, 02:01:36); at n=500 they separated to 0.798 vs 0.782. This is the clearest in-window instance of a 200-sample read being uninformative.

**Paired statistics: yes, and correctly executed.** The McNemar code at L6646–6662 reads `x['scores']['match']['value']=='C'` per item from the retained inspect JSONs and reports discordant pairs plus a two-sided exact p. The scientist promoted on direction with p = 0.322 explicitly on the table. Per `metrics.md`, that is a documented tie-break after a null paired test, not misconduct — and the 500-sample accuracies were not tied (0.798 vs 0.782).

**Largest n = 500. It never read the full test set.** `logs/eval_*.log` confirm every eval was 150/200/500 samples. The relevant cost figure: c01s01's `logs/g350_full.log` shows **1 min 29 s** for all 1,319 samples at `--max-connections 64`, against this cell's 32 s for 500 at `--max-connections 32`. A full-test confirmation of the shipping decision was roughly **one extra minute** of GPU, in a session that had 1:24 of budget left. Its official 1319 score, 0.7817, sits 1.6 pts below its n=500 estimate of 0.798.

**Same path, different artifact, same weights.** `final_default` (0.7933 @150, 02:05:24) and `final_check` (0.80 @150, 03:04:06) are 119/150 and 120/150 on identical weights (`model-*.safetensors` mtime 02:05, L7018–7019); the only intervening change was `chat_template.jinja` + `tokenizer_config.json` at 03:04:00 — and `evaluate.py:120-134` always passes `templates/gemma3.jinja` to vLLM regardless, so that edit could not affect the grade. The 1-item difference is same-artifact repeat variance (ledger #19), and the 03:04 edit was ~0.02 h of unnecessary work.

## 5. Pitfalls — every loss ≥ 0.1 h

| # | window | cost | cause | coverage |
|---|---|---|---|---|
| 1 | ~19:54 → 20:25:34 | **0.50 h** (monitor 19:55–20:25) | SFT-1 exited; `while pgrep -f "train_sft.py"` never returned because the pattern matches the wait loop's own `bash -c` wrapper; the tool call was backgrounded on its 5400 s timeout (L3751), not by the run ending | **covered — #15 / E v2** |
| 2 | ~21:08 → 21:26:21 | **0.30 h** | identical mechanism on `gen_rft.py`; two waits timed out at 21:14:12 and 21:24:22 | **covered — #15 / E v2** |
| 3 | ~21:30 → 23:35:11 | **2.07 h** GPU-free; **2.00 h** conservative non-overlapping post-exit idle (21:34:31 → 23:34:34, the interval with no other tool activity) | SFT-2 OOM at step 50 (`--max-len 1280`, Liger fused-linear CE chunk, L5213), then the same pgrep self-match; compounded by `tail -1` at 21:34:01 returning `3%| | 50/1633 …` because tqdm flushed one more progress refresh *after* the traceback (L4929, L5158–5159) | **covered — #15 / E v2** (the wait) + **#21 / P2** (the OOM) |
| 4 | 01:18:28–01:25:24 | 0.12 h | four GRPO launches: (a) `ValueError: source code string cannot contain null bytes` from `liger_kernel/chunked_loss` — a **silently corrupt install** into the 64 MB `/` overlay that had reported success 6.6 h earlier (L5837, L5870–5879); (b) `default chat template is no longer allowed` (L6030); (c) OOM at bs 32 (L6091) | **covered — #21 / P2** (install) and **#3 / B v2** (vLLM/TRL defaults). Each failure was caught inside ~2 min because every launch was followed by `sleep 115` + a log tail — the same cell that lost 2.07 h to a bad wait handled these correctly |
| 5 | 02:08:50–03:00:54 | 0.87 h | GRPO round 2 (150 steps, harder prompt filter) → 0.788 @500, rejected | exploration with a negative read; adjacent to **#23 / P4** |

**Total GPU idle: 3.06 h of an 8.61 h session (36 %), of which 2.87 h is in the three E-class events.** These are conservative: `system_monitor.log` samples at 60 s and "GPU memory < 2 GiB" would also flag a CPU-only productive step, but the trace shows no other work inside events 1–3 beyond two `Write` calls at 21:32:03 and 21:34:14, which I have already excluded from the 2.00 h figure for event 3.

**The repair the cell found itself** is E v2's frozen text, discovered the expensive way: at 23:35:11 it switched to `echo $! > /tmp/sft2.pid` + `while kill -0 $PID` (L5281, L5326), and after that point `system_monitor.log` records **no idle span ≥ 0.1 h** for the remaining 3.5 h. E v2's secondary metric — "number of proxy-only liveness decisions" — would score this cell 3 before 23:35 and 0 after.

**Traps prior knowledge avoided:** reading `_match.py`/`_common.py` before choosing the answer marker (L1866, 18:36:01); `make_final.py` saving `AutoProcessor` files with the comment *"processor files keep the multimodal config self-consistent for vLLM"* (P3/#22 taken without clobbering the decode config); the HF-safe greedy encoding described in §3.

## 6. Control-arm structure, waiting, and ending

**Structure it imposed on itself: less than c01s01.** Zero `TaskCreate`/`TaskUpdate` calls; the only persistent record is the 3.8 KB `README.md` written in the last 13 seconds (L7326, 03:07:26), which does include a "Things that did not help" section with both rejected experiments and their numbers. During the session, state lived in `runs/*.json` and a `run_eval.sh` wrapper (L5042, 21:34:28) that standardised model path, limit, connections and log directory — a real comparator-protocol discipline, built at the same moment it was about to compare artifacts. The nearest thing to a hypothesis is the one-line SAID turns (nine of them, §1). No `falsified_if` equivalent exists: the second GRPO round was started with no stated stopping rule and killed on a number.

**How it waited: the failure mode of this cell.** Name-pattern `pgrep` loops for the three longest producers (§5), fixed `sleep` for short checks, and — after 23:35 — a PID file with `kill -0`. It never checked exit status; it inferred death from a log read.

**How it ended.** At 03:06:55 it discovered four wait loops still alive, killed them, SAID *"Those were the stale wait-loops I just killed — all real work had already completed."* (L7322), and re-verified `0 MiB` plus every result file (L7111–7123). 1.4 h of budget unused; the last useful measurement was 2.5 minutes before the end.

## 7. Verdict

**Three largest contributors to 0.7817**

1. **A 203k format-matched stage-2 SFT** built from the scorer's own contract (`match(numeric=True, location="end")`, read at 18:36:01) with 28,772 verified RFT traces at 2× weight — this is what takes 7.3 % → 78.2 % @500.
2. **A 90-step LoRA GRPO round**, +1.6 pts at n=500 over its SFT-2 parent (0.782 → 0.798), selected with a paired test.
3. **Evaluation discipline at the decision points**: n=500 with per-item paired comparison for both the promotion and the rejection, and two end-to-end verifications with the grader's default invocation.

**The largest cost:** 2.87 h — a third of the session — of GPU idle behind three broken process waits. That is not a scoring hypothesis I can prove, but the mechanical consequence is visible: this cell ran GRPO for **90 LoRA steps** where c01s01, with the same recipe shape and 0.32 h of idle, ran **350 full-parameter steps** and gained +5.84 pts from its RL stage instead of +1.6.

**The one protocol change most likely to have raised this cell:** the already-frozen **E v2 (#15)**, unchanged. Its predicate — "track the actual producer rather than a launcher or residual engine; check process state and exit evidence" — is exactly what fails here three times, and the cell's own late repair is exactly what the predicate prescribes. I am not proposing new wording; I record the concrete sub-mechanism (a `pgrep -f <script>` pattern matching the wait loop's own shell, and a tqdm line flushed after a traceback) as calibration input for E v2's screen, since its secondary metric already counts proxy-only liveness decisions.

**What this cell did that the protocol arm typically did differently:** it built its comparator protocol as an executable wrapper (`run_eval.sh`) rather than a declared field, and it did paired per-item statistics that no card field requires — while skipping the full-test read that both of my other cells performed for ~1.5 min of GPU.

<!-- END REPORT cell=c01s02 -->

<!-- REPORT cell=c01s03 role=NEW -->
```yaml
cell: c01s03
arm: control (protocol-free; same frozen manifest anchor as c01s01)
accuracy: 0.7202426080363912      # 950/1319, official, stderr 0.0124; tied in scalar with c01s07
hours_used: 8.39                  # time_taken.txt 08:24:28; 18:31:19Z -> 02:54:40Z
hours_to_first_train_launch: 0.11 / 0.20   # 0.11 h = 256-row smoke 18:37:52; 0.20 h = first production SFT-1 18:43:39
protocol_hours: 0                 # no protocol installed; no TaskCreate/TaskUpdate either
waiting_hours: 7.44               # timeline tool-time (37 calls). GPU-free by system_monitor: 0.10 h cumulative, no span >= 0.05 h
greedy_shipped: yes               # finalize.py 18:53:34, pre-validated with a GenerationConfig round-trip at 18:51:26
rl_used: no                       # no GRPO/PPO command anywhere in the trace; cell-reader RL launches = 0
rft_tried: yes (adopted: k=4 rejection sampling -> 40,922 verified traces folded into the 206k SFT-2 mixture)
largest_eval_n: 1319              # five full-test evaluations
stop_reason: "Done. `final_model` holds the best checkpoint." (L6413, 02:54:40Z) with 1:37 left (L6318), after confirming 0 training/eval processes and 0 MiB (L6333-6335)
top_contributors: [the +5.7 pt measured greedy decode fix (63.0 -> 68.7 at n=300), the 206k SFT-2 mixture with 41k rejection-sampled traces (+2.1 pts to 70.8 at n=1319), a broad low-LR third pass (+0.6 to 71.4); the pipeline stopped at SFT with no RL stage, which is where the ~6 pt gap to c01s01 lies]
one_protocol_change: "none of the frozen candidates would have moved this cell materially - it has 0.10 h of GPU idle, a measured decode choice, five full-test evals and a paired per-item comparison. The single change with a plausible path is A v2 (#2), and only because its metric would credit what this cell already did; the actual gap is a recipe choice the protocol is explicitly not allowed to make (ledger #8)"
knowledge_to_transfer: [validate a generation_config by round-tripping it through GenerationConfig.from_pretrained before making it the packaging default; poll for the producer's output artifact (`while ! [ -f <out>/model.safetensors.index.json ]`) rather than for a process name - it produced the lowest GPU idle of the three cells; a narrow low-LR polish pass over an unrepresentative subset cost -5.1 pts at n=1319, keeping the source mixture representative is what mattered]
```

# c01s03 — protocol-free control, official 0.7202 (lowest of my three; tied in scalar with c01s07)

Session `c00b71ce-…`, 18:31:19Z → 02:54:40Z, 180 assistant turns, 6518 trace lines, **0 cards**. This is the cleanest-run and lowest-scoring cell in my group, which is the most useful fact in this report.

## 1. Timeline

| when | what | evidence |
|---|---|---|
| 18:31:19–18:32:25 | reads `evaluate.py`, templates, `inspect_evals/gsm8k/gsm8k.py`, and **the scorer source via `inspect.getsource`** | L82, L230, L736, L888, L995 |
| 18:32:45 | symlinks the frozen snapshot to `base_model` | L1175 |
| 18:32:59 | base eval launched `--limit 150 --max-connections 32 --gpu-memory-utilization 0.5` | L1237 |
| 18:33:34 | 4 OpenMathInstruct-2 shards downloaded in the background | L1293 |
| 18:35:40 | base = **6.7 %** @150 | `logs/base.json` |
| 18:35:46 | **opens the eval's own sample JSON** to read the exact prompt/target shapes | L1383 |
| 18:36:49–18:37:19 | `prep_data.py` → 97,473 rows; contamination input + check (0 matches, `logs/contam1.jsonl` empty) | L1604, L1771, L1967 |
| 18:37:45–18:41:25 | `train_sft.py`; 256-row smoke; FA2 probe; 512/bs16; eager-vs-FA2 sweep at 2048 | L1988, L2192, L2264, L2347, L2598 |
| 18:39:14 | `uv pip install --system liger-kernel` → **succeeds** (`+ liger-kernel==0.8.2`, L2313) | L2306 |
| **18:43:39** | **production SFT-1 launched** (97k, bs 32 × accum 2, lr 1e-5, max-len 1536, fewshot-prob 0.12) | L2643 |
| 18:50:19–18:51:16 | reads base and smoke `generation_config.json`, the eval log's plan/config block, inspect's vLLM provider, vLLM `get_diff_sampling_param`, and `openai_compatible.py`'s parameter builder | L3061, L3132, L3246, L3329, L3489 |
| **18:51:26** | **round-trips a candidate greedy config through `GenerationConfig.from_pretrained('/tmp/gctest')`** before adopting it | L3617–3637 |
| 18:53:34 | `finalize.py` written: greedy config **plus** `templates/gemma3.jinja` baked into `tokenizer_config.json` and `chat_template.jinja` | L3688; `task/finalize.py` |
| 18:55:43–19:08:07 | wait-primitive churn: three background `sleep 1500/1560/1500` fired within 5 s, several short sleeps, one `sleep 3000 & wait`, then it settles on `while ! ls work/sft1_model/model.safetensors.index.json` | L3791, L3830, L3874, L4155, L4250, L4349 |
| 19:56:42 | SAID: *"SFT-1 finished (train loss 0.35). Now let me evaluate it, both with default sampling and with greedy decoding."* | L4633 |
| 19:56:44 → 19:58:41 | **SFT-1 with the stock sampling config: 0.630 @300** (`total time 0:00:27`) | L4639, L4673 |
| 19:58:45 → 20:00:28 | **SFT-1 greedy: 0.687 @300** — **+5.7 pts, measured** | L4686, L4720 |
| 20:00:42 | `final_model` = SFT-1 (greedy); `gen_rft.py` launched, k=4 @T=1.0 over 27,473 problems | L4732–4734 |
| 20:22:37 | 8 more OMI shards downloaded | L4953 |
| 20:37:49 | large data build launched (`--n-gsm 150000 --n-math 40000`) | L5012 |
| 20:41:39 → 22:20:01 | `while ! [ -f work/rft.jsonl ]` wait; RFT yields **40,922** verified traces (82 % of problems solved ≥ once) | L5123; `README.md` |
| 22:20:13–22:20:50 | LaTeX/`$`-filter fix to `prep_data.py`, rebuild, merge → `work/sft2.jsonl` **206,395** rows | L5199, L5222, L5262 |
| 22:21:08 | **SFT-2 launched** (from the base checkpoint, same recipe) — `train_runtime` 9577.9 s | L5329 |
| 23:33:40 | "polish" subset assembled while SFT-2 runs | L5463 |
| 01:02:11–01:04:28 | SFT-2 greedy **0.704 @500**; `final_model` = SFT-2; **polish pass launched** (lr 2e-6/3e-6, narrow RFT-heavy subset) | L5522, L5568 |
| 01:41:51 → 01:46:29 | **polish @1319 = 0.657** — 5.1 pts below the incumbent, rejected | L5645, L5680 |
| 01:46:35 → 01:48 | `final_model` (=SFT-2) **@1319 = 0.7081** | L5691; `logs/sft2_full.json` |
| 01:49:44 | error analysis over the eval JSON (stop reasons, missing answers, lengths) | L5737 |
| 01:49:58–01:53 | `soup.py`; soup(SFT-1,SFT-2) **@1319 = 0.6846**, rejected | L5780, L5858 |
| 01:53:33 | SAID: *"Best so far is SFT-2 at 70.8% on the full test set (already in `final_model`). Let me verify it works with the grader's default flags."* → **0.700 @150** | L5905, `logs/final_default.json` |
| 01:56:56–02:44:03 | SFT-3 = a fresh mixture-preserving 60k subset at low LR; `train_runtime` 2760.6 s | L5978, L6023 |
| 02:44:03 → 02:47:10 | **SFT-3 @1319 = 0.7142**; **paired per-item flips: sft2-only 45, ep2-only 53** | L6064, L6139–6140 |
| 02:47:18 | `final_model` = SFT-3 (`"promoted ep2 (71.4%)"`); soup(SFT-2,SFT-3) built | L6152, L6185 |
| 02:47:52 → 02:50 | soup23 **@1319 = 0.7149** (1 item above SFT-3); SFT-3 kept | `logs/soup23_full.json` |
| 02:51:00 → 02:53:56 | **default `evaluate.py` on `final_model` → 0.720 @150** | L6242, L6279 |
| 02:54:00–02:54:40 | artifact listing + `generation_config.json` (`do_sample: false, temperature: 0.0`) + `Gemma3ForCausalLM`; timer 1:37; 0 processes, 0 MiB; `README.md`; final turn | L6284–6318, L6333, L6338, L6413 |

Hours by category (tool time): waiting 7.44 h / sample_eval 0.46 h / train_launch 0.18 h / data 0.03 h; model generation 0.21 h. **Correction:** `system_monitor.log` shows GPU memory below 2 GiB for **0.10 h cumulative** across the whole session, with **no span reaching 0.05 h** — the best occupancy of the three cells and comfortably inside E's `<0.15 h/cell` bar.

## 2. Recipe decisions and their reasoning

- **Data.** 12 of 32 OMI-2 shards scanned (~5.2 M rows), split by `problem_source` into GSM8K-derived and MATH-derived, integer/decimal answers only, LaTeX-display solutions dropped, `\boxed{}` unwrapped, every solution re-terminated with `\n\nANSWER: <number>`; plus the 7,473 GSM8K-train references. Stated rationale (`README.md`): *"the harness scorer … takes the *last* numeric token, so the response must stop at the answer."* Contamination: 0 across 97,473 / 197,473 / 40,922 documents.
- **Few-shot prefix.** 12 % of rows carry a 1–4-shot system prompt built from GSM8K-train demos, *"so the model keeps its own solution style instead of imitating the terse few-shot demos it sees at eval time"* (`README.md`) — the same reasoning ledger #13 records from two Round-00 cells, arrived at independently again.
- **SFT hyperparameters.** Full-parameter, 1 epoch, lr 1e-5 cosine, effective batch 64, max-len 1536, bf16 + 8-bit AdamW + Liger fused CE + gradient checkpointing, length-grouped, completion-only loss. Attention backend and batch size chosen by the 18:38–18:43 sweep with an `ATTN` env switch (L2264).
- **RFT / STaR.** k=4 at T=1.0 over 27,473 problems (all 7,473 GSM8K-train + 20,000 augmented), 82 % solved at least once, 40,922 traces after answer-checking and near-duplicate removal, folded into a 206k stage-2 mixture retrained **from the base checkpoint** (not from SFT-1). Verdict: adopted, 68.7 % → 70.8 %.
- **On-policy RL: never considered.** `grep -i "grpo|\bRL\b|reinforce|policy grad|ppo"` over the whole trace returns only base64 noise inside redacted thinking signatures — no command, no plan entry, no text mention. The cell-reader's `RL launches = 0` is correct here (unlike g01r04, where the prepared marker and the reader disagreed).
- **The one recipe risk it took and lost.** SFT-3 was attempted twice. The first attempt ("polish", narrow RFT-heavy subset at 3e-6) scored **0.657 @1319** against a 0.708 incumbent and was discarded; the second ("broad", mixture-preserving 60k at 2e-6) scored 0.714 and was shipped. Its own conclusion (`README.md`): *"keeping the source mixture representative was what mattered."*
- **Budget reasoning** is behavioural: 15 timer reads, and one explicit deadline computation at 19:03:48 (`python -c "print((1788460261+36000-$(date +%s))/3600)"`, L4138). No text weighs one direction against another.

## 3. Decode config

**Shipped: yes, and it is the only cell of the three that measured the gain on its own trained model.**

- **Verified before adopting:** at 18:51:26 it wrote a candidate `generation_config.json` to `/tmp/gctest` and round-tripped it through `GenerationConfig.from_pretrained`, accepting the warning *"The following generation flags are not valid and may be ignored: ['temperature']"* (L3636) because vLLM reads the raw JSON — a check neither other cell ran.
- **Packaged at 18:53:34** in `finalize.py` (`do_sample: false, temperature: 0.0, eos_token_id: [1, 106]`), which also bakes the grader's `templates/gemma3.jinja` into the artifact.
- **Measured at 19:56:44–20:00:28:** the same SFT-1 weights, same n=300, same protocol — **0.630 sampling → 0.687 greedy, +5.7 pts** (L4673, L4720). `README.md` states it as *"Decoding default (+5.7 pts)"*.

Under candidate **A v2's** metric — hours from the first post-SFT evaluation to a measured decode choice — this cell scores **0.06 h** with a genuine A/B. It is the positive control for that metric in this group.

## 4. Evaluation practice

Only `evaluate.py`; **0 own-evaluator test-set commands**. Full-test evaluations cost `total time: 0:03:06` at `--max-connections 48` (L5679).

| decision | n | value |
|---|---|---|
| decode choice | 300 (paired configs) | 0.630 → 0.687 |
| SFT-2 accepted | 500, then 1319 | 0.704 → 0.7081 |
| polish rejected | **1319** | 0.6573 |
| soup(1,2) rejected | **1319** | 0.6846 |
| **SFT-3 promoted** | **1319 + paired flips** | 0.7142; sft2-only 45 vs ep2-only 53 |
| soup(2,3) not adopted | **1319** | 0.7149 (1 item above SFT-3) |
| shipped artifact verified | 150 (grader defaults) | 0.700 (SFT-2), then 0.720 (SFT-3) |

**Paired statistics: yes, at full test.** L6111–6126 loads every retained inspect log, keys artifacts by `d['eval']['model']` and sample count, and counts discordant items between `vllm/final_model` @1319 (SFT-2, evaluated 01:48, before the 02:47 swap — the binding is correct) and `vllm/work/ep2_greedy` @1319. No p-value is computed; the 45-vs-53 split is well inside noise, but the accuracy difference it acted on is the same 0.6 pt.

**A one-item judgement worth naming honestly.** soup(SFT-2,SFT-3) measured 0.71494 (943/1319) against SFT-3's 0.71418 (942/1319), and SFT-3 had already been promoted at 02:47:18, before the soup was measured at 02:47:52. The cell describes this as *"Weight soups … gave nothing over SFT-3"* — accurate for a 1-item difference. The official score (0.7202, 950/1319) is above both, so the choice was immaterial.

**Same path, different checkpoints.** `final_model` held SFT-1 (20:00:42), SFT-2 (01:04:28) and SFT-3 (02:47:18). The two default-flag evaluations of "final_model" — 0.700 and 0.720 @150 — are different artifacts, not a repeat measurement. `logs/final_default.json` and `final_default2.json` are only distinguishable by filename and trace timestamp; nothing in the artifact records which checkpoint it is.

**Dev vs official.** Its own full-test read of the shipped SFT-3 was **0.7142** (942/1319, `--max-connections 48 --gpu-memory-utilization 0.6 --max-tokens 1024`); the official read is **0.7202** (950/1319, mc 2, gpu 0.3, max-tokens 4000, `vllm_debug.sif`). The dev estimate is 0.60 pt *below* official here, against 1.06 pt *above* for c01s01 — so the two cells together show the dev/official discrepancy is not one-signed, and I make no causal claim (ledger #24/P5).

## 5. Pitfalls — every loss ≥ 0.1 h

| # | window | cost | cause | coverage |
|---|---|---|---|---|
| 1 | 01:04:28–01:46:29 | **0.70 h** (0.60 h train + 0.10 h full eval) | the "polish" pass: a narrow, RFT-heavy, non-representative 60k subset at lr 3e-6 → 0.657 @1319, 5.1 pts below the incumbent | **uncovered as a protocol item, and I do not propose one.** This is data-mixture composition, which ledger #13/#18 keep as recipe knowledge; it is *not* P1/#20, whose frozen predicate concerns fitted-parent/self-only stages and early-stop observability, and `metrics.md` explicitly warns that a mixed teacher/self stage is not interchangeable with a self-only one. The incumbent artifact was already in `final_model`, so the exposure was bounded |
| 2 | 18:55:43–19:08:07 | **0.21 h of agent turns, 0.00 h of GPU** | wait-primitive churn: three background `sleep 1500/1560/1500` dispatched within 5 s (L3791, L3830, L3874), then `sleep 3000 & wait`, before settling on an artifact-existence poll | **covered — #15 / E v2's surface**, but with zero measured cost: SFT-1 was running throughout, so no GPU time was lost. Useful as a *negative* calibration point: E v2's primary metric (post-exit idle) correctly scores this cell at ~0, even though its waiting *process* was messy |
| 3 | 20:00:42–22:20:01 | 2.32 h | k=4 rejection sampling over 27,473 problems | **not a loss** — GPU busy throughout, produced the 40,922 traces that SFT-2 used. Listed because it is 28 % of the session and the single largest block |

Below the threshold: soup(1,2) build + full eval 01:49:58–01:53 (0.06 h, rejected); the 22:20 `prep_data.py` LaTeX-filter rebuild (0.01 h, caught by inspecting its own data at L5176).

**Traps prior knowledge avoided:**

- The `/tmp/gctest` round-trip at 18:51:26 (L3617) before making greedy the packaging default — this is the check that would have caught the `save_pretrained` rejection c01s01 hit at 03:07:32.
- `finalize.py` writes the grader's `templates/gemma3.jinja` into both `tokenizer_config.json` and `chat_template.jinja` of every artifact — the `template_unreachable` class, handled once in the packaging script rather than per-launch (and it is why this cell never hit c01s02's missing-chat-template failure).
- `base_model` as a symlink to the frozen snapshot (L1175, 18:32:45), so no evaluation could accidentally re-resolve the model.
- `uv pip install --system liger-kernel` succeeded here (L2313) where it broke in the other two cells — evidence that the 64 MB `/` overlay failure in #21/P2 is state-dependent, not deterministic, and therefore cannot be caught by "did the install print ok".

## 6. Control-arm structure, waiting, and ending

**Structure it imposed on itself: the least explicit of the three.** No cards, no `TaskCreate`/`TaskUpdate`, six short assistant text turns, and a 3.9 KB `README.md` written at 02:54:29. What substitutes for structure is *code*: `prep_data.py`, `train_sft.py` with env switches, `gen_rft.py`, `merge_data.py`, `soup.py` and `finalize.py` — every recurring operation, including artifact packaging and decode config, is a script with flags, so the same protocol is applied to every checkpoint by construction. The nearest thing to a `falsified_if` is behavioural: it evaluated the polish pass at full test and discarded it within 5 minutes of the result.

**How it waited: artifact-existence polling.** `while ! [ -f work/<run>/model.safetensors.index.json ]; do sleep 60/120; done` for every long producer (L4349, L5123, L5482, L5605, L6023). This is the pattern the *current* baseline `pitfalls.yaml: run_dies_with_the_session` recommends and that E v2 argues against as an artifact proxy — and here it produced **0.10 h of cumulative GPU-free time, the best of the three cells**. It is a proxy that happened to be safe because every one of this cell's producers completed successfully; had one died before writing its index file, the loop would have run to the tool timeout exactly as c01s02's `pgrep` loops did. I report the outcome and the latent hazard separately.

**How it ended.** `ls -la final_model/` + `cat generation_config.json` + architecture check (L6284–6315), then `ps aux | grep -E "python (train|gen|evaluate)" | wc -l` → `0` and `0 MiB` (L6333–6335), then the README, then the final turn at 02:54:40 with 1:37 left. 1.6 h unused — the largest unused remainder of the three.

## 7. Verdict

**Three largest contributors to 0.7202**

1. **The measured decode fix, +5.7 pts** (0.630 → 0.687 @300, L4673/L4720) — the single largest measured intervention in the cell, and the only one of my three cells that established it experimentally rather than by inference.
2. **The 206k stage-2 mixture with 40,922 rejection-sampled traces**, retrained from base: 68.7 % @300 → 70.8 % @1319.
3. **The broad low-LR third pass** (+0.6 pt to 71.4 @1319), chosen after the narrow variant lost 5.1 pts — a correctly-run negative result that redirected the last 1.5 h productively.

**Why it is 6.9 pts below c01s01: recipe, not operations.** On every operational axis this cell matches or beats the top scorer — 0.10 h vs 0.32 h of GPU-free time, a measured decode A/B where c01s01 had none, five full-test evaluations, a paired per-item comparison c01s01 never ran, and a verified end state. The difference is that its pipeline ends at supervised fine-tuning. c01s01's own numbers price that stage precisely: same parent, same protocol, same n=1319, **0.7415 → 0.7998 = +5.84 pts from GRPO** — almost exactly the gap. This is ledger **#8** (on-policy RL adoption), whose recorded status is *observation, not a protocol target*, and I keep it there: method-adoption rates are a recipe difference, and inferring protocol causality from them is the error the ledger already forbids.

**The one protocol change most likely to have raised this cell:** honestly, none of the frozen candidates. D, B, E v2, H, J and K all address costs this cell did not pay. **A v2 (#2)** is the only one it touches, and only as the positive case its metric should credit. The intervention that would have raised the score is "run an RL stage", which is exactly the kind of rule the line has decided not to write.

**What this cell did that the protocol arm typically did differently:** it moved decode-config packaging, template baking and checkpoint packaging into one script applied uniformly to every artifact, so the comparator protocol was enforced by construction rather than declared per experiment — and it validated the decode config against the library before adopting it, instead of after a failed save.

<!-- END REPORT cell=c01s03 -->

---

## Cross-cell reviewer notes

**Arm/identity checks.** All three are protocol-free controls by the frozen manifest (`exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8.yaml:61-64`: `paths` without `skills/exp_protocol`, `setup: "--tool claude"`), not by AWM SHA. All ran `claude-opus-5[1m]` effort high / 1 M context, CLI 2.1.219, container `opus_5.sif`, node `slurm2-a3nodesetondem-0`, `evaluate.py` md5 `2490079a39a1e242ab1b5286a0087137`. The three official evals report an identical input-token total (2,910,587) and different output totals — consistent with identical prompt construction across cells, though an aggregate token count is not per-item proof.

**Cohort arithmetic.** My three sit inside the six-cell protocol-free control mean 0.7514531210513015 (c01s01 0.7892, c01s02 0.7817, c01s03 0.7202, c01s04 0.7559, c01s06 0.7415, c01s07 0.7202) — I hold the top two and one of the two tied-lowest. Guard6 = 0.7354056103108415; baseline-v3 n=2 = 0.7736921910538286 and is descriptive only. Two scalar coincidences that must not be read as anything: c01s03's official 0.7202426080363912 equals c01s07's (both 950/1319), and c01s01's *developer* full-test read of `work/rft1_bf16` (0.7414708112206216) equals c01s06's official score (both 978/1319). No official per-item logs are retained for any of these, so item-level identity is unverifiable.

**The one thing that separates the three scores is the RL stage size, and it is priced inside a single cell.** c01s01: GRPO 350 full-parameter steps, +5.84 pts over its own RFT parent at n=1319, same protocol. c01s02: GRPO 90 LoRA steps, +1.6 pts at n=500 (McNemar p ≈ 0.322). c01s03: no RL, 0.7202. GPU occupancy does **not** order the scores — the top and bottom cells both had ≈0 idle (0.32 h and 0.10 h) while the middle cell lost 2.87 h. Ledger #8 keeps method adoption as an observation; nothing here changes that, and c01s02→c01s01 suggests the interesting variable is RL *stage size*, not RL yes/no.

**Coverage of every ≥0.1 h loss I found**

| loss | cells | total | status |
|---|---|---|---|
| post-exit idle from name-pattern process waits | c01s02 (×3) | 2.87 h | **covered — #15 / E v2** |
| post-exit idle from a fixed `sleep` overshoot | c01s01 (×1) | 0.13 h | **covered — E v2**; below E's 0.15 h/cell bar |
| TRL truncation-mask / `eos_token_id` 106 → zero gradient | c01s01 | 0.17 h | **covered — #6 / G** (queued; ledger notes the strict guard cohort had no GRPO exposure — this is a fresh NEW-cell exposure) |
| Gemma-3 CE/logits OOM during capacity search | c01s01 0.17 h, c01s02 (inside the 2.07 h event and the 0.12 h GRPO bring-up) | — | **covered — #21 / P2** |
| 64 MB `/` overlay install failure | c01s01 (ENOSPC, 0.04 h), c01s02 (silent null-byte corruption, surfaced 6.6 h later) | — | **covered — #21 / P2** |
| greedy config rejected by `save_pretrained` | c01s01 | 0.02 h | **covered — #14 / D**; foreground save made it cheap |
| late-session runs that returned negative | c01s01 1.55 h, c01s02 0.87 h, c01s03 0.70 h | — | exploration with measured negative reads, bounded by an already-safe `final_model`; adjacent to **#23 / P4**, no new candidate |

**Two calibration inputs for candidates that are already frozen** (offered as design evidence, not as re-proposals):

1. **E v2.** c01s02's three failures share one sub-mechanism: `pgrep -f "<script>.py"` matches the wait loop's own `bash -c` wrapper (proved at L5154–5156, where `pgrep -af` returns only wrapper PIDs and `nvidia-smi` reads `0 MiB, 0 %`). A second signal also lied in the safe-looking direction: `tail -1` of the training log at 21:34:01 returned a fresh tqdm progress line that had been flushed *after* the OOM traceback (c01s02 L4929 vs L5158–5159). E v2's secondary metric already counts proxy-only liveness decisions; this cell would score 3 before its own repair at 23:35:11 and 0 after. Its repair — `echo $! > /tmp/x.pid; while kill -0 $PID` — is E v2's frozen prescription, reached by paying 2.87 h for it. Note also that c01s03's artifact-existence poll produced the *best* idle number in the group (0.10 h) while being precisely the proxy E v2 argues against — the screen should distinguish "safe because every producer succeeded" from "safe by construction".
2. **A v2.** Its metric reads hours from the first post-SFT eval to a *measured* decode choice. In this group, c01s03 passes cleanly (0.06 h, a real 300-sample A/B worth +5.7). c01s01 and c01s02 both shipped greedy **before** their first post-SFT eval, on the basis of reading the inspect vLLM provider and the base-model log line (c01s01 L1466/L1639, 18:35:15–18:35:26) rather than an A/B — and they are the top two scorers of the six controls. If "measured" is read strictly, the metric scores the two best cells as failures. Worth resolving before that screen launches.

**One protocol-surface idea, and its limits.** Every one of my three cells reused the single path `final_model/` for **two or three distinct artifacts** within one session (c01s01: rft1 → g350; c01s02: sft1 → grpo1, plus an in-place tokenizer edit after its verifying eval; c01s03: sft1 → sft2 → sft3), and only c01s01 ever bound the shipped bytes to the selected checkpoint — by `md5sum`, at 03:15:41, 43 seconds before writing its report. `metrics.md` already tells *reviewers* that "a reused `final_model` path needs its timestamp/lineage or content identity"; nothing on the scientist side records it. `card.template.yaml` has `result.output_checkpoint` and an optional `setup.parent_checkpoint.hash`, but no field that binds the *shipped* artifact to the measurement being claimed, and `pitfalls.yaml: final_model_not_loadable` covers loadability only. **Surface:** one optional card field (e.g. `result.shipped_artifact: {path, source_checkpoint, content_hash, verified_at}`). **Source cells:** c01s01, c01s02, c01s03 (3/3 reuse the path; 1/3 verify identity). **Screen metric a 4-cell block can read from the harvested bundle alone:** (a) closed adopt-cards whose recorded artifact identity matches the `final_model` present at session end, target 4/4; (b) count of `final_model` mutations occurring after its last verifying evaluation, target 0 — c01s02 would score 1 on (b). **Guardrail:** the predeclared protocol-baseline pool mean −0.03. **Limits I want on the record:** this is an evidence-integrity item, not an accuracy lever; it adds a field to an already-heavy card, which `metrics.md` warns against when `fields_filled` is the counterweight; and the ledger's stated order is D/B/H first, A/E2 later, with J and K independent — so it belongs behind those, and promotion is not my call.

**Remaining uncertainties**

- **No official per-item evidence exists for any of these three cells.** The official inspect JSON named in `final_eval_1.txt` (e.g. `logs/2026-09-04T03-27-23+00-00_gsm8k_3vyFB2KkbMLVmUmn5Y2T4i.json`) is not in the receipt-backed `result_dir`; only `final_eval_1.txt`, `metrics.json` and the judge outputs survive. Every dev-vs-official comparison in these reports is therefore scalar-only. This is the #29 scratch-lifecycle gap, already recorded.
- **Dev-vs-official gaps on identical weights are real but unattributable**: c01s01 +1.06 pt (dev high), c01s03 −0.60 pt (dev low). Three factors co-vary (`--max-connections` 64/48 vs 2, `--gpu-memory-utilization` 0.85/0.6 vs 0.3, `opus_5.sif` vs `vllm_debug.sif`); P5's adjudication already forbids concurrency rules, and the opposite signs argue against a systematic bias.
- **The n=1319 cost figure is worth checking against other cells before anyone uses it in a screen redesign:** `c01s01/task/logs/g350_full.log` records `total time: 0:01:29` for 1,319 samples at `--max-connections 64`, and `c01s03` records `0:03:06` at `--max-connections 48`. If that generalises, moving a final selection from n=500 to the full test costs 1–3 minutes of GPU. c01s02 shipped on n=500 with 1:24 of budget left and landed 1.6 pts below its estimate. I am not proposing a screen — C v2 was withdrawn and its redesign is the planner's — but this is the missing cost number that redesign needs.
- **Whether the c01s02 waiting pattern generalises** cannot be settled from my group: one cell paid for it and two avoided it with two different techniques. The other three NEW controls (c01s04, c01s06, c01s07) and the guard cells are the place to check whether `pgrep -f`-style waits are common enough to matter for E v2's four-cell screen.
- **The prepared inputs disagreed with the trace in two places I resolved from the actual commands**, consistent with the input-notes warning: `first_train_launch` in all three timelines points at a *smoke* run (0.11/0.16/0.11 h) rather than the first production launch (0.50/0.25/0.20 h); and `final_model_written` for c01s01 (+3.06 h) marks the 21:34:42 safety-net copy of the RFT checkpoint, not the 01:54:07 write of the shipped g350. The cell-reader's `RL launches=14` for c01s01 also counts `sed`/`grep` reads of `grpo_trainer.py`; the actual GRPO training launches are three (21:17:04, 01:24:29, 01:49:02) plus one 3-step smoke (21:13:19). File-size `est_n` was not used anywhere in these reports — every `n` comes from the `gsm8k (N samples)` line in the corresponding eval log.
