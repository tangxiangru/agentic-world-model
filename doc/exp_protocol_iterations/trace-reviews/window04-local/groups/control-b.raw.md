I've finished the read-only review. Reports follow; I did not write any files (per the brief, the planner saves these).

<!-- REPORT cell=c01s04 role=NEW -->
```yaml
cell: c01s04
arm: control (protocol-free; manifest setup "--tool claude", no skills/exp_protocol path)
accuracy: 0.7558756633813495          # 997/1319, stderr 0.011832
hours_used: 8.62                       # time_taken.txt 08:37:30; trace 18:31:11Z -> 03:08:26Z
hours_to_first_train_launch: 0.40      # real SFT 18:55:04Z; first train *command* (smoke) 18:40:19Z = 0.15
protocol_hours: 0                      # 0 cards, 0 locks (collect.csv c01s04/task)
waiting_hours: 7.67 raw / ~0.15 post-exit idle   # raw tool-time in sleep+wait calls; ~7.6 h of it had a GPU job running
greedy_shipped: yes                    # do_sample:false, temperature:0.0 in final_model/generation_config.json; measured +13.5 pts
rl_used: yes                           # full-parameter GRPO (Dr.GRPO, beta=0), 330 + 160 steps, 5.04 h
rft_tried: no (implemented, never executed; verdict "GRPO was the better spend")
largest_eval_n: 1319                   # six full-test dev evals drove the ship decision
stop_reason: "Done. All processes have finished, the GPU is idle, and `final_model/` is written and verified." (L7662, 03:08:26Z; timer 1:22 left, L7639 03:08:11Z)
top_contributors: [full-parameter GRPO +11 pts over SFT, measured greedy generation_config +13.5 pts, full-test (n=1319) candidate selection]
one_protocol_change: SKILL/pitfalls line — stop buying GPU evaluations once the top two candidates are within 1 SE; compute the paired item table from logs already on disk instead
knowledge_to_transfer: [keep the greedy generation_config out of every training parent and only in serving copies, smoke GRPO for <2 min before the real launch, copy only preprocessor_config/processor_config into Trainer checkpoints]
```

## 1. Timeline

Session `728d3a94` starts **18:31:11Z** (L6) and ends **03:08:26Z** (L7662); `time_taken.txt` 08:37:30. Timer read only **4 times** in 8.6 h — 9:59 (L25, 18:31:13), 9:36 (L2854, 18:54:08), 9:25 (L3736, 19:05:30), 1:22 (L7639, 03:08:11).

Stage sequence, resolved from actual commands (the prepared timeline's markers are text matches and three of them are wrong):

| stage | prepared marker | what actually happened |
|---|---|---|
| first eval | 18:31Z | first `evaluate.py` **launch** 18:32:24Z (L985), base model, n=150 → 0.0267 |
| first train launch | 18:40Z | 18:40:19Z (L2238) is a `--out runs/smoke --no-save` **throughput probe**; the real SFT is **18:55:04Z** (L2987, `--data data/sft_v3.jsonl --out runs/sft_a`) = **+0.40 h** |
| first RL | 19:05Z | 19:05:49Z (L3826) is `inspect.getsource(GRPOConfig)`; first GRPO **execution** 20:22:33Z (L5120, smoke), first kept GRPO **20:33:05Z** (L5630) = +2.03 h |
| final_model written | 19:19Z | 19:19:25Z (L4481) is the `Write` of `finalize.py`; the first artifact write is **20:22:20Z** (L5050, from `runs/sft_a`) |

`final_model` was then **rewritten twice at the same path**: 00:08:03Z from `runs/grpo_a/checkpoint-160` (L6275) and 03:02:13Z from `runs/grpo_b/checkpoint-70` (L7291). Only the last one is the scored artifact — any comparison must bind to the checkpoint, not the path.

Category hours (prepared timeline, corrected by reading): `waiting_on_runs` 7.67 h / 50 calls, `train_launch` 0.29 h, `bash_other` 0.21 h, `sample_eval` 0.07 h, `model_generation` 0.28 h. The `sample_eval` figure is an artefact of the classifier: six of seven evaluation batches were launched with `nohup bash -c '… run_eval.sh …' &` (L4927, L6195, L6671, L7006, L7073) and observed through sleeps, so their cost sits inside `waiting_on_runs`. Of the 7.67 h "waiting", **~7.6 h had a GPU job running** (SFT 1.31 h, GRPO-a 3.38 h, GRPO-b 1.66 h, ~1.2 h of batched evals). **Post-exit idle ≈ 0.15 h total**, and all but one event is negligible: SFT exit ≈20:13:44 → 20:14:47 (0.018 h), grpo_a exit 23:56:59 (log tail) → 23:57:35 (0.010 h), grpo_b exit 01:57:00 → 01:57:25 (0.007 h). The wait loops `while [ ! -f runs/<run>/config.json ]; do sleep 60; done` (L4661, L6123, L6610) are the reason.

Time left at the end: **1:22** — it stopped 1.37 h early with the GPU at 0 MiB and no process alive (L7650, 03:08:11Z).

## 2. Recipes and reasoning

**Data.** `prep_data.py` → 161,692 rows written (`logs/prep.log`), filtered to 81,193 used: OpenMathInstruct-2 rows with `problem_source ∈ {gsm8k, augmented_gsm8k}`, one solution per problem, `\boxed{}` stripped, final-number-consistency filtered, plus the 7,473 GSM8K-train gold CoTs; every completion rewritten to `…\n\nANSWER: <number>`; **8 % of rows get a random 1–3-shot prefix** (L7597–7601, 03:08:09Z). Contamination: `contamination_check.py` over 154,219 unique documents → `Contaminated documents 0 / Total matches 0` (`logs/decon.log`), launched 18:39:06Z (L1950).

**SFT.** `train_sft.py`, full-parameter, vision tower frozen, bf16, liger fused linear-CE, 1 epoch, lr 1e-5 cosine, effective batch 96 (bs 48 × accum 2), `--max-len 1200`, length-bucketed. `train_runtime 4720.06 s` = 1.31 h, `PEAK MEM GB 51.19` (`logs/sft_a.log`). Four throughput probes at 18:40–18:54 sized the batch before committing.

**RL.** GRPO chosen and RFT dropped. Both paths were built: `gen_samples.py` 18:58:53Z (L3176) and `build_rft_data.py` 19:03:45Z (L3569) — then never executed. The stated reason, L7623 (03:08:09Z):
> `run_eval.sh` · `gen_samples.py`/`build_rft_data.py` (rejection-sampling path, written but not
> used — GRPO was the better spend).

GRPO config: TRL + colocated vLLM, Dr.GRPO, `beta=0` (no reference model), 32 prompts × 8 rollouts, lr 3e-6 (stage 1) / 2e-6 (stage 2), reward = numeric match of the `ANSWER:` line. Stage 1 `train_runtime 12174.05 s` = 3.38 h, 330 steps; stage 2 `5980.44 s` = 1.66 h, 160 steps. Reward 0.57 → 0.78; `'frac_reward_zero_std': 0.59375` by 21:33 (L5812).

**Budget reasoning** appears only post hoc — the thinking blocks are signature-only, so the trace shows no in-flight budget deliberation, and only 4 timer reads. The retrospective judgement, L7689 (03:08:26Z):
> GRPO stage 2 was essentially flat — the run plateaued once rollout entropy collapsed to ~0.05,
> and every candidate after that sat within one standard error (~1.2%) of the others.

## 3. Decode config — measured, and measured cleanly

**Yes, and this is the cleanest same-weights decode measurement in my group.** At 20:14:53Z (L4920–4927) it built `runs/sft_a_greedy` by **symlinking the same safetensors** and writing only a different `generation_config.json`, then evaluated both at n=200:

- `vllm/runs/sft_a` (stock: `do_sample:true, top_k:64, top_p:0.95`, config printed at L4904, 20:14:47Z) → **0.505** (inspect log 20:16:31Z)
- `vllm/runs/sft_a_greedy` (`do_sample:false, temperature:0.0`) → **0.640** (inspect log 20:18:18Z)

Gain **+13.5 pts on identical weights**. It had verified the mechanism in vLLM's source ~1 h earlier, before it had any checkpoint to test (L4076–4090, 19:08:42Z: builds the greedy `GenerationConfig`, then prints `ModelConfig.get_diff_sampling_param`). Its own note, L7474 (03:07:30Z):
> Measured on the same SFT checkpoint (2026-09-03, 200 GSM8K test items): 50.5% with the inherited
> sampling config vs 64.0% after writing `{"do_sample": false, "temperature": 0.0}` …

Shipped file (verified on the raw volume, `final_model/generation_config.json`, 164 B): `{"bos_token_id":2,"cache_implementation":"hybrid","eos_token_id":[1,106],"pad_token_id":0,"do_sample":false,"temperature":0.0}`, plus a copied `chat_template.jinja`. It also tested `repetition_penalty:1.05` on the winner and rejected it on the full test: 0.7415 vs 0.7528 (L7267, 03:01:59Z).

**Load-bearing design detail:** the greedy config was written **only into serving copies** (`sft_a_greedy`, `prep_ckpt.py` outputs, `finalize.py --greedy`). Every training parent — `runs/sft_a`, `runs/grpo_a/checkpoint-*` — kept the stock, HF-valid config (L4904). That is exactly why this cell never hit candidate D's trap while c01s06 lost 0.86 h to it.

## 4. Evaluation practice

`run_eval.sh` (L3434, 19:01:23Z) pins `--max-connections 32 --gpu-memory-utilization 0.85`; `evaluate.py` defaults are 2 / 0.3. Ladder of n behind decisions (all from the retained inspect logs in the receipt-backed `task/logs/`, actual `completed_samples`, not byte estimates): base 150 → decode A/B 200 → GRPO-a checkpoint triage 200 (g160 0.750, g240 0.725, g330 0.735) → 400 (b160 0.740, b70 0.7425, g160 0.730; soups 0.750/0.7475/0.740) → **six full-test 1319 runs** (soup1 0.74375, soup2 0.74223, **b70 0.75284**, b105 0.74147, b160 0.74754, b70_rp 0.74147).

This is the **only** cell in my group that took the final decision at n=1319, and its `runs/eval_b70` is weight-identical to the shipped model (`prep_ckpt.py runs/grpo_b/checkpoint-70 runs/eval_b70` symlinks the safetensors, L6668, 01:57:25Z; `finalize.py --src runs/grpo_b/checkpoint-70`, L7291, 03:02:13Z).

**No paired statistic was computed.** I recomputed it from the retained logs (streaming id/score scan; one record in one log escapes the pattern, so counts are ±1):

| pair (full test) | discordant A-only / B-only | net |
|---|---|---|
| b70 vs b160 | 219 / 211 | **+8 of 1318** |
| b70 vs b105 | 224 / 208 | +16 |
| b70 vs soup1 | 226 / 213 | +13 |
| b70 vs b70_rp (same weights, only `repetition_penalty`) | 225 / 209 | +16 |
| **b105 vs b70_rp** | **81 / 81** | **0 — identical 0.7414708 accuracy, 162 items differ** |

So the winner's margin over the runner-up is 8 items with 430 discordant (McNemar z ≈ 0.39). The scientist's "within one standard error" is right; the item-level spread is far larger than the accuracy gap suggests, and the b105/b70_rp row is a textbook case of equal scalar accuracy over different correct sets.

**Official vs developer on the same weights:** official 997/1319 (0.75588, `metrics.json`) vs the developer's own full-test read of the same checkpoint 993/1319 (0.75284). The only differences are the serving invocation (`--max-connections 2 --gpu-memory-utilization 0.3 --max-tokens 4000` vs 32/0.85/4000) and batching. Two knobs change together, so neither is isolated; the official per-item log is **not** retained in the result dir (only `task/logs/` up to 03:04Z), so the item-level check is unresolved. It also ran the stock command once as an artifact check: `python evaluate.py` defaults → 0.7133 at n=150 (L7370, 03:02:41Z).

It did a **zero-GPU error analysis** on the winner's full-test log at 02:47:54Z (L7103): `incorrect: 326 / 1319`, `no ANSWER: 27`, `text after ANSWER line: 0`.

## 5. Losses ≥ 0.1 h (and the traps prior knowledge avoided)

| # | window | cost | cause | ledger |
|---|---|---|---|---|
| 1 | 20:22:33Z → 20:33:05Z (L5120→L5630) | **0.175 h** | GRPO bring-up: three failed smokes, three mechanisms — (a) `ValueError: … you must provide a chat template if the tokenizer does not define one` (L5211, 20:23:56Z); (b) TRL EOS → zero gradient, `{'loss': 0.0, 'grad_norm': 0.0, …}` ×8 (L5384–5391, 20:26:32Z), fixed by `tok.eos_token = "<end_of_turn>"` (L5412, 20:27:18Z); (c) `torch.OutOfMemoryError … Tried to allocate 8.00 GiB` (L5507, 20:29:07Z), fixed with `expandable_segments:True` + smaller gen batch | (a) #22/P3-family, (b) **#6 / G**, (c) #21/P2 |
| 2 | 00:08:48Z → 00:15:35Z | **0.113 h** | GRPO stage 2 died **20 s after launch** — `OSError: Can't load image processor for 'runs/grpo_a/checkpoint-330'` (L6417) — and was not noticed because the monitor was a fixed `sleep 420` (L6353, 00:08:32Z). Fixed by copying **only** `preprocessor_config.json`/`processor_config.json` from the base snapshot (L6429, 00:15:44Z) | #22/P3 **and** #15/E (clock wait over a dead producer) |
| 3 | 02:23:11Z → 03:02:13Z | **0.65 h** | nine selection evaluations (three n=400 + six n=1319) after the field was already tied; per the recount above the winner's margin is 8/1318 | #4 / #19 (C v2 withdrawn, redesign pending) |

Below threshold but worth naming: six 2-minute Bash tool timeouts (18:38:33, 18:52:01, 18:54:03, 19:01:14, 19:13:00, 20:16:56) — these cost turns, not wall time, since the underlying jobs kept running; the scientist later switched to the JSON form with explicit `"timeout": 1600000`.

**Traps prior knowledge avoided, with the line:**
- double-`<bos>`: `train_sft.py` tokenises with `add_special_tokens=False` from the first version (L2068, 18:40:08Z).
- stop ids in its own sampler: `gen_samples.py` uses `max_tokens=args.max_tokens, stop_token_ids=[1, 106]` (L3255, 18:58:53Z) — the exact guard c01s06 omitted.
- 262k-vocab logits OOM: `uv pip install --system liger-kernel` at 18:39:36Z (L1967), **before** the first SFT probe.
- Contamination checked over all 154,219 documents before training mattered.

## 6. Control-arm behaviour (no protocol installed)

`protocol_hours = 0`; `collect.csv` reads `c01s04/task,…,0 cards,0 locked,0 pitfalls,0.0 pitfalls_cost_h` and an empty `fields_filled`. Self-imposed structure instead:

- **A five-item task list up front**, all created in 3 s at 18:32:16–18:32:19Z (L936–L980): "Baseline eval", "Build SFT dataset", "SFT training run", "Rejection sampling / STaR round"; a fifth, "GRPO RL run from SFT checkpoint … to boost greedy pass@1", added at 20:22:28Z (L5106) when GRPO was chosen. The RFT task was closed as `"status": "deleted"` at 03:07:48Z (L7552) — an explicit abandonment, recorded.
- **Waiting by artifact existence, not by clock** — `while [ ! -f runs/<run>/config.json ]; do sleep 60; done` (L4661, L6123, L6610). This is why post-exit idle is ~0.15 h across 6.35 h of training. The one place it used a plain `sleep 420` instead is loss #2 above.
- **Ended by writing durable notes rather than cards**: two memory files at 03:07:30Z/03:07:42Z (L7455, L7495) — `eval-sampling-comes-from-generation-config` and `gsm8k-gemma3-4b-postraining-results`, each with the `**Why:** / **How to apply:**` shape — a `MEMORY.md` index (L7517) and `README_RESULTS.md` (L7595). No card, lock, comparator or `falsified_if` anywhere.
- No preflight list and no pitfalls file existed to read; the equivalent function was performed by reading library source (`ModelConfig.get_diff_sampling_param`, `GRPOConfig`) before committing GPU.

Overhead of this self-structure: 5 `TaskCreate` + 7 `TaskUpdate` calls, 0.00 h of tool time in the prepared timeline; the note-writing at the end cost ~0.5 min of generation.

## 7. Verdict

**Three largest contributors to 0.7559:**
1. **Full-parameter GRPO, 5.04 h of it.** SFT alone was only 0.640 at n=200 (20:18:18Z); GRPO stage 1 reached 0.750 at n=200 (23:59:18Z) and the shipped stage-2 checkpoint 0.7528 at n=1319. This is the largest single jump and the biggest single compute allocation in the cell.
2. **Measured greedy `generation_config`, +13.5 pts on identical weights** (0.505 → 0.640, same symlinked safetensors), verified against vLLM source before it was applied.
3. **Selecting at n=1319 rather than at n≤400.** At n=400 the ranking put `soup1` first (0.750) ahead of `b70` (0.7425); at n=1319 that inverts (`soup1` 0.74375, `b70` 0.75284). The full-test pass cost 0.5 h and changed the shipped artifact — even though, per the paired recount, the final margin is not significant.

**What the other arm typically did differently:** with no card, this cell never had to write a comparator or a `falsified_if`, and it abandoned the RFT branch by deleting a todo (L7552) rather than closing a card with a stated reason — the abandonment is legible only because the final report volunteers it.

**One protocol change most likely to have raised this cell** — a single SKILL/pitfalls sentence, *stop-at-tie*: once the top two candidates are within 1 SE, spend no further GPU on larger unpaired reads; instead compute the item-level paired table from the inspect logs already on disk (cost ≈ 0 GPU), and if the discordant counts do not separate, ship the earlier/simpler artifact. Source cells: **c01s04** (0.65 h on nine evals for an 8/1318 margin), **c01s07** (0.716 vs 0.744 rejection on 74/60 discordant), **c01s06** (chose sft2 over soup on 233 vs 232 at n=300 = one item). Surface: SKILL wording or one `pitfalls.yaml` entry. Screen metric: GPU-hours spent on selection evaluations *after* the first pair lands within 1 SE (target ≤0.25 h/cell) and the share of ≤1 SE ship decisions that cite a paired table. Guardrail: accuracy floor. **Relation to frozen text:** this is not C v2 — C v2's observable was `evaluation.protocol.n ≥ 500`, which is already met here (n=1319) and was the stated reason for its withdrawal. This is the discriminating observable ledger row #4 says the redesign still needs, and it overlaps #19 (same-artifact repeat variance) rather than the n-threshold. Whether it becomes a candidate is the planner's call.
<!-- END REPORT cell=c01s04 -->

<!-- REPORT cell=c01s06 role=NEW -->
```yaml
cell: c01s06
arm: control (protocol-free; manifest setup "--tool claude", no skills/exp_protocol path)
accuracy: 0.7414708112206216           # 978/1319, stderr 0.012060
hours_used: 7.87                       # time_taken.txt 07:53:24; trace 18:31:10Z -> 02:23:25Z
hours_to_first_train_launch: 0.185     # pilot 18:42:13Z (OOM'd); first surviving launch 18:45:50Z; full-data SFT 19:04:23Z = 0.55
protocol_hours: 0                      # 0 cards, 0 locks (collect.csv c01s06/task)
waiting_hours: 6.85 raw / ~0.17 post-exit idle   # max single idle event ~0.09 h
greedy_shipped: yes                    # do_sample:false, temperature:0.0, top_p:1.0, top_k:0; measured +15.4 pts on the pilot
rl_used: no                            # GRPO/RL never appears anywhere in the trace
rft_tried: yes (adopted — 2 generation passes, 65k-row round-2 SFT, shipped as final_model)
largest_eval_n: 400                    # ship decision itself taken at n=300; never evaluated at n>=500
stop_reason: "Done. GPU is free, no background jobs remain, and `final_model` is in place and verified." (L5091, 02:23:25Z; timer 2:07 left, L5088 02:23:10Z)
top_contributors: [a strong SFT round-1 (0.757@300) from a 97k <=2-solutions-per-problem corpus, measured greedy generation_config +15.4 pts, rejection-sampling round 2 -- but ~3.8 h of the 7.9 h produced nothing usable]
one_protocol_change: already-frozen candidate D (`parent_generation_config_valid` + `greedy_parent_generation_config`) -- this cell is a textbook instance; the refinement its screen should read is below
knowledge_to_transfer: [set vLLM stop/stop_token_ids on every offline sampling call and read yield before committing the full pass, keep the greedy generation_config out of any directory that will later be an --init parent, read the item-level paired table you already have before choosing between checkpoints]
```

## 1. Timeline

Session `966de841` starts **18:31:10Z** (L7) and ends **02:23:25Z** (L5091); `time_taken.txt` 07:53:24. Timer read **16 times**, last 2:07 (L5088). Stage sequence from actual commands:

| time | event |
|---|---|
| 18:32:05Z (L1102) | base eval launched, n=150 → **0.0667** (metrics readable 18:41:48Z, L2035) |
| 18:42:13Z (L2097) | **pilot SFT launch #1 — OOM at step 0** |
| 18:45:50Z (L2331) | pilot SFT relaunch with liger fused CE + `expandable_segments`; `train_runtime 413.55 s` |
| 18:54:43Z / 18:59:44Z | pilot eval 0.4733, pilot_greedy eval **0.6267** (n=150) |
| 19:04:23Z (L3118) | **full-data SFT `sft1`** — `train_runtime 4712.11 s` = 1.31 h |
| 20:29:37Z | sft1 eval n=300 → 0.7567 |
| 20:33:17Z (L3648) | `cp -r ckpt/sft1 final_model` — first artifact write (prepared marker `final_model_written 20:33Z` is correct) |
| 20:33:33Z (L3690) | **`gen_rft.py` pass 1** — runs to 23:29Z (**2.93 h, discarded**) |
| 23:34:10Z (L4024) | `gen_rft.py` pass 2 with stop tokens — done ≤00:17:48Z (0.72 h) |
| 00:18:27Z (L4315) | round-2 SFT `sft2` — **completes training, dies at save 01:10Z (0.86 h lost)** |
| 01:11:39Z (L4505) | `sft2b` relaunch — `train_runtime 3070.88 s` = 0.85 h |
| 02:06:25Z / 02:13:17Z | sft2 n=300 → 0.7767; soup n=300 → 0.7733 |
| 02:10:54Z (L4776) | `cp -r ckpt/sft2 final_model` (shipped artifact) |
| 02:19:09Z | final_model n=400 → 0.7675 |

Categories (prepared): `waiting_on_runs` 6.85 h / 34 calls, `sample_eval` 0.60 h / 9, `train_launch` 0.18 h / 6, `model_generation` 0.19 h. Corrected: the 6.85 h "waiting" is almost entirely a running producer — gen_rft 2.93 + gen_rft2 0.72 + sft1 1.31 + sft2 0.86 + sft2b 0.85 ≈ 6.67 h. **Post-exit idle ≈ 0.17 h in total**, largest single event ≈0.09 h (sft1 exit ≈20:22:55 from launch + `train_runtime`, next action 20:28:06Z), then ≈0.07 h after gen_rft (file written 23:29, wait call returned 23:33:20Z). Both are under E's 0.15 h/event line. Ended with **2:07 unused**.

## 2. Recipes and reasoning

**Data (round 1).** `prep_data.py` → 97k rows: OpenMathInstruct-2 `gsm8k` + `augmented_gsm8k`, **≤2 solutions per problem**, `\boxed{}` rewritten to an `ANSWER:` line, plus the 7,473 GSM8K-train reference solutions; **18 % of rows carry a random 1–5-shot prefix** (L5105, 02:23:25Z). Decontaminated: `Contaminated documents 0 / Total matches 0` (`runs/decon.log`), and again for the RFT corpus.

**SFT.** `train_sft.py`: `Gemma3ForConditionalGeneration.from_pretrained(args.init, dtype=torch.float32, …)`, liger `apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)`, vision tower + multimodal projector frozen, `bf16=True`, `optim="adamw_bnb_8bit"`, `gradient_checkpointing=True`, `group_by_length=True`, lr 1.4e-5 (round 1) / 7e-6 (round 2), and `model = model.to(torch.bfloat16)` before save (task/train_sft.py:84–158). Note the saved `config.json` still declares `text_config.dtype: "float32"` while the weights are bf16 (safetensors byte-identical to the other two cells) — stale metadata, and the official eval loaded fine, so I make no claim beyond "inconsistent field".

**RL: not considered.** A case-sensitive scan for `GRPO|GSPO|PPO|reinforcement|policy gradient` over the whole trace returns **zero** hits for this cell. Thinking blocks are signature-only, so the *reason* is unknown, not "rejected" — the trace simply shows no engagement. Rejection sampling was the round-2 method and every mention of it is an execution/monitoring description (L3692 onward, 20:33:33Z).

**Round 2 (RFT).** 8 samples per GSM8K-train problem + 4 per 18k augmented problems = 131,784 generations from `ckpt/sft1` at T=1.0; keep correct chains truncated at the first `ANSWER:`, cap 2–3 per problem, mix 18k original rows → 65k rows, continue at lr 7e-6. Its own summary, L5111 (02:23:25Z):
> A 50/50 weight soup of rounds 1 and 2 scored 77.3% — no gain, so round 2 ships as-is.

**Budget reasoning** appears only as timer polling (16 reads) — no quoted deliberation about the remaining budget anywhere in the trace, and none in the final report.

## 3. Decode config

**Yes, measured, and it is the cell's own headline.** It read vLLM's handling in source first — `grep -rn "generation_config" .../vllm/config/model.py` (L2805, 18:59:21Z) and `sed -n 1310,1420p` of the same file (L2830, 18:59:24Z) — then copied the pilot and rewrote only the config (L2955, 18:59:34Z: `cp -r ckpt/pilot ckpt/pilot_greedy && python -c "…"`):

- `vllm/ckpt/pilot` → **0.4733** (n=150, inspect log 18:56:15Z)
- `vllm/ckpt/pilot_greedy` → **0.6267** (n=150, inspect log 19:01:17Z)

**+15.4 pts.** Its statement, L5107 (02:23:25Z):
> **3. Fixed the decoding default — the single biggest win.** … Shipping a greedy
> `generation_config.json` took the pilot from **47.3% → 62.7%**.

From 01:11:35Z (L4491) `train_sft.py` writes the greedy JSON at the end of *every* run — which is exactly what caused loss #2 below. Shipped file (raw volume, 194 B): `{"bos_token_id":2,"eos_token_id":[1,106],"pad_token_id":0,"do_sample":false,"temperature":0.0,"top_p":1.0,"top_k":0,"cache_implementation":"hybrid"}`. No `chat_template.jinja` in the model dir (the harness supplies `templates/gemma3.jinja` via `model_args`, so the official run was unaffected).

## 4. Evaluation practice — the weakest of my three cells

Seven inspect logs, actual `completed_samples`: 150, 150, 150, 300, 300, 300, **400**. `--max-connections` used: 16, 32, 32, 48, 48, 48, 32; `--gpu-memory-utilization` 0.85 except the last (0.3). **It never ran the submission command's defaults (`--max-connections 2`) and never evaluated anything at n ≥ 500**, with 2:07 of budget unspent — the official full-test run took 0:12:27 (`final_eval_1.txt`).

The ship decision was taken at **n=300** on a **one-item** margin: sft2 233/300 = 0.7767 vs soup 232/300 = 0.7733.

It is, however, the **only cell in my group that computed an item-level paired table** — at 02:10:41Z (L4741–4751) it loaded two inspect logs and printed:
> `n=300 both=207 sft1_only=20 sft2_only=26`
> `sft2 stop reasons Counter({'stop': 297, 'max_tokens': 3})`

20 vs 26 discordant is not a separation (McNemar p ≈ 0.55), and the trace shows it proceeded to install sft2 as `final_model` 10 s later (L4764, 02:10:51Z) without commenting on the discordance. So the *capability* was demonstrated and the *inference* was not drawn — a distinction worth keeping.

**Small-n optimism, realised:** shipped estimate 0.7767 (n=300) → own verification 0.7675 (n=400) → **official 0.7415 (n=1319)**. The n=400 read is 2.6 pts above official, ≈1.2 SE of that estimate; the n=300 read is 3.5 pts above.

## 5. Losses ≥ 0.1 h

| # | window | cost | cause | ledger |
|---|---|---|---|---|
| 1 | 20:33:33Z → 23:29Z | **2.93 h, entirely discarded** | `gen_rft.py`'s `SamplingParams` had **no `stop` / `stop_token_ids`** (the fix at L3977, 23:34:06Z adds `stop=["<end_of_turn>","<start_of_turn>"], stop_token_ids=[1, 106]`). Completions ran past `ANSWER:` and restarted the answer (samples printed at L3983, 23:33:46Z). Detected only by dumping a stored completion after the whole pass finished. Corrected rerun 23:34:10Z → ≤00:17:48Z = **0.72 h**; the 72,000-prompt phase alone went **1:33:47 → 24:30** (`runs/gen_rft.log` / `gen_rft2.log`) and correct samples went **32,136 → 96,054** over the same 25,473 problems. The rerun also changed `--max-tokens 640 → 512`, so the 3.8× speed-up is stop-tokens-dominated but not isolated. | **#3 / B v2** (frozen, first wave) |
| 2 | 00:18:27Z → 01:11:39Z | **0.86 h of finished training discarded** | Round-2 SFT trained to completion, then `model.save_pretrained` raised `ValueError: GenerationConfig is invalid:` — `temperature 0.0` with `do_sample False`, and `top_k 0` with `do_sample False` (L4427, 01:11:10Z). The invalid config came from **its own greedy write into `ckpt/sft1`**, inherited through `--init ckpt/sft1`. `ckpt/sft2` contained only `config.json`, dated 01:10 (L4459, 01:11:17Z). Fixed in 22 s by pinning a plain `GenerationConfig` on the model and writing the greedy JSON after save (L4471, 01:11:30Z: `# a greedy generation_config.json (written at the end) fails HF validation, / so keep a plain one on the model while training/saving`). Detection was fast — the crash-to-notice gap is ~1 min. | **#14 / D** (frozen, first releasable wave) — this is D's predicate verbatim |
| 3 | six occurrences, aggregate | **≈0.10 h** | `evaluate.py`'s vLLM server does not release the GPU when the eval finishes: at 18:41:48Z the metrics were already written and `pgrep -fc evaluate.py` returned `1` (L2039). The cell issues `pkill -f evaluate.py` six times (18:41:56, 19:03:13, 20:33:14, 02:10:51, 02:17:20, 02:22:50), each a sleep+kill+verify round trip, and two of them return `Exit code 144` because the pkill matches its own shell. | #3 (orphan engine) |

Below threshold: pilot SFT OOM at step 0 — `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 13.08 GiB` in `cross_entropy` over the 262k vocab (L2140, 18:44:43Z) — 18:42:13Z → 18:45:50Z = **0.06 h**, fixed by installing liger-kernel and switching to fused linear CE (#21/P2).

Netting it out: **≈3.79 h of a 7.87 h session produced nothing that reached the shipped model** (2.93 + 0.86), and the cell still scored 0.7415 — because round-1 SFT alone was already 0.7567 at n=300.

**Traps prior knowledge avoided:** it read vLLM's `generation_config` resolution in source before trusting the decode fix (L2805/L2830, 18:59Z); it decontaminated both corpora; it checked `stop_reason` distributions in the eval log (`Counter({'stop': 297, 'max_tokens': 3})`, L4752). It did **not** carry the offline-sampling stop-token knowledge that c01s07 applied at 18:53:45Z on the same day — the single most expensive difference between the two cells.

## 6. Control-arm behaviour

`protocol_hours = 0`; `collect.csv` shows 0 cards / 0 locks / 0 pitfalls entries / empty `fields_filled`. Structure imposed instead:

- **No task list at all** — unlike c01s04 (5 todos) and c01s07, this cell created zero `TaskCreate`/`TaskUpdate` items. Its only externalised plan is the tool-call `"description"` field, which it used consistently and descriptively ("Launch rejection sampling generation", "Relaunch rejection sampling with proper stop tokens", "Relaunch round-2 training with save fix").
- **Clock-based monitoring throughout**: `sleep 1500/1800/2700/3000/3300` (L3446, L3495, L3536, L3739, L3785, L3832, L3872, L4366, L4553) with a timer read appended. This is why it polled the timer 16 times, and it is also why the 2.93 h bad generation pass ran to completion before anyone looked at its output.
- **Manual GPU hygiene** in place of a lifecycle rule: `pkill -f evaluate.py; sleep 10; nvidia-smi …` before each next stage.
- **Ended with a verification sweep** (L5027, 02:22:54Z: `pgrep -af "evaluate.py|train_sft|gen_rft|vllm" || echo none`, GPU 0 MiB, `ls -la final_model`, cat every `runs/*_metrics.json`) and a final report that volunteers both failures — but wrote **no** durable memory file (contrast c01s04).

Cost of the self-structure: essentially zero tool time; the price was paid in what the structure did *not* do — nothing forced a comparator at a fixed n, nothing forced an output check before a 3-hour generation pass, and nothing recorded the abandoned/failed run other than the final prose.

## 7. Verdict

**Three largest contributors to 0.7415:**
1. **A strong round-1 SFT** — 0.7567 at n=300 from 97k rows with ≤2 solutions per problem and an 18 % few-shot prefix. This is the highest post-SFT number of my three cells (c01s04 0.640@200, c01s07 0.690@200) and it carried the cell.
2. **Measured greedy `generation_config`, +15.4 pts** on the pilot (0.4733 → 0.6267, n=150).
3. **The 3.79 h of discarded compute** capped what round 2 could be. With ~3.8 h back, the cell had time for a full-test comparison of sft1 / sft2 / soup and for a third round; instead it shipped a 1-item n=300 preference with 2:07 unused.

**What the other cells did differently:** c01s07 set `stop_token_ids=[106, 1]` in its sampler at 18:53:45Z, before ever running one — the guard this cell added only at 23:34:06Z, after paying 2.93 h. c01s04 kept its greedy config exclusively in serving copies, so no `--init` parent ever carried it; this cell wrote it into the training output directory itself, which is precisely what broke `save_pretrained`.

**One protocol change most likely to have raised this cell:** the frozen **candidate D** (`8332917` — preflight `parent_generation_config_valid` plus the `greedy_parent_generation_config` pitfalls entry) would have caught loss #2 before the 0.86 h run started; loss #1 is inside **B v2**'s scope (`9f294c3`, "stop ids 不生效"). I am not proposing anything new for those. The one **refinement** this cell supplies to D's screen, traced to two cells: D's observable should be read on the **parent's own directory as it exists at launch**, not on base-snapshot configs — here the unsafe parent was the scientist's *own* round-1 output (`ckpt/sft1`), created 6 h into the session, whereas **c01s04** kept its parents stock-valid by writing greedy only into symlinked serving copies (L4920, 20:14:53Z) and consequently never triggered the fault across two GRPO `--init` chains. So the discriminating measurement for D's 4 cells is "count of `--init` parents whose `generation_config.json` fails HF validation at launch time", alongside the frozen "hours attributable to GenerationConfig-is-invalid = 0". Surface: D's existing preflight check + its test. Guardrail: unchanged score floor. Promotion is not my call.
<!-- END REPORT cell=c01s06 -->

<!-- REPORT cell=c01s07 role=NEW -->
```yaml
cell: c01s07
arm: control (protocol-free; manifest setup "--tool claude", no skills/exp_protocol path)
accuracy: 0.7202426080363912           # 950/1319, stderr 0.012364
hours_used: 8.17                       # time_taken.txt 08:11:25; trace 18:39:33Z -> 02:49:48Z
hours_to_first_train_launch: 0.18      # real SFT 18:50:11Z; first train *command* (smoke) 18:45:22Z = 0.10
protocol_hours: 0                      # 0 cards, 0 locks (collect.csv c01s07/task)
waiting_hours: 6.81 raw / ~0.15 post-exit idle   # largest single idle ~0.10 h after the SFT run
greedy_shipped: yes -- but as `do_sample:true, temperature:0.0` (greedy only under vLLM's temperature-0 rule); measured +9.0 pts
rl_used: yes                           # LoRA r=64 GRPO x3 rounds, ~4.04 h total
rft_tried: yes (adopted as an intermediate: 36,878 rows, rft_v1 0.745@200 / 0.732@500, used as the GRPO-2 parent)
largest_eval_n: 1000                   # ship decision taken at n=500; n=1000 was a post-hoc verification
stop_reason: "Done. No processes are left running and the GPU is idle." (L6490, 02:49:48Z; timer 1:49 left, L6487 02:49:35Z)
top_contributors: [LoRA-only GRPO capacity, a decode fix worth +9 pts, selection at n<=500 inside same-artifact re-read noise]
one_protocol_change: preflight/pitfalls item on the *form* of the shipped decode config -- the two greedy spellings used across this window differ in HF save-validity, and only one of them is safe as a future --init parent
knowledge_to_transfer: [set vLLM stop_token_ids before the first sampling run, read library source for the eval's request body before assuming a decode default, falsify a direction from data you already dumped (maj@6 vs pass@1) instead of building it]
```

## 1. Timeline

Session `6d9367ca` starts **18:39:33Z** (L7) — 8 min after c01s04/c01s06, a job-start offset, not scientist delay — and ends **02:49:48Z** (L6490). `time_taken.txt` 08:11:25. Timer read **23 times**, the most of my three cells; last 1:49 (L6487).

Stage sequence, resolved from actual commands (two prepared markers are text matches, not launches):

| time | event |
|---|---|
| 18:41:51Z (L1015) | base eval launched, n=150 → **0.020** |
| 18:45:22Z (L1917) | `train_sft.py --limit 4000 --max-steps 6 --out ckpt/smoke` — a **probe**, not the run (prepared `first_train_launch 18:45Z`) |
| **18:50:11Z (L2142)** | **real SFT `ckpt/sft_v1`** = **+0.18 h**; `train_runtime 4761.48 s` = 1.32 h |
| 18:53:45–18:55:28Z | writes `gen_samples.py` (with stop ids), `build_rft_data.py`, `train_grpo.py`; reads the inspect vLLM provider and TRL `grpo_trainer` source — **the prepared `first_rl 18:54Z` marker is `sed -n 1290,1420p .../grpo_trainer.py` (L2700), a source read** |
| 20:15:26Z / 20:17:27Z | decode A/B on `ckpt/sft_v1`: 0.600 → **0.690** (n=200) |
| 20:19:16Z (L3495) | first GRPO **execution** — smoke, `TypeError … 'vllm_max_model_len'`; retried 20:19:46Z |
| **20:22:38Z (L3666)** | **GRPO round 1** (LoRA r=64, 400 steps) — `400/400 [2:05:51]`, exits 22:29:22Z |
| 20:27:34Z (L3754) | background `while true` loop copying `ckpt/grpo_v1/checkpoint-*` into `ckpt/grpo_snaps` (the trainer rotates checkpoints) |
| 22:40:58Z (L4288) | `cp -r ckpt/m_400 final_model` — first artifact write, +4.02 h (prepared marker correct) |
| 22:41:28Z (L4590) | `gen_all.py` — 134,838 prompts in `49:00`, `problems with >=1 correct: 19236/22473` |
| 23:36:43Z (L5010) | maj@6 vs pass@1 computed on the dumped samples; SC direction dropped |
| 23:37:21Z (L5126) | RFT SFT `rft_v1` from `ckpt/m_400`; `train_runtime 2628.48 s` |
| 00:27:02Z (L5409–5410) | `final_model ← rft_v1`, **GRPO round 2** launched from `rft_v1` |
| 01:30:54Z (L5605) | GRPO-2 **killed deliberately** at ≥200 of 300 steps |
| 01:44:25Z (L5856) | `final_model ← ckpt/g2_200` — the shipped artifact |
| 01:47:59Z (L5964) | **GRPO round 3** from `g2_200`; killed 02:40:47Z; both snapshots rejected |
| 02:47:46Z | final verification, n=1000 → 0.723 |

Categories (prepared): `waiting_on_runs` 6.81 h / 36 calls, `sample_eval` 0.56 h / 14, `train_launch` 0.43 h / 14, `model_generation` 0.26 h. The `train_launch` figure is composite — several of those calls are `for S in …; do merge_lora; set_gen_config; evaluate; done` loops (L4232, L5700, L6215), i.e. mostly evaluation, not launch overhead. Running-producer time inside "waiting": SFT 1.32 + GRPO-1 2.10 + gen_all 0.82 + RFT 0.73 + GRPO-2 1.06 + GRPO-3 0.88 ≈ **6.91 h**. **Post-exit idle ≈ 0.15 h total**, dominated by one event (below). Ended with **1:49 unused**.

## 2. Recipes and reasoning

**Data.** 65,423 SFT rows: OpenMathInstruct-2 `gsm8k` + `augmented_gsm8k` plus ~8k MATH rows, un-`\boxed`-ed and rewritten to end in `ANSWER: <n>`, formatted byte-for-byte like the eval prompt through `templates/gemma3.jinja`; **12 % carry a 2–10-shot GSM8K-train system prefix** (L6417, 02:49:32Z). Decontamination on both corpora: `Contaminated documents 0 / Total matches 0` (`runs/decon.log`, and again for 36,878 RFT rows at 23:37:14Z).

**SFT.** Full-parameter, `dtype=torch.bfloat16`, liger fused linear CE, `trainer.save_model`, processor files copied after save (task/train_sft.py:97–158). 1 epoch, lr 1e-5, bs 16 × accum 4.

**RL: LoRA GRPO, three rounds** — the only cell of my three that used adapters rather than full-parameter RL. r=64 on the language tower, colocated vLLM, 8 rollouts/prompt, reward = final `ANSWER` matches reference; merged back with `merge_lora.py` (`model.merge_and_unload()`, L3787). Round 1: 400 steps, reward 0.598 → ~0.78 (sampled from `runs/grpo_v1.log`, retained on the raw volume; excluded from the bundle at 2.7 MB). Rounds 2 and 3 both ran on a **difficulty-filtered** prompt set, its stated rationale at L6431 (02:49:32Z):
> …(problems solved 1–5 of 6 times, plus some easy/unsolved) so fewer groups have zero advantage.

**RFT.** 6 samples at T=1.0 over 22,473 problems, ≤2 distinct correct solutions kept → 36,878 rows, 1 epoch at lr 5e-6. Adopted as the GRPO-2 parent, not shipped directly.

**The clearest piece of cheap falsification in my group.** Before building a self-consistency-distillation stage it measured the headroom on samples it already had, at 23:36:44Z (L5033):
> `ALL n=22433 pass@1=0.7147 maj@6=0.7600 unanimous_correct=0.5458 any_correct=0.8575`

and dropped the direction, L6513 (02:49:48Z):
> **A self-consistency-distillation idea was measured and dropped.** Before building it I computed
> maj@6 vs pass@1 on the dumped samples (0.760 vs 0.715) — … the headroom didn't justify the format change

That measurement cost seconds of CPU on data already on disk and freed the budget for RFT + GRPO-2.

**Budget reasoning is behavioural rather than quoted** — 23 timer reads, and both later RL runs were **killed on a clock** rather than run to their configured step count (`for p in $(pgrep -f train_grpo.py); do kill $p; done`, L5605 01:30:54Z and L6078 02:24:03Z), with the background snapshot copiers in place precisely so that a killed run still yields usable checkpoints. That is a deliberate, well-engineered budget strategy; the trace shows no prose about it.

## 3. Decode config — measured, but shipped in the fragile spelling

**Yes, measured.** It read `inspect_ai/model/_providers/vllm.py` for the request body at 18:54:07Z (L2565) before trusting anything, then A/B'd the same checkpoint at 20:15:26Z / 20:17:27Z (L3399, L3447):

- `ckpt/sft_v1`, stock config saved aside as `runs/gc_orig.json` → **0.600** (n=200)
- same directory after `python set_gen_config.py ckpt/sft_v1 --temperature 0.0` → **0.690** (n=200)

**+9.0 pts.** Statement at L6512 (02:49:48Z):
> **Decoding defaults were worth ~9 points.** vLLM seeds its default sampling params from the model's
> `generation_config.json`; … `final_model` ships `"temperature": 0.0`.

**Important correction to the prepared facts.** The cell-reader reports `do_sample=false writes/mentions=0` for c01s07, which reads as "no greedy config shipped". That is a matcher artefact. `set_gen_config.py` (task/set_gen_config.py) edits the JSON **in place** and only touches the keys it is given, so the shipped `final_model/generation_config.json` (raw volume, 231 B) is:

`{"bos_token_id":2,"cache_implementation":"hybrid","do_sample":true,"eos_token_id":[1,106],"pad_token_id":0,"temperature":0.0,"top_k":64,"top_p":0.95,"transformers_version":"4.57.3"}`

vLLM treats `temperature == 0` as greedy and ignores `top_k`/`top_p`, so the artifact **is** greedy at serving time and the +9 pts confirms it took effect. `greedy_shipped: yes` — but by a different spelling from c01s04 (`do_sample:false, temperature:0.0`) and c01s06 (`do_sample:false, temperature:0.0, top_k:0`). The model dir carries no `chat_template.jinja`; `evaluate.py` supplies `templates/gemma3.jinja` via `model_args`, so the official run was unaffected.

Trace evidence on which spelling is save-safe in this image (transformers 4.57.3): c01s07's form survived being an `--init` parent — `ckpt/m_400` carried `do_sample:true, temperature:0.0` when `train_sft.py --init ckpt/m_400` saved `rft_v1` successfully (`runs/rft_v1.log` ends `saved to ckpt/rft_v1`), and `merge_lora.py` saved from a `temperature:0.0` base at 22:34:38Z. c01s06's form did **not** — `ValueError: GenerationConfig is invalid` (c01s06 L4427). Same intent, opposite consequence.

## 4. Evaluation practice

15 inspect logs, actual `completed_samples`: 150, 200 ×8, 500 ×4, 1000. Every `--limit N` run scores the **same first N dataset items** (`shuffled: false` in the log header), so the n=500 comparisons are on identical items — same-items, but no paired statistic was computed. Its own framing, L6508 (02:49:48Z):
> A third GRPO round was trained and **rejected** — 0.716 vs 0.744 on the same 500 items.

Serving knobs it used: `--max-connections 32 --gpu-memory-utilization 0.35 --max-tokens 1024` at n=200, `48 / 0.40 / 1024` at n=500, `48 / 0.40 / 2048` at n=1000. It also ran the **stock command once** on the artifact — `evaluate.py` defaults → 0.7333 at n=150 (L5915, 01:44:34Z).

**Same-artifact re-read noise, measured from the retained logs.** `final_model` is `cp -r ckpt/g2_200` (L5856, 01:44:25Z), so the n=500 run at 01:41:59Z and the n=1000 run at 02:47:46Z score the *same weights and the same generation_config*. Restricting the n=1000 log to the same first 500 dataset ids:

| comparison | result |
|---|---|
| g2_200 @500 (01:41:59Z) | 0.744 |
| same 500 items inside final_model @1000 (02:47:46Z) | **0.724** — 16 items lost, 6 gained, **22 of 500 flipped** |
| items 501–1000 of that same run | 0.722 — so there is **no** first-500 difficulty gradient |

The two runs differ in `--max-tokens` (1024 vs 2048) and in n/batching, so neither knob is isolated; what is isolated is that **the same greedy artifact re-read moves ~2.0 pts at n=500**. That is larger than every margin this cell used to choose:

| decision | recount (same items) | verdict |
|---|---|---|
| g2_200 (0.744) over rft_v1 (0.732) @500 | 50 / 44 discordant, net +6 | not separated (p ≈ 0.6) |
| reject g3_150 (0.716) vs g2_200 (0.744) @500 | 74 / 60 discordant of 499, net +14 | not separated (z ≈ 1.21) |
| g3_150 vs g3_100, both 0.716 | **61 / 61 discordant** | identical accuracy, 122 items differ |

(Streaming id/score scan of the retained logs; one record in the g3_150 log escapes the id pattern, so counts are ±1.)

Sequence for the shipped model: 0.750 (n=200) → 0.744 (n=500) → 0.723 (n=1000) → **0.7202 official (n=1319)**. The n=1000 read is within 0.3 pt of official; the n=200 read is 3 pts high.

## 5. Losses ≥ 0.1 h

This is the **cleanest** of my three cells mechanically — no OOM, no zero-gradient, no orphaned engine, one 14-second typo.

| # | window | cost | cause | ledger |
|---|---|---|---|---|
| 1 | ~20:09:33Z → 20:15:26Z | **≈0.10 h** | Post-exit idle: SFT exited at launch 18:50:11Z + `train_runtime 4761.4806 s`, but the monitor was a fixed `sleep 1400` issued at 19:51:57Z (L3340) that returned 20:15:17Z. The exit time is inferred from the runtime record (bound), the next action is exact. | **#15 / E** (clock wait, not process wait) |
| 2 | 01:47:59Z → 02:49Z | **1.01 h** (0.88 h GRPO-3 + ~0.13 h merge/eval) | GRPO round 3 trained and rejected on 0.716 vs 0.744 at n=500. This is a legitimate negative result, not a fault — but it consumed two thirds of the remaining budget, and per the recount above the rejection margin (74/60 discordant) is not decisive. | #4 / #19 (C v2 withdrawn) + #23/P4 |

Below threshold: `TypeError: GRPOConfig.__init__() got an unexpected keyword argument 'vllm_max_model_len'` at 20:19:32Z (L3539), fixed by `sed -i` 14 s later (L3582); GRPO-1 exit 22:29:22Z → notice 22:31:46Z (0.04 h); three `Exit code 144` self-kills where a `kill`/`pkill` in a composite command matched its own shell (22:31:52Z, 01:35:54Z, 02:41:05Z), each costing one retry turn.

**Traps prior knowledge avoided, with the line:**
- **offline sampling stop tokens** — `stop_token_ids=[106, 1]` is present in `gen_samples.py` when it is first written at **18:53:45Z** (L2412) and in `gen_all.py` at 22:41:25Z (L4542), i.e. ~1.7 h before c01s06 launched the pass that cost it 2.93 h for lacking exactly this.
- **the eval's request body** — `grep -n "temperature|top_p|generation|GenerateConfig|server_args|def " .../inspect_ai/model/_providers/vllm.py` at 18:54:07Z (L2565), before assuming anything about decode defaults.
- **TRL prompt handling** — `grep -n "add_special_tokens|maybe_apply_chat_template|is_conversational" .../grpo_trainer.py` (L2681, 18:54:40Z) and 130 lines of `_generate_single_turn` (L2700, 18:54:43Z), pre-empting the missing-chat-template and EOS faults that cost c01s04 0.175 h at 20:22–20:33.
- **liger** installed at 18:44:04Z (L1514), before the first SFT probe — no OOM anywhere in this cell.
- **checkpoint rotation** — the background snapshot copier (L3754, 20:27:34Z) is what made killing GRPO runs on a clock safe.

## 6. Control-arm behaviour

`protocol_hours = 0`; `collect.csv` shows 0 cards / 0 locks / 0 pitfalls / empty `fields_filled`. Self-imposed structure:

- **Reading the harness and the libraries first.** A 90-second block at 18:53:45–18:55:28Z produced the three source reads above plus `train_grpo.py`. This is the functional analogue of a preflight list, self-administered, and it is why this cell has essentially no bring-up losses.
- **Measure-then-build.** Two directions were killed by measurements on data already on disk: self-consistency (maj@6 0.760 vs pass@1 0.715, L5033) and GRPO round 3 (0.716 vs 0.744 on the same 500 items). Neither used a `falsified_if` field; both are exactly what such a field is for.
- **Budget enforcement by killing runs.** GRPO-2 and GRPO-3 were both terminated at a chosen wall-clock point rather than at their configured step counts, with snapshot copiers guaranteeing usable artifacts. No cell in my group did anything comparable.
- **Fallback discipline on the submitted path.** `final_model` was overwritten three times — 22:40:58Z (`m_400`), 00:27:02Z (`rft_v1`, described as `"fallback updated"`), 01:44:25Z (`g2_200`) — so a valid artifact existed from +4.02 h onward. The cost of this is that the path alone identifies nothing; every comparison must name the checkpoint.
- **Ending**: `README.md` written at 02:49:32Z (L6394) with a full per-checkpoint table, then a process/GPU sweep (L6467, 02:49:35Z). No durable memory file.

## 7. Verdict

**Three largest contributors to 0.7202 (the lowest of my three):**
1. **LoRA-only GRPO.** All three RL rounds trained an r=64 adapter on the language tower and merged it back; c01s04 ran full-parameter GRPO for 5.04 h and gained ~11 pts over its SFT, while this cell's GRPO-1 moved 0.690 → 0.720 at n=200 and rounds 2 and 3 added nothing that survived a 500-item read. Confounded with data recipe, step count and prompt filtering — **not** a causal claim at n=1.
2. **The decode fix, +9.0 pts** (0.600 → 0.690 on the same checkpoint) — real, but the smallest of the three cells' decode gains, consistent with its SFT already producing better-formed output.
3. **Selection inside the noise floor, at n ≤ 500.** Every ship-or-reject decision after 22:30Z was taken on a margin of 6–14 items out of ~500 with 94–134 discordant, while the same artifact re-read moves 22 items on those same 500. The 1.01 h spent on GRPO-3 and its evaluations was spent to resolve a difference smaller than the measurement's own repeatability.

**What the other cells did differently:** c01s04 took its final decision at n=1319 and its ranking *inverted* between n=400 and n=1319 — this cell never evaluated above n=1000 and never above n=500 for a decision, with 1:49 of budget unspent and a full-test run costing ~13 min.

**One protocol change most likely to have raised this cell** — a preflight check with a test (the allowed "preflight check with a test" surface), on the **form** of the shipped/parent decode config: it must (a) resolve to greedy under the serving engine actually used and (b) load and re-save through `transformers.GenerationConfig` without a validation error, with the test pinning which spellings the shipped image accepts. Source cells: **c01s06** (`do_sample:false, temperature:0.0, top_k:0` → `ValueError: GenerationConfig is invalid`, 0.86 h of finished training discarded) and **c01s07** (`do_sample:true, temperature:0.0, top_k:64, top_p:0.95` shipped and, in this image, save-safe). Screen metric: count of `--init` parents and of shipped `final_model` configs failing the check, plus hours attributable to GenerationConfig-invalid saves. Guardrail: unchanged score floor, and the check must not touch the *measured* decode choice — only its spelling. **Relation to frozen text:** this is a scoping refinement of candidate **D** (`8332917`, `parent_generation_config_valid` + `greedy_parent_generation_config`), not a new candidate; D's current framing is "the parent's greedy config breaks the first save", and these two cells show the failure is spelling-dependent, not greedy-dependent — a check that simply flags "greedy" would fire on c01s07's safe artifact and produce exactly the false-positive/override pattern ledger row #22 warns about. It also touches #2 (A v2) only as an observation: all three of my cells reached a measured decode choice within 5 min of their first post-SFT eval (c01s04 2 min, c01s06 5 min, c01s07 2 min), i.e. A's revised observable looks saturated in the protocol-free control arm too.
<!-- END REPORT cell=c01s07 -->

---

## Cross-cell reviewer notes (control group b)

**Arm identity.** All three are protocol-free controls by the frozen manifest (`experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8.yaml`: `awm.setup: "--tool claude"`, paths exclude `skills/exp_protocol`), confirmed by `collect.csv` (0 cards, 0 locks, empty `fields_filled`). `protocol_hours = 0` for all three; there is no card overhead to measure, only what each imposed on itself.

**Prepared-input corrections the synthesis should carry.**
- `first_train_launch` is a smoke/probe in c01s04 (18:40:19Z vs real 18:55:04Z) and c01s07 (18:45:22Z vs real 18:50:11Z), and an OOM'd attempt in c01s06 (18:42:13Z vs surviving 18:45:50Z).
- `first_rl` is a **source read** in both c01s04 (19:05:49Z, `inspect.getsource(GRPOConfig)`) and c01s07 (18:54:43Z, `sed` of `grpo_trainer.py`). Real first GRPO executions are 20:22:33Z and 20:19:16Z; first kept runs 20:33:05Z and 20:22:38Z. c01s06 has **no** RL at all.
- `final_model_written 19:19Z` for c01s04 is the `Write` of `finalize.py`, not an artifact write (real: 20:22:20Z). The markers for c01s06 (20:33Z) and c01s07 (22:40Z) are correct.
- c01s07's `do_sample=false writes/mentions=0` does **not** mean no greedy config was shipped; the artifact is greedy via `temperature: 0.0`.
- `est_n` from file size is not needed: every developer inspect log is retained on the receipt-backed volume with exact `completed_samples`.

**Waiting vs idle.** Raw `waiting_on_runs` (7.67 / 6.85 / 6.81 h) is ~95 % productive running time in all three. Measured post-exit idle: c01s04 ≈0.15 h (one 0.113 h event — a launch that died 20 s in behind a `sleep 420`), c01s06 ≈0.17 h (max event ≈0.09 h), c01s07 ≈0.15 h (one ≈0.10 h event behind a `sleep 1400`). All per-event maxima sit under E's 0.15 h line, but all three of the largest events share the same cause: a fixed sleep over a producer that had already stopped. c01s04's own `while [ ! -f runs/<run>/config.json ]` idiom is the counterexample within the same group — where it used it, idle was 0.007–0.018 h.

**Cost ledger, event-deduplicated (my reading, not card sums — these cells have no cards):** c01s04 ≈0.29 h of failure/repair + 0.65 h of non-discriminating selection evals; c01s06 ≈3.89 h of failure/repair, of which 3.79 h is discarded GPU compute; c01s07 ≈0.10 h of idle and no compute-failure loss. c01s06's `pitfalls_cost_h` in `collect.csv` reads `0.0` for all three cells simply because there are no cards — absence of a card sum is not absence of cost.

**The uncomfortable ordering.** Mechanical cleanliness ran opposite to score here: c01s07 pre-empted every bring-up trap and scored lowest (0.7202); c01s06 lost 3.79 h to two avoidable faults and scored 0.7415; c01s04 lost ~0.29 h and scored 0.7559. With n=3 and confounded recipes (full-parameter GRPO vs LoRA GRPO vs no RL; 81k/97k/65k-row corpora; n=1319 vs n=300 vs n=500 selection), this supports no causal claim about candidates B/D/E — it does say that a screen reading only "avoidable hours" will not see the score.

**What is not saturated in this control arm** (relevant to the withdrawn C's redesign, ledger #4/#19):
- final selection n: 1319 / **300** / 500 — one of three shipped on n<500, contradicting the "already met" reading if controls are in scope.
- paired/item-level reasoning at ≤1 SE: **1 of 3** computed a paired table (c01s06, and it did not act on the result); 0 of 3 cited one in the ship decision.
- budget: 1:22 / 2:07 / 1:49 unused, while a full-test official read costs 0:12–0:17.
- same-artifact repeatability: measured directly for the first time in this window (c01s07, 22 of 500 items flip, −2.0 pts), and it exceeds every selection margin in all three cells.

**Remaining uncertainties.**
1. The official per-item log is not retained in any of the three receipt-backed result dirs (only `task/logs/` up to each session's end). So official-vs-developer differences on identical weights — c01s04 997 vs 993 of 1319; c01s06 0.7415 vs 0.7675 at n=400; c01s07 0.7202 vs 0.723 at n=1000 — cannot be checked item-by-item. This is the same gap the P5 adjudication left open.
2. c01s07's n=500-vs-n=1000 re-read changes both `max_tokens` and batching; the 2.0 pt swing is real but its cause is not isolated, and I have not repeated the measurement in c01s04/c01s06.
3. c01s06's saved `config.json` declares `text_config.dtype: "float32"` while the weights are bf16 (byte-identical safetensors to the other two). The official eval succeeded; I make no claim that this affected anything.
4. Whether transformers 4.57.3 would reject `do_sample:true, temperature:0.0` on save is answered *negatively* by c01s07's own successful saves from such parents — but I have not exercised the full matrix of spellings, and the local interpreter here is transformers 5.15.0, so it is not usable as evidence about the image.
5. My paired recounts are a streaming regex over the retained logs; one record escapes the id pattern in two logs (b70 full, g3_150@500), so every discordant count is ±1. The conclusions are insensitive to ±1 but the numbers should be regenerated with a proper reader if they enter a screen gate.
