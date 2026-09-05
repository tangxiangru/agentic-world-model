```yaml
cell: c01r03
arm: control
variant: control            # null-control, no protocol tree, no cards, no hooks
window_role: calibration    # from the previous window; EXCLUDE from new-cell counts, pooled NEW means and new evidence thresholds
accuracy: 0.778620166793025     # = 1027/1319, stderr 0.011436000004253627
hours_used: 8.73                # trace 2026-09-02T22:02:01Z -> 2026-09-03T06:45:55Z; time_taken.txt 08:44:31
hours_to_first_train_launch: 0.12   # 22:09:04Z probe; 0.40 h to the real stage-1 SFT
protocol_hours: 0.0
waiting_hours: 7.73             # 46 waiting_on_runs calls
greedy_shipped: yes
rl_used: no
rft_tried: yes (adopted, +0.61 pp on the full 1319: 0.77483 -> 0.78089)
largest_eval_n: 1319
stop_reason: "Done. `final_model` holds the best checkpoint and all processes have exited." (L7227, 06:45:55Z); 1:16 unused
top_contributors:
  - stage-1 SFT on 166,772 format-exact examples (7.3% -> 78.6% @500)
  - greedy generation_config, measured same-weight +6.0 pp (0.785 vs 0.725 @200)
  - RFT after the stop-token fix (+0.61 pp on the full 1319)
knowledge_to_transfer:
  - a same-weight greedy-vs-sampling A/B at n=200 costs 3.5 min and settles the decode question with a number
  - save_pretrained refuses a greedy generation config loaded from the parent: 76 min of finished training lost
  - a low RFT pass@1 has two plausible causes; the double-BOS one was real but not the cause, and re-running before testing the second cost 0.75 h
```

**Evidence paths.** Bundle `results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-b-x8-v1/c01r03/`; receipt `.../formal-2026-09-02T123054.012961+0000.json` (job 90493); manifest `experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-b-x8.yaml` (replicate 3); spec `doc/spec/2026-09-02-exp-protocol-round00-null-control.md`; `status.json` complete=true, eligible=true, issues=[], judge_flags=[], node `slurm2-a3nodesetondem-0`, strict_site=true, harvested 2026-09-03T07:12:42Z.

**Calibration handling.** Everything below is a re-measurement of a cell already reviewed in window 02. It is reported so the planner can compare instruments across windows; it must not enter new-cell counts, pooled NEW means, or any new evidence threshold. Where my numbers differ from the window-02 accounting I say so explicitly.

---

### 1. Timeline and the first real training launch

| stage | time | +h | line |
|---|---|---|---|
| session start | 22:02:01Z | 0.00 | L6 |
| baseline eval launched (n=150) | 22:03:14Z | 0.02 | L980 |
| five TaskCreate items (self-imposed plan) | 22:04:06–22:04:11Z | 0.03 | L1238-1295 |
| probe train (6000 ex) | 22:09:04Z | **0.12** | L2490 |
| probe eval (n=150) → 0.5267 | 22:20:11 → 22:24:17Z | 0.30 | L3272/L3388 |
| grader decode observable seen | 22:29:54Z | 0.47 | L3580 |
| stage-1 SFT launched | 22:25:57Z | 0.40 | L3465 |
| greedy vs sampling A/B (n=200 each) | 00:39:21 – 00:42:42Z | 2.68 | L5332/L5414 |
| stage-2 SFT (lost at save) | 00:42:59 → 02:01:34Z | — | L5433 |
| stage-2 SFT re-run | 02:04:01 → 03:22:47Z | — | L5753 |
| RFT gen ×3 (2 discarded) | 03:30:04 → 05:58:19Z | — | L6032/L6273/L6703 |
| RFT train | 05:58:37 → 06:32:15Z | — | L6838 |
| `final_model` = ckpt/sft_rft, greedy | 06:42:13Z | 8.67 | L7068 |
| last turn | 06:45:55Z | 8.73 | L7226 |

Hours by category: waiting_on_runs 7.73, sample_eval 0.45, train_launch 0.16, model generation 0.27. Timer n=30, 9:59 → 1:16.

### 2. Recipe decisions and reasoning

**Data.** 166,772 examples: all OpenMathInstruct-2 `gsm8k` + `augmented_gsm8k` rows with integer answers (≤4 / ≤2 solutions per problem), 25k `math`/`augmented_math` for diversity, plus the GSM8K **train** human solutions; 10% carry a random k-shot prefix (L7240-7243).

**SFT.** SAID (L7245): *"I wrote a custom loop because the 262k vocabulary makes `Trainer` materialise ~25 GB fp32 logit tensors — loss is computed only on assistant positions, in re-materialised chunks, with token-budget batching."* SHOWS: two OOMs during bring-up (`Tried to allocate 22.50 GiB`, L2547; `Tried to allocate 8.00 GiB`, L2995) before `--loss-chunk 2048` made it fit. Full fine-tune of the text tower, vision tower frozen, 1 epoch, lr 1.5e-5, fp32 master weights + bf16 autocast + 8-bit AdamW.

**Stage 2 (rejected).** Fresh OpenMathInstruct-2 + MetaMathQA at lr 8e-6 gave 0.754 @500 against 0.786 @500 for stage 1 (L5968 / L6018) and was discarded — a clean falsification carried out by a control with no `falsified_if` field to fill in.

**Rejection sampling (adopted).** 30,293 verified solutions, pass@1 0.811, 95.6% of problems solved at least once, mixed with a 20k anchor slice, continued at lr 4e-6 (L7251).

**RL.** Never attempted; the trace contains no GRPO/PPO launch and no place where RL is weighed and declined.

### 3. Decode config — the only measured same-weight A/B in the window

- 22:29:54Z L3580: the grader observable — *"Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator."* This is **0.09 h after the first post-SFT eval** returned (22:24:17Z, L3388).
- 00:39:21 – 00:42:42Z: the measurement, `set_gen_config.py` flipping the same checkpoint `ckpt/sft_v1` between the two configs, n=200 each — **greedy 0.785 (L5369), sampling 0.725 (L5418)**. 3.5 minutes of GPU for a +6.0 pp answer.
- 06:42:20Z L7118-7130: the shipped artifact — `"do_sample": false, "temperature": 0.0, "top_p": 1.0, "top_k": 0`.
- SAID (L7248): *"vLLM takes its default sampling params from `generation_config.json` … `final_model` ships greedy decoding instead. This is a change to the model artifact only; `evaluate.py` and `templates/` are untouched."*

**A v2 screen observable, both readings.** By "observable identified": **0.09 h** — pass. By the spec's literal wording ("first post-SFT eval → decode config fixed on a grader observable *with a same-weight comparison*"): first post-SFT eval ended 22:24:17Z, the same-weight comparison closed 00:42:42Z → **2.31 h** — fail. The 2.2 h in between is the stage-1 SFT run, i.e. the delay is "no checkpoint worth A/B-ing existed yet," not indecision. Untested-sampling decision cards in that interval: 0. **The planner should decide which of these two clocks the A v2 screen reads; measured on my three cells the two definitions disagree by 2.3 h on this cell and by 0.00 h on c01r07.**

Read-stability corroboration: greedy here gives own-read 1030/1319 vs official 1027/1319 (**0.23 pp**), against c01r08's sampling 1014 vs 1027 (**0.99 pp**).

### 4. Evaluation practice

| read | n | model | result | source |
|---|---|---|---|---|
| baseline | 150 | base | 0.0733 | `runs/baseline.json` |
| probe | 150 | ckpt/probe | 0.5267 | `runs/probe.json` |
| decode A/B | 200 | sft_v1 greedy | 0.7850 | L5369 |
| decode A/B | 200 | sft_v1 sampling | 0.7250 | L5418 |
| compare | 500 | sft_v2 (stage 2) | 0.7540 | L5968 |
| compare | 500 | sft_v1 | 0.7860 | L6018 |
| compare | 500 | sft_rft | 0.7920 | L6964 |
| full test | 1319 | sft_rft | 0.780895 | L7013 |
| full test | 1319 | sft_v1 | 0.774829 | L7055 |
| default-path check | 150 | final_model | 0.8133 | L7170 |

The strongest evaluation practice of the three cells: **both** the candidate and the incumbent were read on the full 1319 before shipping (06:37:22Z and 06:39:40Z, 2.4 min each). The claimed RFT gain (+0.61 pp) is smaller than one standard error (0.0114) and is *not* backed by a paired item-level statistic — the C v2 rule would ask for one here. No ranking made at small n was overturned at large n in this cell: 0.785 @200 → 0.786 @500 → 0.775 @1319 for sft_v1 is monotone within noise, and the stage-2 rejection at n=500 (−3.2 pp) is ~1.2 SE but was never re-checked at 1319.

The `--limit 150` default-path read of 0.8133 against 0.7809 on the full 1319 is a 3.2 pp front-slice effect on the same greedy weights — consistent with the C v2 note that `--limit N` takes the leading, easier N.

### 5. Every loss ≥ 0.1 h, with candidate coverage

| # | loss | cost | evidence | candidate |
|---|---|---|---|---|
| 1 | **Bogus pass@1 from a missing stop token — two full generation passes discarded.** `gen_rft2` 03:43:29 → 04:50:52 (77,892 prompts, 1:03:33) reported `[rft] pass@1 0.110 solved-any 0.290 wrote 8418` (L6476). First hypothesis, double-BOS, was verified **real** (L6531: `default add_special_tokens: [2, 2, 105, …]` — two `<bos>`) and fixed, but `gen_rft3` 04:51:38 → 05:36:37 still returned `pass@1 0.107 … wrote 5270` (L6670). Only then, at 05:37:07, `stop_token_ids=[1, 106]` with the comment *"`<end_of_turn>` is what the chat template trains the model to emit; the offline API does not pick it up from the generation config"* (L6696-6698) → `gen_rft4` finished in 15:57 with `pass@1 0.811 solved-any 0.956 wrote 30293` (L6779). Neither `rft.jsonl` (8,418) nor `rft2.jsonl` (5,270) was used; `mix_rft.py --rft data/rft3.jsonl` only (L6798). | **1.87 h** | L6032-L6779 | **B** |
| 2 | **Greedy-parent Trainer save failure — the D trap, hit hardest.** Stage-2 training reached `step 1270/1277 … elapsed 75.8m` and died: `ValueError: GenerationConfig is invalid: - temperature: do_sample is set to False. However, temperature is set to 0.0 … - top_k: … set to 0` inside `model.save_pretrained` (L5670-5696, read 02:03:35Z); `ls ckpt/sft_v2` showed only `config.json` (L5698). Patched at 02:03:53 (null the sampling fields before save + a try/except: *"never lose a finished run to a config validation error"*, L5729) and re-run 02:04:01 → 03:22:47. | **1.34 h** wall clock | L5661-5753 | **D** |
| 3 | **Over-scoped generation + orphan engine.** 03:30:04 launched 164,838 prompts, ETA `2:23:37` (L6109); killed 03:42:36; after `pkill -f gen_rft.py` the GPU still read `66017 MiB` with `10928 VLLM::EngineCore` alive (L6211, 03:43:08Z); `kill -9` at 03:43:10; relaunched with `enable_prefix_caching=True, max_num_seqs=512` at 03:43:29. | **0.224 h** | L6032-6273 | **B** |
| 4 | **Post-completion idle, nine events.** sft_v1 done 00:36:20 → cmd 00:39:21 (0.050 h); stage-2 crash 02:01:34 → 02:03:35 (0.034); stage-2b done 03:22:47 → 03:25:36 (0.048); `gen_rft2` done 04:49:00 → 04:51:20 (0.039); `gen_rft3` done 05:33:06 → 05:37:07 (0.067); `gen_rft4` done 05:54:09 → 05:58:28 (0.072); rft train done 06:32:15 → 06:34:55 (0.045); probe train 0.020; probe eval 0.023. | **0.398 h** | `task/system_monitor.log` | **E** |
| 5 | **Gemma-3 262k-vocab logits OOM during bring-up.** `Tried to allocate 22.50 GiB` (L2547, 22:09:5xZ) and `Tried to allocate 8.00 GiB` (L2995, 22:12:0xZ); resolved with a chunked re-materialised loss. | 0.08 h | L2490-3053 | **P2** |

**E for this cell: 0.398 h total, largest single event 0.072 h.** Target <0.15 h/cell → **fail, 2.7×** — but note the *shape* is different from c01r07/c01r08: no event exceeds 4.3 minutes, because this cell used `run_in_background: true` for the long sleep and interleaved short foreground polls (L4749-4789, 22:40:45 and 22:40:48). That is the closest any of the three came to process-driven waiting, and it cut the per-event tail by 4–5×. Worth naming in any E wording: the cheap half of the fix is "poll on a short interval in the foreground while the long wait runs in the background," not necessarily "wait on the PID."

**Agreement with the scientist's own accounting.** L7258: *"a 76-minute run was lost when `save_pretrained` rejected the greedy generation config it had loaded"* — 1.27 h against my 1.34 h launch→detection. The 4-minute difference is detection latency, which the fixed-sleep pattern kept small here.

**Divergence from the window-02 accounting.** The ledger records D as "Window 02 4/8 cells, 3.3 h" across p00r11, p00r14, c01r01, c01r03. My re-measurement attributes **1.34 h to c01r03 alone**. If the earlier figure counted only marginal overhead rather than the re-run wall clock, the two are reconcilable; if it apportioned ~0.8 h/cell, the window-02 D total is understated. **The planner should reconcile this before D's screen threshold is fixed**, since it changes D's expected saving by roughly a factor of 1.6.

### 6. Protocol interaction — control arm

Cards 0, overrides 0, relocks 0, `pitfalls_cost_h` 0.00, `lock_before_launch` 0/0, no `awm exp_protocol`, no hook of any kind (0 matches for `Stop hook`/`stop_hook`/`hook_event`/`PreToolUse`/`blocked by`).

**Self-imposed structure — the most protocol-like of the three controls.** At 22:04:06–22:04:11Z it created five tasks through the harness' own TaskCreate tool (L1238-1295): *"Baseline eval of gemma-3-4b-pt"*, *"Build decontaminated SFT dataset"*, *"SFT training run v1"*, *"Iterate: scale data / RFT / hyperparams"*, *"Finalize final_model"* — each with a one-line description of what it will do. It then closed them with outcome text, e.g. 06:45:38Z (L7207): *"Stage-2 continuation (OMI2 fresh + MetaMathQA) hurt: 75.4% vs 78.6% @500, discarded. RFT round … helped: 78.09% vs 77.48% on the full 1319-item test set."* Total cost: 12 tool calls, 0.00 h of measured tool time. That is a card-shaped ledger — subject, plan, verdict with numbers — built spontaneously at 1/10th the ceremony, and it is the closest natural comparator the control arm offers to `n_closed`/`fields_filled`.

Waiting style: 10 background launches, 48 sleeps, 4 tool timeouts, `bash timer.sh` on nearly every wait (30 readings).

**Live work at stop**: none. 06:45:34Z L7188-7189: `pgrep -af "train_sft|gen_rft|evaluate.py|vllm|VLLM"` empty, GPU `0 MiB`.

### 7. Verdict

**Three largest contributors to 0.7786.**
1. Stage-1 SFT: 7.3% → 78.6% @500 (L6018). Everything after moved ≤0.6 pp.
2. Greedy decoding, and here it is *measured*: 0.785 vs 0.725 on identical weights at n=200 (L5369/L5418), the only direct price tag on the decode config anywhere in this window.
3. The stop-token fix, which turned a 7.4×-degraded RFT yield into 30,293 usable solutions and bought +0.61 pp on the full test set.

**What this cell did that the protocol arm typically does differently**: it A/B-tested the decode config on the same weights instead of arguing about it, read the full 1319 for both the candidate and the incumbent before shipping, and kept a five-item outcome ledger through the harness' own task tool at essentially zero cost.

