I have everything I need. Here are the three reports.

<!-- REPORT cell=g01r03 role=NEW -->

```yaml
cell: g01r03
arm: protocol (guard, --exp-protocol --tool claude --stop-hook, awm 4ae3d87 / tree 189319d6)
accuracy: 0.8278999241849886        # official, n=1319, stderr 0.010397
hours_used: 8.43                     # trace 18:31:04Z -> 02:57:05Z; time_taken.txt 08:26:31
hours_to_first_train_launch: 0.51    # first card-locked training 19:01:32Z; first train_sft.py of any kind 18:44:58Z (+0.23 h, uncarded smoke)
protocol_hours: 0.10                 # 0.004 h tool + <=0.092 h card-composing generation; see Q6
waiting_hours: 7.24                  # raw; 6.13 h pure sleep/poll + 1.11 h composite launch+wait; post-exit idle >=0.85 h
greedy_shipped: yes                  # measured +7.7 pts @n=300, grader observable verified
rl_used: no
rft_tried: no (script scripts/sample_rft.py written 19:06:23Z, never executed)
largest_eval_n: 800                  # developer; the shipping comparison. 150 = final default check; 1319 = official
stop_reason: "Done. Everything has finished; no processes are running and the GPU is idle." (L9329, 02:57:05Z; timer 1:33 left)
top_contributors: [greedy decode config +7.7pts@300, data scale 120k->253k rows +7.0pts@300, exp-04/exp-05 weight soup +2.1pts@800]
one_protocol_change: name Gemma-3 `preprocessor_config.json`/`processor_config.json` in the loadability pitfall (already queued as P3/#22) — it is the only >=0.1 h loss in this cell
knowledge_to_transfer: [record-high cell reached 0.828 with 0 RL and 0 RFT: decode fix + more solutions-per-problem + a 2-way soup, all screened at n>=300 and settled at n=800]
```

## 1. Timeline

Session 2026-09-03T18:31:04Z → 2026-09-04T02:57:05Z, 262 assistant turns, 173 tool calls, `stop_reason: end_turn`, `terminal_reason: completed`.

| stage | time | +h | evidence |
|---|---|---|---|
| bootstrap `Skill: exp_protocol` | 18:31:04 | 0.00 | L10 |
| skill + pitfalls + template read (`cat`) | 18:31:10–18:31:11 | 0.00 | L178, L332 |
| first eval launched (base, n=300) | 18:36:35 | 0.09 | L2583, after exp-01 lock 18:36:30 |
| first `train_sft.py` (uncarded smoke) | 18:44:58 | 0.23 | L4195 `--out ckpts/smoke` |
| **first card-locked training** (exp-02 SFT) | 19:01:32 | **0.51** | L5006, lock 19:01:24 (L4973) |
| first post-SFT eval scored | 20:16:22 | 1.76 | inspect log `2026-09-03T20-15-33…` |
| measured greedy decision closed | 20:30:23 | 1.99 | exp-03 result appended, L6534 |
| exp-04 (253k rows) launched | 20:35:13 | 2.07 | L6867, lock 20:35:06 |
| exp-05 (350k rows) launched | 23:02:42 | 4.53 | L7708, lock 23:02:36 |
| n=800 head-to-head | 02:07:27–02:11:49 | 7.61 | L8288 |
| soup written, `final_model` installed | 02:43:41–02:44:46 | 8.23 | L8917, `match=yes` on both shards |
| final default-args check (n=150) | 02:47:32–02:49:08 | 8.30 | inspect log `…02-47-32…` |
| end | 02:57:05 | 8.43 | L9329 |

Hours by category (my re-derivation, correcting the prepared timeline):

| category | h | note |
|---|---:|---|
| waiting on runs (raw) | 7.24 | 6.13 h pure `sleep`/`tail` polls (31 calls) + 1.11 h calls that also *launched* the job they then waited on (14 calls) |
| — of which post-exit idle | ≥0.85 | 10 monitor-anchored events ≥3 min, ±1 min each; largest 02:11:49→02:22:47 = **0.18 h** |
| train launch | 0.28 | 9 calls |
| sample/eval | 0.18 | 2 calls the classifier caught; most evals ran inside `run_eval.sh` composites counted as waiting |
| protocol | 0.10 | see Q6; the prepared timeline's 0.08 h figure is *dominated by one misclassified call* — `02:29Z 0.08h protocol L8734` is the soup patch + relock + rerun, not bookkeeping |
| data / bash_other / code writes | 0.07 | |
| model generation | 0.49 | thinking/writing gaps |

Time left at the end: `timer.sh` returned **1:33** at 02:56:49 (L9285 result). Stated reason, quoted (L9329, 02:57:05Z):
> Done. Everything has finished; no processes are running and the GPU is idle.

That is a completion claim, not a budget claim — the trace SHOWS the GPU idle and `ps` returning 0 producer processes at 02:56:49, so the cell stopped with ~1.5 h of budget and no in-flight work. The scientist gave no reason for not starting another card.

## 2. Recipes and reasoning

- **Data**: `nvidia/OpenMathInstruct-2` rows whose `problem_source` is gsm8k/augmented_gsm8k, pinned to revision `469216e3f4…`. exp-02 = 120,000 rows (2 solutions/problem), exp-04 = 253,045 (`--max-per-problem 5`), exp-05 = 350,000 (`--max-per-problem 20`). Contamination checked on every build (0/120k, 0/253k, 0/350k).
- **Prompt rendering**: `scripts/render.py` embeds the grader's own `templates/gemma3.jinja` with a `TEMPLATE_SHA256` constant, corrected to the real hash at 18:40:11 (L3247) and verified against the tokenizer at 18:40:21 (L3260). This is the `template_unreachable` pitfall handled before any training.
- **Few-shot prefix**: none — exp-02 trains bare question→solution rows; the base failure diagnosed in exp-01 was continuation of the grader's few-shot pattern ("203 of 300 completions ran on into invented problems", L9329).
- **SFT hyperparameters** (identical across exp-02/04/05): lr 2e-5, 1 epoch, batch 64, grad_accum 2, max_seq_len 1280, fp32 master weights with bf16 autocast and `adamw_bnb_8bit`, liger kernels. Chosen from four timed GPU probes at 18:46–18:57, all recorded afterwards in `exp-02.yaml: situation.smoke_runs` (3 entries).
- **RL**: never considered in any visible text. The only `grpo|dpo` strings in the whole trace are the skill's own family enum (L143, L202, L273, L2191) — the prepared timeline's `first_rl 18:44Z` marker is a **false positive**: its regex is `grpo|dpo|ppo` and `su-p-p-o-rted` contains `ppo`; the match is `verdict: supported` in the exp-01 close at 18:44:43 (L4144).
- **RFT**: `scripts/sample_rft.py` was written at 19:06:23 (L5292) with a docstring describing k-sample rejection sampling, and then **never invoked** — no `sample_rft` execution appears in the 173 tool calls. The scientist left no written reason; the trace SHOWS it chose data scaling (exp-04, exp-05) with the same hours instead.
- **Budget reasoning** appears only inside cards. exp-03 `situation.alternatives_rejected` (card key, artifact `memory/cards/exp-03.yaml`):
  > {option: "Leave decoding alone and spend the time on more training", reason: "this costs one 7-minute eval and changes no weights; if it is worth points it is the cheapest point on the board"}

  and the card-shape constraint, same key:
  > {option: "Sweep temperature 0.3/0.5/0.7 as well", reason: "one card is one intervention, and the eval is single-sample, where greedy is the standard choice"}

## 3. Decode config

**Yes, measured, and verified in the grader's own output.**

- Mechanism established *before* the first post-SFT eval: at 19:04:54 (L5092) the scientist printed `vllm.config.ModelConfig.get_diff_sampling_param` source, and at 18:33:50 (L2080) read `inspect_ai/_openai.py` to confirm inspect sends no temperature. `exp-03.yaml: situation.trigger_evidence[1]` records "line 233: params['temperature'] only set when config.temperature is not None".
- Intervention (exp-03, 20:21:23, L6177): a symlink-only copy of exp-02/final with `temperature: 0.0, do_sample: false`, `top_k`/`top_p` removed. Identical weights.
- Measured gain: **0.6833 → 0.760 at n=300 (+7.7 pts)**, `eval/exp-03_greedy_dev300.json`. Latency from first post-SFT eval scored (20:16:22) to the measured decode choice recorded (20:30:23) = **0.23 h** — inside A v2's ≤0.5 h screen bar.
- Half the stated mechanism was falsified and the card says so: `result.diagnostic_result.notes` — "completions at the 4000-token cap went 4 -> 6 of 300, so greedy decoding slightly increases repetition loops rather than removing them; that half of the mechanism is contradicted."
- Grader observable confirmed at 02:56:20 (L9223), on `final_model` in a fresh process:
  > Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator.
- Shipped artifact (`final_model/generation_config.json`, receipt path): `do_sample: false`, `temperature: 0.0`, `eos_token_id: [1, 106]` — the semantically-clean form, written to JSON after save (transformers' `GenerationConfig.validate` rejects that pair on save; see Q5).

## 4. Evaluation practice

Actual n from the retained Inspect logs on the receipt-backed result volume (the prepared facts' byte-derived `est_n=[149,301,307,313,317,747,795,798,812]` is wrong in every entry):

| scored | artifact | n | acc |
|---|---|---:|---:|
| 18:38:43–18:43:01 | base snapshot | 300 | 0.060 |
| 20:15:33–20:16:22 | exp-02/final (sampled decode) | 300 | 0.6833 |
| 20:24:04–20:24:47 | exp-03_greedy | 300 | 0.760 |
| 22:44:31–22:45:15 | exp-04/final | 300 | 0.830 |
| 02:01:14–02:01:53 | exp-05/final | 300 | 0.800 |
| 02:08:28–02:09:37 | exp-04/final | 800 | 0.8175 |
| 02:10:43–02:11:49 | exp-05/final | 800 | 0.81625 |
| 02:35:39–02:36:43 | exp-07_soup | 800 | 0.83875 |
| 02:47:32–02:49:08 | final_model, grader defaults | 150 | 0.8467 |

- Every decision up to exp-05 rested on **n=300**; the ship decision rested on **n=800**.
- **An inversion was caught.** At n=300, exp-04 (0.830) beat exp-05 (0.800) by 3.0 pts. At n=800 the ordering collapsed to 0.8175 vs 0.81625 and the card's own paired test says so: L8372–8373, 02:22:47 — "paired items 800 A-only-correct 46 B-only-correct 45 / mcnemar se 9.54 z 0.1". exp-06's verdict is `contradicted`, decision `reject`.
- **Paired statistics were used** (McNemar, `analysis/exp-06_paired.json`, L8358–8364), and again for the soup: on the 91 items the parents split, the soup takes 65 (L9329).
- All developer evals are prefixes of the *official* `openai/gsm8k` test set (`evaluate.py --limit N`; inspect log `eval.dataset.name = openai/gsm8k`, 1319 items). Selection therefore happened on subsets of the graded set — inherent to the harness's only evaluator, not a protocol violation, but it means the n=150 official-default number (0.8467) is not out-of-sample. Full-test evaluation was never run by the scientist; the official 1319-item score, 0.8279, is 1.1 pts below the n=800 dev estimate and 1.9 below the n=150 one, consistent with the first-N prefix being easier.

## 5. Pitfalls (every loss ≥0.1 h, plus one recorded as zero)

Card-summed `pitfalls_cost_h = 0.85` over 6 entries (collect). Event-deduplicated:

| # | cause | cost | evidence |
|---|---|---:|---|
| 1 | **`final_model_not_loadable`** — `save_pretrained` does not write `preprocessor_config.json`/`processor_config.json`; vLLM loads Gemma-3 as multimodal and refuses the dir. Killed the first exp-02 eval. | **0.15 h** (card); trace-bounded ~0.12 h: launch 20:06:35 (L5920) → `OSError: Can't load image processor for '/home/ben/task/ckpts/exp-02/final'` read at 20:13:38 (L5971) → relaunch 20:13:56 (L6042) | card `exp-03.yaml: situation.pitfalls_hit[0]`; fixed permanently by patching `train_sft.py` at 20:34:12 (L6692) |
| 2 | OpenMathInstruct-2's gsm8k-derived pool is only ~70k distinct problems; four extra shards added **7** new problems | 0.20 h | `exp-04.yaml: pitfalls_hit[0]` |
| 3 | `--max-per-problem 10` over 7 shards yielded only 291k rows (+15 %) — not worth a 2.4 h run | 0.30 h | `exp-05.yaml: pitfalls_hit[0]` |
| 4 | GPU config probes: grad-checkpointing off at micro-batch 16 OOMs at 79.1/79.2 GiB; micro-batch 8 runs at 3.7k tok/s | 0.05 + 0.10 h | `exp-02.yaml: pitfalls_hit[0..1]` |
| 5 | one built row carried a second `ANSWER:` line inside the body (`double_answer_format`) | 0.05 h | `exp-02.yaml: pitfalls_hit[2]`; caught by inspection at 18:41:20 (L3559), before training |
| 6 | **Unrecorded**: `scripts/soup.py` aborted on its first run — `ValueError: GenerationConfig is invalid: temperature is set to 0.0 … do_sample is set to False` (L8697–8722, 02:29:11). Both soup parents carry the greedy config this cell writes. | ~0.09 h (02:24:08 launch → 02:29:25 patched relaunch); card records **0.0** | `exp-07.lock.json: relocked_from[0].reason` — the loss is visible only in the relock reason, so `pitfalls_cost_h` understates this cell |

Post-exit idle (E2's metric), monitor-anchored (`task/system_monitor.log`, 60 s sampling, ±1 min): **cumulative ≥0.85 h over 10 events**, all fixed-`sleep` overshoot. Largest three: 02:11:49 (both n=800 evals done) → 02:22:47 next observation = **0.18 h**; 02:49:08 → 02:56:20 = 0.12 h; 02:36:43 → 02:43:33 = 0.11 h. This cell fails E2's <0.15 h/cell bar on the cumulative measure and on one single event.

**Traps prior knowledge avoided**: template mismatch (hash-checked render before training, L3234/L3260); `<bos>` duplication and stop tokens (the render module owns termination); contamination (checked on all three builds); session death (all waiting is foreground `sleep; tail`, and `ps aux` at 02:56:49 shows 0 producers left).

## 6. Protocol interaction

- **Per-card cost.** `awm exp_protocol new` → `lock` spans: exp-01 18:35:04→18:36:30 (1.4 min), exp-02 19:00:36→19:01:24 (0.8), exp-03 20:21:34→20:22:21 (0.8), exp-04 20:34:15→20:35:06 (0.85), exp-05 22:54:39→22:55:29 (0.8), exp-06 02:06:47→02:07:23 (0.6), exp-07 02:23:07→02:24:04 (0.95), exp-08 02:45:11→02:45:49 (0.6). Closes take 10–30 s (`cat >>` + `awm exp_protocol close`). Total protocol tool time 0.004 h over 39 pure-protocol calls; adding the model-generation gaps that precede them bounds the whole ceremony at **≤0.10 h of 8.43 h**.
- **Bootstrap's first action**: `Skill: exp_protocol` at 18:31:04 (L10), then `cat card.template.yaml` / `pitfalls.yaml` / `example-card.yaml` at 18:31:10–18:31:11. First substantive environment action 18:31:15. Cost ≈11 s of wall time.
- **Preflight changed decisions twice, both by override, not by repair.**
  - 18:36:24 (L2542) `not locked: preflight failed on data_files_exist` for an eval-only card; overridden 18:36:30 with "exp-01 trains nothing; setup.data is required by the schema…". No fake file was created — the card points `setup.data` at the metrics JSON the eval will write. Direction #5 / candidate H, third variant of the workaround.
  - 02:07:19 (L8274) `FAIL comparator_same_protocol — /home/ben/task/eval/exp-06_exp04_dev800.json does not exist`; overridden 02:07:22: "Both arms of this card are measured inside the card under the one protocol it defines (n=800); the comparator file is one of the two outputs, so it cannot exist before the run." This is exactly candidate K's (#27) in-card head-to-head dependency, reproduced in a NEW guard cell.
- **The card format shaped the plan, visibly.** (a) "one card is one intervention" killed a temperature sweep (exp-03 `alternatives_rejected`). (b) Rule 6 (`keep the checkpoints you said you would`) is why exp-05 — verdict `contradicted`, decision `reject` — still had a live checkpoint 3.5 h later for the soup that produced the shipped model. (c) The comparator rule forced the n=800 head-to-head that overturned the n=300 ordering.
- **Two relocks, both honest and documented.** exp-02 relocked 20:21:08 because `train_sft.py` was edited *after* the run to copy the processor files ("It adds no training behaviour and could not have affected the checkpoint this card produced"); exp-07 relocked 02:29:25 for the soup save fix.
- **Stop hook: 0 blocks.** `memory/.stop_hook.json` is absent from the harvested bundle and the raw `task/memory/`; the hook writes that counter on every block (`hooks/stop_open_cards.py`, `COUNTER = Path("memory")/".stop_hook.json"`). The guard's *pitfall text* was nevertheless read out on every one of the 10 lock invocations (`run_dies_with_the_session`, e.g. L2541, L9152) and the behaviour matches it: 22 background launches, every one waited for in the foreground, 0 processes killed at exit.
- **Launch scope (#26/J) reproduces.** Six GPU training probes ran between 18:44:58 and 18:57:29 with no card open (L4195, L4262, L4376, L4420, L4465, L4602); the first training card locked at 19:01:24. They are documented *afterwards* in `exp-02.yaml: situation.smoke_runs` (3 entries) — a later `smoke_runs` entry is not a pre-launch lock. The cell-reader's `lock_before_launch = 3/3` counts only card-matched main training commands; do not read it as full compliance.
- **Complaint/workaround**: none written. `n_cards 8, n_closed 8, n_locked_open 0, fields_filled 1.0`.
- **Rule 7 was broken once.** `exp-02.yaml: problem.failure_examples[0]` stores a verbatim GSM8K test question and its gold answer:
  > question: "Cody eats three times as many cookies as Amir eats. If Amir eats 5 cookies, how many cookies do both of them eat together?", gold: "20"

  Rule 7 says "No item from it in `failure_examples`, `watch_set`, or training data." exp-03/04/05 use paraphrases but still record ids + gold values. No score consequence (cards never enter training), but the rule is unenforced.

## 7. Verdict

**Three largest contributors to 0.8279** (the highest score in the Window04 roster; NEW guard mean 0.7354, control 0.7515, baseline-v3 0.7737):

1. **The decode config, +7.7 pts on identical weights** (0.6833→0.760 @n=300, `eval/exp-03_greedy_dev300.json`). Cost: one 7-minute eval. Nothing else in the cell bought points that cheaply.
2. **Solutions-per-problem scaling, +7.0 pts** (exp-04: 253k rows @ 5 solutions/problem → 0.830 @n=300 vs exp-02's 0.760 greedy-equivalent). The cell established that the *problem* pool is capped at ~70k (`exp-04` pitfall, 0.20 h) and correctly pivoted the axis to solutions rather than problems.
3. **The two-way soup, +2.1 pts at n=800** (0.8175 → 0.83875), and it was only available because a *rejected* card's checkpoint was kept under rule 6. Verified as a real average, not inheritance: on the 91 items the parents split, the soup takes 65 (L9329).

**What this cell did that the other guard cells did not**: it settled its shipping decision at n=800 with a paired test *and* re-verified the packaged artifact end-to-end under the grader's exact defaults from a fresh process (exp-08), including reading vLLM's own override line back out of the log. g01r04 shipped on point estimates with no paired test; g01r05 measured at n=1319 but never re-ran the grader's default invocation on a comparably scoped check.

**One protocol change most likely to have raised this cell**: none of the covered candidates would have added score — the only ≥0.1 h loss (0.15 h) is the missing Gemma-3 `preprocessor_config.json`/`processor_config.json`, which is direction #22 / P3, already queued. D (`8332917`) would have prevented the unrecorded 0.09 h soup abort, but see the scope note below.

**Proposal (one item, uncovered)** — `test_item_text_absent`, a new preflight check plus its test:
- *What*: at `lock`, scan the card's free-text fields (`problem.failure_examples[*]`, `watch_set`, `problem.evidence`) for verbatim question text or gold answers from the benchmark test copy the contamination checker already reads; fail the lock, overridable with a reason.
- *Source cells (≥2)*: g01r03 `exp-02.yaml: problem.failure_examples[0]` (verbatim test question + gold "20"); g01r05 `exp-02/03/04.yaml: failure_examples[*].question = "(text not reproduced - benchmark test item, rule 7)"`, `gold: "(withheld)"` — the same rule read strictly, at the cost of the card's diagnostic content. g01r04 sidesteps it entirely with a GSM8K-*train* watch set.
- *Surface*: one preflight check with a test (allowed). It enforces existing frozen SKILL.md rule 7; it adds no clause.
- *Screen metric a 4-cell block can read*: cells with any verbatim test-item text or gold value in a card = 0/4; overrides on the new check ≤1/cell; mean `failure_examples` per card and `fields_filled` not below the baseline pool (so the check does not just empty the field).
- *Guardrail*: protocol-baseline pool mean −0.03.
- *Coverage checked against frozen text*: not in A v2 `f6cdccc`, B v2 `9f294c3`, D `8332917`, E2 `c6f11d8`, H `b52e5f2`, J `549e25a`, K `58a6992`, P1 `a4c4954`, or ledger #16–#29. Honest limitation: **no score effect is observed or claimed** — this is an evidence-integrity item.

<!-- END REPORT cell=g01r03 -->

<!-- REPORT cell=g01r04 role=NEW -->

```yaml
cell: g01r04
arm: protocol (guard, --exp-protocol --tool claude --stop-hook, awm 4ae3d87 / tree 189319d6)
accuracy: 0.690674753601213          # official, n=1319, stderr 0.012732 — lowest of the six NEW guard cells
hours_used: 7.75                     # trace 18:31:04Z -> 02:16:00Z; time_taken.txt 07:45:27
hours_to_first_train_launch: 0.37    # first card-locked training 18:53:27Z; first train_sft.py of any kind 18:41:40Z (+0.18 h, uncarded smoke)
protocol_hours: 0.05                 # 0.002 h tool + <=0.050 h card-composing generation
waiting_hours: 6.56                  # raw; 4.34 h pure sleep/poll + 2.22 h composite launch+wait; post-exit idle >=0.75 h
greedy_shipped: yes                  # measured +5.8 pts @n=500; shipped file keeps do_sample:true (inert under vLLM) — see Q3
rl_used: no
rft_tried: yes (verdict supported, adopted; +3.0 pts @n=500 — but it cost 2.15 h of the card's 2.28 h pitfall total)
largest_eval_n: 1000
stop_reason: "Done. All experiment cards are closed and no processes are running." (02:16:00Z; timer 2:15 left at 02:15:41)
top_contributors: [SFT termination/format fix from 0.050 to 0.614, greedy decode +5.8pts@500, RFT +3.0pts@500 bought at 2.15 h of pitfall cost]
one_protocol_change: a pitfalls entry requiring a long unattended generation run to make its first durable write before its final transform step — 1.6 h of completed sampling was destroyed by a post-generation grading crash
knowledge_to_transfer: [vLLM offline generate() stops on tokenizer eos only, not generation_config's [1,106] — 60937/63892 samples ran on; pkill -f matches the Bash tool's own command line]
```

## 1. Timeline

Session 2026-09-03T18:31:04Z → 2026-09-04T02:16:00Z, 209 assistant turns, 138 tool calls, `stop_reason: end_turn`, `terminal_reason: completed`.

| stage | time | +h | evidence |
|---|---|---|---|
| bootstrap `Skill: exp_protocol` | 18:31:04 | 0.00 | L10; `Read` of pitfalls/template/example 18:31:06–18:31:08 (L177, L251, L399) |
| base eval launched (n=200) | 18:35:42 | 0.08 | L2227, after exp-01 lock 18:35:38 |
| first `train_sft.py` (uncarded smoke) | 18:41:40 | 0.18 | L3774 `--out ckpts/smoke` |
| uncarded GPU eval of `ckpts/smoke2/final` (n=50) | 18:48:39 | 0.29 | L4055 |
| **first card-locked training** (exp-02 SFT) | 18:53:27 | **0.37** | L4370, lock 18:53:21 (L4347) |
| first post-SFT eval scored | 20:30:13 | 1.99 | inspect log `…20-29-51…`, n=200, 0.565 |
| measured greedy decision closed | 20:52:08 | 2.35 | exp-03 result appended, L5470 |
| RFT sampling, 1st attempt | 20:52:40 | 2.36 | L5591 |
| RFT sampling, final attempt starts | 22:49:10 | 4.30 | L6530 (after two aborts + one crash) |
| exp-04 SFT launched | 23:30:03 | 4.98 | L7129, lock 23:29:59 |
| `final_model` installed + verified | 02:03:26 → 02:07:53 | 7.54 | L7510, L7709 |
| end | 02:16:00 | 7.75 | turn 209 |

Hours by category (re-derived):

| category | h | note |
|---|---:|---|
| waiting on runs (raw) | 6.56 | 4.34 h pure poll (25 calls) + 2.22 h calls that launched then waited (16 calls) |
| — of which post-exit idle | ≥0.75 | monitor-anchored, ±1 min; event-summed ≈0.96 h; **largest single 0.36 h** (01:41:52 → 02:03:23) |
| train launch | 0.41 | 9 calls |
| sample/eval | 0.30 | 9 calls |
| protocol | 0.05 | 25 pure-protocol calls |
| data / bash_other / code | 0.02 | |
| model generation | 0.34 | |

Time left: `timer.sh` returned **2:15** at 02:15:41 (L7919 result). Stated reason, quoted (02:16:00):
> Done. All experiment cards are closed and no processes are running.

The scientist did name what it would have done next — "Untested leads: … a preference method would be the natural attack" — but gave no reason for not spending the remaining 2.25 h. The trace SHOWS the GPU idle from 02:07:53 onward.

## 2. Recipes and reasoning

- **Data**: `openai/gsm8k` train + `nvidia/OpenMathInstruct-2` (gsm8k / augmented_gsm8k). exp-02 = 140,494 rows (`--n-omi 200000 --n-fewshot 3000`), including a deliberate few-shot-prefixed slice; exp-04 = 164,789 rows = the same offline mixture plus **24,295 answer-verified self-generated rows** (`rft_v2.jsonl`, 2 kept per question). 500 GSM8K-*train* items held out as `data/dev_train500.jsonl`.
- **Few-shot handling** is explicit and unusual among the three: `merge_data.py` "re-assigns the few-shot-prefixed slice" (L6191, 21:28:13), i.e. a fraction of training rows carry the grader's 10-shot prefix. `scripts/fmt.py` (L2518, 18:37:53) verifies both renderings byte-for-byte against `templates/gemma3.jinja` (sha `7de1c58e208e`), recorded in `exp-02.yaml: smoke_runs[0]`.
- **SFT hyperparameters** (exp-02 and exp-04 identical): lr 1e-5, 1 epoch, token-budget batching at 24,576 tokens, grad_accum 8, max_seq_len 3072, bf16. exp-04 retrains **from base**, not from exp-02 — so it never met the greedy-parent save trap.
- **RL**: not used and not discussed. The only `grpo|dpo` strings are the skill's family enum (L143, L269, L340, L1696). The prepared cell-reader line `RL launches=0` is correct; the prepared timeline's `first_rl 18:41Z` is the same `ppo`-inside-`supported` false positive noted for g01r03.
- **RFT: tried, adopted, and expensive.** `exp-04.yaml: hypothesis.claim` predicted "+2 to +5"; measured +3.0 @n=500. The reasoning that made it worth doing, `exp-04.yaml: situation.trigger` and the closing text (02:16:00):
  > The unique-problem count looks like the binding constraint — training loss was flat from step 80 of 320 in both runs, and half-epoch and full-epoch checkpoints tie.
- **Budget reasoning**: the visible instances are scope cuts under time pressure, not written deliberations — the sampling pool was cut `--n-omi 35000 → 33000 → 28000 → 9000` across four launches (L5591 20:52:40, L5960 21:01:01, L6077 21:07:15, L6530 22:49:10).

## 3. Decode config

**Yes — measured earliest of the three cells, and the mechanism was observed from a runtime log line before any card claimed it.**

- Observable seen at 18:55:38 (L4497), in the *smoke* eval log, i.e. 2 h before the decode card:
  > (APIServer pid=2536) Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator.

  Cited as `exp-03.yaml: situation.trigger_evidence[0]` (`logs/smoke2_eval.log`).
- Intervention (exp-03, 20:44:41 lock, L5377): `scripts/mkvariant.py --src ckpts/exp-02/final --dst ckpts/exp-02/final_greedy --set temperature=0.0 --drop top_k top_p`, weights symlinked.
- Measured gain: **0.614 → 0.672 at n=500 (+5.8 pts)**, `eval/exp-02_final_dev500.json` vs `eval/exp-03_greedy_dev500.json`. Latency from first post-SFT eval scored (20:30:13) to the measured decode choice recorded (20:52:08) = **0.37 h** — inside A v2's ≤0.5 h bar.
- **Shipped file is not what the card's first write produced.** At 02:03:26 the packaging step wrote `do_sample: False, temperature: 0.0` and the tool result confirms it (L7548). At 02:04:18 (L7660) the scientist deliberately overwrote it to make the artifact byte-identical to the *measured* checkpoint:
  > cp ckpts/exp-04/final_greedy/generation_config.json final_model/generation_config.json … && echo "IDENTICAL to the measured artifact"

  The diff returned `IDENTICAL to the measured artifact` (L7706, 02:04:30) and the file that shipped therefore carries **`do_sample: true` with `temperature: 0.0`** (verified on the receipt-backed volume: `…_90650/final_model/generation_config.json`).
- SAID vs SHOWS: `exp-04.yaml: conclusion.next_step` says `final_model` is "byte-identical to the measured `ckpts/exp-04/final_greedy`". The trace SHOWS the *config* was diffed and identical; the *weights* were `cp -rL ckpts/exp-04/final` (L7510) and were never hash-compared — `final_greedy` symlinked those same weights, so the claim is very likely true but only half-verified.
- **Effect on the grade: none.** vLLM reads `temperature/top_p/top_k/min_p/repetition_penalty` from `generation_config.json` and ignores `do_sample`; `temperature: 0.0` is what makes it greedy. The post-change verification run at 02:06:11 (n=150 → 0.6933) and the official 1319-item score (0.6907) are both consistent with the n=1000 greedy dev number (0.681). The `do_sample: true` is a transformers-level inconsistency in the shipped record, not a decoding failure.

## 4. Evaluation practice

Actual n from the retained Inspect logs (the prepared facts report 10 logs and `est_n=[49,148,193,195,480,484,485,486,970,974]`; there are **11** logs — one was renamed to `evallogs/exp-01_base_dev200_full.json` — and every byte-derived n is off):

| scored | artifact | n | acc |
|---|---|---:|---:|
| 18:37:34–18:39:00 | base snapshot | 200 | 0.050 |
| 18:50:17–18:50:26 | ckpts/smoke2/final | 50 | 0.320 |
| 20:29:51–20:30:13 | exp-02/final | 200 | 0.565 |
| 20:31:43–20:32:06 | exp-02/checkpoint-160 | 200 | 0.590 |
| 20:37:25–20:38:11 | exp-02/final | 500 | 0.614 |
| 20:46:17–20:47:02 | exp-02/final_greedy | 500 | 0.672 |
| 01:24:08–01:24:54 | exp-04/final_greedy | 500 | 0.702 |
| 01:26:38–01:27:26 | exp-04/c227_greedy | 500 | 0.716 |
| 01:37:43–01:39:18 | exp-04/c227_greedy | 1000 | 0.684 |
| 01:40:20–01:41:52 | exp-04/final_greedy | 1000 | 0.681 |
| 02:06:11–02:07:53 | final_model, grader defaults | 150 | 0.6933 |

- **An inversion was seen and correctly resisted.** At n=200 exp-02's half-epoch `checkpoint-160` (0.590) beat `final` (0.565); `exp-03.yaml: pitfalls_hit[0]` records "the gap is inside the noise of a temperature-1.0 sampler at n=200", cost 0.0 h. At n=500 the two exp-04 checkpoints split 0.716 (c227) vs 0.702 (final); at n=1000 they tied 0.684 vs 0.681, and the card says so: "the two kept checkpoints differ by 1.4 pts at n=500 … but are tied at n=1000, so that gap is dev noise, not a real ordering."
- **No paired statistic anywhere.** `grep -i "mcnemar|paired"` returns nothing in this trace, unlike g01r03 and g01r05. The tie call at n=1000 rests on overlapping point estimates only, and the cell then shipped `final` (0.681@1000) over `c227` (0.684@1000) — a defensible tie-break, but not a tested one.
- **A watch set was declared and never scored.** `watch_set: {path: data/dev_train500.jsonl, n: 500}` appears in exp-02, exp-03 and exp-04; all three carry `watch_set_result: null`. `fields_filled = 1.0` regardless — a filled field is not an exercised one.
- The 500 held-out items are GSM8K *train*, so no test text entered training; the ship decisions themselves still ran on prefixes of the official test set via `evaluate.py --limit N`.
- Official 1319 = **0.6907**, slightly *above* the n=1000 dev estimate (0.681); the last 319 items were not harder for this model.

## 5. Pitfalls (every loss ≥0.1 h)

Card-summed `pitfalls_cost_h = 2.28` over 7 entries — the highest of the six NEW guard cells, and 2.15 of it sits on one card.

| # | cause | cost | evidence |
|---|---|---:|---|
| 1 | **RFT sampling completed and then destroyed itself.** 139,892/139,892 prompts generated in 1:37:31, then the in-process grading loop raised `OverflowError: cannot convert float infinity to integer` in `graded_number()`; the script wrote nothing until the end, so every sample was lost. | **1.6 h** | L6410, observed 22:48:19; launch was 21:07:15 (L6077). Card `exp-04.yaml: pitfalls_hit[1]`. Repaired by dumping raw generations first — `data/rft_v1.jsonl.raw.jsonl` (77.7 MB) appears in `status.json: skipped` |
| 2 | **vLLM's offline `generate()` does not stop on `<end_of_turn>`** — its stop set comes from tokenizer eos (id 1), not `generation_config`'s `[1, 106]`; **60,937 of 63,892** samples ran on into a second `<start_of_turn>model` turn. Grading the last number of that run-on text reported 27.6 % when the true figure was 59.6 %. | 0.1 h | `exp-04.yaml: pitfalls_hit[0]`; the closing summary repeats the counts (02:16:00). This is exactly candidate B v2's (#3) mechanism, in a NEW guard cell |
| 3 | No FlashInfer in this image, so top_k/top_p sampling fell back to a PyTorch sort over the 262k vocab; generation ran at 7.3k tok/s | 0.4 h | `exp-04.yaml: pitfalls_hit[2]` |
| 4 | `pkill -f rft_sample.py` matched the Bash tool's **own** command line and killed the calling shell — the call returned `Exit code 144` | 0.05 h | L5822, 21:00:01; card `pitfalls_hit[3]`, fix "kill by pid from ps". Recovered at 21:00:18 (L5870) with `ps -eo pid,cmd \| … \| xargs kill -9` |
| 5 | Piping the trainer through `grep` in a Bash call buffered all output; a 5-minute smoke run looked hung | 0.08 h | `exp-02.yaml: pitfalls_hit[1]` |
| 6 | inspect-ai wrote its per-sample log into `./logs` mixed with training logs | 0.05 h | `exp-02.yaml: pitfalls_hit[0]`; fixed by `INSPECT_LOG_DIR=evallogs/...` from 18:48:39 on |

Post-exit idle: **≥0.75 h monitor-anchored (≈0.96 h event-summed) over 8 events**, all fixed-`sleep` overshoot. The largest is unambiguous: both n=1000 evals finished scoring at **01:41:52** (inspect `completed_at`), the GPU released ~01:42, and the agent's next action was at **02:03:23** — two consecutive `sleep 800`/`sleep 780` calls with no producer alive = **0.36 h**, the largest single post-exit idle across the three cells I reviewed. This cell fails E2's <0.15 h/cell bar by a wide margin.

**Traps prior knowledge avoided**: template mismatch (byte-for-byte check in `scripts/fmt.py` before training); contamination (0/164,789); session death (18 background launches, all foreground-waited, 0 killed at exit); the greedy-parent save abort (avoided structurally — exp-04 retrains from base, so `parent_generation_config_valid` would have passed).

## 6. Protocol interaction

- **Per-card cost**: exp-01 `new` 18:33:54 → lock 18:35:38 (1.7 min, including creating the placeholder file); exp-02 write 18:53:16 → lock 18:53:21; exp-03 write 20:44:04 → lock 20:44:41 (0.6 min); exp-04 write 23:29:56 → lock 23:29:59. Closes 10–20 s. Total pure-protocol tool time 0.002 h over 25 calls; with composing gaps, **≤0.05 h of 7.75 h** — the cheapest ceremony of the three cells, and it wrote the fewest cards (4).
- **Bootstrap's first action**: `Skill` at 18:31:04 then three `Read`s of pitfalls/template/example, first environment action 18:31:12 → ≈8 s.
- **Preflight: 1 observed failure, 4 overrides, 0 repairs.** At 18:35:31 (L2183) `not locked: preflight failed on stop_token_consistent, answer_marker_single` for the eval-only exp-01. The scientist did not fix the data — it had already, at 18:35:26 (L2145), *manufactured* one:
  > printf '%s\n' '{"note":"placeholder: exp-01 is a measurement card … trains on nothing","completion":"none"}' > eval/no_training_data.jsonl

  and then overrode both checks (18:35:37, L2193). At exp-03 (20:44:40, L5377) it passed the same two overrides **pre-emptively**, with no failing attempt. This is direction #5 / candidate H's target behaviour in its strongest form seen this window: a fabricated data file plus two content overrides, on two of four cards.
- **The card format shaped the plan.** With only 4 cards, each covered a large step; the comparator rule is what produced the n=500 re-measure of exp-02 (`exp-03.yaml: smoke_runs[1]` — "0.614 +/- 0.022 — the comparator") before the greedy claim could be made. No `falsified_if` abandoned a direction here.
- **Stop hook: 0 blocks** (`memory/.stop_hook.json` absent). The `run_dies_with_the_session` text was printed on all 4 lock invocations (L2182, L2214, L4366, L5388, L7125) and the behaviour matches: 18 background launches, all foreground-waited, `ps` clean at 02:15:41.
- **Launch scope (#26/J) reproduces**: two `train_sft.py` smoke runs (18:41:40 L3774, 18:42:20 L3822) and one GPU eval of `ckpts/smoke2/final` (18:48:39 L4055) ran before the first training card locked at 18:53:21; all three are recorded afterwards in `exp-02.yaml: smoke_runs`. `lock_before_launch = 2/2` counts only card-matched main training commands.
- **Complaints/workarounds**: none written; the only workaround is the placeholder data file. `n_cards 4, n_closed 4, n_locked_open 0, n_relocked 0, fields_filled 1.0`.
- Notable non-protocol side effect: at 02:15:18–02:15:36 (L7823, L7859, L7906) the scientist wrote three files into `/home/ben/.claude/projects/-home-ben-task/memory/` (`ptb-grader-inherits-generation-config.md`, `vllm-offline-generate-ignores-chat-eos.md`, `MEMORY.md`). That is the CLI's own memory directory, not the protocol's `task/memory/` — it does not appear in `collect` and is outside the frozen protocol surface.

## 7. Verdict

**Three largest contributors to 0.6907** (lowest of the six NEW guard cells; guard mean 0.7354):

1. **The SFT termination/format fix carried the whole floor→0.61 jump.** Base 0.050 with 130/200 completions at the token cap and only 27/200 ending on `ANSWER:`; after exp-02, non-termination 65 % → 0.5 % (closing summary, 02:16:00). Everything after that is a 5–9 point layer.
2. **Greedy decode, +5.8 pts on identical weights** (0.614 → 0.672 @n=500), decided 0.37 h after the first post-SFT eval.
3. **RFT, +3.0 pts @n=500 — bought at 2.15 h of pitfall cost.** This is the causal explanation for the low score: the cell spent 20:52→23:26 (2.56 h wall) on one sampling round that produced 24k rows, and only reached its main training launch at 23:30 with 2.8 h left. It ran **one** post-SFT training round; g01r03 ran three and souped two of them, g01r05 ran three. The 1.6 h crash directly forced the sampling pool from 28k+7k prompts down to 9k+7k, so the RFT layer that did land was the small version of the one that was planned.

**What this cell did that the other guard cells typically did not**: it committed a single large method bet (rejection sampling) early, with no paired test and no second training axis in reserve; and it is the only one of the three whose shipped `generation_config.json` disagrees with what its own card text implies about `do_sample` (inert here, but it would not be inert under a transformers-based grader).

**One protocol change most likely to have raised this cell**: not the decode candidates (A v2's bar was already met) and not D (it trained from base). The 1.6 h loss is the whole gap.

**Proposal (one item)** — a new `pitfalls.yaml` entry, `long_run_defers_its_only_write`:
- *What*: a long unattended generation/training run must make a durable write before its final transform/serialize step — dump raw generations before grading them; write the first checkpoint early. `check: null` (not machine-checkable), with the two source incidents named.
- *Source cells (≥2)*: **g01r04 exp-04** — 1:37:31 of completed generation destroyed by an `OverflowError` in the post-generation grading loop (L6410, 22:48:19), repaired only by adding a raw dump; **g01r05 exp-03** — 50 min of training destroyed at the first checkpoint save, leaving `ckpts/exp-03/checkpoint-620/` holding only a `config.json` and no weights (`exp-03.yaml: result.failure`). Same consequence class: the run did all the work and died at its single durable-write step.
- *Surface*: one `pitfalls.yaml` entry (allowed).
- *Screen metric a 4-cell block can read*: hours attributed to "run completed its compute but produced no durable output" = 0/cell; count of launches longer than 20 min whose first durable write is deferred to completion.
- *Guardrail*: protocol-baseline pool mean −0.03.
- *Coverage checked against frozen text*: B v2 `9f294c3` covers vLLM offline sampling defaults (stop-id loss at n>1, parser inf, `finish_reason` probing) — it does not say "write raw before transforming". D `8332917` fixes the *cause* of the g01r05 instance (greedy parent) but its guidance is about parent configs and Trainer saves, not output durability. **Disclosure**: the g01r05 instance's root cause is D's; only the consequence class is shared. If the planner judges that too thin, this should be recorded as an observation from one cell rather than a candidate.

<!-- END REPORT cell=g01r04 -->

<!-- REPORT cell=g01r05 role=NEW -->

```yaml
cell: g01r05
arm: protocol (guard, --exp-protocol --tool claude --stop-hook, awm 4ae3d87 / tree 189319d6)
accuracy: 0.7611827141774071        # official, n=1319, stderr 0.011744
hours_used: 7.92                     # trace 18:31:01Z -> 02:26:06Z; time_taken.txt 07:55:25
hours_to_first_train_launch: 0.44    # first card-locked training 18:57:46Z; first train_sft.py of any kind 18:44:16Z (+0.22 h, uncarded probe)
protocol_hours: 0.09                 # 0.016 h tool + <=0.069 h card-composing generation
waiting_hours: 6.70                  # raw; 6.27 h pure sleep/poll + 0.43 h composite; post-exit idle >=0.46 h
greedy_shipped: yes                  # designed into the training script before the first launch; mechanism verified, never measured against a sampled comparator
rl_used: no
rft_tried: yes (two rounds; round 1 +4.25 pts @n=1319, McNemar z=3.54 — adopted; round 2 +0.83, z=0.74 — inconclusive but shipped)
largest_eval_n: 1319                 # full official test set, four times
stop_reason: "I left the last ~2h unused rather than run a third round I couldn't have verified and would have been selecting on the test set for a fourth time." (02:26:06Z; timer 2:05)
top_contributors: [SFT stop-token compliance 0.59->0.99 giving +65 pts, RFT round 1 +4.25 pts @n=1319 (paired z=3.54), full-1319 selection eval that reversed an n=200 reading]
one_protocol_change: candidate D (`8332917`, parent_generation_config_valid) — its frozen predicate would have failed both locks that cost this cell 0.85 h; no new candidate is needed for this cell's losses
knowledge_to_transfer: [artifact-appearance waits (`while [ ! -f <out>/config.json ]`) produced ~0 post-exit idle on 3 runs while fixed sleeps produced 30 min on 3 others — and hung risk: a dead run never writes the artifact]
```

## 1. Timeline

Session 2026-09-03T18:31:01Z → 2026-09-04T02:26:06Z, 249 assistant turns, 146 tool calls, `stop_reason: end_turn`, `terminal_reason: completed`.

| stage | time | +h | evidence |
|---|---|---|---|
| bootstrap `Skill: exp_protocol` | 18:31:01 | 0.00 | L10; `Read` pitfalls/template/example 18:31:04–18:31:05 |
| vLLM default-sampling mechanism read | 18:35:36–18:36:06 | 0.08 | L2420, L2874 (`get_diff_sampling_param`, `default_sampling_params`) |
| base eval launched (n=200) | 18:34:37 | 0.06 | L2291, after exp-01 lock 18:34:33 |
| first `train_sft.py` (uncarded probe) | 18:44:16 | 0.22 | L3889 `--max-steps` |
| **first card-locked training** (exp-02 SFT) | 18:57:46 | **0.44** | L5066, lock 18:57:41 |
| exp-02 scored (n=200) | 21:08:48 | 2.63 | inspect log `…21-08-12…` |
| RFT round 1 generation | 21:11:50 → ~21:58 | 2.68 | L6043 |
| exp-03 training launched | 22:04:14 | 3.55 | L6530, lock 22:04:10 — **died at step 620/1238** |
| exp-04 (same run, repaired) launched | 23:05:40 | 4.58 | L7163, lock 23:05:36 |
| exp-05 full-1319 selection eval | 00:29:23 → 00:36:33 | 5.97 | L7572, lock 00:29:20 |
| RFT round 2 → exp-06 training | 00:57:52 → 02:11 | 6.45 | gen L7864; `lock && train` composite L8145 01:32:07 |
| exp-06 scored on all 1319 | 02:13:04–02:15:15 | 7.70 | 0.76194 |
| `final_model` installed, hash-verified | 02:15:35–02:18:01 | 7.74 | L8341; `final_model 7806f35e0b3cbb0b` == `ckpts/exp-06/final 7806f35e0b3cbb0b` |
| exp-07 soup evaluated and rejected | 02:22:01–02:24:26 | 7.85 | 0.7536 |
| end | 02:26:06 | 7.92 | L9060 |

Hours by category (re-derived):

| category | h | note |
|---|---:|---|
| waiting on runs (raw) | 6.70 | 6.27 h pure poll (20 calls) + 0.43 h composite (2 calls) |
| — of which post-exit idle | ≥0.46 | 3 monitor-anchored events; **largest 0.29–0.30 h** (00:36:33 → 00:54:28) |
| sample/eval | 0.37 | 8 calls |
| train launch | 0.25 | 8 calls |
| protocol | 0.09 | 32 pure-protocol calls; the prepared timeline's 0.06 h includes the 01:32Z `lock && train` composite (0.04 h) |
| data / bash_other / code | 0.04 | |
| model generation | 0.39 | |

Time left: `timer.sh` last read **2:05**. Stated reason, quoted (02:26:06):
> I left the last ~2h unused rather than run a third round I couldn't have verified
> and would have been selecting on the test set for a fourth time.

This is the only explicit, reasoned early stop among the three cells — the scientist priced the remaining time against what it could still *verify*, which is precisely direction #23 / P4's target behaviour, exhibited rather than violated.

## 2. Recipes and reasoning

- **Data**: `openai/gsm8k` train + OpenMathInstruct-2 gsm8k-family. exp-02 = 95,642 rows (`--max-per-problem 1 --gsm8k-train-repeat 2`); exp-03/04 = `rft_v1.jsonl` 79,179 rows (59.2k self-generated verified solutions + 20k replay); exp-06 = `rft_v2.jsonl` 58,699 rows from **fresh** questions (`gen_rft.py --exclude-questions data/rft_exp03_raw.jsonl` excluded 30,727 round-1 questions — `exp-06.yaml: smoke_runs[0]`). Contamination: 95,642 / 79,179 / 58,699 documents, 0 matches.
- **Prompt rendering**: `train_sft.py` "renders every training row" through the grader's own `templates/gemma3.jinja`; `exp-02.yaml: smoke_runs[0]` records "ran end to end; grader-template check passed".
- **Hyperparameters**: exp-02 lr 1e-5, 2 epochs, bs 32, ga 4, msl 2048, fp32 master weights + bf16 autocast + `adamw_bnb_8bit`, beta2 0.95. RFT rounds anneal the lr: exp-03/04 7e-6 (2 → 1.5 epochs), exp-06 5e-6 (1 epoch). Chosen from three timed GPU probes at 18:45–18:50 and two OOM findings (see Q5).
- **RL**: not used, not discussed. The only `grpo|dpo` strings are the skill's family enum (L143, L269, L340, L1651). The prepared facts line `RL launches=1 e.g. python scripts/finalize.py --src …` is a **false positive** of the cell-reader's `(grpo|dpo|ppo|rloo)` + `.py` rule: the composite call at L7708 (00:55:09) runs `finalize.py` *and* appends `verdict: supported` to `exp-04.yaml` in the same command — `su-ppo-rted`.
- **RFT: two rounds, explicitly reasoned.** `exp-03.yaml: problem.statement`:
  > exp-02 is unreliable rather than incapable: at temperature 1.0 it solves 94.6% of held-out-style training questions at least once in four tries, but gets all four right on far fewer.
- **Budget reasoning** appears in `exp-05.yaml: situation.alternatives_rejected`:
  > {option: "Run a third training round with the remaining time instead", reason: "another round would land with the same n=200 ambiguity and no budget left to re[solve it]"}

## 3. Decode config

**Yes — shipped, but by design rather than by measurement.** This is the material difference from g01r03 and g01r04.

- The mechanism was established at **18:35:36–18:36:06** (L2420, L2874), before any training: the scientist printed `inspect_ai.model._providers.vllm` and `vllm/config/model.py`'s `get_diff_sampling_param` / `default_sampling_params`, including the source string "Default sampling parameters have been overridden by the " (L2934).
- `scripts/train_sft.py` therefore writes the greedy config into **every** saved final checkpoint from the first run onward (`g01r05/task/scripts/train_sft.py:248–257`: `"do_sample": False, "temperature": 0.0`), and `scripts/finalize.py:45–51` asserts it (`assert gen.get("temperature") == 0.0, "vLLM reads temperature from generation_config; want greedy"`).
- The runtime observable was seen at L5813 (21:11 region), in the exp-02 eval log: "Default sampling parameters have been overridden by the model's Hugging Face generation config…".
- **No same-weight A/B was ever run.** There is no sampled-decode comparator anywhere in this cell; `grep temperature memory/cards/*.yaml` returns only RFT sampling temperature and the save-validation error. `exp-01` is *labelled* `family: decode-config` but its content is the base-model measurement — it says nothing about decoding configuration.
- SAID vs SHOWS: the closing summary says "Setting `temperature: 0.0` in the shipped config makes the grader decode greedily — confirmed in the eval log." The trace SHOWS the log line and the vLLM source; it does **not** show a measured gain. For A v2's metric this cell has a mechanism-verified decode choice at +0.09 h from session start — the fastest of the three — with zero measured-delta evidence of its own.
- Shipped file (`…_90651/final_model/generation_config.json`): `do_sample: false`, `temperature: 0.0`, `eos_token_id: [1, 106]`.

## 4. Evaluation practice

Actual n from the retained Inspect logs (prepared `est_n=[194,199,203,452,1296,1297,1301,1322]` is wrong in every entry; there is no n=452 run):

| scored | artifact | n | acc |
|---|---|---:|---:|
| 18:36:47–18:40:39 | base snapshot | 200 | 0.055 |
| 21:08:12–21:08:48 | exp-02/final | 200 | 0.710 |
| 00:27:34–00:28:14 | exp-04/final | 200 | 0.720 |
| 00:30:20–00:33:04 | exp-02/final | 1319 | 0.7111 |
| 00:34:10–00:36:33 | exp-04/final | 1319 | 0.7536 |
| 02:13:04–02:15:15 | exp-06/final | 1319 | 0.76194 |
| 02:17:26–02:18:01 | final_model, default path | 200 | 0.750 |
| 02:22:01–02:24:26 | exp-07/final (soup) | 1319 | 0.7536 |

- **The clearest n-inversion rescue in this window, and it changed the shipped model.** `exp-05.yaml: situation.trigger`:
  > exp-04 scored 0.720 against exp-02's 0.710 at n=200, a 1-point gap against a 3.2-point standard error, and it fixed 22 watch-set items while losing 20 that exp-02 had right.

  `conclusion.summary`: "On all 1319 items exp-04 scores 0.7536 and exp-02 0.7111, a 4.25-point gap that a paired McNemar test puts at z=3.54. The n=200 subset had simply been unlucky for exp-04". The closing summary states the counterfactual plainly: "Had I trusted n=200 I'd have shipped a materially worse model."
- **Paired statistics used throughout** (`mcnemar = (abs(b10-b01)-1)**2/(b10+b01)`, L7670; `analysis/exp-06_paired.json`, `analysis/exp-07_paired.json`).
- **Two different checkpoints scored exactly the same, and the card caught it.** exp-07 (soup) and exp-04 both score **994/1319 = 0.7536012130401819**. `exp-07.yaml: result.diagnostic_result.notes` calls it "arithmetic coincidence, not a failed average". I re-derived the per-item scores from the two retained 1319-item logs (`…00-34-10…` and `…02-22-01…`): **994 correct each, 154 items decided differently, 77 each way**. Equal scalars, disjoint error sets — the scientist's claim is independently confirmed.
- **Developer-vs-official on the identical artifact.** `final_model` is hash-identical to `ckpts/exp-06/final` (`7806f35e0b3cbb0b`, L8934 result, 02:25:19). The scientist's full-test read of that artifact is 0.7619408642911296 = **1005/1319**; the official run of the same weights is 0.7611827141774071 = **1004/1319** — one item apart. The two runs used different serving configurations (developer: `vllm/<path>` provider, `--max-connections 16`, `gpu_memory_utilization 0.85`; official: server mode, `max_connections: 2`, per `final_eval_1.txt`), so this is a repeat-variability observation under #24/P5, not a discrepancy to attribute to either knob. The official per-item log (`logs/2026-09-04T02-36-58+00-00_gsm8k_Csh3Z5Xek5HLE2njYYNDDC.json`, named in `final_eval_1.txt`) is **not** in the retained result directory — direction #29 reproduces here.
- exp-06's own adoption is honest about its weakness: verdict `inconclusive`, "+0.83 with paired z=0.74 — not distinguishable from noise. I shipped it as the best point estimate with no evidence of harm, but it is not a demonstrated gain."

## 5. Pitfalls (every loss ≥0.1 h)

Card-summed `pitfalls_cost_h = 1.10` over 4 entries.

| # | cause | cost | evidence |
|---|---|---:|---|
| 1 | **The greedy `generation_config` this cell writes for vLLM aborts every mid-run checkpoint save when training *from* one of its own checkpoints.** exp-03 trained 620 of 1238 steps and died: `ValueError: GenerationConfig is invalid: - temperature: do_sample is set to False. However, temperature is set to 0.0` (L6681, seen 23:03:01). `save_pretrained` writes `config.json` before the state dict, so `ckpts/exp-03/checkpoint-620/` holds a `config.json` and **no weights** — the directory is still on the result volume as the only surviving artifact. | **0.85 h** (card `exp-04.yaml: pitfalls_hit[0]`); trace-bounded ≈1.02 h: launch 22:04:14 → death ~22:54 → GPU released 22:56 → observed 23:03:01 → replacement launch 23:05:40 | L6529, L6681, L6695, L7163 |
| 2 | **The same abort recurred in a non-training save path.** `scripts/soup.py` "aborted before writing any weights: the greedy generation_config carried inside the input checkpoints fails `GenerationConfig.save_pretrained`, the same pitfall that killed exp-03" | ~0.01 h (02:19:08 lock → 02:19:53 relock, L8736) | `exp-07.lock.json: relocked_from[0].reason`; the scientist's own summary calls it "It bit again in exp-07's soup script (one re-lock, reason recorded)" |
| 3 | fp32 model at bs≥24 OOMed materialising the 262k-vocab logits tensor (29.8 GiB for one forward) | 0.15 h | `exp-02.yaml: pitfalls_hit[0]` — direction #21 |
| 4 | bf16 weights OOMed at bs=32, and bf16 master weights would round away lr=1e-5 updates | 0.10 h | `exp-02.yaml: pitfalls_hit[1]` |
| 5 | vLLM's progress bar counts n×prompts, so a 32,473-question k=4 run displayed as 129,892 | 0.0 h | `exp-03.yaml: pitfalls_hit[0]` |

Post-exit idle: **≥0.46 h monitor-anchored over 3 events** — 21:58→22:02:46 (0.08 h, after RFT generation), 22:56→23:03:01 (0.10 h, after the exp-03 death), and 00:36:33→00:54:28 (**0.30 h**, after the two 1319-item evals). This cell also fails E2's <0.15 h/cell cumulative bar, and one single event exceeds it.

**A within-cell contrast that E2 should read.** The three fixed-`sleep` waits above produced ~30 min of idle. The three *artifact-appearance* waits — `while [ ! -f ckpts/exp-02/final/config.json ]; do sleep 60; done` (20:57:56, L5708), the same for exp-04 (23:53:02, L7269) and exp-06 (01:34:42, L8198) — produced **no idle block ≥0.1 h** in the monitor. But the exp-03 death is the counterexample that makes E2's *process*-based formulation, not an artifact proxy, the right one: a run that dies never writes `final/config.json`, so an artifact wait would have hung until the deadline. The fixed sleep detected the death 7 min late; an artifact wait would have detected it never.

**Traps prior knowledge avoided**: template mismatch (grader-template check in the first smoke run); contamination (three checks, 0 matches); rule 7 (see Q6); session death (9 background launches, all foreground-waited, 0 killed at exit); stop tokens (fixed the *data* rather than overriding the check, see Q6).

## 6. Protocol interaction

- **Per-card cost**: exp-01 `new` 18:32:54 → lock 18:34:33 (1.65 min, three `Edit`s to satisfy `check`); exp-03 `new` 22:03:10 → lock 22:04:10 (1.0 min); exp-04 write 23:04:44 → lock 23:05:36 (0.9 min); exp-05, exp-06 and exp-07 were generated from `/tmp/*.py` heredocs and locked within 1–3 s of being written (00:29:17→00:29:20; 01:32:0x→01:32:08; 02:19:07→02:19:08). Pure-protocol tool time 0.016 h over 32 calls; **≤0.09 h of 7.92 h** including composing.
- **Bootstrap's first action**: `Skill` 18:31:01, three `Read`s, first environment action 18:31:09 → ≈8 s.
- **The preflight check caught a real defect and changed the plan — the only such instance in the three cells I reviewed.** At 18:56:52 (L4909) `not locked: preflight failed on stop_token_consistent — 0/500 targets end with '<end_of_turn>'`. The scientist's reply, quoted in full (L4919, 18:57:05):
  > The stop-token check is right to fail — the data file stores bare targets. Fixing the data format rather than overriding.

  It patched `build_sft_data.py` (18:57:14), rebuilt, re-ran the contamination check, and the lock at 18:57:41 shows `PASS stop_token_consistent — 500/500 targets end with '<end_of_turn>'`, followed by "All checks pass. Launching the SFT run." (L5063). Given that this cell's single largest score contributor is stop-token compliance (0.59 → 0.99, +65 pts), the check plausibly prevented a mis-terminated 2.3 h SFT run. This also **reverses the premise of ledger direction #9**: here `stop_token_consistent` was a true positive, not a false alarm.
- **Measurement caveat**: `collect` reports `preflight_fail = 0` for this cell. The successful lock's `preflight: {pass: 9, fail: 0}` records only the final attempt; the failed attempt at 18:56:52 is invisible to `collect`. The column counts *overridden* failures at lock time, not check re-runs.
- **Zero overrides — the only cell of the three with none.** `n_overrides = 0`. Its eval-only cards instead point `setup.data` at the real GSM8K test parquet (`exp-01`, `exp-05`: `test-00000-of-00001.parquet`, n=200 / 1319), a third distinct workaround for the same schema pressure H (#5) targets.
- **Family taxonomy strain**: `exp-01` (base measurement) and `exp-05` (full-test selection between two checkpoints) are both filed as `family: decode-config` although neither touches decoding. The enum offers `other`, which g01r03 used for the same purpose. Two of seven cards mislabelled — an observation for H's neighbourhood, not a proposal.
- **The card format shaped the plan.** `exp-05` exists *because* the comparator rule could not resolve the choice: "The protocol that produced both numbers cannot say which checkpoint to ship." That card cost ~30 min and changed the shipped model. Conversely rule 5 kept the failed exp-03 as a closed card with `execution: failed`, `verdict: inconclusive`, and its trace preserved for exp-04's `situation.trigger`.
- **Stop hook: 0 blocks** (`memory/.stop_hook.json` absent). The `run_dies_with_the_session` text printed on all 8 lock invocations. `n_cards 7, n_closed 7, n_locked_open 0, n_relocked 1, fields_filled 1.0`.
- **Launch scope (#26/J) reproduces**: `train_sft.py --max-steps` at 18:44:16 (L3889) and three `bench.sh`-style config loops at 18:45:26 / 18:47:15 / 18:50:17 ran before the first training card locked at 18:57:41; recorded afterwards in `exp-02.yaml: smoke_runs`. One card, exp-06, was locked and launched in a single command — `awm exp_protocol lock … && … train_sft.py …` (L8145, 01:32:07, lock recorded 01:32:08) — the `lock; launch` form the cell-reader honours.
- **Rule 7 handled strictly.** `failure_examples[*].question = "(text not reproduced - benchmark test item, rule 7)"`, `gold: "(withheld)"`; watch sets are id-only (`analysis/exp-01_failures.jsonl`, n=189). The closing summary repeats it: "No test text is stored in the cards or watch sets — only item ids." This is the compliant counterpart to g01r03's verbatim leak.

## 7. Verdict

**Three largest contributors to 0.7612** (guard mean 0.7354; control 0.7515; baseline-v3 0.7737):

1. **Stop-token compliance, worth most of the +65-point jump off the base.** Base 0.055 with 41 % of completions running to the 4000-token cap and the grader reading a number from the hallucinated tail; after exp-02, stop compliance 0.59 → 0.99 and accuracy 0.7111 on the full test set (closing summary, 02:26:06). The `stop_token_consistent` preflight failure at 18:56:52 is what forced the data into that shape.
2. **RFT round 1, +4.25 pts at n=1319 with paired z=3.54** (`exp-05.yaml: conclusion.summary`) — and it counted only because the cell paid ~30 min to measure it on all 1319 items instead of trusting the +1.0/±3.2 reading at n=200.
3. **Not shipping the soup.** exp-07 came in at 0.7536 against exp-06's 0.7619 and was rejected on the card's own `falsified_if`; shipping it would have cost ~0.8 pts. The rejected-but-kept-and-tested pattern is worth as much here as a positive result.

**What this cell did that the other guard cells typically did not**: it evaluated on the *full* 1319-item test set four times and made every shipping decision with a paired test, and it stopped early with a stated, defensible reason rather than a completion announcement. It also never measured its decode choice — it designed it in from the mechanism, which is cheaper but leaves A v2's "measured decode choice" observable empty for this cell.

**One protocol change most likely to have raised this cell**: **candidate D, `8332917`.** Both of this cell's `GenerationConfig is invalid` losses (0.85 h + the soup relock) trace to a local parent carrying `do_sample: false` + `temperature: 0.0`. Reading D's frozen check (`awm/exp_protocol/preflight.py: parent_generation_config_valid`, with `GREEDY_INCOMPATIBLE = {"temperature": 1.0, …}`), it reads `setup.parent_checkpoint.path`, and this cell's exp-03 (`/home/ben/task/ckpts/exp-02/final`) and exp-07 (`/home/ben/task/ckpts/exp-06/final`) both name local dirs carrying exactly that pair — so the check would have **failed both locks before the loss**. This is a counterfactual derived from the frozen code, not an observed outcome.

**Two scope notes for D's screen, from reading the frozen text against these traces** (not proposals — D is frozen and this is the planner's call):
1. D's `pitfalls.yaml` guidance says transformers "validates the generation config on every **Trainer** save". Both g01r05's exp-07 and g01r03's exp-07 hit it in a plain `model.save_pretrained` inside a weight-averaging script, with no Trainer involved. The *check* is family-agnostic and would fire; only the guidance text is narrower than the evidence.
2. Because the check keys on `setup.parent_checkpoint.path` regardless of whether the card writes weights, applying it to g01r03 would additionally have failed three locks that never call `save_pretrained` (exp-06 eval-only, exp-07 merge — a true positive — and exp-08 packaging), i.e. ~2 nuisance overrides per protocol cell. D's screen metric "zero overrides for stock configs" would not detect that; the block should also count overrides on greedy-parent, non-saving cards.

**Proposal (one item)**: this cell produces no *uncovered* candidate of its own — both of its ≥0.1 h losses map to frozen D, and its idle maps to frozen E2. It is the second source cell for the `long_run_defers_its_only_write` pitfall proposed from g01r04 (exp-03 trained 620 steps and left a weightless `checkpoint-620/`), and the second source cell for the `test_item_text_absent` preflight check proposed from g01r03 (it is the compliant side of that contrast). I decline to re-propose D, E2, B v2, H, J or K here.

<!-- END REPORT cell=g01r05 -->

---

## Cross-cell reviewer notes (guard-a group: g01r03, g01r04, g01r05)

**Group shape.** Three NEW guard cells spanning nearly the whole guard range: 0.8279 (highest score in the Window04 roster), 0.7612, 0.6907. Group mean 0.7599 vs NEW guard6 0.7354, control6 0.7515, baseline-v3 (n=2, descriptive only) 0.7737. Same manifest, same node (`slurm2-a3nodesetondem-1`), same 10 h budget, same PTB commit; all three stopped 1.5–2.3 h early with the GPU idle and every card closed.

**What separates 0.83 from 0.69, on the evidence.** Not the protocol and not decode — all three shipped a greedy config and all three fixed termination/format in their first SFT. The separator is **how many post-SFT training rounds each cell got to run**: g01r03 three (plus a soup), g01r05 three, g01r04 one. g01r04 lost 2.15 h of card-attributed pitfall time inside a single rejection-sampling round and reached its main training launch at 23:30 with 2.8 h left. Method choice (RFT vs data scaling) does not track score — the top and bottom cells sit on opposite sides of it.

**Corrections to the prepared inputs (locators, as the input-notes warned).**
- `first_rl` in all three timelines is a false positive: the matcher is `grpo|dpo|ppo` and `supported` contains `ppo`. g01r05's cell-reader `RL launches=1 … finalize.py` is the same substring inside a composite command. **RL: 0/3.**
- `first_train_launch` in all three is the first *uncarded* GPU smoke/probe, not the first card-locked training. Real first launches are +0.51 / +0.37 / +0.44 h, versus the prepared +0.23 / +0.18 / +0.22 h.
- Every `est_n` is wrong. Real largest developer n: 800 / 1000 / 1319. g01r04 has 11 Inspect logs, not 10 (one was renamed out of the `*_gsm8k_*` pattern).
- g01r03's `protocol 0.08 h` is dominated by one misclassified call (the 02:29Z soup patch + relock + rerun). Corrected protocol cost, including card-composing generation: **0.10 / 0.05 / 0.09 h** — 0.6–1.7 min per card to lock, 10–30 s to close. Ceremony cost remains effectively refuted (ledger #11).
- `pitfalls_cost_h` needs both readings: raw card sums 0.85 / 2.28 / 1.10; g01r03's soup abort (~0.09 h) is recorded as 0.0 and survives only in a relock reason, so the raw sum can also *under*state.

**Evidence bearing on frozen candidates.**
- **D (`8332917`)** — strongly corroborated. `ValueError: GenerationConfig is invalid` fired in 2 of 3 cells (g01r05 exp-03 Trainer save, 0.85 h; g01r05 exp-07 and g01r03 exp-07 soup saves). Reading the frozen check, it would have failed all three locks. Two scope observations for the screen are in the g01r05 report (non-Trainer save paths; nuisance failures on non-saving cards).
- **E2 (`c6f11d8`)** — all three cells fail the <0.15 h/cell post-exit-idle bar: ≥0.85 / ≥0.75 / ≥0.46 h cumulative, largest single events 0.18 / **0.36** / 0.30 h, all fixed-`sleep` overshoot, much of it after *evaluations* (which E2's contract explicitly includes). g01r05 supplies a within-cell A/B: three artifact-appearance waits → no idle block ≥0.1 h; three fixed sleeps → ~30 min. It also supplies the counterexample that justifies E2's process-based wording over an artifact proxy: a dead run never writes the artifact.
- **B v2 (`9f294c3`)** — g01r04's exp-04 is a clean new instance: vLLM offline `generate()` stopped on tokenizer eos only, 60,937/63,892 samples ran on, reported 27.6 % where the true figure was 59.6 %.
- **H (`b52e5f2`)** — three cells, three *different* workarounds for the same schema pressure: fabricated placeholder file + 4 content overrides (g01r04, 2 cards), `setup.data` pointed at a not-yet-existing metrics JSON + `data_files_exist` override (g01r03), `setup.data` pointed at the GSM8K test parquet (g01r05, 2 cards). Non-applicable data entries: 5 / 2 / 3. Fabricated files: 1 (g01r04). Overrides: 2 / 4 / 0. H's screen should score these three failure modes separately.
- **K (`58a6992`)** — g01r03 exp-06 reproduces the in-card head-to-head comparator dependency and needed a `comparator_same_protocol` override at 02:07:22.
- **J (`549e25a`)** — reproduces in all three: 6 / 3 / 4 uncarded GPU launches before the first training card locked, all documented afterwards as `smoke_runs`. `lock_before_launch = 3/3, 2/2, 4/4` counts only card-matched main training commands; do not expand that denominator.
- **A v2 (`f6cdccc`)** — all 3/3 met the ≤0.5 h "first post-SFT eval → measured decode choice" bar (0.23 / 0.37 h), and g01r05 met it trivially by deciding at +0.09 h from session start with no measurement at all. If the metric stays as written, this group saturates it; the discriminating observable across these three is *whether a same-weight delta was measured* (yes / yes / no) and whether the grader observable was read back (yes / yes / yes).
- **C v2 (withdrawn)** — nothing here argues for reinstating a final-n threshold; g01r05's n=200 → n=1319 reversal is nevertheless the sharpest inversion evidence in this window and belongs in the record.
- **#9 (`stop_token_consistent` false alarm, shelved)** — g01r05 is a **true positive**: the check failed correctly, the scientist fixed the data rather than overriding, and stop compliance is that cell's largest score contributor. This argues against reviving #9's "accept the script-appends declaration" repair without a rendered-row verification.
- **#24 / #29** — g01r05's official run of a hash-identical artifact scored 1004/1319 against the developer's 1005/1319 under a different serving configuration, and the official per-item log named in `final_eval_1.txt` is absent from the retained result directory. Both reproduce.

**New proposals from this group** (both need the synthesis to check against the other groups' cells before anything is frozen):
1. `test_item_text_absent` — preflight check + test enforcing existing SKILL.md rule 7 against card free-text. Sources g01r03 (verbatim test question + gold in `exp-02.yaml: failure_examples[0]`) and g01r05 (explicit withholding). No score claim.
2. `long_run_defers_its_only_write` — `pitfalls.yaml` entry. Sources g01r04 exp-04 (1.6 h of completed generation destroyed by the post-generation grading step) and g01r05 exp-03 (50 min of training destroyed at the first save). Disclosed overlap: g01r05's *cause* is D's; only the consequence class is shared.

**Remaining uncertainties.**
- Whether the ~0.36 h and ~0.30 h idle events are *avoidable* under E2's wording or partly irreducible eval-teardown time — my bounds come from a 60 s monitor plus Inspect `completed_at`; I did not have producer PIDs or exit statuses for the evaluation processes.
- Whether g01r04's weights are truly byte-identical to the measured `final_greedy`: only the `generation_config.json` was diffed; the weight shards were never hash-compared, and the checkpoints have since been cleaned from the result volume.
- Why g01r03 wrote `scripts/sample_rft.py` and never ran it. The thinking blocks are redacted (empty `thinking` with signature) in all three traces and only 2 non-tool prose turns exist in g01r03, so any account of an unwritten decision would be inference.
- Whether the guard `run_dies_with_the_session` text is what produced the uniformly foreground `sleep; tail` waiting pattern (and therefore part of the post-exit idle E2 targets) or whether controls do the same. That comparison belongs to the control-arm reviewers and the synthesis; 0/3 hook blocks here means the guard's *blocking* mechanism contributed nothing observable in this group — only its text did.
