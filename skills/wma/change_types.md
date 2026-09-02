# The manual: verifier tiers, change types, priors, traps, noise

Distilled from `doc/reference/verifier_tiers_and_change_types.md` (166
annotated PostTrainBench runs, 5,574 changes, 3,050 verifications; 2026-09-02).
Every effect carries its evidence grade — **controlled** (single variable,
everything else aligned), **semi-controlled** (one known misalignment),
**observed** (cross-run aggregate, no control), **confounded** (several
things changed at once). Do not quote a number without its grade.

## 1. The shape of the problem

| | training (C3/C4) | verification (any tier) |
|---|---|---|
| cost | hours; 61–72 % of a run's budget | minutes; 20 % of the budget in total |
| what the corpus shows | 42 % of training launches die within 5 min (crash / OOM restart chains); agents rarely change one variable at a time | a full gsm8k eval is 3.0 min median; 27 % of runs never ran one, 34 % never read a full score |

So: what a cheap tier can decide, decide before the GPU is spent; what only a
training can reveal gets a wide interval and a cost decision.

## 2. The four verifier tiers

| tier | what it does | cost | noise | decides | cannot decide |
|---|---|---|---|---|---|
| **1 static** | read the eval pipeline and chat template; unit-test the scorer; token-level comparison of a rendered training sample against the real eval prompt; diff `generation_config.json` | seconds–minutes, **no GPU** | none (deterministic) | scorer behaviour, template symmetry, prompt alignment, which sampling fields the server reads, whether a stop token is in the targets | anything about model ability |
| **2 smoke** | a reduced full pipeline (`--max-samples 600`, output `smoke/`), or a watch folded into the real launch (`until grep -qiE "error\|traceback\|OutOfMemory" train.log` in the first 60 s) | **0.7 min median**, none over 30 min | n/a | deps, argument names, VRAM, disk, that a checkpoint lands | training effect — a passed smoke ≠ a working recipe |
| **3 partial eval** | `evaluate.py --limit N`, N < full | n=20–50 ≈ 2 min; 150–300 ≈ 2–5 min; 600–800 ≈ 25–45 min | n=20–50 **±10–15 pp**; 150–300 ±4–8; 600–800 ±2–4 | effects ≥ 10 pp (e.g. a decode fix) | small effects — below n=150 a gap under 5 pp means nothing. `--limit N` takes the **first N** items, not a random subset: n=50 vs n=150 differed by +26–29 pp four times on the same weights |
| **4 full eval** | `evaluate.py --limit -1` | **2.5 min median** (gsm8k 3.0, healthbench 19) | a function of n × temperature — §5 | the final ranking of candidates | differences below the noise floor |

Runs with no early-failure detection at all (28 % of the corpus) crash more;
the commonest crash is TRL renaming `max_seq_length` to `max_length` (6 of 9
runs of one agent hit it).

## 3. Change types: cheapest tier, what to check, prior, cost

"Check offline" = what can be read from the card's own commands, hyper-
parameters and data description when no workspace is mounted; "probe online"
= what the tier allows when the session is live.

| type | what it is | cheapest tier | check offline / probe online | effect prior (grade) | cost & L3 note |
|---|---|---|---|---|---|
| **C1a** decode-parameter tuning | temperature / top-p / top-k / penalties in `generation_config.json` | 1 + 3 | which fields changed; whether `do_sample` was flipped in the same edit (it is ignored by vLLM — the field that acts is `temperature`) / tier-3 compare at n ≥ 150 | **no sign, ±10 pp**: t 0.6→0.2 gave −30 (gemma/gpqa), t→1.0+top_k −12, greedy→0.6 gave +10 (SmolLM3/aime); tuning branch +7.5–20 (semi-controlled) | minutes; worth a tier-3 search, never a copyable fix |
| **C1b** stop-token / length-cap fix | add `<\|im_end\|>` to `eos_token_id`; set or remove `max_new_tokens` | **probe** | is the eos in the config *and* in the training targets; was `max_new_tokens` deleted in a full rewrite / **sample_probe: does the model emit that token under the eval prompt distribution?** | **+41.3** when the model emits the token (SmolLM3/gsm8k, controlled) vs **+0.67** when it does not (Qwen3-1.7B, 10-shot, controlled). The precondition decides everything | minutes; editing the config always "succeeds" — the probe is the verdict |
| **C2** format alignment | train sample ≡ eval prompt byte-for-byte; answer marker (`ANSWER: X` not `####`); stop token trained; few-shot context mixed in | **1** | answer marker single and consistent; targets end with the stop token; does the recipe bypass `apply_chat_template` (qwen3 inserts an empty `<think>` block at train time only) / token-level render comparison | prompt-attainable share of the headroom (controlled): humaneval 76 %, **gsm8k 63 %**, gpqa 53 %, healthbench 21 %, aime 11 %, bfcl 0 %. One format-only change (30 % few-shot mix-in) gave 3.3 → 33.3 % (+30, semi-controlled) | zero GPU; almost always worth doing before any C3/C4 |
| **C3** data source / recipe | where the data comes from and in what mix | 2 + 4, **nothing in between** | data size and provenance; whether hyperparameters moved at the same time (50 % of trainings changed both) / — | **±0** in the one near-clean ablation (44 % of the base data swapped + 10k verified RS rows → score identical to the digit, semi-controlled, n=1); corpus deltas ±1–2 pp, all confounded | hours — the core L3 question; ask what C2 and C5 could give first |
| **C4** training method / hyperparameters | full vs LoRA; SFT/DPO/GRPO; lr, epochs, batch, sequence length | 2 + 4 | **LoRA without `modules_to_save` for `lm_head`/`embed_tokens` cannot learn to emit the stop token**; `batch × accumulation` unchanged means the effective batch did not move / — | LoRA trap: **−20 to −40 pp, binary** (6 agents reproduced; +28 pp when fixed, semi-controlled). GRPO +1 to +13 (confounded). Method choice otherwise below noise | hours; rule out the trap at tier 1 first |
| **C5** checkpoint selection / early stop | pick among saved steps; kill mid-way and submit a checkpoint | 3 | how many checkpoints the plan saves (`save_steps`); a plan with one final checkpoint forfeits this lever / 2–5 min per checkpoint | **±15 pp, controlled** — the only strict single-variable contrast in the corpus (720 / 840 / 934 steps: 80.0 / 68.7 / 83.3). "Trained to the end is often worse" (87.11 at epoch 1 vs 86.7 full) | no training; the highest effect per minute — a cheaper alternative to a new C3/C4 |
| **C6** weight averaging (uniform soup) | average checkpoints of one trajectory | 3 / 4 | which checkpoints, same tokenizer / one eval | ≈ 0 on average, but two final submissions were uniform soups beating every ingredient (0.8522 vs 0.8476/0.8423); a variance-reduction move, the default when time is short | minutes + one eval |
| **C7** home-made proxy verifier | a cheap scorer standing in for the official one | 1 + 3 | is the proxy a verbatim copy of the official scoring path, or its own thing / correlate on a few items | can drift the wrong way: one proxy fell 0.934→0.919 while the official rose 0.95→0.99 | it does not score points; it decides the quality of every other decision |
| **C8** runtime / feasibility fix | OOM → smaller batch, deps, flag conflicts, disk, missing config, throughput | 2 | pitfalls catalogue; whether it silently changes C4 values (batch, lr, seq len) / smoke | binary: without it nothing runs | minutes; only 24 % ever judged by a verifier |
| **C9** submission guard | decide which existing candidate goes to `final_model` now | — | is there a regression guard (an unconditional overwrite put 0.567 over 0.627); is the best-scored candidate the one being packaged / — | its *absence* is the costliest deduction: a run holding an 80 % model submitted 61 % | seconds; always worth it |
| **C10** compliance / decontamination | filter suspect samples, audit data | 1 | hit counts / — | expected effect 0 or negative; decides validity, not height | minutes |
| **C11** verifier instrumentation | turn the official scorer's *output* into signal: truncation rate, stop reason, item sets, repeat means | 1 | is truncation rate or stop reason read at all / — | a deterministic criterion from the same eval: truncation 47 % → 0 % → 3 % while six evals of the same weights scattered 0.067–0.167 — **it makes a C3 change judgeable that scores cannot** | zero GPU; only 9 % ever judged |
| **C12** verifier invocation | `--max-connections`, `--max-tokens`, `--limit`, `--gpu-memory-utilization` — touches nothing but how the official scorer runs | — | **§4 mechanisms**: `--max-tokens` vs `max_model_len`; concurrency; `--limit` first-N bias; default `--limit` is not the official protocol (a run kept the worse candidate because it trusted n=50; the official score was on 448) / the self-checks in §4 | **0 → 91 points** on the same weights; five mechanisms, do not pool them | seconds; two arms with different invocation parameters are not a comparison (a final decision rested on 0.823 vs 0.807 with concurrency 4 vs 2) |
| **C13** cross-run knowledge | memory / NOTES.md for later episodes | — | — | structurally 0 for this run | seconds |
| **C14** delivery self-check | sha256 of shards, config fields, offline load of `final_model` | 1 | is it there at all / run it | expected 0; prevents §4 deductions | minutes |
| **C15** white-box probe | read next-token probabilities, embedding norms, tensors | — | — / seconds, no noise | the only action in several runs that located a mechanism (`<\|im_end\|>` probability 0.007 → 0.922 explained 0/30 → 3/30) | seconds; propose it as a precondition for C1b |
| **C16** weight surgery / non-uniform merge | weighted interpolation (α searched, even −0.25), scaling rows of `lm_head`, tensor picking | 3 / 4 | — / evals | +15 items in one run; six negatives (−0.5 to −2.2) in another; opposite sign to C6 | minutes + evals; needs its own tier-3/4 reads |
| **C17** candidate destruction | delete the only trained candidate; abort a healthy run | — | is the deleted or aborted thing the only scored candidate / — | one run with 8 h left deleted its only scored candidate and submitted nothing | L3 `no` unless a scored candidate survives |
| **C18** repeat evaluation | re-run the official scorer on byte-identical weights, effective parameters identical | 4 | does the plan compare a gap smaller than the noise floor without repeats / — | it buys the standard error, not a verdict; a final decision rested on 1049 vs 1048 correct (one question) | one full eval each; the *only* correct move when the gap is below the floor |

Reading history: retrieve precedents by the same base model, the same change
type and a comparable evaluation size; a delta measured at n=20 and one at
n=1319 are not the same evidence. `delta_official` mixes C1 with C3/C4 — when
a precedent's comparator ran under a different decode config, its delta is
not the training's.

## 4. Silent failures: a valid-looking number that is not one

Each of these produces a score (often 0.0) that enters every table looking
normal. Check for them at L1 whenever the evaluation command or the config
changed, and name the mechanism in the note.

| mechanism | what happens | self-check (tier 1, from the eval output) |
|---|---|---|
| requests rejected en bloc | model `max_model_len` 8192 vs the default `--max-tokens 16000`: vLLM rejects every request, the refusal text is scored as the answer → **0.0 → 0.91 with `--max-tokens 1024`** | are there outputs at all; does the "answer" read like an error message |
| garbage at high concurrency | `--max-connections 8` vs 2 on the same greedy weights: 0.2733 → 0.8533; 99/150 outputs began with `!!!!` | count outputs starting with a garbage prefix and the request index where it starts |
| truncated before the answer | generation budget cut before `ANSWER:` → scored 0 (0/10 at 8k, 3 correct at 16k on the same weights) | mean output tokens sitting at the cap |
| engine death | concurrency 8/16 → `EngineDeadError`; no reading at all | the log says so; the candidate looks "bad" only because it has no score |
| sampling drift | concurrency reorganises batches; moves the score only under sampling decode | greedy is immune to this one alone |
| config fields vanish | `save_pretrained` writes only values that differ from the library default: a base with `do_sample: false` loses the field; gemma (`do_sample: true`, top_k 64, top_p 0.95) keeps them and must have them removed to decode greedily; Trainer collapses `eos_token_id: [1, 106]` to `1` before saving; the agent's own save path may strip fields too | diff the saved `generation_config.json` against what the card claims was set |
| config field written as `null` | `temperature`/`top_k`/`top_p` = JSON null → vLLM falls back to sampling; fixing it: 0.460 → 0.613 on identical weights | "is the field there" is not enough — is it null |
| save-time validation | `top_k = -1` (a vLLM sentinel) written into a checkpoint makes the **next** `save_pretrained` raise: a finished GRPO, a full-parameter run and a bf16 conversion were each lost this way (≥ 11 hits, 35 min – 1.25 h each) | grep the config for sentinel values before a training that resumes from it |
| right diagnosis, wrong field | editing `do_sample` instead of `temperature` (19 points) | the field the server reads is `temperature` |

**A diagnosed format floor is not a capability estimate.** Apply this rule
only to the first intervention after a pretrained/non-instruct checkpoint has
been shown, under the card's actual chat template and end-anchored grader, to
lose most answers through termination or answer-format failure. Anchor the L2
lower end in that measured floor. Anchor the upper end in known capability of
the *same checkpoint* under an aligned format when one exists; otherwise use
the C2 headroom share from §3 against the card's observed headroom. Cite both
anchors. Once termination is restored, later training cards use ordinary
C3/C4 priors: never carry the format-restoration jump forward as capability.

Other deductions with a price tag: session death (a run holding an 80 % model
submitted 61 %); no regression guard on `final_model` (0.567 over 0.627); shell
self-harm (40 min); LoRA trap (§3 C4).

## 5. The noise floor of an accuracy read

Repeat evaluations of byte-identical weights under identical effective
parameters (the corpus's own C18 events):

| benchmark | n | decoding | spread |
|---|---|---|---|
| gsm8k | 1319 | greedy (byte-identical copies) | 0.15–0.45 pp; near-greedy t=1e-6 1.6 pp |
| gsm8k | 150 | greedy / sampling | 1.3 pp / 3.3 pp |
| humaneval | 150 | greedy / t=0.05 / 0.1 / 0.2 | 1.3 / 0.7 / 2.0 / **10.7 pp** |
| gpqamain | 448 / 200 / 50 | greedy / greedy / t=0.6 | 1.5 / 4.0 / **14.0 pp** |
| aime2025 | 30 | greedy / t=0.6 | every greedy pair flipped one question (3.33 pp) / 10.0 pp |
| bfcl | 100 | greedy | 0.0 pp, bit-exact |

Rule of thumb by question count, used by the ledger as the floor an interval
is compared against: n ≥ 1000 → 0.01; 500–999 → 0.02; 150–499 → 0.03; 50–149 →
0.07; below 50 → 0.12; and never below 1/n. Sampling decoding widens all of
these. A card whose evaluation is n=20 cannot resolve anything under 5 pp; a
plan that will decide on a smaller gap needs a C18 repeat or a larger n as a
precondition — say so in the suggestions.

An L2 interval wider than **3× this noise floor** must cite, in its `basis`,
either a same-type precedent at comparable `n` or a probe that changed L2.
Without that evidence, contract the interval to the evidence-graded effect
prior for the classified type around the card's checked anchor; uncertainty is
not a license to span unrelated mechanisms.
