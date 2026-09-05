# Round 01 strict guard cohort — full eight-cell trace-review synthesis

**Status: evidence document, not a decision.** The planner owns the guard safety gate, candidate selection, all queue actions and any promotion. Nothing below authorises a submission, release or cancellation. This synthesis is read-only; no file, manifest, protocol tree, receipt or queue item was modified.

## 0. Scope, cohort and provenance

This is the **exact prescribed strict cohort g01s01–g01s08**, not a new eight-clean window. `g01s04` was already consumed inside frozen Window 03; the other seven are incremental. All eight are receipt-backed, PTB-validator-complete, eligible, non-quarantined and judge-clean. Old `g01r01`/`g01r02` and every control are excluded from every denominator here.

| item | value |
|---|---|
| batch / manifest | `experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2.yaml` |
| receipt | `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/formal-2026-09-02T204221.237369+0000.json` (held 2026-09-02T20:42:22Z → released 2026-09-03T09:46:56Z; `release_safety_override` authorized_by=user 09:39 UTC) |
| spec | `doc/spec/2026-09-02-exp-protocol-round01-session-guard.md` (§ "Spillover 後的 strict-site replacement buffer" + 09-03 09:39 UTC release decision) |
| variant | `awm.sha 4ae3d87c…`, `protocol_tree 189319d63d30…`, setup `--exp-protocol --tool claude --stop-hook`, `run_index 2` |
| bundles | `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/g01s0{1..8}/` |
| raw results | `data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2_g01s0N_formal_r2/gsm8k_google_gemma-3-4b-pt_907NN` (8/8 present) |
| jobs / placement | 90791–90798, all `slurm2-a3nodesetondem-1`, partition `ptb-a3`, PTB `dcf5da0`, top `67440e9`, container `opus_5.sif` — **verified from `runtime_provenance.json`, 8/8 inside the frozen nodelist, no placement quarantine** |
| reports | `doc/exp_protocol_iterations/trace-reviews/round01-strict-guard-addendum/cells/g01s0{1,2,3,5,6,7,8}.md` + `doc/exp_protocol_iterations/trace-reviews/window03-local/cells/g01s04.md`; group notes `four-cell-notes.md`, `final-three-notes.md`; `guard-collect.csv`, `launch.json` (reviewer sessions `73c154b4…`, `35a6a6d1…`) |
| prior context | `window03-local/planner-decision.md`, `window03-local/p1-predicate-audit.md`, `doc/exp_protocol_iterations/directions-ledger.md`, Round 00 record (historical only) |

I verified independently: official `n` from `p(1−p)/SE²+1` = **1319.0 for all eight**; `status.json` complete/eligible/unquarantined with empty `issues` and `judge_flags` for all eight; `judgement_general.json` `general_anomaly: false` for all eight.

---

## 1. Per-cell header fields

| field | g01s01 | g01s02 | g01s03 | g01s04 | g01s05 | g01s06 | g01s07 | g01s08 |
|---|---|---|---|---|---|---|---|---|
| arm / variant | protocol (guard) ×8 | | | | | | | |
| accuracy | 0.695982 | 0.724792 | 0.725550 | 0.735406 | 0.710387 | 0.708112 | 0.735406 | 0.725550 |
| correct / 1319 | 918 | 956 | 957 | 970 | 937 | 934 | 970 | 957 |
| stderr | .012670 | .012302 | .012292 | .012151 | .012494 | .012523 | .012151 | .012292 |
| hours_used | 8.06 | 9.46 | 8.08 | 7.62 | 8.96 | 8.44 | 8.78 | 7.94 |
| unused budget (h) | 1.94 | 0.54 | 1.92 | 2.38 | 1.04 | 1.56 | 1.22 | 2.06 |
| hours_to_first_train_launch | 0.52 | 0.37 | 0.55 | 0.63 | **0.27** | 0.43 | 0.51 | **0.68** (1.01 to the surviving run) |
| protocol_hours (pure CLI) | 0.03 (17 s) | 0.03 (48 calls) | 0.02 (8 s) | 0.15 (0.11 net of bundled dry-runs) | 0.06 | **0.001** (4 s) | 0.01 (51 calls) | 0.02 (2 s) |
| waiting_hours (raw tool) | 5.96 | 8.33 | 6.65 | 4.62 | 7.98 | 6.45 | 7.05 | 6.03 |
| **post_exit_idle_h (all producers)** | 1.10 [1.05–1.15] | 1.13 [0.99–1.22] | 1.73 [1.65–1.78] | **~1.93** ‡ | 1.13 [1.00–1.28] | 0.74 [0.70–0.79] | 2.72 [2.60–2.80] | 0.69 [0.65–0.73] |
| — training/sampling only | 0.144 | 0.573 | 0.609 | 0.48 | 0.518 | 0.122 | 0.667 | 0.336 |
| — max single event | 0.270 | 0.483 | 0.502 | 0.40 | 0.247 | 0.175 | 0.667 | 0.250 |
| greedy_shipped | yes | yes | yes | yes | yes | **NO** | yes (never A/B-measured) | yes |
| rl_used | no ×8 | | | | | | | |
| rft_tried | yes (contradicted) | yes (contradicted, non-lineage) | yes (null) | yes (mixed) | yes (contradicted) | yes ×2 (r1 supported, r2 contradicted) | **no** (rejected 3×) | yes (contradicted) |
| largest_eval_n (actual) | 500 | **1319** | 500 | 800 | **400** | 1000 | **1319** | 500 |
| n_cards / closed / locked_open | 8/8/0 | 10/10/0 | 6/6/0 | 5/5/0 | 6/6/0 | 6/6/0 | 7/7/0 | 9/9/0 |
| n_relocked (cards / events) | 2/2 | 2/2 | 1/1 | 1/1 | 0/0 | 0/0 | 1/1 | 1/1 |
| n_overrides (collect = current+history) | 0 | **4** (3 current + 1 relock-history) | 2 | 0 | 0 | 2 | 0 | 0 |
| preflight_fail (collect) | 0 | 3 | 2 | 0 | 0 | 2 | 0 | 0 |
| pitfalls_hit / cost_h (collect) | 5 / 1.30 | 6 / 2.22 | 4 / 0.88 | 6 / 1.60 | 5 / 2.20 | 9 / 0.75 | 4 / **4.20** | 7 / **3.25** |
| pitfalls cost, **distinct events** | 1.30 | 2.22 | 0.88 | 1.60 | 2.20 | 0.75 | **2.20** | **3.15** |
| fields_filled | 1.0 ×8 | | | | | | | |
| stop_reason (quoted, abridged) | "All processes are finished, all eight experiment cards are locked and closed" (17:52:46Z) | "Done. All processes are finished, the GPU is idle, and every card is closed." (L9593/19:16:15Z) | "No processes are running; all six experiment cards are locked and closed." (17:53:09Z) | "All cards closed, no processes left running, GPU clear." (L9126/17:26:21Z) | "All runs are finished, no processes are alive, the GPU is idle, and all six cards are locked and closed." (L8100/18:46:50Z) | "`final_model/` holds the exp-03 checkpoint: 73.3% on GSM8K (n=1000)" (18:15:39Z) | "All runs finished, all cards closed, nothing left running." (L8024/18:35:33Z) | "All runs finished, all nine experiment cards are locked and closed, and `final_model/` is verified." (17:45:31Z) |
| one_protocol_change (cell's own) | D | B v2 | D | B v2 | D | B v2 | **E** | 10-shot prefix entry |

‡ **g01s04 is not measured to the same standard.** Its Window-03 report scored E on **training/sampling producers only** (0.48 h). Using the same evidence class as the other seven (`[W903 …] destroy_process_group` exit stamps in each eval log → next assistant action), the eight evaluations in that cell add **≈1.45 h**, of which three single events are 0.372 / 0.369 / 0.376 h. That figure is **my computation, not the report's**, is not overlap-audited to the other reviewers' standard, and is flagged as such wherever it appears. The audited, comparable number for g01s04 does not exist.

---

## 2. Aggregate header fields

| aggregate (n=8) | value |
|---|---|
| accuracy mean | **0.7201478392721758** (pooled 7599/10552, identical because every n=1319) |
| sd / SEM | 0.013988 / 0.004946 |
| min / median / max | 0.695982 / 0.725171 / 0.735406 |
| hours_used mean | 8.42 (range 7.62–9.46) |
| unused budget mean | 1.59 h; **5/8 cells ended with ≥1.5 h unused** |
| hours_to_first_train_launch mean | **0.495** (range 0.27–0.68) |
| protocol_hours mean (pure CLI) | **0.040** (range 0.001–0.15) |
| waiting_hours mean | 6.63 |
| post-exit idle, cohort total | **≈11.17 h** across 8 cells; mean **1.40 h/cell** = **9.3×** E's 0.15 h target |
| — of which evaluation-attributable | **≈7.72 h (69%)** |
| — of which a crashed producer nobody polled | ≈2.49 h (22%), 6 of 8 cells |
| E pass count | **0/8** all-producer; **2/8** under the narrower training/sampling-only convention (g01s01 0.144, g01s06 0.122) |
| cards | **57 written, 57 locked, 57 closed, 0 locked-open, 0 unreadable**, `fields_filled` 1.0 ×8 |
| relocks | 8 cards / 8 events across 5 cells; **every one records a substantive repair or re-plan, none is churn** |
| training launches after their lock | **28/28, zero exceptions** (independently re-derived from trace launch lines vs `*.lock.json`, including `lock; launch` one-liners) |
| evaluation launches **before** their card's lock | **3 events in 2 cells** (g01s07 ×2, g01s08 ×1) |
| pitfalls_cost_h, collect sum / distinct-event sum | 16.40 h / **14.30 h** |
| greedy shipped | 7/8 |
| largest_eval_n ≥500 | 7/8 (mean 792); shipping decision backed by a valid n≥500 comparison **7/8** |
| RL launches / zero-grad lines / RL trainer classes | **0 / 0 / 0 in 8/8** |

**Score context, explicitly not a promotion claim and not concurrent matched evidence.** Historical v3 clean pool mean **0.6885628** (14 clean of 16; p00r08 failed, p00r16 official-scorer incomplete). Strict guard mean is **+0.031585** above it (Welch t≈2.37, df≈17 against the v3 pool's sd 0.04632 — a *historical* pool, run in a different wave, so this is descriptive only). Historical no-protocol control pools: 0.755724 (n=10, Window 02) and 0.756229 (n=15, round-00 core); strict guard sits **−0.036** below them, again non-concurrent. The Window-03 NEW control mean 0.757240 (n=5) is likewise a different window. **No matched control ran alongside these eight.**

---

## 3. Ranked mechanisms

There is no concurrent control arm, so this ranks (a) what explains the 3.9 pp within-cohort spread and (b) what consumed hours. Top-vs-bottom (g01s04/g01s07 0.7354 vs g01s01 0.6960) is 3.9 pp ≈ 2.3 SE of the difference — marginal, not a clean separation.

**1. The initial SFT on terminated CoT targets — 8/8 cells, +52.0 to +68.7 pp.** Universal and dominant; it is where the score comes from in every cell. g01s01: format compliance 0.28 → 1.00, token-cap hits 0.44 → 0.00 — *"the whole base-model deficit was termination, not arithmetic."* g01s07 is the extreme: one 152k-row full-parameter SFT, 0.060 → 0.747 at n=150, clean termination 49.3% → 98.7%. Because every cell did this, it explains the level, **not** the spread.

**2. Prompt-distribution match with the grader's 10-shot block — 8/8 touch it, 1 paid catastrophically, 1 measured the residual.** This is the largest identified single-card score effect anywhere in the cohort. g01s08 exp-03 trained a full 1.95 h epoch on single-problem prompts and graded **0.060** with `ends_with_answer_line 0.0067`, `hit_token_cap 0.74`, `multiple_answer_markers 0.987`; exp-04 then re-trained **375 steps on 12k of the same targets** with the grader's own k-shot block folded in and moved 0.060 → 0.700 (**+64.0 pp**). Its card is explicit: *"the stop_token preflight passed because the training targets DO end in `<end_of_turn>` — the mismatch was in the prompt distribution, not the target."* Prefixed-row share across the cohort: 0.00→1.00 (g01s08), 0.06 (g01s01, residual measured at **−3.0 pp**), 0.08 (g01s02, later measured as costless), 0.10 (g01s07), 0.12 (g01s05, never re-examined), 0.15 (g01s04), 0.20 (g01s03), 0.20 (g01s06). **Every existing preflight passed on the run that failed.**

**3. Decode configuration — 7/8 shipped greedy; measured same-weight gains +2.7 / +6.7 / +9.3 / +10.0 / +10.67 / +12.7 pp in the 6 cells that measured it.** The mechanism is large and real. Its *marginal* contribution to this cohort's spread is nevertheless **not identified**: g01s06 shipped `do_sample: true, top_k: 64, top_p: 0.95` and never made a decode measurement in 8.44 h, yet scored 0.7081 — above g01s01's 0.6960, which did ship greedy. g01s06 had the grader observable in two of its own eval logs and printed `do_sample: true` from a checkpoint twice (L5775/13:02:46Z, L6610/15:33:19Z) while asking a different question. g01s07 shipped the right config from the first SFT but never isolated it — its pre-registered ablation (exp-03) was repurposed to a data-scaling stage.

**4. Hours burned on mechanical failures, which bought or cost the third intervention — 8/8 cells, 14.30 h of distinct carded cost + 11.17 h of post-exit idle.** Ranked by cohort hours:
 - **E (post-exit idle): ≈11.17 h, 8/8 cells.** Largest single block of recoverable time in the cohort by a factor of two.
 - **D (`GenerationConfig is invalid` on a greedy parent): 4.13 h of failed compute, 5/8 cells** (g01s01 0.85, g01s03 0.60, g01s05 1.47, g01s07 1.17, g01s02 0.04) plus **1.17 h of post-exit idle that belongs to E and must not be added to D's savings**.
 - **B v2 (offline vLLM sampling defects): ≈4.05 h, 4/8 cells** (g01s02 1.80, g01s04 1.15, g01s06 0.65, g01s05 0.45). Two cells pre-empted it entirely; one ran RFT cleanly; one never sampled.
 - **P2 (262k-vocab fp32-logits OOM, plus the 64 MB overlay): ~1.1 h, 7/8 cells**, small per cell but nearly universal.
 - **P3 (checkpoint without processor/tokenizer): 0.17 h, 2/8 cells**; prevented by construction in 3 more.

**5. Evaluation n and paired discipline — decides whether a noise-level "gain" gets shipped.** 7/8 back the shipping decision with n≥500. Three cells ran exact/continuity-corrected McNemar (g01s04, g01s06, g01s07 — g01s07 three times); four used paired item counts without a test; **g01s05 used neither and made every decision at n=150**, then discovered by accident that the same hardlinked weights scored 0.7733 and 0.740 on the same 150 items — the same size as the +5.3 pp it had already adopted. Two genuine rank inversions: g01s02 (n=150 order flipped at n=1319) and g01s06 (double inversion, n=150 → n=500 → n=1000).

**6. After SFT and decode, almost nothing moved the score outside noise.** Post-decode interventions that cleared their own measured noise floor: g01s02's lr 1e-5→2e-5 (**+3.0 pp at n=1319** on byte-identical data), g01s06's RFT round 1 (+6.7 pp at n=150, 0.733 at n=1000), g01s07's 50/50 soup (+1.1 pp at n=1319, p=0.223 — sign-consistent, not significant). Everything else landed inside noise or negative: g01s01 exp-04/05/06 all −1.3 to −2.0 pp; g01s03 exp-04 +2.0 pp inside one SE; g01s05 exp-06 **−5.3 pp** (the same data axis that had just given +5.3); g01s07 exp-04 0.0 at n=150 / −1.8 at n=500; g01s08 exp-06 −3.3/−2.8 and exp-08 −1.3.

---

## 4. What the protocol cost

**Direct ceremony is not a cost.** Pure `awm exp_protocol` CLI time is **2–17 seconds per cell in six of eight cells** (mean 0.040 h). `new → lock` authoring is 0.04–0.15 h for 5–10 cards. The apparent `protocol` category in the timeline tool is an artefact of composite calls: g01s01's 0.25 h is 0.22 h of two calls that bundle `lock` with a background eval launch and a `sleep 400`; g01s08's 0.25 h is 829 of 831 seconds spent in five `lock … && setsid nohup … &` compounds. g01s06 found the floor — one `python - <<'PYEOF'` per card, lock in the same call, **4 seconds total**.

**Time to first real training launch: 0.495 h mean**, faster than any prior window's protocol arm and comparable to controls. The two slowest (g01s07 0.51, g01s04 0.63) spent the extra minutes on a memory/throughput sweep and a byte-for-byte template check, both of which visibly prevented later losses.

**What the card format bought, with counts:**
- The **comparator rule forced real evaluations that changed conclusions in 5/8 cells** — g01s01 exp-07, g01s02 exp-06/07, g01s03 exp-06, g01s06 exp-05/06, g01s08 exp-07 all exist only because a gap sat inside its own stderr. g01s02's recorded rank reversal was found by a card written to satisfy that rule.
- The **pitfalls list generated code before the failure**: g01s01 wrote `verify_template.py` at L2500/09:55:30Z *"Guards pitfall `template_unreachable`"* and `make_final.py` *"Guards pitfall `final_model_not_loadable`"*; g01s01 pre-empted both B v2 mechanisms in its sampler at L6488/10:53:05Z before a 45-minute run; `double_answer_format` caused MetaMathQA to be rejected by name in three cells.
- **The failed run had to be written up rather than silently retried** (g01s05 exp-05/exp-06 are the same hypothesis, which is the only reason the D mechanism is legible there).
- **The best single instance of the protocol paying for itself is g01s06 at 18:15:18Z**, after its final verification: *"One record correction: I deleted `ckpts/exp-02/final` for disk headroom after exp-03 superseded it, which contradicts what exp-02's card promised."* It reopened the closed card, recorded the broken `keep=last` promise, and re-closed it. No check could have caught that.

**What the card format cost, with counts:**
- **`comparator_same_protocol`: 5/8 cells, 8 FAIL events** (§6, Proposal 1). Cost: 6 override instances, **3 evaluation launches before their card's lock**, and 0.05 h carded directly in g01s06.
- **`setup.data` on non-training cards: 4/8 cells created a file or named a source file purely to satisfy the schema** (§7 H).
- **`stop_token_consistent` / `answer_marker_single` on the wrong file: 3/8 cells, 4 events, 0 overrides** — including a 7,473-row data rewrite plus a trainer patch (§6, Proposal 3).
- **The `run_dies_with_the_session` reminder was reprinted at all 57 locks and offered two waiting branches as equals.** Across the cohort, post-exit idle averaged **≈0.11 h per clock-waited producer** and **≈0.018 h per condition-waited producer**. g01s08 shows both branches in one session: 0.064 h vs 0.011 h per event, and its one clock wait over a dead process cost 0.250 h alone.

---

## 5. Guard-safety evidence matrix

The **predeclared guard gate** is narrow. Round 01 spec §判据: cells lost to session end; `n_locked_open`; hook blocks per cell; accuracy not more than 0.03 below baseline; plus the risk item "hook traps a scientist with no run (cap 12)". Round 02 spec §四(1) restates it as *"no cell lost training to session end, accuracy not below baseline − 0.03"*.

### 5a. Guard-specific gate

| criterion | g01s01 | g01s02 | g01s03 | g01s04 | g01s05 | g01s06 | g01s07 | g01s08 | cohort |
|---|---|---|---|---|---|---|---|---|---|
| **Live scientific work lost at session end** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0/8** |
| — trace evidence (`ps`/`pgrep` + `nvidia-smi` at last turn) | 0 procs, 0 MiB L10107/17:51:36Z | empty L9503-9534/19:15:27Z | 0 procs L7560/17:52:56Z | empty compute-apps L9070/17:26:06Z | 0 procs, 0 MiB L7929/18:45:55Z | empty L7695/18:15:11Z | 0 MiB 18:34–18:35 | 0 procs, 0 MiB L8383/17:45:14Z | 8/8 |
| — **independent check (`system_monitor.log`)**: last GPU-process sample before monitor end | −6 min | −6 min | −9 min | −23 min | −12 min | −8 min | −36 min | −4 min | **8/8 GPU idle at exit** |
| — background tasks killed/stopped at exit | 0 | 0 | 0 | **2**, both non-scientific shell wait-loops (`bo3tiank1`, `burqrgge4`) | 0 | 0 | 0 | 0 | 0 scientific, 8/8 |
| **Stop-hook blocks** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0/8** |
| — verification | `memory/.stop_hook.json` absent in all eight; the hook writes that counter on its **first** block (`_record_block(session, blocks+1)`); `.claude/settings.json` confirms the Stop hook was installed in all eight; `.claude/` dotfiles are demonstrably harvested. **Positive evidence, not a harvesting gap.** | | | | | | | |
| **False blocks / scientist trapped** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0/8** |
| **Locked-open cards at stop** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0/8 (57/57 closed)** |
| **Tool timeouts killing a producer** | 0 | 0 | 0 | 0 (2 wait-loops timed out; no producer) | 0 | 0 | 0 | 0 | 0/8 |
| **Long runs launched detached (`setsid nohup … < /dev/null &`)** | 31 | all | all | all | all | all | all | all | 100+ runs, 8/8 |
| **Score guardrail** (≥ v3 clean mean − 0.03 = 0.658563) | +0.0374 | +0.0662 | +0.0670 | +0.0768 | +0.0518 | +0.0495 | +0.0768 | +0.0670 | **8/8 pass; mean margin +0.0616** |

**Verdict on the predeclared gate: every criterion is met, 8/8, with independent corroboration for the load-bearing one.** Mean 0.720148 is +0.031585 *above* the v3 clean mean, not below it.

**Two honest qualifications the planner should carry into the gate decision:**

1. **The hook never fired, so the guard's blocking mechanism is untested in this cohort.** The evidence establishes that the guard is *harmless* — zero blocks, zero false blocks, zero trapped scientists, zero lost work — not that it is *effective*. Whatever protective effect exists came from SKILL rule 9 and the `run_dies_with_the_session` reminder reprinted at all 57 locks, and from the `setsid nohup` idiom that all eight adopted. The cap-12 risk item is untriggered and therefore unfalsified.
2. **The comparison baseline is historical.** No matched control ran concurrently with these eight. The +0.0316 versus v3 and the −0.036 versus the control pools are both cross-wave.

### 5b. Overall protocol compliance — reported separately, not folded into the gate

The guard candidate's own observable is `lock_before_launch` on **training** launches. That observable is clean and would not see the following, which I therefore report on its own line rather than narrowing the rule-1 invariant to training.

| compliance item | count | cells | evidence |
|---|---|---|---|
| Training launches after their card's lock | **28/28** | 8/8 | re-derived from trace launch lines against `*.lock.json`, incl. `lock; launch` one-liners; every launch 0–8 s after its lock |
| **Evaluation launches BEFORE their card's lock** | **3 events** | **2/8** | g01s07 exp-05: eval launched L6590/16:21:58Z, card locked 16:37:04Z (**15.1 min**); g01s07 exp-07: eval L7627/17:55:14Z, lock 17:55:50Z (**36 s**); **g01s08 exp-07: eval L7273/16:42:05Z, lock 16:44:17Z (2.2 min)** — a third instance not identified in any cell report |
| — of those, mechanically caused by `comparator_same_protocol` | **2 of 3** | g01s07, g01s08 | both compound commands ran `awm exp_protocol lock …; setsid nohup python evaluate.py …` — the lock FAILed, the eval launched anyway, and the card was locked only after the comparator arm had produced the file the check demanded |
| — self-reported as a rule-1 deviation | 1 of 3 | g01s07 | exp-07 `training_summary`: *"PROTOCOL DEVIATION, recorded rather than hidden: the eval command was launched … a few minutes BEFORE the lock … It still breaks rule 1 and should not be repeated."* |
| **Preflight failures invisible to `collect`** | **6 episodes** | **4/8** | g01s01 (1), g01s02 (1), g01s04 (2), g01s07 (1), g01s08 (1) — see §6 |
| Fabricated / fictional `setup.data` entries on non-training cards | 4 cells | g01s01, g01s06, g01s07, g01s08 | §7 H |

**None of these three evaluation launches involved training, none lost work, and none produced a session-end failure.** They are rule-1 (lock before you launch) compliance findings, and two of the three are caused by a preflight check with no correct answer. They belong in the planner's compliance read, not in the guard-harm verdict — in either direction.

---

## 6. Adjudication of the five critical checks

### Check 1 — E is already retained; do not re-propose what E v2 contains

The retention proof stands and is corroborated by the full cohort. The two conservative single-event lower bounds (g01s03 ≥0.433 h, g01s08 ≥0.234 h) already made ≥7/8 saturation impossible. With all eight now read, **the cohort passes 0/8 on the all-producer convention and 2/8 on the training/sampling-only convention** — saturation is impossible under *either* reading, which is a stronger statement than the two-cell bound alone. **Retain E; the G/P1 replacement branch is moot on saturation grounds.**

I read the frozen E v2 text (`c6f11d8`, tree `ceb68549`) in full. It already contains, explicitly: training **and sampling and evaluation** as producers; one-foreground-script launch/wait/exit-status handling; same-shell `wait "$pid"` with the explicit note that another Bash call cannot wait on that shell's child; a retained exit result via launch wrapper or task handle, with "unknown" rather than inferred success; bounded producer-state polling **at most 60 seconds apart**; the explicit refutation that an unchanged tail, an existing file or GPU memory proves life or death; the launcher/leftover-engine distinction; and current-run artifact identity/freshness/contents verification before accepting results or chaining. **I propose no addition to E.**

Applied literally, E v2's 60-second polling clause would cap every one of the cohort's ~90 post-exit gaps at ≤60 s. Against the measured 11.17 h that is **≈10.4 h recoverable across eight cells (≈1.3 h/cell)** — the target the E v2 screen exists to measure. The relevant caveat for the screen is not the wording but adoption: g01s06 used a condition wait for every training and sampling run and a clock sleep for every evaluation, in the same session, at a 4× difference; g01s08 switched branches mid-session and cut its per-event idle from 0.064 h to 0.011 h. Whether the rewrite reaches evaluation processes is exactly the observable.

Registration and release remain forbidden independently: **90820 (control cell c01s08) is RUNNING on `slurm2-a3nodeset-1`, outside the frozen `slurm2-a3nodesetondem-[0-1]`, with `ReqNodeList=(null)`** — a real ownership/placement FAIL that the subqueue summary reports as OK (a documented false negative). Old E jobs 91064–91067 stay held; the replacement is unregistered.

### Check 2 — producer-exit stamps: what the `W903` line actually is

The reports treat `[rank0]:[W903 hh:mm:ss…] ProcessGroupNCCL … destroy_process_group() was not called before program exit` as an exact producer-exit stamp. **I verified the emitting process and sequence, and the claim needs one qualification.**

- The `W903` line is emitted by the torch **rank-0** process at its own exit, and it is followed in the same log by `(APIServer pid=NNNN) INFO: Shutting down` → `Waiting for application shutdown` → `Application shutdown complete`. So the parent `evaluate.py` exits **at or slightly after** the `W903` stamp, not before it. The stamp is a **lower bound** on the parent's exit, and post-exit idle computed from it is biased **high by seconds**.
- The bound is tight. Independent corroboration from `system_monitor.log` (60 s cadence) in g01s07: EngineCore holding 25,158 MiB at 17:32:57 → `W903 17:33:05` → 0 MiB, no processes at 17:33:57; and 17:59:01 busy → `W903 17:59:08` → 18:00:01 clear. **In both spot checks the stamp falls inside the monitor's [alive, free] bracket, i.e. it is accurate to ≤60 s.**
- One caution for future tabulation: in g01s07, `exp-05.log` and `exp-05_comparator.log` carry the **identical** stamp `W903 16:24:08.315498106` and the same `APIServer pid=10960`. They are one process's output, not two producers. Counting both would double an interval.

**Net effect on the conclusions: none.** The bias is seconds against events of 0.2–0.7 h. For the sub-0.02 h rows the sign matters and those rows should be read as upper bounds.

I did **not** substitute file mtimes or GPU-memory proxies anywhere: harvest rewrote every bundle mtime to 2026-09-03 18:36–19:49.

### Check 3 — reconciling `collect` with the reports

All four discrepancies named in the brief are real, and I found a fifth. Raw reports stay unmodified; these are corrections to carry alongside them.

1. **`pitfalls_cost_h` double-counts the same event twice in the cohort, not once.**
 - **g01s07**: `exp-03` and `exp-04` each carry `cost_h: 2.0` for the *same* crash → collect 4.20 h, **distinct 2.20 h**. Already flagged by the report.
 - **g01s08 (new)**: `exp-02` and `exp-03` each carry an identical `cost_h: 0.1` entry, *"first smoke run OOMed in cross_entropy at bs 8 with no gradient checkpointing (fp32 master weights…)"* → collect 3.25 h, **distinct 3.15 h**. Not identified in any report.
 - Cohort: **16.40 h → 14.30 h.** The shape recurs whenever a failed card is re-run as a new card, which the protocol encourages. Any screen reading "hours attributable to mechanism X" must de-duplicate.
2. **The same crash's failed compute and post-exit idle are disjoint and must not be summed into one saving.** g01s07 exp-03: **1.17 h burnt before the exit, 0.667 h of wall clock after it** — the first is D's, the second is E's. Same for g01s03 (0.60 / 0.445), g01s05 (1.47 / 0.039), g01s01 (0.85 / 0.017).
3. **`pitfalls_hit` counts card entries, reviewer tables count mechanisms; both are correct under their own definition.** g01s01: collect 5, report 4 — the difference is a zero-cost exp-08 entry (*"Assumed greedy decoding made the eval deterministic…"*, `cost_h: 0`). g01s03: collect 4, report 3 (a 0.03 h `cd`-persistence entry). g01s06: collect 9, report 5 — four of the nine are zero-cost or protocol-friction entries, **including two the report never surfaced: 0.03 h + 0.02 h carded directly against `comparator_same_protocol` and the plan-hash check**. g01s08: collect 7, report 6.
4. **`n_relocked` counts cards, not events.** In this cohort they coincide (8 cards / 8 events across 5 cells), unlike g01r02. Every relock is substantive: a trainer rewrite after a pyarrow stall, a save-crash repair, a budget re-plan with the discarded run's step count on the record, a post-hoc script fix recorded against the shipped weights.
5. **`n_overrides` is current + relock history; H's number is a strict subset.** Verified in g01s02: 3 current + 1 in `exp-07.relocked_from[0].overrides` = **4** in collect, while the report says 3. Of those 4, **3 are `comparator_same_protocol` and only 1 is `data_files_exist`** — H's figure is 1, not 4.
6. **`preflight_fail` counts only failures carried into a written lock via `--override`. Every failure resolved another way is invisible.** Confirmed structurally: only the successful preflight is persisted (`exp-05.preflight.json` in g01s07 has `ran_at: 16:37:04Z` and shows `comparator_same_protocol: pass` — the 16:21:52Z FAIL leaves no trace in the card record at all).

 Cohort tally from the traces: **≈11 distinct preflight-failure episodes; `collect` records 7** (g01s02 3, g01s03 2, g01s06 2). The **6 invisible ones** are g01s01 exp-03, g01s02 exp-02, g01s04 exp-02, g01s04 exp-03, g01s07 exp-05, g01s08 exp-07.

### Check 4 — the `comparator_same_protocol` loop, audited across all eight

This is materially larger than the reports concluded (they had it at "2 cells / 3 instances").

| cell | card | FAIL | how it was discharged | visible in `collect`? |
|---|---|---|---|---|
| g01s02 | exp-06 | 16:10:33Z | `--override` | yes (fail 1, override 1) |
| g01s02 | exp-07 | 16:17:07Z + 16:19:30Z | `--override` ×2 (one in relock history) | yes |
| g01s03 | exp-06 | L7045/17:03:38Z | `--override` | yes |
| g01s06 | exp-05 | L7232/17:34:39Z, again 17:34:46Z | `--override` (+0.03 h carded) | yes |
| g01s06 | exp-06 | L7503/17:51:22Z | `--override` | yes |
| **g01s07** | exp-05 | L6578/16:21:52Z | **launched the comparator eval 6 s later and locked 15.1 min afterwards** | **no** |
| **g01s08** | exp-07 | L7305/16:43:05Z | **the compound command's eval ran anyway; card patched with the resulting 0.72 and locked 2.2 min later** | **no** |

**Total: 5 of 8 cells, 8 FAIL events, 6 override instances, 2 ordering violations, 0.05 h carded.** The mechanism is identical every time and structural, not scientist error: a within-card two-arm measurement produces its comparator file as an *output*, so it cannot exist at lock time. Quoted reasons: *"the comparator's n=500 eval does not exist yet by construction: this card's whole purpose is to produce both arms under the same n=500 protocol in one run"* (g01s02 exp-06 lock); *"reusing the n=500 number would be the protocol mismatch this check guards against"* (g01s06 exp-06).

The g01s08 instance is new and is the cleanest demonstration. The trace shows the failure and the launch in one tool result:

> `FAIL comparator_same_protocol — the comparator's eval file exists and used the same n: /home/ben/task/eval/exp-06_dev500.json does not exist`
> `not locked: preflight failed on comparator_same_protocol` … `launched exp-06 eval` (L7305–7309, 16:43:05Z)

The eventual lock at 16:44:17Z records `5 pass, 1 warn, 0 fail, 3 skip`, `overrides: {}`. **A successful lock provably hides an earlier failure that was resolved by starting the work the check was gating.** Zero recorded `preflight_fail` is not proof of no prior failed check.

A separate Codex reviewer is auditing this source. I did not wait for it; these findings are my own reads of the bundles and may be superseded. Residual uncertainties I could not settle: whether any *further* preflight failures were resolved inside compound commands whose output was truncated in the trace; and whether the g01s02 exp-07 relock-history override and its current override describe one episode or two.

### Check 5 — P1 against the full frozen `a4c4954` predicate

The frozen predicate is a conjunction: **(i)** the samples were drawn from the checkpoint being trained; **(ii)** the loss starts within about 0.05 of the parent's **final** loss; **(iii)** it does not move over the first 20 optimizer steps; **(iv)** the round returns the parent at n≥500 (or declines on a second round). Its own guidance excludes *"the first round from a parent that had not seen its own samples."* The Window-03 clarification adds that a mixed teacher/self stage is not interchangeable with a self-only stage, and that terminal logged loss ≠ whole-run `train_loss`.

I read the retained training logs directly (not card summaries, not trace tails) and the cards' `setup.data[].source`.

| stage | lineage | data composition (from `setup.data[].source`) | parent **terminal logged** loss | child first logged loss | Δ | flat over first steps? | matched n≥500 outcome | verdict |
|---|---|---|---|---:|---:|---|---|---|
| **g01s03 exp-05** | ✅ `ckpts/exp-04/final` sampled and trained | **self-only** (replay explicitly rejected) | **0.2481** (`train_loss` 0.2767) | **0.2043** | **0.044** ✅ | 0.2043 / 0.1952 / 0.2040 / 0.1980 ✅ | **n=500: +0.2 pp — a null** ✅ | **confirmed hit** |
| **g01s08 exp-06** | ✅ `ckpts/exp-04/final` | **self-only** (the 35% is a *prompt-format* re-render, not a data source) | **0.2515** (`train_loss` 0.2528) | **0.2241** | **0.027** ✅ | 0.2241 / 0.2200 / 0.2155 / 0.2132 ✅ | **n=500: −2.8 pp** ✅ | **confirmed hit** |
| g01s06 exp-04 (round 2) | ✅ `ckpts/exp-03/final`, itself an RFT product | self **+ 15,000 anchor rows resampled from `train_sft.jsonl`** | 0.2064 (`train_loss` 0.2120) | 0.1673 | 0.039 ✅ | flat ✅ | n=1000: −1.8 pp ✅ | **mixed data → out of the predicate's scope**; outcome consistent |
| g01s06 exp-03 (round 1) | ✅ `ckpts/exp-02/final` | self **+ 25,000 anchor rows** | 0.2691 (`train_loss` 0.2912) | 0.2109 | **0.058 ✗** | flat | n=1000: **+ gain** (0.733, best in cell) | **correctly excluded twice** — by the 0.05 band and by data composition |
| g01s01 exp-05 | ✅ `ckpts/exp-03/greedy` | **36% replay** (53,121 self + 30,000 replay) | **0.3224** (card quotes `train_loss` 0.3307) | 0.2550 | **0.067 ✗** | flat | none (n=150 only, −2.0 pp) | **not a hit** |
| g01s05 exp-03 | ✅ hardlinked | 42% teacher | 0.2589 (`train_loss` 0.30066) | 0.3323 @ step 20 | 0.073 ✗ (0.032 vs the mean) | flat | none | partial |
| g01s02 exp-05 | ✗ trained from base | 68% teacher | 0.2632 | 0.6531 @ step 10 | 0.39 ✗ | steep descent ✗ | n=1319: +0.68 pp | not a hit |
| g01s04 exp-04 | ✗ 83% teacher | mixed | 0.319 | descent | ✗ | ✗ | dev-400 +2.25, graded 150 net 0 | not a hit |
| **g01s07 exp-04** | ✗ **zero self rows** (120k fresh 405B teacher) | teacher-only | **0.3049** (`train_loss` 0.2815) | **0.3277** | **0.023** | **flat** | **n=500: −1.8 pp** | **shorthand false positive** |
| **g01s08 exp-08** | ✗ zero self rows (12k teacher) | teacher-only | **0.2515** | **0.2565** | **0.005** | **flat** | n=150: −1.3 pp | **shorthand false positive** |

**Three findings the planner should act on.**

1. **The prescribed "≥2 guard cells with the P1 signature" condition is now satisfied — 2 confirmed hits in the exact strict cohort** (g01s03 exp-05, g01s08 exp-06), each satisfying every clause including matched n≥500 evidence. This is the first time the full predicate has been met in a prescribed cohort. It arrives *after* E was retained on the non-saturation bound, so P1 is no longer competing as an E replacement; it stands on its own evidence.
 **Recoverable hours from a step-20 stop, restricted to those two: 1.05 h + 1.55 h = 2.60 h across 2 of 8 cells (0.33 h/cell).** Sampling that ran before step 20 (0.30 h and 0.83 h) is excluded and cannot be counted.
2. **The lineage clause is doing real work and must be kept.** Two teacher-only stages (g01s07 exp-04 at Δ 0.023, g01s08 exp-08 at Δ 0.005) match the shorthand "flat first-step loss" perfectly and have no self-generated rows at all. Stopping them at step 20 would have been the right call for entirely the wrong mechanism. The shorthand is not a substitute for the predicate.
3. **The 0.05 band is thinner than it looks.** The only self-distillation round in the cohort that *gained* — g01s06 round 1, +6.7 pp at n=150 and 0.733 at n=1000 — sits at **Δ 0.058**, a 16% margin outside the band. Two independent clauses (the band, and the anchor-row data mixture) each excluded it, so the frozen predicate produced no false stop; but a screen that widened the band to 0.06, or dropped the data-composition clause, would have vetoed the cohort's best RFT round.

**Corrections to the raw reports on P1, kept separate from them:**
- **g01s01's report calls exp-05 "a clean positive instance with the loss visible." That is not supported under the full predicate.** The data is 36% replay; the comparison is against exp-02's **whole-run `train_loss` 0.3307**, not its terminal logged loss **0.3224**; Δ vs the terminal value is **0.067**, outside "about 0.05"; and there is no matched n≥500 measurement — the outcome is −2.0 pp at n=150 against a 3.7 pp stderr.
- **`final-three-notes.md` calls g01s05 "only partial" and then cites "≈0.20 h recoverable."** That saving is conditional on a predicate the stage does not satisfy (mixed 58/42 data; Δ 0.073 against the terminal loss). It is a hypothetical, not a confirmed hit, and should not enter any savings total. Its 0.21 h training is already inside P1's `<0.5 h/cell` target.
- **G: zero GRPO/RL exposure in 8/8 is non-exposure, not success.** Independently verified: 0 RL trainer classes, 0 zero-grad lines, 0 `family: grpo` cards in all eight. GRPO string counts are 4–6 per cell, with a floor of 4 from the protocol's own family enumeration; only g01s01, g01s02 and g01s04 discuss it substantively, and all three rejections are budget/setup arguments — *"neither is set up and a first GRPO launch inside 4.6 remaining hours risks the whole batch for an unmeasured gain"* (g01s04 L7900/15:02:48Z). A `trl_grpo_gemma_zero_gradient` entry could not have changed any of these decisions, because none of them was about the trainer. **G has no baseline it could move in this arm.**

---

## 7. Candidate-by-candidate cohort adjudication

**A v2** — primary observable (first post-SFT eval completion → completed same-weight measured decode choice, ≤0.5 h): **5/8 clearly pass** (g01s04 0.03 h, g01s02 0.10 h, g01s01 0.14 h, g01s05 0.14 h, g01s03 0.19 h); **1/8 ambiguous** (g01s08: 1.06 h from the literal first post-SFT eval, 0.115 h from the first eval of a working artifact — the gap *is* exp-04, which existed only because exp-03's artifact was unmeasurable); **2/8 produce no clock at all** (g01s06 never measured; g01s07 shipped the right config and repurposed its own pre-registered ablation). Secondary (post-choice decision cards under an unmeasured sampling config, ≤1/cell): **7/8 pass**, g01s06 fails at 6/6. Final greedy shipped 7/8. **The Window-02 conclusion that A's original observable is saturated does not survive this cohort**, and the screen needs a defined outcome for the "shipped but never measured" state, which is 2/8 here.

**B v2** — target <0.3 h/cell of sampling-attributable RFT hours. Baseline: **4/8 cells above** (g01s02 1.80 h, g01s04 1.15 h, g01s06 0.65 h, g01s05 0.45 h), 4/8 at ~0 (g01s01 and g01s03 by pre-emption, g01s08 clean, g01s07 no exposure). g01s06 is the strongest single confirmation — three named mechanisms plus one unnamed in one card, and a measured pass rate of **0.0025 against the same checkpoint's 0.633 on the benchmark**. **A B screen must expect a non-zero base rate of cells that already know this**: g01s01 wrote `add_special_tokens=False` and `stop_token_ids=[1,106]` into its sampler at L6488/10:53:05Z before launching. One mechanism is uncovered and single-cell: `top_p=1.0` making Gemma-3 sampling incoherent and non-terminating (g01s06, 0.10 h) — carry forward, insufficient alone.

**C v2** — shipping decision backed by a valid n≥500 comparison: **7/8** (only g01s05 fails; every decision at n=150 with a single post-hoc n=400 confirmation). Largest actual n: 1319 ×2, 1000, 800, 500 ×3, 400. Paired statistics with p-values: 3/8 (g01s04 McNemar z=0.40; g01s06 twice; g01s07 exact three times). Paired item counts without a test: 4/8. Neither: 1/8. Genuine rank inversions: 2/8. **Two honest tie-breaks were labelled as tie-breaks, not gains** (g01s07 exp-05, g01s08 exp-09) — under the standing convention that is not misconduct. The screen should expect all four states (formal test / counts only / pre-registered tie-break / nothing) as distinct outcomes.

**D** — root cause hit in **5/8 cells**; D's frozen `parent_generation_config_valid` would have caught **4/8** (g01s01 0.85 h, g01s03 0.60 h, g01s05 1.47 h, g01s07 1.17 h = **4.09 h of failed compute**). The fifth (g01s02, 0.04 h) is the *same validator error* from a different path: `soup.py` set `do_sample=False, temperature=0.0` in memory before `save_pretrained` while the parent carried a stock sampling config — the check reads the parent's file and would correctly have said "nothing to validate." That is a scope limit, not a false negative to fix. **Stock false positives: 0 observed.** The worst variant is g01s03's: the crash landed on an *intermediate* save at step 780/1562, killing the run mid-flight and forcing intermediate checkpoints off entirely. Structurally immune cells used a symlink/hardlink greedy directory (g01s04, g01s08) or never wrote greedy (g01s06) — but g01s03 used a symlink for exp-03 and still hit the trap one card later, because its *SFT* parent had been saved greedy. **The check must read the training parent's `generation_config`, which is exactly what it does.** Report the 4.09 h of failed compute, the repair time, and the 1.17 h of post-exit idle as three separate quantities; the last belongs to E.

**E** — **0/8 pass all-producer; 2/8 pass training/sampling-only; ≥7/8 saturation impossible under either convention.** Cohort total ≈11.17 h, mean 1.40 h/cell. Composition: **69% (7.72 h) is waiting past evaluations that had already finished** — sleeps of 5–38 minutes for processes that took 7 seconds to 3 minutes, with the duration printed as a `total time:` line in the very log being tailed; **22% (2.49 h, 6/8 cells) is a crashed producer nobody polled**. g01s07 alone is 2.72 h (31% of its session): a full 1319-item evaluation on this node takes **2:52** and it waited **38 minutes** for one (`sleep 2280`) and **25 minutes** for the other. Within-session branch A/Bs make the mechanism unambiguous: g01s06 used condition waits for every training/sampling run (0.013–0.037 h each) and clock sleeps for every eval (0.051–0.175 h each); g01s08 averaged 0.011 h per condition-waited producer against 0.064 h per clock-waited one, and 0.250 h on the one clock wait over a dead run.

**G** — **non-exposure, 8/8.** No baseline to move. Two tooling artefacts to correct in any tabulation: g01s06's facts `RL launches=1` is a heredoc patching `rft_sample.py`, and the timelines' `first_rl` stage marker is a keyword match in card text in every cell (g01s02 09:54Z, g01s05 12:06Z, g01s07 10:01Z).

**H** — the original "zero fabricated entries, zero `data_files_exist` overrides" metric **fails**, and `n_overrides` ranks the cells in exactly the wrong order.

| cell | fabricated/fictional file | non-applicable entries (`mixture_weight: 0.0`) | `data_files_exist` overrides | note |
|---|---|---|---|---|
| g01s01 | **1** — `data/eval_fewshot_10.jsonl`, 10 rows generated purely to validate, after an empty `data/none.jsonl` placeholder was rejected with `ERROR setup.data[0].n_examples: must be a positive integer` (L2008–2040/09:52:38–46Z) | 5 | 0 | |
| g01s02 | 0 | 1 (+6 truthful-but-inapplicable entries at weight 1.0) | **1** | the honest route |
| g01s03 | 0 | 3 | **1** | *"eval-only card … copying test items into the session dir is forbidden by rule 7"* — **declined to fabricate and overrode instead** |
| g01s04 | 0 | 3 | 0 | read-only benchmark/diagnostic paths |
| g01s05 | 0 | 0 (1 truthful applicable provenance reference at weight 1.0) | 0 | mildest; the case the planner's clarification protects |
| g01s06 | **3 fictional** — the harness **source file** `…/inspect_evals/gsm8k/gsm8k.py` declared as a data file with `n_examples` 150/500/1000 | 3 | 0 | file exists, so no override |
| g01s07 | **1** — `eval/README_eval_only.md`, 4 lines, created 2 min before the first lock: *"This file exists only so that setup.data can point at a real path, as the card schema requires."* Declared `n_examples` **150/500/500/1319 against a 4-line file** | 4 | 0 | `.md` makes `data_n_examples_match`, `stop_token_consistent`, `answer_marker_single` all **skip** |
| g01s08 | **1** — `eval/README_no_training_data.md`, 3 lines: *"setup.data is required by the card schema, so it points here."* | 4 | 0 | same skip-by-extension route |

**Cohort: 4 of 8 cells fabricated or fictionalised a non-training data entry; 23 `mixture_weight: 0.0` entries plus ≥7 inapplicable weight-1.0 references; only 2 `data_files_exist` overrides, both in the two most honest cells.** A concrete, checkable screen observable emerges: **the share of non-training cards whose `setup.data[0].path` is not a `.jsonl`** — that extension is precisely what converts three checks into `skip` and produces `overrides = 0, fail = 0`.

**I (`stop_token_consistent` ownership)** — the ledger shelved this as "evidence disappeared; restore if a v3/guard cell overrides again." **The evidence reappeared in a form the ledger's own direction #17 predicted: as data rewrites, not overrides.** 3 of 8 cells, 4 events, 0 overrides, 0 recorded `preflight_fail` (§ Proposal 3).

**P2** — 7/8 cells hit the 262k-vocab fp32-logits OOM (0.05–0.35 h each, ~1.1 h cohort), plus g01s03's 0.10 h 64 MB overlay fill. One new mechanism, single-cell: g01s08's *"with `group_by_length` the long-row batches arrive later than any smoke run reaches, so a smoke run on a `--limit` slice does not prove the memory config"* (0.35 h, an OOM at step 100/4931). Keep P2 queued as a single-item candidate; **do not fold P3 in.**

**P3** — 2/8 paid (g01s02 0.02 h, g01s04 0.15 h); 3/8 prevented by construction. Independent of P2. Any fix must preserve the measured decode config and must not copy all base configs indiscriminately.

**P4** — **5/8 cells ended with ≥1.5 h unused** (1.94 / 1.92 / 2.38 / 1.56 / 2.06 h; cohort mean 1.59 h). But the mechanism P4 targets is weakly supported here: g01s01, g01s05, g01s07 and g01s08 all priced the untaken option from *measured* cost — g01s01 exp-08: *"the fix needs ~55 minutes of the 1:58 remaining, and a 3-point effect could not be distinguished from the eval's own run variance even if it worked."* That is an early stop priced from measurement, i.e. the behaviour P4 wants, occurring under the frozen text. **Record the 5/8 baseline; keep P4 an observation, not a proposal, on this cohort.**

**P5** — the investigation grows and stays confounded. Same-weights repeat reads: 2.7 pp (g01s01, md5-identical shards, 0.740 vs 0.7133), 0.98 pp (g01s02 at n=1319 with only **51.9% byte-identical completions**), 0.67 pp (g01s03), ~1.3 pp (g01s04), 3.3 pp (g01s05, 0.7733 vs 0.740 on hardlinked weights), 0.6 pp (g01s07 at n=500), 0.0 pp (g01s08). **Seven cells, seven different answers; no single repeat is a usable variance estimate.** Developer-vs-official on md5-identical shipped artifacts: g01s02 965 vs 956 of 1319 (max_connections 32 vs 2), g01s07 **991 vs 970** (16 vs 2). Those gaps are 1.5–3.5× the within-config jitter, and serving config, process identity and repeat variance vary together in every one. First-N prefix bias against the official 1319: **+3.0 pp (n=500), −0.55 pp (n=500), +2.2 pp (n=1000), +2.25 pp (n=500), +1.2 pp (n=800), ~+2 pp measured in-cell (g01s07)** — **the sign is not uniform**, so "the front of the test set is easier" is too strong as stated. Keep P5 as an investigation with matched n/weights/decode/process settings required before any causal concurrency claim; do not open a screen slot.

---

## 8. Proposals

Rules applied: at most four; each exactly one item on the allowed surface (`SKILL.md` wording / a `pitfalls.yaml` entry / a preflight check with a test / an optional card field); ≥2 actual source cells from this cohort; a 4-cell observable; a falsifier; a score guardrail. Nothing is proposed that E v2 already contains, and no slot is filled for the sake of filling it.

### Proposal 1 — relax or annotate `comparator_same_protocol` for within-card head-to-head measurement cards
- **Surface:** one preflight check plus its test.
- **Source cells: 5 of 8** — g01s02 (exp-06, exp-07), g01s03 (exp-06), g01s06 (exp-05, exp-06), g01s07 (exp-05), g01s08 (exp-07). **8 FAIL events.**
- **Why it is the strongest item in the cohort:** the check has no correct answer for a card whose comparator arm is produced by the card's own launch, and it is the only protocol element in this cohort that *caused* a violation of the protocol's own first rule. Two of its three evaluation-before-lock events are directly attributable to it; the third was self-reported by the same scientist that the check had already pushed off the ordering rule once.
- **4-cell observable:** `comparator_same_protocol` FAIL/override events per cell (baseline **8 events across 5 of 8 cells**), and evaluation-launch-before-lock events per cell (baseline **3 events across 2 of 8 cells**). Both are readable from `*.lock.json`, `*.preflight.json` and trace launch timestamps.
- **Falsifier:** a card passes the relaxed check while its comparator was in fact scored at a different `n` or under a different protocol. If that occurs, the relaxation is wrong and the check should revert.
- **Guardrail:** block accuracy mean ≥ protocol-baseline pool mean − 0.03; `fields_filled` must not drop; `n_overrides` must fall *without* the invisible discharge routes rising.

### Proposal 2 — a `pitfalls.yaml` entry (`check: null`) naming the graded 10-shot prefix
- **Surface:** one `pitfalls.yaml` entry.
- **Source cells: g01s08 (2.20 h carded plus a wasted 1.95 h epoch, +64.0 pp recovered) and g01s01 (residual measured at −3.0 pp at a 6% prefixed share)**; pre-empted at no cost by g01s03 (20%) and g01s06 (20%); a related renderer mismatch cost g01s04 0.10 h (its 10-shot reproduction stripped the `<<a*b=c>>` calculator annotations). **8/8 cells touch the lever.**
- **What it must name:** the grader always prepends ten solved examples; a model trained only on single-problem prompts continues the in-context pattern and never ends its turn; **every existing check passes, because they inspect targets and the mismatch is in the prompt.** Observables: `ends_with_answer_line`, `hit_token_cap`, `multiple_answer_markers` on the first post-SFT eval, plus "render one training row the way the grader will and check the model still ends the turn."
- **4-cell observable:** cells whose first post-SFT eval shows `hit_token_cap > 0.2` (baseline **1/8**, at a cost of 2.20 h carded and a 1.95 h epoch), and the prefixed-row share per cell (baseline 0.00 / 0.06 / 0.08 / 0.10 / 0.12 / 0.15 / 0.20 / 0.20).
- **Falsifier:** a cell with ≥20% prefixed rows whose first post-SFT eval still fails on prompt-distribution grounds.
- **Guardrail:** block mean ≥ protocol pool mean − 0.03, **and the entry must not read as "always train on 10-shot prompts."** g01s08's own `alternatives_rejected` says why: *"Train only on 10-shot prompts … would lose the zero-shot behaviour needed for the rejection-sampling round."*

### Proposal 3 — let `stop_token_consistent` / `answer_marker_single` accept a declared script-appended terminator, validated on one rendered row
- **Surface:** one preflight check plus its test.
- **Source cells: 3 of 8, 4 events, 0 overrides, 0 recorded `preflight_fail`.**
 - **g01s02 exp-02** (10:11:29Z, L3780): FAIL `500/1000 targets end with '<end_of_turn>'`. The offending file was the **few-shot exemplar pool**, not a target file. The fix rewrote all **7,473 rows** of `data/fewshot_pool.jsonl` into a `{"shot": …}` schema **and patched `scripts/train_sft.py`** to read it, at 10:11:39Z. Lock at 10:11:40Z records `9 pass, 0 fail`.
 - **g01s04 exp-02** (10:23:28Z, L4974) on a real training card: *"The preflight is right: the stop token is appended by the trainer, not carried in the data. Making the data self-documenting instead of overriding."* (L4995/10:23:49Z) — a data rewrite driven by the check.
 - **g01s01 exp-03** (12:31:41Z, L7242–7243) and **g01s04 exp-03** (13:12:58Z, L6600–6601), both eval-only cards: both **nulled `stop_token` and `answer_marker`**, converting two FAILs into WARNs and locking within 6 seconds — *"nothing is trained in this card, so the declaration itself is what's wrong"* (g01s04 L6618/13:13:08Z).
- **4-cell observable:** check-driven data-or-script rewrites and declaration-nulling events per cell (baseline **4 events across 3 of 8 cells**), and `stop_token_consistent`/`answer_marker_single` FAIL events (baseline **4**).
- **Falsifier:** a cell that declares `appended_by: script`, passes the rendered-row validation, and still ships a target set missing the terminator.
- **Guardrail:** block mean ≥ protocol pool mean − 0.03; the rendered-row validation must actually fail when the terminator is absent (an exercised-check reading, as for D).

### The fourth slot is deliberately left empty
The remaining well-evidenced items are already placed: **P1 now has 2 confirmed hits in this cohort and should be considered on its own evidence**, not as an E replacement, and its frozen commit `a4c4954` / tree `7294c236` already encodes the predicate this cohort validated — nothing needs adding to it. **P2** deserves one clause on `group_by_length` but that clause is single-cell; **P3** must stay independent of P2; **P4**'s mechanism is undercut by four cells that priced their early stop from measurement; **P5** remains an investigation under the standing planner disposition; **A/B/C/D/E/H** are frozen and their cohort readings are in §7. Forcing a fourth variant would violate the one-item-per-commit construction and dilute attribution.

---

## 9. What the protocol — and the analysis around it — should stop doing

1. **Stop reading `n_overrides` and `preflight_fail` as the friction or fabrication signal.** 6 of ~11 preflight-failure episodes are invisible to `collect`. The two cells with fabricated schema-satisfying files (g01s07, g01s08) report 0 overrides and 0 failures; the two that behaved most honestly (g01s02, g01s03) carry the overrides. Cells: 4/8 invisible episodes, 4/8 fabricated entries.
2. **Stop summing `situation.pitfalls_hit[].cost_h` as distinct-event cost.** Two double-counts in the cohort: g01s07 4.20 → 2.20 h, g01s08 3.25 → 3.15 h; cohort 16.40 → 14.30 h. Cells: 2/8, and the shape recurs whenever a failed card is re-run as a new card.
3. **Stop treating `lock_before_launch` on training launches as the protocol-compliance invariant.** 28/28 training launches are clean, and that observable is blind to all 3 pre-lock evaluation launches, 2 of which a preflight check caused. Cells: 2/8.
4. **Stop scoring E on training/sampling producers only.** g01s04's 0.48 h is not comparable to the other seven; the same-method all-producer figure is ~1.93 h, which moves it from best-in-class to second-worst. The convention change alone flips the cohort from 2/8 passing to 0/8.
5. **Stop presenting the two waiting branches as equals** — already fixed in frozen E v2 `c6f11d8`; no further proposal needed. The cohort quantifies the cost of the old wording at ≈0.11 h per clock-waited producer against ≈0.018 h per condition-waited one.
6. **Stop calling A's observable saturated.** 7/8 shipped greedy but 2/8 produce no A v2 clock at all, and one of those shipped a sampling config after having the grader's own warning line in two of its own logs.
7. **Stop treating "flat first-step loss" as P1.** Two teacher-only stages in this cohort match the shorthand exactly and have zero self-generated rows.
8. **Stop reading `est_n` from log bytes.** The measured rate here is **≈42 KB/sample**, not 44 KB, so `est_n` runs ~5% low (489→500, 502→500, 490→500); g01s06's `inspect logs: 0` is an `INSPECT_LOG_DIR` redirection artefact, not an absence. Use `evaluate.py`'s stderr, `evaluation.protocol.n`, or `p(1−p)/SE²+1`.

Items 1–4 and 8 are measurement conventions for the meta skill (`skills/exp_protocol_meta/metrics.md` and `trace_review.md`), **not** changes to any frozen scientist-facing protocol tree.

---

## 10. Open questions for the next wave

1. **Does E v2's 60-second polling clause actually reach evaluation processes in practice?** That single question decides ≈7.7 h of the cohort's 11.17 h. g01s06 and g01s08 show the same scientist applying different branches to trainings and evaluations within one session, so adoption — not comprehension — is the variable.
2. **What is the correct A v2 clock origin when the first post-SFT eval grades a broken artifact?** g01s08 is 1.06 h or 0.115 h depending on the answer, and the screen cannot score the two cells (g01s06, g01s07) that produce no clock at all.
3. **What decides the outcome of a fitted-parent RFT round, given g01s06's two rounds have indistinguishable loss curves and opposite signs?** The cohort's answer is "whether the parent is itself an RFT product," but that is 1 cell for the contrast. The 0.05 band's 16% margin on the one profitable round needs a second observation.
4. **Is the same-weights repeat variance 0.0, 0.67, 0.98, 1.3, 2.7 or 3.3 pp?** Seven cells, seven single draws, none a usable estimate. A matched design (same weights, same items, same serving config, ≥3 repeats) would cost ~10 minutes and would set the floor under every C, A and P5 reading.
5. **Does relaxing `comparator_same_protocol` remove the ordering violations without letting a genuinely mismatched comparator through?**
6. **Would the `.md`/non-jsonl skip route survive an H tree that makes `setup.data` optional for non-training families**, or would it simply migrate to another invisible discharge? 4/8 cells found a route this time and only 2/8 used the visible one.
7. **Is the 8/8 zero-RL rate a budget phenomenon that a longer horizon would dissolve?** Every rejection in this cohort is a time/setup argument; none is a belief about the trainer.

---

## 11. Boundaries this synthesis does not cross

- **The guard safety gate is the planner's decision.** §5 supplies the evidence for both of its predeclared conditions and both qualifications; it does not declare the gate passed.
- **Ownership/placement FAIL is live and independent of every statistical conclusion here.** Job **90820** (control cell `c01s08`) is RUNNING on `slurm2-a3nodeset-1`, outside the frozen `slurm2-a3nodesetondem-[0-1]`, with `ReqNodeList=(null)`; the subqueue summary reports `OWNERSHIP OK` (a documented false negative). No new submission or release is authorised. Native two-node isolation remains unsatisfied. I did not query or modify Slurm.
- **Wave order is unchanged by this evidence: D / B v2 / C v2 + drift A v2 remain the first scientific wave; A v2 and H follow; E v2 last.** Nothing in the cohort justifies a different first wave. D's cohort hours (4.09 h catchable across 4/8 cells) and B v2's (4.05 h across 4/8) both hold up; C v2 is close to but not at saturation (7/8).
- **Old E jobs 91064–91067 stay held.** E v2 (`c6f11d8` / `ceb68549`, manifest `…-r02-e-wait-on-process-x4-v2.yaml`, cells `e02s01–04`) is built, frozen and **unregistered**; a valid replacement held receipt must exist before the old block is withdrawn, and no RUNNING job is touched.
- **P1's status changed on evidence, not on the saturation branch.** E was retained, so P1 is not a replacement; it now has 2 confirmed hits in the prescribed cohort and can be considered on its own merits when a slot opens. Its tree and manifest are untouched.
- **No baseline is promoted; AIME2025 is untouched; the raw cell reports are unmodified.** Every correction in §6 is recorded here, separately from the helper output it corrects. A separate Codex reviewer is auditing the `comparator_same_protocol` source; my findings there are independent reads of the bundles and may be superseded by it.
