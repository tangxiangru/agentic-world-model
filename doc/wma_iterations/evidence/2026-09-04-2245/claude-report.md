# WMA evolve — diagnostic report for event `20260904T214603Z-dc9ade9335`

**Context.** The hook fired `new_clean_complete>=8` at 2026-09-04T21:46:03Z against repo head
`b3732f8f7feb85d97deb9af7c4f72599f8db525d` on `gangda_wma_evolve`. This report is the read-only
diagnosis of that window plus the design of the next evidence-efficient wave. It is advisory only:
no file in the repo was edited, no commit or push was made, no PR was touched, and no Slurm job was
submitted, cancelled or requeued. The event directory contains **no `operator.status.json`**, so this
handoff has not been executed and this is not a repeat; the five prior events each have one, and the
immediately preceding one (`20260904T164313Z-792dc7f482`) is `state: reviewed`,
`disposition: accepted_with_corrections`, `promotion: null`, with an explicit do-not-repeat note.
Claims that the operator already rejected there are not re-asserted.

**Confidence marking.** `[V]` = independently verified by me in this session against the named
artifact. `[O]` = operator's own recorded number, reproduced here but not re-derived by me. `[S1]` =
single specialist source, adversarial verification did **not** run (the verifier stage was stopped on
budget) — treat as a lead requiring operator confirmation, not as an established result.

---

## 1. Validity and compliance gate

The window contains **9 clean-complete cells in three non-poolable strata**. They differ in benchmark,
scientist model, WMA presence and protocol, so no cross-stratum arithmetic is legitimate.

| Stratum | Batch | Cells | Bench | Scientist | WMA skill | n clean |
|---|---|---|---|---|---|---|
| A | `wma-crossbench-opus48-r05-bfcl-protocol-x4` | c54r01–04 | BFCL | `claude-opus-4-8` high 200k | **none (arm P)** | 4/4 |
| B | `wma-crossbench-opus48-r05-gsm8k-protocol-x4` | c52r03 | GSM8K | `claude-opus-4-8` high 200k | **none (arm P)** | 1/4 |
| C | `wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4` | w14r01–04 | GSM8K | `claude-opus-5[1m]` | H `a536a0af24d7` | 4/4 |

**Provenance [V].** Strata A/B carry `agent claude_vertex_high_200k_awm`, model `claude-opus-4-8`,
spec `doc/spec/2026-09-04-wma-opus48-crossbench.md`, jobs 92185–92188 (A), `ptb_commit
e62036f0c244995a6f45496522d3310b239383c6` / `top_commit 225bd584f35ecaf0ec3fac4c2fb02d946030180c` (A)
and `0bb448cca7dbc5f54178507a66cfdfc15d682df3` / `c914ef98ff84382193ec7509cd4fe78ce747e75e` (c52r03).
Stratum C carries `claude_vertex_high_awm` / `claude-opus-5[1m]`, jobs 91445–91448, `ptb_commit
62203e498d1007875b32a3415c11caa95ccf4a2e`, `top_commit 487483842337a51057f0e45f1933d9bc7fcbfd06`, spec
`doc/spec/2026-09-03-wma-round04-probe-selection.md`. Different `ptb_commit` values across strata are a
second, independent reason A/B and C cannot be pooled.

**`lock.wma.state` [V].** For strata A and B this field does not exist and its absence is *correct*, not
a compliance failure: `experiments/posttrainbench/wma-crossbench-opus48-r05-bfcl-protocol-x4.yaml`
contains `setup: --exp-protocol --tool claude --decision-mode single` and **no `wma:` block**, so arm P
ships no WMA sidecar. Five of the nine clean cells therefore generate **zero** WMA verdicts and cannot
speak to skill quality at all. Only stratum C has lock evidence, and there the operator's record and the
ledger agree that all retained finals are `delivered` [O].

**Verdict-before-launch [O/S1].** Preserved across the H cells per the operator record; 25 retained
finals, all delivered. One paid verdict file was **rejected by the validator** and is excluded from the
n=25 ledger: `results/ptb/.../w14r02/task/memory/cards/exp-01.verdict.json.rejected`, $1.733 [S1]. If
confirmed, this is a real accounting gap — the ledger's cost is a lower bound in a second, previously
undocumented way (beyond the known relock-overwrite path).

**Waits, timeouts, skips [O/S1].** H shows no lock timeout and no `not_attached`. `[S1]` reports **38
WMA reviews executed for 26 cards**, i.e. **12 superseded reviews destroyed by `--relock`**, which is
consistent with the known instrument behaviour (superseded reviews are invisible to `uptake.py`;
retained cost is a lower bound). Mean per-verdict wall is 6.3608 min [O/S1].

**Judge flags [V].** Stratum A: `issues: []` and `judge_flags: []` on all four cells — the cleanest
block in the window. c52r03: clean. c52r02 is **excluded** (`judge_flags: ["general_anomaly"]`,
accuracy .49583017437452614) and correctly absent from `clean_complete_cells`. Stratum C: all four pass
PTB completion and automatic judges, but carry **11 original WMA access/scope flags** in the ledger [O].

**Not results [V].** Job `92312 / v91r01` is a validation-only context smoke test for S0 and is PENDING;
it is not a scientific cell. S0 cells `w66r01–04` are staged and **not submitted** — absent from the
queue at 21:52:14Z. Queue at that time: ownership OK, `gangda_wma_evolve` 16/16 GPUs on
`slurm2-a3nodesetondem-[2-3]`, 33 pending (32 scientific + 92312).

**Gate outcome: the window is valid but thin and lopsided.** Every arm is below the 8-clean-cell
threshold (A: 4, B: 1, C: 4). By the standing rule all three are **provisional**, and no promotion may
be recommended from this window on sample size alone — before any guard is even consulted.

---

## 2. Ledger, uptake, timing, cost, harms, PTB comparison

### 2.1 Ledger by WMA skill hash

Skill hashes recomputed by me with `awm.wma.schema.skill_sha` over `git archive <commit> skills/wma` [V]:

| Source commit | Skill hash | Role |
|---|---|---|
| `ae46724` | `176f0a464986` | v0.2 baseline |
| `125a434e` | `e4402ffa6bca` | G probe-scope (falsified) |
| `7e69e5c5` | `a536a0af24d7` | **H soup-ingredients (this window)** |
| `4bd8ed51` | `17be8a23046a` | redesigned policy = current HEAD |
| `f8e4c228` | `176f0a464986` | S0 legacy archive — **byte-equivalent to v0.2** |

H is a genuine single edit [V]: `git diff ae46724 7e69e5c5 -- skills/wma/` is *1 file changed, 1
insertion, 1 deletion* — the C6 row only, replacing "a variance-reduction move, the default when time is
short" with "a possible variance-reduction move, not a default and never justified by the clock alone".
H's commit parent is G, and H removes G's 13-line probe block, so H's net diff versus v0.2 is exactly the
C6 clause and nothing else.

**H ledger (`a536a0af24d7`, claude / opus-5 / high / online) [O, reproduced S1]:** n=25 finals,
n_scored=14, **n_leak_suspected=11**, L0_hit 1.0, L1_hit 1.0, L2_coverage 0.727 (n=11),
L2_width_mean 0.1373, **width/noise 4.9453**, L3_hit 0.600, `gpu_h_saved` 0, `gpu_h_wrongly_killed` 0,
cost_usd_sum **$49.117**, mean $1.9647, mean wall 6.3608 min.

### 2.2 L0–L3 structure — the decisive negative result

`[S1]`, extending the prior accepted finding: on H's retained finals **L3 = "yes" 25/25**, and on the
four cards whose ground truth was *reject*, H scored **0/4**. The same structure holds for v0.2 (0/8) and
G (0/6). The single non-"yes" L3 recovered anywhere in the three cohorts was in a **superseded, destroyed
review** (`L3=defer`). Two independent structural facts corroborate this without needing the specialist:
`gpu_h_saved` and `gpu_h_wrongly_killed` are **both exactly 0** for H, G and v0.2, and
`awm/wma/ledger.summarize` only accumulates those two counters over rows with
`L3_answer in ("no","defer")` — so 0/0 is not a measurement artifact here, it is the arithmetic
signature of *no `no`/`defer` rows ever surviving to the retained final*.

**This is the central diagnostic fact of the window.** L3 is behaving as a constant classifier. Its
apparent hit rate (0.600) is entirely inherited from the base rate of cards that should be approved.
`width/noise 4.9453` says the L2 intervals are ~5× the noise floor — wide enough to be almost
unfalsifiable. Under a constant-`yes` L3 and a 5×-noise L2, *no C6-style wording edit can move the
primary metric*, because the metric is not sensitive to the text being edited.

### 2.3 Uptake funnel and timing

H: 38 reviews → 26 cards → 25 retained finals → 14 scored → 11 L2-scorable [O/S1]. The 38→25 collapse
(12 destroyed superseded reviews) is the widest leak in the instrument and is invisible to `uptake.py`.
Retained-final waits and lifecycle minutes are recorded but are **not** GPU idle and must not be read as
utilization [O]. Allocated time 35.0261 h across four H cells [O].

### 2.4 Score effects and PTB comparison — strictly within stratum

**Stratum C (GSM8K, opus-5[1m], H vs matched v0.2 w10) [V from snapshot, O for statistics]:**
73.6164 / 70.7354 / 80.0607 / 68.4610 → mean **73.2183%**, sample SD **5.0257 pp**, n=4.
Matched w10 v0.2: **72.3180% ± 2.7382 pp, n=8**. Descriptive difference **+0.9003 pp**, i.e. **0.18 within-arm
SD** — far inside noise, and H's own SD is nearly double the baseline's. w14r03 (80.0607%) is the highest
observed WMA single score to date; a maximum is not a treatment effect. For calibration, G in the same
runtime was 73.5406% ± 2.6897 (n=4) and was falsified. H, G and v0.2 are mutually indistinguishable.

**Stratum A (BFCL, opus-4.8, arm P):** .91 / .88 / .94 / .90 → **90.75% ± 2.50 pp, n=4** [V from
snapshot]. Scientist spend $70.021497, allocation 12.1992 GPU-h [O].

**Stratum B (GSM8K, opus-4.8, arm P):** one clean cell, **58.3017%** [V]. n=1 supports no inference.

**The crossbench primaries are currently undefined [V].** BFCL raw (R) and GSM8K raw (R) each have
**zero** clean cells, so **P−R cannot be computed on either task**, and J−R / J−M are likewise unavailable.
Stratum A is a well-measured *protocol-only* point with no counterfactual attached to it. Under the
standing rule ("never compare Opus 4.8 cells to the old Opus 5 cohort as though only the WMA skill
changed"), A and B cannot borrow C's baseline.

### 2.5 Cost and variance

H: WMA $49.117 retained + scientist $224.60378525 = **$273.72078525** known combined, excluding judges,
excluding 12 superseded reviews, and excluding the $1.733 validator-rejected file [O + S1]. Cost per
usable statistical unit is the real finding: **~$68 per H replicate to move a mean by 0.18 SD.** Variance
is the binding constraint, not budget: at SD ≈ 5 pp, detecting the ~1 pp effects these edits produce
needs n in the hundreds. Four repeats are an exploratory screen, as the crossbench spec itself states.

### 2.6 Harms — and their correct attribution

The dominant harm in this window is **harness/CLI, not skill**. The lifecycle failure class (CLI ends
with a text-only `end_turn` reply promising to wait for background work; background tasks stop; no final
model is produced) destroyed **c51r01, c53r01, c55r04, c56r03**, killed exp-04 in c52r04, and is the same
pattern behind the `general_anomaly` flags on c53r03/c53r04 [V, crossbench log]. That is what emptied
both raw arms and made P−R undefined. Seven occurrences are now on record and `long_training_lifecycle`
is an open blocked task with no accepted runtime candidate.

Two further harms are protocol-side, not skill-side: c56r03 trained on a 110,689,404-byte SFT file that
passed the n-gram check and was later semantically flagged (279 documents, 67 HumanEval-idiosyncratic
function families, Magicoder OSS/Evol lineage) — a data-boundary defect with the WMA absent; and H's
**11 original access/scope flags**, which are skill-side and fatal to H's guard.

---

## 3. Ranked causal diagnosis

1. **The WMA decision layer is a constant approver; wording edits cannot reach the metric.** (High
   confidence.) L3 = yes on every retained final across v0.2, G and H; 0/0 saved/wrongly-killed is the
   arithmetic signature; L2 width is ~5× noise. *Counterevidence:* the one recovered `defer` sat in a
   destroyed superseded review, so the policy is not literally incapable of a non-yes — the **retention
   path** may be filtering them. That alternative is testable and matters enormously for the next wave.
   *Uncertainty:* the 38→26→25 funnel is `[S1]`; the operator should reproduce the count before acting.
2. **Empty raw arms are a harness lifecycle defect, not a skill or protocol effect.** (High confidence.)
   Four cells lost to text-only `end_turn`; the two `general_anomaly` BFCL cells share the pattern. Arm P
   is unaffected precisely because it ships no WMA sidecar, which is a *confound*, not evidence for P.
   *Counterevidence:* none found; the failure is model-visible in the transcripts.
3. **The window has no power to separate any candidate.** (High confidence.) Within-arm SD 2.7–5.0 pp
   against ~1 pp differences; the round-01 lesson (n=8 resolves nothing under ~4 pp) reproduces exactly.
4. **H is correctly falsified, and for the guard reason, not the score reason.** (High confidence.)
   11 original scope flags vs. a preregistered zero-flag guard [O]. Independently, the spec's mechanism
   threshold ("fewer than three eligible soup cards means insufficient opportunity") was not met earlier
   in the arm. Both routes reject H without reference to 73.2183%.
5. **Cost accounting systematically understates WMA spend.** (Medium confidence.) Two distinct leaks:
   relock-destroyed reviews (known) and validator-rejected-but-paid verdict files (`[S1]`, one instance,
   $1.733).
6. **Neither G's nor H's text is in the deployed policy.** (High confidence [V].) HEAD is
   `17be8a23046a`; `skills/wma/change_types.md:47` still reads "the default when time is short" and
   `skills/wma/SKILL.md` has no probe-scope block. So the falsification of G and H says nothing about the
   policy actually running in the crossbench — a fact that must not be lost when reading Opus4.8 results.

---

## 4. Promotion decision

**No promotion is justified from this window.** This is not a close call and it is over-determined:

- Every arm has fewer than eight clean-complete cells (4 / 1 / 4) → provisional by rule.
- H fails its own preregistered common guard: 11 original scope flags against a zero-flag requirement.
- H's score difference (+0.9003 pp) is 0.18 of its own SD, and its mechanism-opportunity threshold
  (≥3 eligible soup cards) was not met.
- Strata A and B contain **no WMA at all**, so they cannot promote any skill candidate in either
  direction.
- The one arm with a clean 4/4 block (BFCL P) has no counterfactual: its comparator arm has zero clean
  cells.

The only forward-moving item already accepted by the operator — the **S0 legacy-v0.2 policy reference**
(`f8e4c228`, skill `176f0a464986`, manifest `wma-crossbench-opus48-r06-gsm8k-legacy-v02-x4.yaml`, cells
w66r01–04) — remains **gated**, not promoted, pending validation job 92312.

---

## 5. Next wave — independent single-edit candidates

Design principle forced by §3.1 and §3.3: **stop spending $270 per arm on prose edits measured through a
downstream benchmark mean whose noise is 5 pp.** Two of the four candidates below are measured on
*decision-quality* metrics that need no GPU at all; the two that do consume GPU are aimed at the harness
defect that is currently destroying arms. Each is one edit, independently launchable, none pooled.

### N-A — Reject-truth probe set (instrument, zero GPU) — *highest priority*

- **Edit:** one new file under `tools/wma-rca/` implementing a frozen, offline set of ~40 archived cards
  with adjudicated ground truth, of which ≥15 are true rejects; scores L3 against them. No change to
  `skills/wma/`, no change to any scorer, judge or manifest.
- **Mechanism:** if L3 is a constant classifier, its accuracy on a balanced set collapses to the reject
  base rate; if the retention path (not the policy) is eating `no`/`defer`, offline L3 will produce them.
  This is the single experiment that discriminates diagnosis 1 from its counterevidence.
- **Primary metric:** L3 recall on the true-reject subset (`n_reject_correct / n_reject`).
- **Falsification:** if offline L3 recall on rejects is ≥0.5, the constant-classifier diagnosis is wrong
  and the next wave must target retention instead of policy text.
- **Guards:** leak — cards are already-run artifacts, no held-out prompt/gold enters any context; cost —
  CPU only, target <10 min, $0 scientist spend; PTB — reads no live cell and rewrites no ledger, scorer,
  flag or historical judge output.
- **Replication / baseline / gate:** deterministic, so n=1 per skill hash; run against `176f0a464986`,
  `e4402ffa6bca`, `a536a0af24d7` and HEAD `17be8a23046a`. Gate: report all four recalls with exact card
  citations; **no promotion attaches to this candidate at all** — it is an instrument that makes later
  promotions decidable.

### N-B — Review-retention accounting fix (instrument, zero GPU)

- **Edit:** one change so `--relock` preserves superseded reviews under a distinct suffix and
  `uptake.py` counts them; also count validator-rejected `*.verdict.json.rejected` files in cost.
- **Mechanism:** closes both cost leaks and makes the 38→25 funnel measurable, which is a precondition
  for trusting any future uptake or cost comparison.
- **Primary metric:** recovered review count and recovered USD on the three existing archived cohorts.
- **Falsification:** if recovered reviews are <5% of retained finals and recovered cost <2%, the leak is
  immaterial and the fix should not ship.
- **Guards:** must not alter any existing `exp-NN.verdict.json`, ledger row, flag or score; append-only.
- **Replication / baseline / gate:** re-derive on H, G and v0.2 archives; gate is exact reproduction of
  each existing published number *plus* the newly surfaced rows, with a diff showing no historical value
  changed.

### N-C — Background-work lifecycle guard (harness, GPU-bearing)

- **Edit:** one harness change that treats a text-only `end_turn` with unfinished background training as
  a non-terminal state (block the turn or fail loudly) instead of silently ending the cell.
- **Mechanism:** directly targets the cause of the four destroyed cells and the two `general_anomaly`
  BFCL cells; it is the only intervention that can make the raw arms produce clean cells and therefore
  the only route to a defined P−R.
- **Primary metric:** clean-complete rate in the raw arms (currently **0/8 observed across BFCL R and
  GSM8K R**).
- **Falsification:** if clean-complete rate in a fresh 4-cell raw block does not exceed 2/4, the guard
  does not address the failure and should be reverted.
- **Guards:** leak — none, no benchmark data touched; cost — must not increase per-cell wall by >10%;
  PTB — must not be retrofitted into running or pending cells and must not selectively rescue any failed
  result; existing first-wave outcomes stay their own cohort.
- **Replication / baseline / gate:** 4 cells GSM8K R + 4 cells BFCL R under the guard vs. the frozen
  pre-guard raw attempts as historical reference. Gate: report clean-complete counts per task
  separately; **do not pool across tasks**; promotion of the guard requires ≥8 clean cells total *and*
  a per-task rate strictly above the pre-guard rate on both tasks.

### N-D — S0 legacy-vs-current policy reference (already staged, GPU-bearing)

- **Edit:** none new — launch the already-frozen `wma-crossbench-opus48-r06-gsm8k-legacy-v02-x4.yaml`
  (w66r01–04, legacy `176f0a464986`) once 92312 clears.
- **Mechanism:** supplies the only matched old-vs-new policy contrast inside the Opus4.8 runtime, which
  currently has *no* WMA-bearing clean cell at all.
- **Primary metric:** GSM8K accuracy, legacy `176f0a464986` vs. current `17be8a23046a`, both under
  `claude-opus-4-8` high 200k, same `ptb_commit`.
- **Falsification:** if |Δ| < 1 within-arm SD at n=4, the comparison is declared underpowered and
  explicitly not extended by adding repeats to a losing arm.
- **Guards:** the 40 nominal GPU-h added budget is a ceiling; no G/H manifest or crossbench receipt is
  altered; verdict-before-launch preserved; all timeouts/skips reported.
- **Replication / baseline / gate:** n=4 is an exploratory screen only. Gate: **8 clean cells per arm
  before any promotion language is used**; report the legacy arm's flag count against the same
  zero-original-flag guard that falsified G and H.

**Sequencing.** N-A and N-B are zero-GPU and should run first and in parallel — together they cost
essentially nothing and they determine whether any GPU-bearing skill experiment is worth funding.
N-C unblocks the crossbench primaries. N-D proceeds on its existing gate. **No skill-text candidate
(a fifth C6/probe-style edit) should be launched until N-A returns**, because §3.1 says the metric
cannot currently see such an edit.

---

## 6. Operator handoff — what Codex should do next

**Verify (read-only, before anything else):**
1. Reproduce `uv run awm wma ledger results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/*/task`
   and confirm the three `[S1]` claims: L3 = yes on 25/25 retained finals; 0/4 on reject-truth cards;
   38 reviews for 26 cards. These are single-source and unverified.
2. Confirm the validator-rejected paid file
   `results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r02/task/memory/cards/exp-01.verdict.json.rejected`
   ($1.733) and whether it is excluded from every published H cost figure.
3. Confirm `[V]` that HEAD `17be8a23046a` contains neither H's C6 fix
   (`skills/wma/change_types.md:47` still reads "the default when time is short") nor G's probe block —
   and decide explicitly whether that is intended, since the Opus4.8 crossbench is running that policy.

**Edit:** nothing in `skills/wma/`. The only edits worth making this round are the two zero-GPU
instruments, N-A (`tools/wma-rca/` reject-truth probe) and N-B (relock/rejected-verdict accounting),
each as its own single candidate commit. Defer the HumanEval semantic guard as already recorded (the
PostTrainBench submodule holds a preserved external update).

**Launch:** nothing new on GPU until 92312 clears. Then S0 `w66r01–04` under its existing gate. N-C
(lifecycle guard) needs an accepted runtime candidate first; `long_training_lifecycle` stays blocked
until one exists.

**Record:** write `operator.status.json` into
`/rmeng_data/robtang/wma-evolve-hook/gangda_wma_evolve/events/20260904T214603Z-dc9ade9335/` with
`promotion: null` and the explicit reasons (all arms <8 clean; H's 11 scope flags; strata A/B carry no
WMA; BFCL/GSM8K raw arms have zero clean cells so P−R is undefined). Append to
`doc/wma_iterations/2026-09-04-opus48-crossbench.md`: BFCL P **90.75% ± 2.50 pp (n=4)** as a
protocol-only point *with its comparator explicitly marked undefined*, and the four lifecycle-destroyed
cells named as harness harm rather than arm outcomes.

**Do not:** pool A/B with C; read BFCL P's clean 4/4 as evidence for arm P; extend H; or launch another
skill-wording candidate before N-A reports.

---

### Coverage note

The six-dimension verified workflow (`wf_b345e5fb-992`) was stopped on budget after Dimension 1
returned; Dimensions 2–6 and all twelve adversarial verifiers did not run. Sections 1, 2.4, 2.6 and 3.6
rest on my own direct artifact reads `[V]` and the operator's records `[O]`; §2.2 and the funnel counts
in §2.3/§2.5 rest on the single completed specialist `[S1]` and are flagged for operator reproduction in
the handoff above. Nothing in this report was inferred from a PR comment.
