# WMA evolution — event `20260904T164313Z-792dc7f482` diagnostic report

**Advisory only. Nothing was edited, committed, pushed, submitted, cancelled or requeued.**

## Context

The hook froze a new validator-clean window on `gangda_wma_evolve` at 2026-09-04 16:43 UTC:
7 clean-complete cells (3 no-WMA control + the first 4 cells of candidate **G `g-probe-scope`**).
The event dir has **no `operator.status.json`**, so this handoff has not been executed and this is
not a repeat. The question is whether G's preregistered gate is met, what the window actually
establishes, and what the next evidence-efficient wave should be.

Short answer: **no promotion is justified**, G is falsified on its own primary readout, and the two
highest-value next moves cost **zero GPU hours** because they are measurement repairs, not skill edits.

---

## 1. Validity and compliance gate

**Window.** `repo_head 7a21bb47`, trigger `partial_clean_complete>=4_aged_21600s`. Clean-complete
cells: `wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1/{c10r05,c10r07,c10r08}` and
`wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4/{w13r01..w13r04}`. `c10r06` is in the batch's
`clean_complete=4` but **not** in the payload — `state.json:analyzed_cells` already holds it from a
prior window. This window therefore contributes 3 new control cells, not 4.

**Judges and validator.** All 8 snapshot rows: `issues: []`, `judge_flags: []`, `run_purpose: formal`,
agent `claude_vertex_high_awm`, model `claude-opus-5[1m]`, effort high, `ptb_commit 62203e49`. Clean.

**Slurm is not the criterion, in either direction.** All four G jobs 91441–91444 are Slurm **FAILED**
(elapsed 08:56:07 / 08:24:34 / 08:18:44 / 06:46:37) while being validator-complete and judge-clean.
Two distinct causes in `slurm.err.tail`, both in the **sbatch epilogue** of
`third_party/PostTrainBench/src/commit_utils/slurm/single_task.sbatch`, i.e. after the agent session,
the graded eval and the judges had all produced artifacts: a `line 240: syntax error near unexpected
token 'fi'` (all four G cells + c10r08) and `error reading input file: Stale file handle`
(c10r05, c10r07). All 7 payload cells' `slurm.out.tail` ends at `PREFLIGHT PASSED`. The FAILED state
carries no information about experimental validity. It affects 12 of 20 cells.

**`lock.wma.state`.** G **25/25 `delivered`**, zero failed/timeout/not_attached. Control 27/27
`not_attached`, correct — the control arm ships no WMA. Total final recorded wait 8543.7 s =
**142.395 min**; all-request lifecycle 251.305 min. Neither is measured GPU idle. Delivery per cell
(processed / responses / delivered finals / wait s): w13r01 13/13/7/2232.3, w13r02 10/10/7/2467.5,
w13r03 8/8/6/2047.0, w13r04 11/11/5/1796.9.

**42 responses but 25 retained finals — 17 review sessions were overwritten** by `awm exp_protocol
lock --relock`, which replaces the retained verdict. Retained cost is a **lower bound**; do not fill
missing cost with zero and do not scale by 42/25.

**`verdict_before_launch` is mismeasured and every prior round's compliance number is wrong.**
`tools/wma-rca/uptake.py` reports G 2/25 and baseline 18/57, contradicting the blocking-lock design.
Root cause is `tools/wma-rca/rcalib.py:274 launch_time()` with the `LAUNCH` regex at
`rcalib.py:27`: it accepts any Bash call from `locked_at − 3 min`, `locked_at` is the *start* of the
now-blocking lock call, and the "names this card" test is satisfied by (a) the
`awm exp_protocol lock --relock/--override` command itself, whose message text quotes
`scripts/train_sft.py`, and (b) `out_dir` basename `eval` for decode-config cards, which appears in
almost every command. First-detection classification:

| arm | lock/check cmd | card-edit heredoc | genuine launch | other | none |
|---|---:|---:|---:|---:|---:|
| G (25) | 16 | 4 | 4 | 1 | 0 |
| v0.2 (58) | 24 | 4 | 21 | 8 | 1 |

Successive corrections converge only after per-case adjudication:

| method | G | baseline |
|---|---:|---:|
| tool as shipped | 2/25 | 18/57 |
| exclude `exp_protocol` lock calls only | 15/25 | 37/57 |
| + exclude card-edit heredocs, anchor to lock `completed_at` | 19/25 | 47/58 |
| **all 17 residual exceptions adjudicated case-by-case** | **25/25** | **58/58** |

**TRUE GATE BYPASS: 0.** All 17 residuals classify as not-a-launch (3), legitimate pre-lock
probe/smoke/dry-run (6), a prior or other card's run (3), pre-lock soup artifact build (2), or a
minute-rounding tie (3). The ties resolve because `responses/*.json:completed_at` **lags** true
delivery by +5.7/+15.6/+29.7 s (min/median/max over all 83 cards, never negative) — e.g.
`w13r03` prints `verdict: L0_runs=yes@0.93 … read it before launching` at 15:29:55 while the
response file says 15:30:09. An independent detector built from each card's own locked
`setup.command.argv` resolved 73/83 cards and found no bypass either. **The blocking lock is
working; the reported violation rate was entirely an instrument artifact.** Three artifact
generators beyond the two named: `cat > memory/cards/exp-NN.yaml <<EOF` card writes whose body
quotes `train_sft.py`; `nohup` on non-run work (dataset downloads, `contamination_check.py`, soup
builds, the lock itself, e.g. `w10r04/exp-02 08:34 nohup awm exp_protocol lock … &`); and
`family=='other'` cards, which accept any launch. `launch_time` also uses the *final* lock's
`locked_at`, so relock chains are scored against the wrong verdict.

**Provenance.** G private source `125a434e`, skill `e4402ffa6bca`, public protocol `ae46724`, PTB
`62203e49`. Verified independently by recomputing `schema.skill_sha`: `ae46724 → 176f0a464986`,
`125a434e → e4402ffa6bca`; `skills/wma/` is unchanged between `ae46724` and `125a434e`'s parent
`1c96976a`; parent→G diff is exactly `SKILL.md | 13 +++`.

**"One edit" is true of the skill but not of the shipped archive.** `git diff --stat ae46724 125a434`
is 4,391 files; excluding harvested `results/`, `doc/` and `experiments/`, the changed source paths are
`AGENTS.md`, `awm/wma_client.py`, `awm/wma_evolve_{hook,timer}.py`, `skills/wma/SKILL.md`,
`skills/wma_meta/SKILL.md`, `tools/wma-rca/*`, `tools/wma-evolve-hook`, and tests. All are offline
analysis/meta code and `skills/wma_meta` is never installed into a scientist cell (all 15 cells carry
exactly `task/.claude/skills/exp_protocol`) — **except `awm/wma_client.py`**, which adds
`say() = print(…, flush=True)` as the default `out=` for `wait_for_verdict`/`review_and_wait`, making the
backgrounded lock log unbuffered. That is scientist-visible: it is why scientists can `tail logs/lock-*.log`
and poll for delivery. The runtime-relevant comparison is the *top* commits, and there it resolves
cleanly: `6f6a5755 → 48748384` (w10r05–08 + all 3 controls vs G) differs **only** by
`awm/wma_evolve_timer.py` and its test, unreachable from a cell — so the baseline ext half and the
controls are effectively runtime-identical to G. **w10r01–04 (`76d75dec`) additionally carry the
print-flush difference**, which is one more reason the baseline arm is two sub-batches, not eight
identical cells.

All 20 cells
share `status.awm_sha ae46724`, `ptb_commit 62203e49`, model, effort, 1M context, cli 2.1.219,
10 h budget, 1 GPU, `gemma-3-4b-pt @ cc012e0a`, container shas, `judge_profile official`. Verdict
skill hashes are clean: 58/58 w10 = `176f0a464986`, 25/25 G = `e4402ffa6bca`.

Two provenance caveats worth recording:
- **`source.top_commit` splits every arm 4+4** (c10r01–04 & w10r01–04 `76d75dec`; c10r05–08 &
  w10r05–08 `6f6a5755`; w13r01–04 `48748384`). The scientist-facing protocol is constant but the
  harness tree is not, so "n=8" is two 4-cell sub-batches per arm.
- G's sidecar `wma_runtime.checkout_sha` is `125a434e`, not `ae46724`. Provenance alone cannot
  certify skill-only difference; the operator's `git diff ae46724 125a434 --` over every non-skill
  `WMA_PRIVATE_SHIP` path returning empty is what certifies it. Keep both facts together.

**Stratification fence — three live skills, never to be pooled.**

| skill hash | what | runtime | scientist / WMA | cells |
|---|---|---|---|---|
| `176f0a464986` | v0.2 baseline | `ae46724` | Opus5[1m] high | w10r01–08 |
| `e4402ffa6bca` / `a536a0af24d7` | G / H, one edit each vs v0.2 | `ae46724` | Opus5[1m] high | w13r01–04 / w14r01–04 |
| `17be8a23046a` | **redesigned skill** (introduced `4bd8ed51`) | `31b854bb` | Opus4.8 high 200k | 60-cell crossbench, 0 complete |

The crossbench cohort differs in skill, runtime, scientist model, WMA model, context window and
benchmark set **simultaneously**. Nothing in it may be read as a WMA-skill effect.

---

## 2. Ledger, uptake, timing, score, cost, harms, PTB

### 2.1 Ledger by skill hash (independently reproduced — 22/22 operator numbers PASS)

| | G `e4402ffa6bca` | v0.2 `176f0a464986` |
|---|---:|---:|
| n / n_scored / n_leak | 25 / 21 / 4 | 58 / 36 / 22 |
| L0_hit | 1.000 (21/21) | 0.972 (35/36) |
| L1_hit | 1.000 | 1.000 |
| L2_coverage | **0.722** (13/18) | **0.900** (27/30) |
| L2 width mean / over noise | 0.1441 / **5.5606** | 0.1185 / **4.4546** |
| L3_hit | 0.647 (11/17) | 0.704 (19/27) |
| gpu_h_saved / wrongly_killed | 0 / 0 | 0 / 0 |
| cost sum / mean / wall min | $47.6701 / $1.9068 / 5.946 | $114.6868 / $1.9774 / 6.2591 |

Reconciles exactly to `verdicts.py` raw coverage (G 18/25, baseline 47/57) once the access fence
(`ledger.summarize:145`), the self-measurement rule (`schema.score:299`) and the
`delta_vs_comparator` fallback (`schema.truth_from_card:247`, affecting only `w10r07/exp-07`) are applied.

### 2.2 The single largest finding: **L3 is a constant classifier and always has been**

L3 answer is `yes` on **25/25 G and 58/58 baseline — 83/83, zero `no`, zero `defer`, across two skill
hashes.** Truth is `schema.WORTH` (`awm/wma/schema.py:39`): `adopt→True`, `reject→False`,
`abandon_line→False`, `iterate→unscorable`. Card decisions: G {adopt 16, reject 6, iterate 3};
w10 {adopt 36, reject 13, iterate 9}. Therefore **L3_hit is exactly the adopt base rate**:

- G 11/17 = the 11 remaining clean adopt cards and **0 of 6** clean reject cards
  (`w13r01/exp-05`, `w13r01/exp-06`, `w13r02/exp-06`, `w13r02/exp-07`, `w13r04/exp-04`, `w13r04/exp-05`)
- v0.2 19/27 = 19 adopt and **0 of 8** clean reject cards
  (`w10r03/exp-05`, `w10r03/exp-09`, `w10r04/exp-06`, `w10r05/exp-05`, `w10r06/exp-04`,
  `w10r06/exp-07`, `w10r07/exp-05`, `w10r08/exp-03`)

**Discrimination on reject-truth cards: 0/14 clean, 0/19 including flagged.** The WMA has never once
identified a card the scientist would go on to reject. `gpu_h_saved` and `gpu_h_wrongly_killed` are
then **0 by construction** — `ledger.summarize` accumulates only rows with `L3_answer in ("no","defer")` —
not empirically. They carry no information about G, and never have, in any round or any `--by type` slice.

### 2.3 Leak flags — G removed the classes it named

G 4/25 = 0.160 vs baseline 22/58 = 0.379 (Fisher two-sided **p = 0.070**). All four G flags are
`/home/ben`-directed capacity/existence reads: `w13r02/exp-05` (`df -h /session /home`),
`w13r03/exp-01` (`ls -d /home/ben`, `.../hf_cache`), `w13r03/exp-04` (`ls -la /home/ben/`),
`w13r03/exp-05` (`df -h /session /home/ben`). **Zero G verdicts read installed package trees** —
the baseline's `/usr/local/lib/python3.10/dist-packages/{vllm,inspect_ai,inspect_evals}` class and its
`grep -rl … /history` class both go to zero. The residual is the single under-specified case: a command
naming `/session` **and** a home directory together.

Plus the confirmed manual semantic exposure at **`w13r01/exp-01`**, missed by the scanner: 150 GSM8K
main-test rows exported by the scientist (trace 1843–1853) into `task/eval/exp-01_inputs.jsonl`
(42,004 bytes, 150 records, sha256 matches the frozen lock); WMA private transcript lines 78–79 load
all 150 and print two question/gold records into model-visible context before the card ran; final
evidence `e4` uses it. Score causality **not** established; recorded as a separate manual exclusion,
original ledger unchanged at 25/21/4.

**The fence removes disproportionately many hits.** Of the 7 G cards dropped from raw L2, **5 were raw
hits**; of the baseline's 28 dropped, **21 were**. Out-of-fence reads were *buying* calibration — the
mechanism by which G's guard and G's calibration trade off against each other.

The coverage loss is **not confined**: over the 10 change types with n≥3 in both arms, L2_coverage falls
in 8 and L3_hit in 7. Load-bearing rows are C3 (13 scorable, Δcov −0.237) and C4 (10, −0.209) with
width/noise flat (+0.13, +0.21) — genuine miscalibration, not narrower intervals. The two largest
deltas (C9 −0.323, C12 +0.100) rest on n_L2_scorable 5 and 1 and should not be read.

### 2.4 Uptake funnel — hand-counted, all 83 cards read

Funnel: G 42 requests → 42 responses → **25 retained finals** (17 overwritten); baseline 76 → 76 → 58 (18 overwritten).

| hand class | G | v0.2 |
|---|---:|---:|
| **RELOCK-CITED** (scientist wrote the reason into the relock) | **13/25 (52%)** | **15/58 (25.9%)** |
| acted, not cited | 5 | 32 |
| read, no action | 4 | 7 |
| never read | 3 | 4 |
| **acted at all** | 18/25 (72%) | 47/58 (81%) |

**The difference is legibility, not volume.** The baseline acts on advice inside the card body
without re-locking (32/58 = 55%); G writes the reason into the relock. G relocks 56% of cards vs
27.6%. Since relocks are what make uptake auditable, G's loop is far more legible — this is the one
axis on which G clearly beats v0.2, and it is not the axis G was preregistered on.

The mechanical `adopted` class is an upper bound and is scored only on the *retained* verdict.
Hand-verified false positives: `w13r02/exp-02` (regex hit the verdict-printing command; the launched
argv has no `--save-steps`), `w13r02/exp-06`, `w10r02/exp-06`. False negatives: `w10r01/exp-05`,
`w10r02/exp-07`. Three G cards mechanically `never-read` are RELOCK-CITED — the *first* verdict was
acted on and the re-review was then never opened.

**Timing.** Per-request latency is indistinguishable between arms (~360 s median): G 42 requests,
median 359.6 s, total **4.19 h**; baseline 76 requests, median 364.4 s, total **8.06 h**. The lock
files understate true review wall time by **77% (G, 2.37 → 4.19 h)** and **38% (baseline, 5.83 → 8.06 h)**
because the overwritten reviews are invisible to `uptake.py`. The arm difference is in *how many*
requests, not latency.

**Gate cost is not GPU idle, and is small.** Merging launch intervals against each wait window with
the corrected detector: G 1.66 h of 2.37 h wait had no job in flight (70%); baseline 3.52 h of 5.83 h
(60%). Per cell that is ~25 min of unoverlapped scientist time out of a ~14 h budget — **~3%**.

### 2.5 Score effects — GSM8K / gemma-3-4b-pt only, never pooled

| arm | n | mean | SD | range |
|---|---:|---:|---:|---|
| control (no WMA) c10r01–08 | 8 | **74.9716%** | 4.3408 pp | 68.7642 – 81.5011 |
| WMA v0.2 w10r01–08 | 8 | **72.3180%** | 2.7382 pp | 68.8400 – 77.1797 |
| WMA G w13r01–04 | 4 | **73.5406%** | 2.6897 pp | 70.2047 – 76.2699 |

- **G − v0.2 = +1.2225 pp.** Welch t=0.7378, df=6.20, **p=0.4876**, 95% CI **[−2.8005, +5.2456]**;
  exact Mann-Whitney **p=0.4606**. All four G cells lie **strictly inside** the w10 range; two w10 cells
  individually beat G's arm mean.
- **G is 1.4310 pp BELOW the no-WMA control** (Welch p=0.5005, CI [−6.0313, +3.1692]).
- **control − v0.2 = +2.6535 pp** (the only 8-vs-8 contrast): Welch t=1.4624, df=11.81, **p=0.1697**,
  95% CI **[−1.3071, +6.6141]**; MW exact **p=0.2345**. Null; not a causal harm claim.
- Power: achieved **0.274** (2.65 pp at n=8), **0.135** (1.22 pp at n=8), **0.084** (n=4).
  For 80% power you need **~31 cells/arm** at 2.65 pp and **~82 cells/arm** at 1.2 pp.

**G's +1.2225 pp is fully absorbed by a lever nobody controlled.** G trained **0.994 h less** than w10
(3.38 ± 0.27 h vs 4.37 ± 0.52 h) — a **1.91 w10-SD** gap, 4.3× the size of the 0.45-SD score gap. The
arm-demeaned OLS slope over the 16 R02 cells (−1.3286 pp/h) predicts **+1.3203 pp = 108% of the
observed difference** from that one lever alone. Fewer cards (+0.49 pp) and fewer hours used (+0.31 pp)
over-explain it further. The slope is not significant (t=−0.99) and its sign contradicts Round 01's
+0.48 — this is not a causal claim, it is a demonstration that the gap is not attributable to the skill.

**Round 01's score levers do not replicate.** Only training volume clears significance over all 20 cells
(max_rows_k ρ=+0.556, epoch_rows_k ρ=+0.496), and epoch-rows degrades to **+0.332** once a
**`tools/wma-rca/cells.py` row-counting bug** is corrected: cards declaring `source: "derived: …"` are
excluded by the tool's regex, so 11 of 20 cells under-count rows and `w10r05` reads 0.0 despite training
~379.3k epoch-rows. First-SFT dev +0.60 → +0.239 (**dead**); training hours +0.48 → +0.010 (**sign-flipped**);
max eval n +0.49 → −0.005 (**flips between arms**). No within-arm |ρ| reaches the n=8 threshold of 0.738.

### 2.6 Cost and variance (results in their own right)

| arm | n | cost total | cost/cell | agent-h mean |
|---|---:|---:|---:|---:|
| control | 8 | $284.69 | $35.59 | 8.16 |
| v0.2 | 8 | $375.76 | $46.97 | 8.35 |
| G | 4 | $195.17 | $48.79 | 7.69 |

Running a WMA costs **+$11.38/cell (+32%)** over no WMA. WMA-side per-verdict cost is $1.9068 (G) vs
$1.9774 (v0.2): all-finals ratio **0.9643**, unflagged-finals **0.9899** — the ≤1.5× guard is satisfied
**on retained finals only**; 17 overwritten reviews make this a lower bound.

### 2.7 Harms — all 17 overwritten reviews recovered (17/17)

Recovered from `task/logs/lock-exp-NN{,b,c,d}.log` (11) and the trace echo (6); private transcripts
hold only the last session, and no `.rejected` files exist.

**The recovery strengthens §2.2 rather than softening it: L3 = `yes` on 42/42 G reviews and 76/76
baseline reviews, not merely the retained 25 and 58.** `gpu_h_saved = 0` therefore holds over every
review either arm ever issued. The recovery *does* revise L0/L1: true non-yes rate is G **2/42 (4.8%)**
vs baseline **4/76 (5.3%)** — the arms are indistinguishable on willingness to say no, and both say it
only at L0/L1, never at L3.

- **Heeded-no harm: 0 instances in either arm, and 0 opportunities.** All six L0/L1 "no"s produced
  modify-and-proceed; every modified card then ran to completion. No card in either arm was ever
  abandoned or deferred because of a WMA verdict.
- **Blocker value, both operator claims CONFIRMED** with trace citations: `w13r04/exp-02`
  (`L0=no@0.58`, 95 GB logits arithmetic, loss rewrite, crediting relock → delta **+0.6467**) and
  `w13r04/exp-04` (`L0=no@0.8; L1=no@0.75`, scientist independently **reproduced** `ValueError
  GenerationConfig is invalid`). A third, missed by the operator: `w13r01/exp-02` pre-empted the same
  save class before `exp-03`'s 1.54 h run.
- **The one G miss, confirmed:** `w13r03/exp-03` got `L0=yes@0.9; L1=yes@0.85`, its four preconditions
  flagged dtype/regression/few-shot/watch-set and *not* the save path, and it lost **1.151 h** observed
  training wall. Not attributable to the probe-scope rule — the same fact is available in-scope from
  the manual §4 pitfall table, which is exactly how G's own `w13r04/exp-04` derived it. Baseline's
  cumulative loss in this class is comparable (0.42 h at `w10r08/exp-04` **after two explicit
  warnings**, 0.05 h at `w10r04/exp-02`), and G has no warned-and-still-lost counterpart.
- **Over-optimistic L2 → wasted run: G 4/21 (2.37 h) vs baseline 4/50 (2.11 h).** Three of G's four are
  sub-0.03 shortfalls inside the n=150 noise floor. Only **`w13r02/exp-02`** is substantive — and it is
  the sharpest single finding in the harm set: the *same verdict's* `cheaper_variants[1]` named the
  exact failure mechanism that then occurred and priced a probe at ~12 min, yet the verdict returned
  L2 higher[0.15,0.55]@0.6 and L3 yes@0.88. The correct blocker existed at suggestion tier and was
  priced as neither an L2 downgrade nor an L3 no. Downstream books the whole 1.3 h run as a cost.
- **G's rule closed the channel it targeted:** 0/25 G final verdicts cite an installed-package evidence
  path vs **5/58** baseline; `dist-packages` mentions 9/25 vs 80/58. Four verdicts contain the rule's
  explicit refusal, and in all four the escape hatch worked — each became a costed precondition the
  scientist acted on. **No suppressed blocker is demonstrated.**
- **But the falsification channel is `unconfirmed`, not cleared.** The one artifact that could
  adjudicate whether the rule suppressed a probe — `w13r03/exp-03`'s *first* private transcript — was
  destroyed by its own relock. Per spec lines 151–153 that channel does not pass.
- Self-reported lost GPU-h: G 5.68 h / 25 cards (0.227 h/card) vs baseline 10.63 h / 58 (0.183 h/card).

---

## 3. Ranked causal diagnosis

**C1 — L3 carries no information, and never has. (harness + skill, high confidence)**
Constant `yes` on 83/83; hit rate = adopt base rate; both GPU-hour columns structurally 0; discrimination
0/14 on clean reject-truth cards. Two candidate causes are **not yet separated**: (a) *skill* — the L3 prompt
("is this the right use of the next hours?") admits a trivially-yes reading; (b) *harness* — `WORTH` equates
`reject` (the experiment's **result** was negative) with "not worth running", i.e. sunk-cost reasoning encoded
as ground truth, and a correctly calibrated world model *should* say yes to an informative experiment that
turns out negative. Counterevidence to pure-(b): 19 reject cards is a large, stable class and some were
plausibly knowable at lock time. Counterevidence to pure-(a): the definitional mismatch is real and
unaddressed. **Separating them costs zero GPU and must precede any L3 candidate.**

**C2 — `verdict_before_launch` is mismeasured. (harness, high confidence)**
`rcalib.py:274` + `:27`. Adjudicated truth is **G 25/25, baseline 58/58, zero true bypasses** against a
reported 2/25 and 18/57. The blocking lock works; the instrument was reporting a ~90% violation rate
that does not exist. Every prior compliance claim derived from this statistic is wrong in the same
direction, and no round's number was ever a skill signal.

**C3 — G's probe-scope rule works on what it names and fails on what it under-specifies. (skill, medium)**
Package-source and history reads → 0; all 4 residual flags are the "names `/session` and a home directory in
one command" case. Counterevidence: n=25 verdicts from 4 cells, Fisher p=0.070 is not significant, and the
preregistered guard was **zero**, so any residual falsifies regardless of rate.

**C4 — the same fence costs calibration. (skill, medium, adverse)**
L2 coverage 0.900 → 0.722 (p=0.132) spread over 8 of 10 comparable types with widths flat. Mechanism, not
just correlation: the fence drops disproportionately many *hits* (5/7 G, 21/28 baseline). Redundant
explanation not excluded: G's 4 cells are a smaller, self-selected, differently-configured draw.

**C5 — L2 intervals are wide *and* wrong. (design, high confidence)**
width/noise 4.4546 (v0.2) and 5.5606 (G) — roughly 5× the noise floor — while still missing 10–28% of cards.
No candidate to date targets width. This is a finding, not yet a candidate: there is no mechanism proposed.

**C6 — the in-flight crossbench cannot attribute anything to the skill redesign. (design, high confidence)**
Skill `17be8a23046a`, runtime `31b854bb`, scientist model, WMA model, context window and benchmark set all
change at once. Whatever it returns, no cell in it will isolate `skills/wma/`.

**C7 — score effects are unreachable at this replication. (protocol, high confidence)**
Within-arm SD 2.7–4.3 pp; the largest contrast in the cohort is null at p=0.17; 31–82 cells/arm are needed.
**PTB score must be used as a guard, never as a primary metric, for any skill candidate.**

**C8 — measurement tooling has at least three independent defects.** `launch_time()` (C2), the `cells.py`
`derived:` row-count exclusion (§2.5), and the undisplayed denominator on the GPU-hour columns (§2.2).
The instrument is currently a larger error source than any skill edit under test.

---

## 4. Promotion

**None is justified.** Explicitly: no candidate is promoted from this window.

- **G fails its own preregistered primary readout** — `doc/spec/2026-09-03-wma-round04-probe-selection.md`
  requires "Zero original-fence flags and zero outside mutations". Four original access flags survive.
- **G independently trips its own falsification clause** — "G is falsified if … an apparent validity gain
  comes from … indirect access that merely evades path detection": the confirmed `w13r01/exp-01` held-out-input
  exposure was missed by the path scanner entirely.
- **The window is provisional by rule** — 4 clean-complete cells < 8, so no promotion may be recommended from
  it regardless of the flags. Post-hoc power for its headline difference is 0.084.
- **The +1.2225 pp is not G's** — it is fully absorbed by G's 0.994 h training-hours deficit, and G still sits
  1.4310 pp below the no-WMA control.
- **H has no readout** — jobs 91445–91448 all RUNNING at 07:59:06 / 07:56:01 / 07:50:35 / 07:49:55, 0/4 complete.
- A *modified* G is a new candidate requiring its own preregistration and cells, not an extension.

---

## 5. Next wave — independent single-edit candidates

Constraints: the queue is **16/16 GPUs allocated with 46 pending** Opus 4.8 cells; H returns within roughly
the hour and is the next scientific readout. Every skill candidate is one edit against v0.2 `176f0a464986` on
runtime `ae46724`, four repeats per manifest, 8 cells = base `x4` + `-ext-x4`. **The wave leads with the
zero-GPU items and holds every GPU candidate until H reports and capacity frees.**

### N1 — zero-GPU adjudication: is L3's ground truth wrong? *(precondition, not a candidate)*
- **Mechanism.** For each of the 19 reject-truth cards (14 clean, listed in §2.2), read `conclusion` and the
  lock-time state and classify **knowable at lock** (redundant, dominated, misspecified, or an already-rejected
  line) vs **unknowable at lock** (a well-posed experiment whose result was negative).
- **Primary metric.** Knowable share.
- **Preregistered decision rule.** ≥ 0.60 knowable → skill cause, N2 authorized. ≤ 0.40 → harness cause,
  `WORTH` is the defect and **no skill edit is authorized**. Between → both causes; neither is authorized.
- **Falsification.** If two independent readers agree on < 80% of cards, the classification is unusable and
  the question stays open.
- **Guards.** Read-only over frozen artifacts; no GPU, no cells, no definition changed.
- **Replication.** Two independent readers over the same 19 cards; disagreements listed, never averaged.
- **Baseline.** n/a.
- This is **not** the rejected P3 ("redefine saved") or L ("raise the no rate"): it changes no definition and
  no rate, it measures whether the existing definition is answerable at all.

### N2 — skill candidate: make L3 discriminate *(conditional on N1 returning "skill cause")*
- **Edit.** One change to the `L3_worth_now` row of `skills/wma/SKILL.md`: require the verdict to name, from
  the card's own `situation`, the specific condition under which the answer would be `no`/`defer`, and to
  answer `no`/`defer` when that condition holds. Skill only; no protocol change.
- **Primary metric.** L3 hit rate on **reject-truth** cards (currently 0/14 clean, 0/19 all).
- **Harm metric.** L3 hit rate on **adopt-truth** cards (currently 30/30 clean, by construction).
- **Falsification.** Falsified if reject-truth hits stay at 0, **or** adopt-truth hit rate < 0.85, **or**
  `gpu_h_wrongly_killed > 0`.
- **Guards.** Leak-suspected verdicts 0; accepted-verdict cost ≤ 1.5× matched v0.2; PTB not below
  72.3180% beyond the observed 2.7382 pp spread; no unjustified killed work; verdict-before-launch preserved
  (measured with the **repaired** detector from N3); all timeouts/skips reported.
- **Replication.** 8 cells: `wma-gsm8k-gemma4b-high-r05-n2-l3-condition-x4` + `-ext-x4`, four repeats each.
- **Baseline.** v0.2 `176f0a464986`, runtime `ae46724`, w10r01–08.
- **Analysis gate.** At 8/8 clean-complete: promote only if reject-truth hit rate ≥ 0.35 on n ≥ 10 reject-truth
  cards **and** adopt-truth hit rate ≥ 0.85 **and** all four guards hold. Anything else → no promotion.

### N3 — harness repair, zero GPU: fix `launch_time()`
- **Edit.** `tools/wma-rca/rcalib.py`: exclude the lock command itself and card-file edit heredocs from the
  launch detector; anchor to lock `completed_at` from `task/.wma/responses/*.json` rather than the lock call's
  start; drop the `family == "other"` wildcard or require the card's own output path.
- **Primary metric.** Recomputed `verdict_before_launch` per arm. The hand-adjudicated target is
  **G 25/25 and baseline 58/58** — the repair is only correct if it reproduces those.
- **Falsification.** If the repaired detector disagrees with the case-by-case adjudication on more than
  2 of the 83 cards, it is insufficient and must not be adopted.
- **Also fix:** anchor to the *per-lock* `locked_at` rather than the final lock's, so relock chains stop
  being scored against the wrong verdict; and treat `completed_at` as a +5.7…+29.7 s lagging proxy.
- **Guards.** No result directory, verdict, card, lock, ledger or score is modified. Old numbers retained
  alongside as superseded, never silently replaced. Per-cell classification must print.
- **Replication.** None — deterministic recomputation over frozen artifacts.
- **Analysis gate.** Adopt as the reporting metric only after all 17 exceptions are individually adjudicated.

### N4 — harness repair, zero GPU: make structural zeros visible
- **Edit.** Report `gpu_h_saved` / `gpu_h_wrongly_killed` with their denominator (count of rows with
  `L3_answer != "yes"`), so `0/0` is distinguishable from an empirical 0. Same for the `cells.py` `derived:`
  row-count exclusion: count and print the excluded cards rather than silently reading 0.0.
- **Primary metric.** Both columns print a denominator; the 11 under-counted cells print their exclusion.
- **Falsification.** None required — the causes are located (`ledger.summarize` filter; `cells.py` source regex).
- **Guard.** No definition of "saved" changes. This is **not** P3.

### N5 — de-confound the redesigned skill *(queued behind the crossbench harvest; do not launch now)*
The crossbench cannot attribute anything to skill `17be8a23046a`. The minimum de-confounder is one 4-cell
GSM8K/Opus4.8 arm shipping **v0.2's `skills/wma/`** on the frozen `31b854bb` runtime, everything else identical
to the crossbench GSM8K WMA cell. Manifests are frozen and the queue is full — record it as the first item
after the harvest, not as part of this wave.

**Held / not proposed.** A modified G: leak count is a *guard*, not a primary metric, and G's own evidence
says the fence costs calibration — spending 8 cells to shave 4 flags is poor value. An L2-width edit: C5 is a
finding with no mechanism yet. Anything from the rejected set P1–P4, K, L, M, N, J.

---

## 6. Operator handoff for Codex

**Verify (before anything else):**
1. That `/rmeng_data/robtang/wma-evolve-hook/gangda_wma_evolve/events/20260904T164313Z-792dc7f482/`
   still has **no `operator.status.json`**.
2. The three findings that are **new to the record** and absent from `evidence/2026-09-04-1630/`:
   L3 answer distribution 25/25 and 58/58 all `yes`; L3_hit ≡ adopt base rate with **0/14 clean reject-truth
   hits**; `gpu_h_saved`/`gpu_h_wrongly_killed` structurally zero via the `L3_answer in ("no","defer")` filter
   in `ledger.summarize`.
3. The `launch_time()` defect at `tools/wma-rca/rcalib.py:274` (+ regex `:27`) and the adjudicated
   **G 25/25 / baseline 58/58 with zero true gate bypasses**; and the `cells.py` `derived:` row-count
   exclusion affecting 11 of 20 cells. **This is good news the record currently reports as bad news.**
5. That "one edit" is true of `skills/wma/` but not of the shipped archive — `awm/wma_client.py`'s
   unbuffered `say()` is scientist-visible, and `w10r01–04` differ from G by it while `w10r05–08` and
   the controls do not.
6. That the G falsification channel is **`unconfirmed`, not cleared**: `w13r03/exp-03`'s first private
   transcript, the one artifact that could adjudicate spec lines 98–100, was destroyed by its own relock.
   This is a standing argument for preserving superseded reviews rather than overwriting them.
4. That G's `+1.2225 pp` is absorbed by its 0.994 h (1.91 w10-SD) training-hours deficit, and that G is
   1.4310 pp **below** the no-WMA control.

**Edit:** nothing in `skills/wma/` this round. The only edits this analysis authorizes are the zero-GPU
harness repairs **N3** and **N4**, each as its own commit, with superseded numbers retained beside the new ones.

**Launch:** nothing. 16/16 GPUs are allocated with 46 pending; H (91445–91448) is the next readout; no skill
candidate is authorized until N1 returns.

**Record:** in `doc/wma_iterations/2026-09-03-round-04-online.md` and a new `evidence/2026-09-04-<ts>/` —
the L3 constant-classifier finding with its exact denominators and card lists, the `launch_time()` and
`cells.py` defects with before/after per arm, the three-skill stratification fence, the G-vs-control result,
and an `operator.status.json` for this event marking the handoff **executed** with **promotion `None`**.

**Then:** harvest H through `awm ptb results` when its four cells land — Slurm state is not the criterion
(all four G cells read FAILED while validator-clean).

---

### Coverage note

**All five specialist dimensions are complete**: ledger by skill hash, uptake/timing (83 cards hand-read),
score levers, harm cases (17/17 overwritten reviews recovered), and compliance/provenance (all 17 residual
gate exceptions adjudicated). Every claim above is anchored to a cell, card, manifest, receipt or result
directory.

Two of my own earlier figures were superseded by the adjudication and are corrected in place:
verdict-before-launch is **25/25 and 58/58 with zero true bypasses**, not the 19/25 and 47/58 I first
computed; and the Slurm epilogue failure has **two** causes, not one. Both corrections make the protocol
look *better*, and neither touches §4 — G's promotion gate fails on its own preregistered zero-flag
criterion and on the n<8 rule independently of anything measured here.
