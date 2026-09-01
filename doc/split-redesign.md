# Redesigning `gsm8k-gemma-holdout-v1`

Status: **proposal**, 2026-08-30. Nothing here is committed to `splits/` yet.
Every number below is reproducible with `tools/splitdx/` (see the last section).

Reconciled against the project pitch on 2026-09-01: where this record contradicts
`doc/iclr-27-idea.md`, the contradiction is written up in that document's
2026-09-01 changelog entry, and each affected claim there carries a footnote
pointing at it. Six premises did not survive; §3's architecture, §4's tower and
§1's positioning are untested here, not refuted.

## Why the shipped split has to change

`splits/posttrainbench/gsm8k-gemma-holdout-v1` holds out the *base model*: 50 test
runs, all gsm8k, all `google_gemma-3-4b-pt`. Measured on the catalogue:

| what test shares with train | share |
|---|---|
| `agent_model` seen in train | **98 %** |
| `agent_family` seen in train | **98 %** |
| same `experiment` as a train row | **96 %** |
| `trained_model` seen in train | 0 % |

The one dimension it closes explains **7.5 %** of accuracy variance. The dimension
it leaves 98 % open explains **66.3 %**. So a per-agent lookup table — no
trajectory, no learning, one `dict` — scores Spearman **0.7507** and

> **top-3 regret 0.0000.**

Top-k regret is the primary metric in `doc/iclr-27-idea.md` §6. A learned
predictor cannot beat zero; the best it can do is tie. The split is saturated.

Two smaller problems come with it:

* **The target contradicts §3.** The idea doc says predict *Δ, not the absolute
  score*, because "absolute score is dominated by how hard is this task and how
  strong is this base model". The shipped split predicts absolute accuracy. On
  the full corpus `benchmark` alone explains 47.9 % of absolute-accuracy variance.
* **The choice set is one set of 50.** Picking 3 of 50 candidates on one
  (benchmark, base model) cell is not a decision anyone makes, and a single
  choice set means the metric has no variance to average over.

## Three axes, not one

The saturation is usually described as "the holdout is wrong". It isn't only the
holdout. Three independent choices each set the answer, and each can saturate a
design on its own:

1. **Split** — what is held out.
2. **Target** — what number is predicted (absolute accuracy, or Δ against a
   reference).
3. **Choice set** — the group of candidates the top-k pick is made within.

Sixteen designs were measured across all three (`tools/splitdx/designs/`). Four
structural facts came out, and they constrain what is even worth proposing.

### 1. A base-model holdout and a cell-referenced Δ are mutually exclusive

If Δ is measured against the train median of the row's (benchmark, base model)
cell, and the split holds out a whole base model, then no test row's cell exists
in train. **0 of 50 test rows get a label.** The combination is not "bad", it is
undefined. (Design OWNER-4.)

### 2. When the choice set *is* the cell, a cell-referenced Δ cannot move top-k regret

Δ against the cell median subtracts the same constant from every member of a
cell. A within-set monotone shift leaves the within-set *ranking* untouched, so
every top-k regret is byte-identical to the absolute-target version. Verified on
two pairs that differ only in target — OWNER-10/11 and OWNER-12/13 — identical at
every k for every baseline, and again per-fold in the 5-fold design below
(0.0231 / 0.0469 / 0.0444 / 0.0259 / 0.0272 under both targets).

This does **not** make the Δ target pointless. It changes Spearman, RMSE and
calibration, and that is where it earns its place — see the recommendation.

### 3. A small choice set manufactures a passing score

regret@3 inside a set of 3 is zero by construction. Four designs "passed" this
way before the choice-set sizes were printed:

| design | median choice set | regret@3 |
|---|---|---|
| OWNER-1 | 2 | 0.0000 |
| OWNER-5 | 1 (13 singletons of 24) | 0.0000 |
| OWNER-6 | 2 | 0.0000 |
| OWNER-9 | 1 (5 singletons of 6) | 0.0000 |

Any proposal has to report the choice-set size distribution next to the regret,
or the regret means nothing. One of the five independently-proposed designs
("hack-aware arenas", median 4 candidates) scores regret@3 = 0.0023 and fails for
exactly this reason.

### 4. A tuned metadata model loses to a parameter-free lookup

A `HistGradientBoostingRegressor` over the five metadata columns, with its
capacity chosen by configuration-grouped CV *inside train* (never test) and
averaged over five 85 %-resampled fits, **loses to the best parameter-free lookup
on 14 of 16 designs**, and on 5 of 5 folds of the recommended design. Metadata is
not the bottleneck; there is real room above it for something that reads content.

## The measurement that ranks the designs

Comparing a baseline's regret to a per-run standard deviation is not a
comparison — one is a max-order statistic over a choice set, the other is a
spread around a single run. The quantity that actually bounds the headroom:

* **dumb@3** — the best parameter-free lookup's top-3 regret. A model has to beat
  this, so it is the top of the useful range.
* **floor@3** — the top-3 regret a **perfect** predictor still pays, because the
  labels carry re-run noise. Simulated from the replicate groups of the design's
  own test set (`_run2` / `_run3` / `_old_container` are replicate suffixes; 58
  experiments collapse to 33 configurations). Nothing can go below it.
* **winnable** = dumb@3 − floor@3, reported as a share of the mean accuracy
  spread inside a choice set — because an absolute regret gap means different
  things on a benchmark where runs differ by 40 points and one where they differ
  by 4.

A design whose winnable share is zero or negative is saturated, whatever its
holdout looks like on paper.

## The comparison

Full output in `tools/splitdx/compare.out`. `lk_*` are leakage shares.

| win% | winnable | dumb@3 | floor@3 | gbdt@3 | n_test | sets | med | lk_ag | lk_exp | design |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **13.8 %** | +0.0598 | 0.0619 | 0.0021 | 0.0865 | 297 | 28 | 11 | **0 %** | **0 %** | OWNER-12 — agent families ≈25 % of mass, absolute |
| **13.8 %** | +0.0598 | 0.0619 | 0.0021 | 0.1457 | 297 | 28 | 11 | **0 %** | **0 %** | OWNER-13 — same split, Δ vs cell |
| 9.9 % | +0.0545 | 0.0593 | 0.0048 | 0.0894 | 614 | 28 | 23 | 35 % | 0 % | OWNER-16 — blocked by configuration (50 %), Δ |
| 8.9 % | +0.0397 | 0.0447 | 0.0050 | 0.0749 | 103 | 7 | 14 | 50 % | 0 % | OWNER-14 — configuration **and** base model |
| 8.8 % | +0.0511 | 0.0669 | 0.0158 | 0.0766 | 63 | 4 | 15 | 0 % | 0 % | OWNER-2 — gsm8k, 3 largest agent families out |
| 7.8 % | +0.0470 | 0.0478 | 0.0008 | 0.0642 | 59 | 4 | 14 | 32 % | 0 % | OWNER-3 — gsm8k, blocked by configuration |
| 6.1 % | +0.0304 | 0.0386 | 0.0082 | 0.0179 | 305 | 7 | 44 | 100 % | 97 % | OWNER-7 — the shipped rule, widened to 7 benchmarks |
| 6.0 % | +0.0373 | 0.0405 | 0.0032 | 0.0774 | 113 | 8 | 14 | 35 % | 0 % | OWNER-15 — verifiable benchmarks only |
| 5.2 % | +0.0280 | 0.0298 | 0.0018 | 0.0599 | 392 | 28 | 14 | 64 % | 0 % | OWNER-10/11 — blocked by configuration (30 %) |
| **−1.3 %** | −0.0083 | 0.0061 | 0.0144 | 0.0083 | 172 | 4 | 43 | 100 % | 100 % | OWNER-8 — hold out a whole benchmark |
| **−3.3 %** | −0.0245 | 0.0000 | 0.0245 | 0.0634 | 50 | **1** | 50 | 98 % | 96 % | **CONTROL — the shipped split** |
| — | — | — | — | — | 50 | — | — | — | — | OWNER-4 — **UNDEFINED**, 0/50 labelled |

Reading the two negative rows: the dumb baseline already scores *below* the noise
floor. That can only happen when the choice-set structure hands it the answer —
which is the sharpest available statement that a split is saturated.

Widening the shipped rule to all seven benchmarks (OWNER-7) does help — 6.1 % —
but it leaves `agent_model` and `experiment` leakage at 100 % / 97 %. It buys
headroom by adding rows, not by closing the leak.

## The knob nobody was reporting: which families you hold out

Two of the five independently-proposed designs are the same shape as OWNER-12/13
and report dumb@3 of **0.0338** and **0.0453** where this one measures **0.0619**.
Same rule, different rosters, and the number moves by a factor of two.

So: fix the rule, vary the roster over every draw the rule allows, and look at
the spread (`tools/splitdx/roster.out`). Thirteen rosters, each ~25 % of runs:

* dumb regret@3: median 0.0380, range **0.0104 – 0.0711**
* winnable share: median 7.1 %, range **1.7 % – 17.3 %**
* **every roster positive**

The *sign* is robust; the *magnitude* is a factor of ten. Quoting 13.8 % as the
headline would be tuning on the metric the split is supposed to report.

## Recommendation: a 5-fold agent-family partition, Δ vs cell

Remove the roster knob rather than choosing a value for it. Partition all 26
agent families into 5 folds (greedy largest-first into the lightest fold — no
seed, no search). Every family is in test exactly once; **every one of the 1,175
runs is scored exactly once out-of-fold**; the fold spread becomes the error bar
instead of a degree of freedom.

* **Split** — hold out whole agent *families*, so `claude-opus-4-6[1m]` never
  trains while `claude-opus-4-6` tests. Leakage: `agent_model` 0 %,
  `agent_family` 0 %, `experiment` 0 %.
* **Target** — Δ against the train median of the (benchmark, base model) cell.
  Every cell is in train by construction, so all labels are defined.
* **Choice set** — the (benchmark, base model) cell. 28 sets, median 8–9, no
  singletons, median 6 distinct configurations per set — the candidates are
  genuinely different recipes, not replicates of one.
* **Excluded columns** — `stderr`, `num_turns`, `total_cost_usd`, `duration_ms`,
  `time_taken`, `session_count`. All six are written after the run finishes and
  two of them beat the honest baseline; see the next section for the measurement.

`tools/splitdx/kfold.out`:

| fold | n_test | dumb@3 | floor@3 | winnable | win% |
|---|---:|---:|---:|---:|---:|
| 0 | 235 | 0.0231 | 0.0004 | +0.0227 | 5.0 % |
| 1 | 234 | 0.0469 | 0.0012 | +0.0457 | 12.1 % |
| 2 | 237 | 0.0444 | 0.0019 | +0.0425 | 11.2 % |
| 3 | 234 | 0.0259 | 0.0028 | +0.0231 | 6.3 % |
| 4 | 235 | 0.0272 | 0.0028 | +0.0244 | 7.6 % |
| **pooled** | **1175** | **0.0335** | **0.0018** | **+0.0317** | **8.5 %** |

Fold spread 5.0 – 12.1 %, sd 2.8 points, **all five positive**. Identical under
both targets, as fact 2 requires.

**Why Δ, given that it cannot move top-k regret.** It moves everything else, and
it moves it in the direction §3 asks for:

| | absolute | Δ vs cell |
|---|---:|---:|
| one-way R² of `benchmark` on train | 0.4788 | 0.1308 |
| one-way R² of `trained_model` on train | 0.0136 | 0.0065 |
| metadata GBDT Spearman, pooled over folds | **+0.691** | **+0.129** |
| metadata GBDT beats the dumb lookup | 0 / 5 folds | 0 / 5 folds |

Under the absolute target a metadata-only model reaches Spearman +0.69 by
learning which benchmark is easy. Under Δ that shortcut is gone and the same
model drops to +0.13 — while 97 % of the Δ variance is still explainable from the
replicate groups, so the signal has not been removed, only the shortcut. That is
the configuration a trajectory-reading model needs in order to be attributable.

**Cost:** free. No new runs; a re-partition of the 1,175 already in the
catalogue.

## The clause every design above is missing: excluded columns

A split decides *which rows* a model may learn from. None of the 17 designs
above — including the recommended one — says anything about *which columns* it
may read, and the PostTrainBench catalogue row is written **after** the run
finishes. Six of its fields are post-execution facts, and one of them is the
label in disguise.

`stderr` is not correlated with the label; it *is* the label:

```
stderr = sqrt(accuracy * (1 - accuracy) / n)
```

with `n` the benchmark's item count. Solving `n` back out returns a constant to
machine precision — aime2025 **29**, gpqamain **447**, gsm8k **1318**,
humaneval **163** (relative spread ~1e-15; arenahardwriting and healthbench are
not fixed-`n` and spread 0.75 / 1.58). Below p = 0.5 the map is strictly
increasing, so on any benchmark whose accuracies all sit under 0.5 — aime2025
(max 0.3000) and gpqamain (max 0.3973) — ranking by `stderr` is an **exact**
oracle: Spearman +0.9995 and +0.9999, regret@3 **0.0000**.

That is a fact about the column. The number that matters is what it costs the
recommended design, priced the way an attacker actually gets it — choosing per
benchmark, **on train only**, whichever ranker looks best:

| ranker | regret@3 | headroom over the 0.0018 floor | headroom destroyed |
|---|---:|---:|---:|
| best parameter-free lookup (the honest bar) | 0.0335 | **+0.0317** | — |
| `stderr` alone, pooled | 0.0388 | +0.0370 | none — *it looks harmless* |
| `stderr`, chosen per benchmark on train | 0.0274 | +0.0256 | **19 %** |
| all six post-hoc columns, chosen per benchmark | 0.0269 | +0.0251 | **21 %** |

The second row is why this nearly went unnoticed. Pooled over seven benchmarks
`stderr` scores *worse* than the honest lookup, because the average mixes two
benchmarks it solves exactly with five it is bad at. A single pooled number
declares the column safe. Measuring per benchmark, and then letting the attacker
pick per benchmark the way any real one would, reverses the verdict.

Every column, alone, against the 0.0335 bar (`tools/splitdx/stderr_leak.out`):

| column | coverage | regret@3 | vs honest | |
|---|---:|---:|---:|---|
| `num_turns` | 100 % | **0.0269** | −0.0067 | **leaks on its own** |
| `stderr` | 90 % | 0.0388 | +0.0053 | leaks *per benchmark*, see above |
| `total_cost_usd` | 100 % | 0.0378 | +0.0043 | no help alone |
| `time_taken` | 100 % | 0.0462 | +0.0127 | no help alone |
| `duration_ms` | 100 % | 0.0475 | +0.0139 | no help alone |
| `session_count` | 100 % | 0.0691 | +0.0356 | no help alone |

`num_turns` beats the honest lookup outright, pooled, with no per-benchmark
selection at all — a longer session is a harder task badly handled.

**So any split that ships must carry an excluded-columns list**, and it must
exclude all six, not only the two that leak today. The four that "do not help
alone" are still post-execution facts: they cost nothing to exclude, they are
selected into the full attack in every fold above, and whether they help is a
property of this corpus rather than of the design.

**How bad is this?** Not fatal, and it should not be overstated. The design
survives — +0.0251 of headroom is still positive under the worst attack — but
the honest headline drops from +0.0317 to +0.0251, and on **2 of 7 benchmarks**
the benchmark is simply solved. The correct claim is "the excluded-columns
clause is load-bearing and worth a fifth of the headroom", not "the whole
benchmark is a lookup on `sqrt(p(1-p)/n)`".

### Why masking the agent is not a substitute for holding it out

The obvious cheaper fix for §1 is to leave the split alone and simply delete
`agent_model` from the input. Measured on the shipped split, it changes nothing:

| lookup | keys | regret@3 | |
|---|---|---:|---|
| per-agent | `agent_model` | 0.0000 | masked away |
| per-agent-family | `agent_family` | 0.0000 | masked away |
| **per-experiment** | `experiment` | **0.0000** | **survives masking** |
| per-format | `trace_format` | 0.0844 | survives masking |
| global-mean | — | 0.1543 | |

`experiment` recovers the agent **100 %** of the time — all 58 experiments are
single-agent, as is every one of the 1,175 `run_name`s — and 48 of the 50 test
rows share an experiment with a train row. So the saturating lookup is still
there after masking, at exactly regret@3 = 0.0000, keyed on a different column.

Masking is also the weaker guarantee in principle. Holding out a family makes
the exploit *structurally impossible*: there is no train row with that agent to
look up, so a model that identifies the agent perfectly from the trajectory
gains nothing. Masking only makes it *inconvenient*: the row is still in train,
and the trajectory carries the agent's house style whether or not the name field
is present — re-identification is exactly the sort of thing the model under test
is good at. An honour-system constraint on a benchmark whose whole purpose is to
be attacked is not a constraint.

**The two compose, and both are needed.** Hold out families (structural), *and*
list the columns nobody may read (procedural) — and that list has to include
`experiment` and `run_name` alongside the six post-execution fields, because
they are agent identity under another name.

## The confound no split can fix

Worth stating plainly, because it bounds everything above: **all 58 experiments
are single-agent.** "Which agent produced this run" and "which recipe this run
followed" are the same variable in 1,175 of 1,175 rows. The corpus is fully
nested, not crossed.

No choice of split and no masking separates a nested pair — they are one column
wearing two labels. Only new rollouts that deliberately cross the two can: the
**same recipe executed by different agents**, and **different recipes executed
by the same agent**. Until then, any result of the form "the trajectory predicts
the outcome" is indistinguishable from "the agent predicts the outcome", which
§1 already showed is worth 66 % of the variance.

There is one hint that the crossing is worth paying for. At the coarse level
`recipe_signal.py` reads the recipes at, some pipeline signatures *are* shared
across agents — and they do not behave like one recipe:

| pipeline signature | distinct agent families | accuracy spread |
|---|---:|---:|
| `sft` | 19 | **0.8036** |
| `sft>merge` | 9 | 0.6679 |
| `sft>package` | 7 | 0.6027 |
| `sft>sft` | 7 | 0.4261 |
| `sft>sft>sft` | 4 | 0.1941 |

Nineteen different agents ran plain `sft` and landed 0.80 of accuracy apart. So
the algorithm family carries almost nothing; whatever signal exists is in finer
detail — data mix, dataset choice, LR, epochs — or in execution quality.

Note what is and is not to blame here. The extraction *schema* already records
that detail: `datasets[].dataset_id`, `n_examples`, `share`, `filtering`, and a
`hyperparams` block with `lr`, `epochs`, `batch_size`, `grad_accum`,
`max_seq_len`, `scheduler`, `warmup`, `weight_decay`, `precision`. What throws
it away is `feat()` in `tools/splitdx/recipe_signal.py`, which reduces each
record to nine coarse summaries before the probe ever sees it. Reading the
recipes at a finer grain costs no new extraction — only a different `feat()`.
Coverage does: 143 of 1175 rows. That is the reason the extraction has to be
re-run wide before a split is committed, and a first reason to believe a crossed
rollout would resolve something.

## The blocker this exposed, which is bigger than the split

A better split is necessary and not sufficient. The split diagnostics show
metadata is not *sufficient* to predict the outcome; the thesis needs the
stronger claim that trajectory content is *available* — that a model reading the
recipe knows something the catalogue row does not.

Measured on the only rows where recipes exist today (`recipe_signal.out`, the 143
gsm8k train rows of the shipped split, 3 cells, 24 agent families):

**Pass 1** ranked the recipe top: `dataset_names` R² 0.7917 against the best
metadata feature's 0.7029.

**Pass 2** killed that. `dataset_names` has 107 levels over 143 rows — it is
nearly a row identifier, and a one-way R² on a near-unique categorical is high
whatever the labels are. Against a within-cell permutation null it scores
**+0.0409 excess, z = +0.9** — nothing. Ranking features of different cardinality
by raw R² measures cardinality. Corrected, the honest ordering is `agent_family`
+0.5392 (z +13.1), then `n_inference_tricks` +0.4555 (z +23.8) and `n_datasets`
+0.4409 (z +15.8).

**Pass 3** is the one that matters, because the recommended split removes agent
identity from train — so the question is not "does the agent predict the score"
but "does the recipe predict it *once the agent is known*". Residualising the
within-cell Δ on `agent_family` and re-testing against a within-family
permutation null:

| feature | levels | R² | null | excess | z |
|---|---:|---:|---:|---:|---:|
| `n_datasets` | 9 | 0.0773 | 0.0396 | +0.0377 | +1.8 |
| `n_stages` | 9 | 0.0521 | 0.0341 | +0.0180 | +1.1 |
| `dataset_kinds` | 9 | 0.0599 | 0.0430 | +0.0170 | +0.9 |
| `uses_rl` | 2 | 0.0048 | 0.0016 | +0.0032 | +1.4 |
| `pipeline_signature` | 49 | 0.2090 | 0.1959 | +0.0131 | +0.3 |
| `n_discarded` | 14 | 0.0978 | 0.0889 | +0.0088 | +0.3 |
| `n_inference_tricks` | 6 | 0.0172 | 0.0204 | −0.0032 | −0.2 |

**0 of 7 survive.** On this slice every recipe feature collapses to the agent's
house style: which agent wrote the recipe explains what the recipe contains
explains the score, and the middle term adds nothing of its own.

This is not yet a verdict on the thesis, for three reasons, and each is a
concrete next step rather than a hedge:

1. **Power.** Conditioning on 24 families inside 143 rows leaves a residual
   holding 30 % of the within-cell variance. `n_datasets` at z = +1.8 is
   suggestive and unresolvable at this n; reaching z = 3 needs roughly (3/1.8)²
   ≈ 2.8× the rows, i.e. ~400. The full corpus has **1,175**, so this *is*
   resolvable — but only after the extraction is run beyond gsm8k.
2. **Coverage.** One benchmark, and the train side of the split being replaced.
3. **Feature coarseness.** These are seven summary counts derived from the
   extraction, not the recipe text. A model reading the trajectory has more to
   work with — but "more to work with" has to be demonstrated, and right now the
   coarse version shows nothing.

**So the order of work is:** run the recipe extraction over all 1,175 runs, redo
pass 3 at full power, and only then commit a split — with the excluded-columns
clause in the spec file, not in this document. Committing the 5-fold design
first would be committing a well-built measuring instrument before knowing there
is anything to measure.

## The answer, at full power

The extraction now covers all **1,175 runs** (`tools/extract_recipes.py`,
15.4 minutes, 1,279 model calls, ~$244). Every record passes a mechanical anchor
check: each `evidence_i` is an event index that exists in that run's digest and
each `evidence_quote` is a literal span of that event. 1,171 passed on the first
extraction or its one repair; the remaining 4 passed after the checker was
hardened against a non-object list element.

Two things had to be established before the answer means anything.

### Is the cheap extraction the same measurement as the expensive one?

`tools/extract_agree.py` re-scores the 143 runs both pipelines saw. Neither side
is ground truth; disagreement bounds how much of an extraction is the extractor
rather than the run.

| | agreement |
|---|---:|
| pipeline signature, exactly | 62.9 % |
| same after collapsing repeats (`sft>sft` ≡ `sft`) | 80.4 % |
| same set of families, any order | 86.7 % |
| dataset-id Jaccard (mean / median / exact / disjoint) | 0.787 / 1.000 / 64.3 % / 2.1 % |
| `lr`, where both report one | 85.8 % |
| `epochs` / `batch_size` / `grad_accum` / `max_seq_len` | 92.6 / 93.0 / 91.4 / 91.0 % |
| `precision` | 70.0 % |

The cheap tier is not worse on coverage — it fills `n_examples` on 54 % of
datasets against gold's 45 %, and the same 76 % of hyperparameter slots — and it
reports "low confidence" far less often (6 vs 22 of 143), which is the one place
to distrust it.

The interesting result is the ordering. **The coarse skeleton is the least
reproducible part of the extraction and the fine detail is the most.** Two
extractions of the same trajectory disagree about the pipeline signature 37 % of
the time, and 17 of those 37 points are purely *how many re-runs of the same
family count as shipped stages rather than as discarded attempts* — a definition
question, not an observation. They agree about the learning rate 86 % of the
time. So the feature the earlier passes leaned on is the noisy one.

### What the recipe adds once the agent is known

`tools/splitdx/recipe_signal2.py`, 1,175 rows, 28 cells, 26 agent families,
21 features at the grain the recipe is written at, every number bucketed, each
scored as R² minus its own within-family permutation null (500 draws), bar set
at z > 3.5 for Bonferroni over 21 tests.

Before conditioning, the recipe looks strong: `peft` scores +0.1749 excess at
z = +93. After residualising on `agent_family` it scores **−0.0021**. Whether a
run used LoRA or a full finetune is a fact about which agent it was.

Three features survive:

| feature | z | excess R² | share of residual |
|---|---:|---:|---:|
| `uses_rl` | +7.3 | +0.0049 | 0.49 % |
| `n_datasets` | +4.4 | +0.0173 | 1.73 % |
| `lr` | +3.8 | +0.0116 | 1.16 % |

`top_dataset` has the largest excess of anything tested (+0.1183) and does not
clear the bar — 394 levels over 1,175 rows is a near row identifier, and pass 2
of the original probe exists precisely to stop that from counting.

Three hits out of 21 tests is the regime where a result is most likely to be the
search. Split-half, holding agent families intact so a family never straddles
the two halves, 12 halves of ~587 rows. Scored against the *attenuated*
expectation — halving the rows lowers the z a real effect can reach by about
√2, so re-imposing z > 3.5 on half the data would be demanding a bigger effect
than the one that was found:

| feature | full z | expected on a half | observed median | ratio | worst half |
|---|---:|---:|---:|---:|---:|
| `uses_rl` | +7.3 | +5.2 | +3.4 | 0.65 | +0.3 |
| `n_datasets` | +4.4 | +3.1 | +2.8 | 0.90 | −0.0 |
| `lr` | +3.8 | +2.7 | +1.7 | 0.62 | +0.1 |

Only `n_datasets` attenuates no faster than the halving explains, and all three
vanish on at least one half.

**The answer, in three sentences.** Something in the recipe is real — three
features clear a corrected bar against a permutation null after the agent is
conditioned out. It is small: 0.5–1.7 % of the residual variance each, against
the 66 % `agent_model` takes on its own. It is not sturdy, and 1,175 runs is
about the smallest corpus that could see it at all, with no more runs to add.

This closes the power question that §"The blocker" left open. It was not a power
problem. A choice-set predictor built on these features would be predicting the
agent, which is exactly what the 5-fold agent-family split exists to forbid — so
the split is still the right instrument, and there is still nothing here for it
to measure. What remains untested is the claim one step up: that a model reading
the *trajectory* beats these 21 summaries of it. That is the next section.

## The summaries were the problem, not the corpus

The section above is a null on 21 features. It is *not* a null on the recipe,
and the difference turns out to be almost the whole result.

**The question, made answerable.** Take two runs from the same cell (same
benchmark, same base model) that were *also written by the same agent family*,
and ask which scored higher. Within-family pairing is the conditioning — it asks
what is left once the agent is known, the same question the 5-fold split asks,
but as a decision instead of an R². Pairwise puts the floor at exactly 0.5 and
removes calibration from the problem. Restricting to an accuracy gap ≥ 0.05
leaves **540 pairs over 13 families and 27 cells**, median gap 0.124. Every pair
is asked twice, A-first and B-first; a self-contradiction scores 0.5, as does an
abstention, so arms with different coverage stay comparable.

**The threat that had to be measured first.** Agents run their own evals and
print the numbers. A regex for score-shaped numbers next to an eval word finds
**4,074 of them across 593 runs**; a rule that reads nothing but the largest such
number and picks the run with the bigger one is right 61.8 % of the time on the
75.6 % of pairs it can decide. So "a model reads the trajectory" had to be split
into reading the recipe and reading the answer, or the measurement would have
been an OCR benchmark.

| what the arm is shown | acc | 95 % CI |
|---|---:|---:|
| effort only — turns, wall time, cost, all bucketed, fitted | 43.1 % | [39.1, 47.2] |
| the 21 bucketed features, **fitted** leave-one-family-out | 52.6 % | [48.3, 56.9] |
| the largest self-reported number, and nothing else | 58.9 % | [55.2, 62.5] |
| the same 21 bucketed features, **read by a model** | 59.0 % | [55.1, 62.8] |
| the full extraction record, read by a model | 76.8 % | [73.6, 79.9] |
| the trajectory digest, **score numbers blanked** | 85.7 % | [83.0, 88.4] |
| the trajectory digest, unmodified | 86.7 % | [83.9, 89.4] |

Paired sign tests over the shared pairs, rung by rung: effort → features
164–113 (p = 0.003), features → summary 151–112 (p = 0.019), summary → recipe
190–77, recipe → redact 101–35 (both p < 1e-4), redact → raw 23–14 (p = 0.19).
Two of those deserve to be read as results in their own right. Blanking 95 % of
the score-shaped numbers costs one point and cannot be distinguished from noise.
And `selfreport` vs `summary` is **155–154, p = 1.0** — a model given the 21
bucketed features is exactly as accurate as a regex that reads nothing but the
largest number the agent printed about itself. Those two arms know nothing in
common; they are simply both nearly useless.

**Four ways of checking it is not the leak.**

1. Redaction removes 955 → 47 score-shaped numbers per 200 runs, and costs
   1.0 pp.
2. On the **132 pairs where the regex cannot compare the two runs at all**, the
   digest arms hold **83.0 %** [76.5, 88.6]. On the 22 where *neither* run
   printed a readable score, 81.8 % [63.6, 95.5].
3. On the 312 pairs where the model's own stated reason never quotes a measured
   score, `raw` is at **82.9 %**.
4. `raw` agrees with the self-report rule on 63.7 % of the 383 pairs that rule
   decides. If `raw` were that rule with extra steps, agreement would be 100 %.
   Two arms this accurate (87.9 % and 61.8 % on those pairs) that decided
   *independently* would agree 58.9 % of the time by arithmetic alone. Observed
   is 63.7 %: they overlap a little, and almost all of that is both being right.

**And not the effort proxy either.** The obvious rival explanation is that the
model is reading which run finished and worked, not what it did. Turns, wall
time, cost and elapsed time, bucketed and fitted the same way, come in at
**43.1 %** — significantly *below* chance. Working harder predicts scoring worse
here, so an arm that used effort as its signal would have to be beaten by a coin,
and adding effort to the 21 features moves nothing (50.6 %).

**Where the information actually went.** The three-rung gap is the finding:

- **59.0 → 76.8** (17.8 pp) is what bucketing into 21 categoricals destroys. The
  model reading those 21 values has all the prior knowledge in the world and
  still cannot do better than 59 %, so this is not a sample-size artefact of the
  fitted arm — the summary genuinely does not contain the answer.
- **76.8 → 85.7** (8.9 pp) is what the *extraction* destroys, and this one is
  uncomfortable: the schema was built to capture the recipe, and it still loses
  nearly nine points against the raw trajectory. What survives in the digest and
  not the record is mostly fit-to-task — whether the chat template matches the
  grader, whether the eval harness was reproduced locally, what was tried and
  rejected and why.
- **52.6 → 59.0** (6.4 pp) is the fitted-vs-prior-knowledge gap, and it is the
  smallest of the three. 540 pairs is thin, but thinness is not what is wrong
  with the 21 features.

Confidence is usable: `raw` is right 98 % of the 66 pairs it rates 5, and 68 % of
the 44 it rates 2. No family carries the result — the weakest is
`claude-opus-4-6` at 73.5 % over 100 pairs, the strongest `claude-opus-4-8` at
98.1 % over 54. Position bias, the reason for asking every pair twice, turned out
to be small: `raw` says "A" 51.2 % of the time and contradicts itself on 5.2 % of
pairs.

**What this does and does not license.** It says the trajectory contains, in
readable form, most of what separates a good run from a bad one within a cell and
within an agent family, and that every summary tried so far throws that away. It
does not say a *predictor* can be built: the arm is a discriminator shown both
sides, not something that scores one run in isolation, and 86.7 % on pairs
separated by ≥ 0.05 is not a claim about the 0.05-and-under population the choice
set actually contains — accuracy there is 79.4 %, against a 50 % floor. The
honest next step is the reverse direction: rank a whole choice set rather than
compare two, and the crossed rollout in §"The confound no split can fix" to break
the agent/recipe confound at the source.

## The reverse direction: ranking a whole choice set

The pairwise arm is a discriminator shown both sides. The thesis needs a
predictor: hand it a cell's worth of candidates and take the three it likes best.
The metric is `regret@3 = best_in_the_set − best_of_the_three_you_picked`, and
zero means the winner was in your three.

Two stages were built, which is what a deployment would plausibly do — an `O(n)`
scorer to narrow and an `O(k²)` comparator to decide:

| stage | what it sees | calls | cost |
|---|---|---:|---:|
| A | one redacted digest, alone, rate it 0–100 | 1,175 | $124.90 |
| B | round-robin the top 6 of every choice set, both orders, Copeland | 4,568 | $884 |
| B, K = 10 | the same at a wider shortlist | +4,086 | +$793 |

Stage B took three attempts to measure and the first two were both wrong, in
opposite directions — see §"Where the remaining regret is". What it is worth, at
full pair coverage: −0.0043 on full cells (5–1, still under its MDE of 0.0072),
−0.0022 within scaffold, and a wash within family. Solved@3 goes 67.9 % → 82.1 %
on full cells. Everything below is reported for both stages, and the
recommendation at the end is to ship both — with the caveat that the comparator's
contribution is never a significant contrast, only a consistent point estimate.

Stage A is the harder half and the new thing: scoring in isolation means
supplying your own standard, where a pair only needs a relative difference. It
works as a *ranker* and not as an estimator — within-cell Spearman against true
accuracy has median **+0.818** across the 28 cells (range +0.061 to +0.946),
while its point estimate of the benchmark score is off by **0.062** on average.

### The sort key was reading the wrong field

Before the table below is worth anything, one defect in how it was produced. The
stage-A call returns two usable numbers per run — `quality`, a 0–100 integer the
prompt asks it to spread, and `predicted_accuracy`, a float — and the shortlist
read only `quality`. That field does spread, across the middle; at the top it
saturates. **79.3 %** of full-cell runs share a quality value with a set-mate,
and the rank-6 cut falls *inside* a tie block in **24 of the 28 cells**, median
block size 5 competing for 3 places. Those places were then handed out by
`r["run"]`, the job id — alphabetical order deciding the metric.

The two fields are not redundant where it matters. Over a whole cell, Spearman
against true accuracy is +0.814 for `quality` and +0.825 for `predicted_accuracy`
— indistinguishable. Over each cell's **top third**, the only region the top-3
cut ever sees, it is **+0.406 against +0.842**, and `predicted_accuracy` wins in
23 of 28 cells (p = 0.00091). A double dissociation from one call: the field
being used is the one that stops working exactly where the metric lives.

The fix is `z(quality) + z(predicted_accuracy)`, z-scored *within the set* so no
run is ever compared across cells, with the job id demoted to a last-resort
deterministic tiebreak. The weight is 1.0 and is **not fitted**: sweeping it, 2.0
also improves both populations and 0.5 does not, so the choice is not knife-edge,
but any tuned value would be one more thing selected on 28 cells.

Ranking on `predicted_accuracy` *alone* is better still on full cells (0.0099)
and **worse within a family** (0.0053 against the old 0.0047). That is precisely
the arm this document's own design rule rejects — anything that only works where
the agent-family lookup table already works has not been shown to work. The sum
improves both, which is why it is the one that ships. Stated plainly so it can be
held against the result: leave-one-cell-out selection scored on *full cells only*
picks the `predicted_accuracy` arm in 27 of 28 folds. The both-populations
criterion is a design principle declared before the sweep, not a filter chosen
after seeing which arm it favours.

The same key now also settles Copeland ties in stage B. It used to fall back to
bare `quality` there, so on any cell where the comparator ties — and at 1.0 %
within-family pair coverage it ties on nearly all of them — the alphabetical
order the shortlist fix removes came straight back in through the fallback.

### The baseline was averaging accuracies across cells, and it was carrying the headline

The same scrutiny the sort key got has to be applied to the thing it is measured
against, and the baseline fails it harder. The agent-family lookup table scored a
family by the **raw mean accuracy** of its runs, pooled over every other cell.
Those cells' own mean accuracy runs from **0.003 to 0.820**. So a family that
happened to appear mostly in `bfcl` cells outscored a family that appeared mostly
in `aime2025` cells for reasons that have nothing to do with either family: the
"baseline" was largely ranking families by which benchmarks they were sampled on.

Averaging the **within-cell z-score** instead — the same leave-one-cell-out
construction, the same data, the only change being that each run is standardised
against its own cell before it is pooled — takes the table from **0.0248 to
0.0070** on the full cell, and from 50.0 % to 78.6 % of cells solved.

That single fix retracts this document's full-cell headline. The published claim
was that stage A beats the table by **0.0167, 10–2, sign p = 0.0386**. Against a
baseline on the right scale it is **+0.0011, 3–6, p = 0.508** — stage A is
nominally *behind*, and the interval [−0.0069, +0.0085] contains zero comfortably.
Neither direction is detectable: the MDE at 28 cells is 0.0110, an order of
magnitude above the effect. It always was — the published 0.0167 was itself under
the old MDE of 0.0266, with a t-test at p = 0.0908, and this document reported the
sign test without the power calculation next to it.

The obvious objection is that within-cell z is a baseline chosen after seeing that
it hurts. It was checked the only way that objection can be answered: by enumerating
the alternatives rather than picking one. Over **28 defensible definitions** of the
table — {raw accuracy, within-cell z, within-cell rank} × {mean, median, 20 %-trimmed
mean, max, …} — raw+mean is the **worst of the 28**, and the **only one** under
which `stage A − table` reaches p < 0.05. Leave-one-cell-out selection over the
28 definitions, scored on the held-out cell, picks a corrected form in **28 of 28
folds** and lands on the same 0.0070. There is no reading of the data on which the
published number is the right one to have reported.

### Where the remaining regret is, and the shortlist bug it exposed

Decomposing stage A + B's residual over full cells — 0.0072 at the time, i.e. the
whole pipeline within 0.7 accuracy points of oracle on the median cell:

| bucket | share | cells |
|---|---:|---:|
| the winner was never shortlisted at all | 52.7 % | 3 |
| shortlisted, but the comparator was never shown it | 14.4 % | 3 |
| shortlisted and compared, and ranked below third | 33.1 % | 1 |

Stage A's top 6 contains the cell's winner in **25 of 28 cells**. The earlier
"90.8 % / 9.2 %, no more than 0.0013 recoverable" split was computed when Copeland
was still ranking uncompared runs last, so most of what it filed under "the
comparator ranked it wrong" was the comparator having no data at all.

That middle row turned out not to be a budget decision but a bug. `stage_b` built
its shortlist from `cells(rows)` alone, while the board scores three populations,
and the top 6 of a within-family sub-set is mostly *not* the top 6 of its parent
cell. So the comparator was being credited on pairs it had never been shown:

| population | pair coverage of its own shortlist | shortlisted runs with no comparison at all |
|---|---:|---:|
| full cell | 37.6 % | 65 of 168 |
| within scaffold | 13.3 % | 382 of 496 |
| within family | 1.0 % | 359 of 369 |

In 12 of the 28 cells one of the uncompared runs was that cell's own winner.
`stage A+B` on those rows was stage A plus noise, which is the whole of the
"stage B costs points" reading. `choice_sets` now enumerates what the report
actually scores — the cell, plus each within-family and within-scaffold sub-set
of four or more runs — with pairs still keyed on the parent cell so a pair two
populations both need is fetched once. 3,728 new ordered comparisons, coverage
98.8 / 100 / 100 %:

| population | stage A+B before | after | solved@3 |
|---|---:|---:|---:|
| full cell | 0.0072 | **0.0038** | 75.0 % → 82.1 % |
| within scaffold | 0.0061 | **0.0047** | 78.6 % → 79.8 % |
| within family | 0.0029 | 0.0036 | 92.1 % → 89.5 % |

Within family it is a wash and slightly negative (2-4, p = 0.688, MDE 0.0027) —
reported rather than selected away, and the obvious structural excuse does not
hold: splitting by whether K = 6 actually narrows the set (71 of those 76 sets
have n ≤ 6) puts stage B behind on *both* sides, 1-2 either way, which is three
sets deciding a direction.

`aime2025 × Qwen3-4B-Base`, the single 0.0667 cell that was the entire "genuinely
misranked" bucket, is now 0.0000.

### Nothing else moves it: four levers, all measured, all null

With coverage fixed, the residual is 0.0038 full cell against an oracle-over-the-
top-6 of 0.0030 — 0.0008 of comparator headroom left at that budget. Four ways to
spend more, in the order anyone would try them:

| lever | cost | result |
|---|---|---|
| widen the shortlist to K = 10 | 4,086 calls, $793 | 0.0038 → 0.0033 full cell (1-0), 0.0047 → 0.0048 within scaffold, exactly nothing within family |
| weight Copeland by the comparator's confidence | free | identical on full cells, worse within scaffold; dropping confidence ≤ 2 is worse on both |
| Bradley-Terry instead of Copeland | free | indistinguishable at every blend weight |
| blend Copeland with stage A instead of overriding | free | best in-sample weight is worth ≤ 0.0007; chosen leave-one-cell-out it is 0.0000 / −0.0005 / −0.0007 |
| average the two stage-A passes (the run-id ablation is a second independent read) | free | +0.0004 / 0.0000 / −0.0016 at k = 3, though it helps at k = 1 on all three (0.0200 → 0.0188, 0.0265 → 0.0245, 0.0359 → 0.0322) |

K = 10 is the informative one: the oracle floor *does* drop (0.0030 → 0.0018 full
cell, 0.0017 → 0.0001 within scaffold), so the extra 4,086 comparisons genuinely
put the winner in front of the comparator more often, and it converts almost none
of that. A wider list gives a fixed-accuracy judge more chances to be wrong. Both
sizes stay on the board rather than the better one being chosen, because picking
K off 28 cells is fitting a hyperparameter on the test set.

The blend null has a twist worth recording. On the 772 comparisons cached before
the fix, stage A scored 83.8 % pairwise against stage B's 76.1 %, which says the
lexicographic form is backwards — the weaker judge overriding the stronger one.
On the same 7,753 comparisons it is stage B 79.0 % against stage A 76.8 %. The two
stages had been graded on different pair sets, and the pre-fix set was the easy
one. Split by how far apart the two runs really scored:

| true gap | n | stage B | stage A |
|---|---:|---:|---:|
| under 0.02 | 1468 | 59.1 % | 56.6 % |
| 0.02–0.05 | 2153 | 71.2 % | 70.8 % |
| 0.05–0.15 | 2218 | 85.2 % | 80.7 % |
| over 0.15 | 1914 | 95.7 % | 94.6 % |

Stage B is the better judge, and only outside the narrow band — which is exactly
the top of a shortlist. That is the ceiling, stated as a property of the task
rather than of the method.

### What the floor is made of

Five cells of 28 carry all the remaining full-cell regret:

| cell | regret | one graded item | items missed |
|---|---:|---:|---:|
| `aime2025 × gemma-3-4b-pt` | 0.0333 | 0.0333 | 1 |
| `gpqamain × Qwen3-1.7B-Base` | 0.0312 | 0.0022 | 14 |
| `gpqamain × SmolLM3-3B-Base` | 0.0201 | 0.0022 | 9 |
| `humaneval × Qwen3-1.7B-Base` | 0.0122 | 0.0061 | 2 |
| `arenahardwriting × Qwen3-1.7B-Base` | 0.0087 | — | continuous |

The other 23 are solved exactly. The largest miss in the corpus is one AIME
question. Beyond this the constraint is not the ranker but the design: the 28
cells are 7 benchmarks × 4 models fully crossed, there is no 29th to add, and the
515 further trajectories on disk fall inside those same 28 — they would deepen the
sets without adding a cluster, and the clustered tests count clusters. At nc = 28
the MDE is ~0.008–0.010, and the within-scaffold contrast (−0.0095, 13-5,
p = 0.0963, 78 % tie-break robustness) sits just under its own. Even a perfect
comparator there reaches −0.0125 against an MDE of ~0.010, so this population
clears 0.05 only barely and only with a ranker that does not exist. **More power
has to come from a second corpus, not from a better method on this one.**

### On a whole cell, the lookup table is not beaten

Median cell: 44 candidates, spread 0.655. `tie-break` is the same arm re-scored
under 200 random tie-breaks, and the percentile the shipped job-id order sits at
inside that distribution.

| arm | regret@3 | 95 % CI | cells solved | regret@1 | tie-break |
|---|---:|---|---:|---:|---|
| random three | 0.1307 | [0.0903, 0.1756] | 11.4 % | 0.2640 | — |
| self-report (largest printed number) | 0.1126 | [0.0608, 0.1719] | 17.9 % | 0.1748 | 0.1047, p90 |
| 21 bucketed features, fitted | 0.0242 | [0.0134, 0.0359] | 50.0 % | 0.0678 | no ties |
| **stage A alone** | 0.0081 | [0.0027, 0.0148] | 67.9 % | **0.0200** | 0.0077, p68 |
| **agent-family lookup table** | 0.0070 | [0.0019, 0.0138] | 78.6 % | 0.0480 | 0.0091, p22 |
| stage A + B | 0.0038 | [0.0007, 0.0075] | 82.1 % | **0.0145** | no ties |
| stage A + B, K = 10 | **0.0033** | [0.0006, 0.0070] | **85.7 %** | **0.0145** | no ties |

**On a whole cell, a table of "which agent averages what" is as good as reading
the trajectories, and this document should stop claiming otherwise.** Stage A is
+0.0011 behind at 3–6 (p = 0.508), under the 0.0110 the population can resolve.
The one thing stage A does own here is `regret@1` — 0.0200 against the table's
0.0480 — i.e. it is better at naming the single winner even though the table is
at least as good at getting the winner *somewhere* in three. That is a real
difference and it is not the metric the thesis declared.

The two `stage A + B` rows are ahead of the table on the point estimate, and they
are still not a result: −0.0033 at 4–2, p = 0.688 with only **30 %** tie-break
robustness, and −0.0037 at 4–1, p = 0.375 at 41 %. Six cells and five cells
respectively differ at all. A row decided by five cells, half of which survive a
re-draw of the tie-break, is a number to report and not to claim.

The design rule this document set for itself was that any arm which only looks
good where the agent table works has not been shown to work. Applied honestly, it
now cuts the other way: the full-cell population is the one where the table works,
and there is nothing to show there. The result lives in the two populations below.

**Stage B is worth buying, and every previous sentence in this document about
whether it was is void.** At full coverage it is −0.0043 against stage A alone,
5–1 out of 28 (p = 0.219, MDE 0.0072 — still under, so "worth buying" is a point
estimate and a solved-rate, not a significant contrast). Its three earlier
headline numbers were each about the scaffolding. The **−0.0008 on 2 non-ties**
immediately above was measured at 37.6 % pair coverage against a shortlist stage
B had never been given. The older **+0.0026 "stage B costs points"** was an
artefact of Copeland counting wins:
a shortlisted run the comparator was never shown scored zero wins and therefore
sorted *below* a run that lost every comparison it had, and 38.7 % of shortlisted
runs are in that state. Scoring `(wins − losses) / comparisons decided` — 0 both
for "never compared" and for "even record" — removes it. And the older
**−0.0072, 8–0, p = 0.008** in stage B's favour was real for the mirror-image
reason: stage A was handing it a shortlist ordered by alphabetical tiebreak, so
the comparator's whole measured value was undoing a defect that now costs $0 to
fix. Every one of stage B's headline numbers, in every direction, has been about
the scaffolding around it rather than about the comparator; the 79.0 % pairwise
accuracy above is the first one that is not.

### 76 sets cut from 28 cells are not 76 independent observations

Before either sub-set population is read, the sample size has to be fixed. The
within-family sets are cut out of the *same* 28 cells — five sets from one cell
share its benchmark, its base model and most of its runs — and an arm's advantage
on a sub-set correlates **+0.54 to +0.79** with its advantage on the parent cell.
Treating them as 76 draws overstates n by a factor of ~2.7 in the standard error
and by more than that in a sign test, which counts each sub-set separately.

Every paired test below therefore **clusters on the 28 cells**: the per-set
differences are averaged within a cell first, and the bootstrap resamples cells,
not sets. `n` drops from 76 to 28 within family and from 84 to 28 within scaffold.
The set-level number is still printed in parentheses by `choice_rank_report.py`
so the two can be compared, and one arm changes verdict between them (below).

### Where the table has no information, the picture inverts

Restrict each set to one agent family, which is what the 5-fold split holds out.
76 sets in 28 cells, median size 5, median spread 0.129:

| arm | regret@3 | 95 % CI | sets solved | vs random, clustered | vs random, by set |
|---|---:|---|---:|---|---|
| agent-family lookup table *(job-id order — see below)* | 0.0546 | [0.0257, 0.0935] | 53.9 % | 10–18, p = 0.19 | 40–31, p = 0.34 |
| random three | 0.0271 | [0.0179, 0.0404] | 67.6 % | — | — |
| 21 bucketed features, fitted | 0.0197 | [0.0115, 0.0291] | 71.1 % | 14–14, **p = 1** | 49–22, p = 0.002 |
| self-report | 0.0092 | [0.0040, 0.0155] | 78.9 % | 22–6, p = 0.004 | 59–12, p = 1e-8 |
| **stage A** | **0.0029** | [0.0005, 0.0061] | **92.1 %** | **25–3, p = 3e-05** | 65–6, p = 1e-13 |

**The table's 0.0546 is an artefact and this document previously leaned on it.**
Inside one family the table's score is constant, so it is not ranking at all —
`r["run"]` is, and job ids are issued in time order, so the row is literally
"pick the three oldest runs in the family". Under random tie-breaks the same
table scores **0.0247**, and the shipped 0.0546 is worse than **all 200 draws**.
Reversed — newest first — it is 0.0102. Saying the table "lands worse than
random" was a rhetorical point built on an arbitrary ordering; the correct
statement is that inside a family the table carries **no information** and lands
on random's 0.0271. Stage A beats it at **23–2 of 25 cells, p = 1.9e-05**, and
that survives the tie-break control — under 200 random tie-breaks the clustered
sign test still clears 0.05 in **94 %** of draws, at a mean effect of −0.0294.
The −0.0517 this document previously quoted was the set-level mean; clustered, the
mean effect is −0.0796 with a bootstrap of [−0.164, −0.024].

**The fitted-features arm does not survive the clustering.** At 76 sets it beat
random 49–22, p = 0.002. Averaged into its 28 parent cells it is **14–14, p = 1**,
and the bootstrap [−0.0525, +0.0069] crosses zero. Its apparent win was 28 cells'
worth of signal counted 76 times. Stage A and self-report both survive; the
features arm is the one row that changes verdict, and it is the row a
metadata-only baseline would have been built on.

Stage A over self-report is **13–5, p = 0.096** clustered (15–6, p = 0.078 by set)
— still not significant, and now also below its own MDE of 0.0077, because with
five candidates and three picks there is very little left to win. The
within-family half of the sort-key gain is likewise the weak half: the joint
bootstrap that resamples cells *and* reshuffles tie blocks puts it at −0.0023
[−0.0054, +0.0005], P(no gain) = 0.048, against a solid −0.0156 [−0.0299,
−0.0047], P(no gain) = 0.003 on full cells. The claim the sum supports is "it
does not cost anything within a family and it clearly helps on full cells", not
"it helps equally on both".

A note on reading the MDE column: within family, `stage A − random` is flagged
`<MDE` (effect 0.0347, MDE 0.0404) while the clustered sign test returns
p = 3e-05 at 25–3. That is not a contradiction. The MDE prices a *t*-test on cell
means, and the cell means here are heavy-tailed — a couple of cells carry a large
difference and the rest carry a small one in the same direction. The sign test
reads the consistency and ignores the magnitude, so it is the load-bearing test
on this population and the mean effect is the fragile number.

**Stage A + B is still not measured on this population.** Stage B only compared
each *cell's* top six, and those six almost never fall inside one family — pair
coverage of the within-family shortlists is **1.0 %**, and **359 of 369**
shortlisted runs (97.3 %) had no cached comparison at all, against 37.6 % / 38.7 %
on full cells, and the two rows were identical to four decimal places with **0
non-ties** — an absence of data, not a tie. At full coverage stage A + B is
0.0036 against stage A's 0.0029, 2–4, p = 0.688 against an MDE of 0.0027: a wash,
and the only population where the comparator does not pay for itself. Median set
size here is 5, so there is nothing for a shortlist to narrow, but that excuse
does not survive checking — the five sets with n > 6 go 1–2 as well.

### A third population, where the table is neither useless nor sufficient

Within-family sets are small (median 5) and full cells are where the table wins,
so both extremes are easy to dismiss. The population in between is the set of runs
sharing a **scaffold** — 84 sets in the same 28 cells, median size 13, median
spread 0.338. The agent-family table is *not* constant here (0 % of sets), so it
is genuinely ranking, and there are enough candidates for a ranker to be wrong.

| arm | regret@3 | 95 % CI | sets solved | vs table, clustered |
|---|---:|---|---:|---|
| random three | 0.0768 | [0.0633, 0.0912] | 29.8 % | 2–26, p = 3e-06 |
| self-report | 0.0396 | [0.0246, 0.0570] | 53.6 % | 8–19, p = 0.052 |
| 21 bucketed features, fitted | 0.0234 | [0.0151, 0.0330] | 57.1 % | 8–17, p = 0.11 |
| agent-family lookup table | 0.0142 | [0.0077, 0.0218] | 73.8 % | — |
| **stage A** | 0.0069 | [0.0030, 0.0122] | 76.2 % | 13–6, p = 0.17 |
| **stage A + B** | **0.0047** | [0.0026, 0.0071] | **79.8 %** | 13–5, p = 0.096 |
| stage A + B, K = 10 | 0.0048 | [0.0027, 0.0071] | 79.8 % | 12–5, p = 0.14 |

Stage A halves the table's regret (0.0069 against 0.0142) and the sign test does
not reject at 28 clusters — 13–6, p = 0.17, effect −0.0073 against an MDE of
0.0085. Adding the comparator takes it to 0.0047, a third of the table, and moves
the contrast to −0.0095 against an MDE of 0.0103 — still not rejecting at
p = 0.0963, but the tie-break robustness goes from 24 % to **78 %**, i.e. the sign
of the row stops being a property of the job ids. This is the cleanest statement
the corpus supports on a population where the table has real information: **the
trajectory reader is at worst as good as the table and probably better, but 28
cells cannot tell those apart.** Reported as a positive result it would be exactly
the overclaim the full-cell headline was.

It is also as close as this corpus gets. The oracle over stage A's top 6 here is
0.0017, so even a perfect comparator reaches −0.0125 against an MDE of ~0.010 —
clearing 0.05, but only just, and with a ranker that does not exist. The 515
unextracted trajectories on disk all fall inside these same 28 cells, so they
deepen the sets without adding a cluster and the clustered test counts clusters.
(Measured 2026-09-01: `index.parquet` holds 1,745 rows but only 1,690 distinct
run keys; all 1,175 scored runs are present on disk, and 515 converted runs are
unscored — 166 contamination-flagged, 316 carrying a score, 165 both scored and
unflagged, 199 with no score at all. Their 28 `task_id`s are exactly the 28
cells already in play, none outside. The earlier figure of 787 was never
derived and is withdrawn.)
**The next real gain is a second corpus, not a better method on this one.**

### The leak control

Stage A reads the redacted digest, so the question is what is left in that text.
Answering it by conditioning on "no run printed a score" is useless — exactly
**1 of 76** within-family sets qualifies. The usable control is to run the score
regex on the identical string stage A was handed:

| arm | full cell | within family | within scaffold |
|---|---:|---:|---:|
| self-report, raw text | 0.1126 | 0.0092 | 0.0396 |
| self-report, on the redacted text | 0.0622 | **0.0519** | 0.0509 |
| stage A, on the redacted text | 0.0081 | **0.0029** | **0.0069** |
| random | 0.1307 | 0.0271 | 0.0768 |

One correction to the self-report arm itself, which moved its numbers here and in
both tables above. The parser inferred percent-versus-fraction from the
*magnitude* — divide by 100 if the number is above 1 — so every percentage below
1 % was read at fraction scale: `pass@1 0.83%` became 0.83, `win rate (stderr:
±0.64%)` became 0.64, and a bare `1%` became a **perfect 1.000**. That is 56
matches across the corpus, 20 of the 1,175 run maxima, and **16 of the 42 runs
that appeared to report a perfect score**. The failure is one-sided — every one
of the 16 was inflated. The scale now comes from which pattern matched, and the
fraction pattern carries a `(?!\s*%)` so `0.83%` cannot also match it and win the
`max()`.

The fix makes the self-report arm **worse**, from 0.1019 to 0.1126 on full cells
(17.9 % solved, down from 21.4 %), which is worth stating plainly: it widens the
gap this document reports in stage A's favour on that population. `redact()`
blanks the matched digits without ever reading their value, so the redaction
control is unchanged by it.

Redaction takes quotable scores from 1,003 of 1,175 runs down to 290. On that
text the regex is worse than random within a family (0.0519 vs 0.0271) while
stage A, reading the same characters, is at 0.0029. Whatever stage A is doing, it
is not transcribing a printed score. What this control does *not* rule out is
that the within-family headline is separable from model-free surface features of
the redacted string — length, section counts, how much of the log survived
redaction. The next section takes the largest such feature and ablates it.

### What the digest still carries: the run id

`traj_read._digest_of` renders the trajectory with `RC.render(row["run"], …)`, so
the first thing stage A reads is the run id — and a PTB run id ends in the job
number, which is issued in time order. That is not a redaction bug in the score
sense; it is metadata that happens to correlate with the label, because the corpus
got better over the months it was collected.

How much is it worth? Rank each set by job number, newest first, and pick three:

| arm | full cell | within family |
|---|---:|---:|
| random | 0.1307 | 0.0274 |
| oldest three | 0.3187 | 0.0546 |
| **newest three** | **0.0461** | **0.0102** |
| stage A | 0.0081 | 0.0029 |

Newest-first recovers **69.0 %** of stage A's whole gain over random on full cells
(cluster-bootstrap 95 % CI [44 %, 92 %]) and **70.5 %** within a family
([28 %, 98 %]). Oldest-first is far *worse* than random in both, which is what
rules out this being an artefact of how ties are broken: the ordering carries
signed information about the label.

That a *free* feature reproduces most of the headroom this section claims for a
$125 model is enough to make "stage A reads the trajectory" unestablished. So it
was tested rather than argued: `--stage noid` replaces the job number with an
order-free hash of itself and re-runs the whole pass — same prompt, same
redaction, same 1,175 digests, differing by eight characters each. It was run on
**all 28 cells**, because running it only where the job-id arm looks strong would
select the sample on the statistic under test. It cost **$124.89**.

| population | random | newest three | stage A, id present | **stage A, id removed** |
|---|---:|---:|---:|---:|
| full cell | 0.1307 | 0.0461 | 0.0081 | **0.0077** |
| within family | 0.0271 | 0.0102 | 0.0029 | **0.0029** |
| within scaffold | 0.0768 | 0.0262 | 0.0069 | **0.0082** |

**Stage A keeps 100 % / 100 % / 98 % of its gain over random with the id gone**
(cluster bootstrap [100 %, 102 %], [100 %, 100 %], [95 %, 100 %]); the largest
change on any population is 0.0012 on 4 non-tied cells, p = 0.63. The two passes
are genuinely independent — **0 of 1,174** rationales are identical — and they
agree at Spearman **0.994** on `quality` and **0.997** on `predicted_accuracy`,
which doubles as the test–retest reliability of stage A and is the reason the
ablation has any power at this n.

So the job-id arm and stage A are two things that both correlate with the label,
not one thing wearing two hats. Stage A does not need the id and does not use it.

What this closes and what it does not: it closes "the model is transcribing an
identifier", the way the redaction control closes "the model is transcribing a
printed score". It does not close the deeper version — the crossed rollout below
measures that a corpus accuracy is ~90 % the executing agent, so whatever stage A
reads off a trajectory to predict that number may still be agent identity rather
than recipe quality. Two controls down, that one open.

### What this settles and what it does not

A model reading trajectories picks a set's winner into its top three **82.1 %** of
the time on a whole cell, **79.8 %** within a scaffold, and **92.1 %** within an
agent family — the first two with the comparator, the third without it.

**It does not beat the agent-family lookup table on a whole cell.** That claim
appeared in earlier versions of this document at 0.0167, p = 0.039, and it was an
artefact of a baseline that averaged raw accuracies across cells with wildly
different score scales. Corrected, the contrast is +0.0011 at p = 0.508 on an
effect the population could not have resolved either way. What is left is:

- **within an agent family**, where the table has nothing to say, stage A is
  0.0029 against random's 0.0271 and beats the table 23–2, p = 1.9e-05, robust to
  the tie-break control;
- **on the 24 cells where the winner's family also produced a below-median run**
  — the cells where "just look up the family" is genuinely ambiguous — stage A is
  0.0135 / 75.0 % solved against the table's 0.0265 / 54.2 %;
- **within a scaffold**, 0.0047 against the table's 0.0142, directionally the same
  and not significant at 28 clusters (p = 0.0963), though now robust to the
  tie-break in 78 % of draws rather than 24 %.

Four things this does not show. It is one $125 stage-A pass plus a $884 comparator
over 8,654 comparisons, not a trained predictor. The comparator's contribution is
a point estimate and a solved-rate, never a significant contrast — −0.0043 on
5 non-tied cells full-cell, −0.0022 on 10 within scaffold, and a wash within
family. The fitted-features baseline, once the sub-sets are
clustered, is not distinguishable from random within a family — so "a metadata
model does the job" is not supported either, in either direction. And none of it
touches the underlying confound — every recipe in this corpus was written *and*
executed by the same agent, so "the recipe predicts the score" and "the agent
predicts both" remain observationally identical here. Only the crossed rollout
separates them, and it is now run: see §"What the crossing measured".
## Crossing the confound: a deterministic recipe executor

`rollout/agents/hv_recipe/solve.sh` is the executor half of the crossing in §"The
confound no split can fix". There is no LLM anywhere in it: one script, six
recipes, everything else held fixed. Which half of a run is "the recipe" is not a
judgement call — it is read off the extraction schema.

| VARIES — what `tools/extract_recipes.py` recorded | FIXED — what it never captured |
|---|---|
| datasets, subsets, caps, repeats | learning rate and schedule, precision |
| epochs, nominal batch size | sequence length, target format, EOS handling |
| | which checkpoint ships, decode settings |

`learning_rate` is `None` on **all 24** SFT-only runs of the gsm8k ×
Qwen3-1.7B-Base cell, which is why it sits in the FIXED column: nothing was
recorded to vary. That column is also, precisely, where the agents differed
invisibly.

The six recipes span the cell, 0.733 down to 0.042. The two ends are the point:
**the 0.733 run and the 0.042 run have essentially the same recipe on paper.** So
the 0.69 gap lives somewhere the extraction never looked, and the null here — that
under a fixed executor the spread collapses — was a live outcome rather than a
strawman. It is very nearly what happened: see §"What the crossing measured". Two
seeds per recipe, because a single-seed contrast in this cell would be
uninterpretable against seed noise, plus an untrained control (`hv_noop`) so a
collapsed spread can be told apart from a trainer that trains nothing.

Submitted as two 7-cell packs on two exclusive a3 nodes (jobs 87585 / 87586),
seeds alternating within a node and swapping between them, node B's order rotated
so no recipe lands on the same card index twice.

### Three defects the CPU dry run caught before any GPU time

Each would have produced a clean-looking wrong answer, and two of them would have
produced *my own hypothesis' null*:

- **`MAX_LEN` was 1024; the eval's own 10-shot system message is 1715 tokens.**
  100 % of rows truncated, and under `completion_only_loss` it is the completion
  that goes — so every row carries zero loss tokens, every arm scores the base
  model's floor, and the run still exits 0. Measured p50 1929 / max 2551 across
  both sources; `MAX_LEN` is now 2560 (0.00 % truncated) and a runtime guard fails
  the run above 2 %.
- **Targets ended in `tok.eos_token`.** On a *Base* checkpoint that is
  `<|endoftext|>`, while the template — and therefore vLLM at grading — stops on
  `<|im_end|>`. Train on the wrong one and the model never emits a stop token and
  buries its answer in the tail. The terminator is now read from the template.
- **The template was unreachable.** `src/eval/tasks/gsm8k/evaluate.py` hands vLLM
  `templates/qwen3.jinja`, but that lives at `src/eval/templates/` and
  `run_task.sh` copies only `solve.sh` into the sandbox. The fallback was whatever
  the tokenizer happened to ship. It is now embedded byte-for-byte and
  hash-checked, so training and grading render the same string.

A target-format assertion ships alongside them: MetaMathQA bodies carry gsm8k's
own `#### 752` line, which teaches a second answer format that the grader then
reads instead of the intended one.

### What the crossing measured

Both packs completed (jobs 87585 / 87586, exit 0, 14 cells, two exclusive a3
nodes). The untrained control lands at **0.3332**, against the 0.3321 the corpus
records for this base model — the trainer is training and the grader is grading.

| recipe | corpus accuracy | rollout, seed 0 | seed 1 | mean |
|---|---:|---:|---:|---:|
| r733 | 0.733 | 0.7271 | 0.7225 | 0.7248 |
| r699 | 0.699 | 0.6801 | 0.6672 | 0.6736 |
| r600 | 0.600 | 0.7506 | 0.7369 | **0.7437** |
| r544 | 0.544 | 0.6755 | 0.6755 | 0.6755 |
| r401 | 0.401 | 0.7513 | 0.7384 | **0.7449** |
| r042 | 0.042 | 0.7104 | 0.7051 | 0.7077 |
| *base, untrained* | — | 0.3343 | 0.3321 | 0.3332 |

**The recipe is worth about a tenth of what the corpus attributes to it, and the
ordering does not survive at all.** The six runs span 0.691 in the corpus and
**0.0713** under a fixed executor — **10.3 %** reproduced. Measured as a standard
deviation across the six rather than a range, 0.0291 against 0.2329, **12.5 %**.
And the two best rollouts are the corpus's **third and fifth** recipes: Spearman
between the corpus order and the rollout order is **−0.200**, exact permutation
p = 0.71 two-sided, p = 0.67 for the directional "the corpus order predicts the
rollout order". The recipe the corpus scores at 0.042 lands at 0.708 here, above
four of the five recipes that beat it in the corpus.

This is not seed noise. The two seeds of a recipe differ by 0.0082 on average
(per-run sd **0.0069**), so the 0.0713 spread is **10×** the noise floor — the
recipes really do differ, by about seven accuracy points. They just differ in a
different order, and by a tenth as much.

Two things follow, and they cut in opposite directions for this document.

The first is the point the crossing was built to make: **~90 % of a corpus score
is the executing agent, not the recipe it wrote.** That is the confound stated as
a measured quantity rather than a caveat, and it explains why the agent-family
lookup table is such a strong baseline — the table is reading the thing that
actually varies.

The second is a threat to the predictor's interpretation, and it is the one
control that is still open. Stage A is evaluated against corpus accuracies that
are ~90 % agent, so "the trajectory reader predicts the score" is substantially
"the trajectory reader identifies the agent". The two surface-transcription
explanations are closed — not the printed score (redaction) and not the run id
(the `noid` ablation) — but reading a trajectory and recognising *how this agent
works* is not surface transcription and neither control touches it. Settling it
needs the blocked half of the crossing below: the same recipe under different
agents.

Two limits on this result, both real. **This 6-set cannot rank rankers**: with six
candidates and three picks, the oracle scores 0.0000, taking them in corpus order
scores 0.0011, and picking three at random scores 0.0052 — every arm is inside the
0.0069 seed sd, so nothing about *choice-set ranking* can be concluded from these
runs, only about how much of the corpus spread is recipe. And **"recipe" here is
the extraction schema's recipe** — sources, subsets, caps, epochs, batch size. The
corpus rows also carry learning rate, scheduler, precision, sequence length, PEFT
and RL flags, which the executor holds fixed. A fairer statement than "the recipe
is worth 10 %" is: *the part of the recipe this project can extract from a
trajectory is worth 10 %.* Whether the unextracted part carries the rest, or the
agent does, this crossing does not separate.

### The other half is blocked, not descoped

The crossing has two halves. This is the one that varies the recipe under a fixed
executor. The other — **the same recipe executed by different agents** — needs
agent credentials inside the PTB container, and the live `.env` carries none: only
`POST_TRAIN_BENCH_*` infra variables and `HF_HOME`. That half is blocked on
credentials or Vertex plumbing, not on design, and nothing below should be read as
covering it.

## Reproducing

```bash
bash rollout/setup.sh                   # pinned private PTB checkout + both agents
sbatch rollout/hv_pack.sbatch r733.s0 r699.s1 r600.s0 r544.s1 r401.s0 r042.s1 base
sbatch rollout/hv_pack.sbatch r042.s0 r401.s1 r544.s0 r600.s1 r699.s0 r733.s1 base

pip install scikit-learn                # ceiling.py only; the battery needs nothing extra
cd tools/splitdx
python3 run.py designs/owner.py designs/owner2.py   # per-design detail
python3 compare.py                                  # the ranking table
python3 kfold.py                                    # the recommended design
python3 recipe_signal.py                            # is there anything in the recipe? (143 rows)
python3 recipe_signal2.py                           # the same question at 1175 rows, finer features
python3 stderr_leak.py                              # which columns are the label in disguise
python3 mask_probe.py                               # would masking the agent do instead of holding it out?

cd ../..                                            # extraction, from the repo root
python3 tools/extract_recipes.py                    # all 1175, resumable, ~15 min, ~$244
python3 tools/extract_agree.py                      # cheap tier vs the 143 heavy records

python3 tools/traj_read.py --arms selfreport,features,effort   # free
python3 tools/traj_read.py --arms summary,recipe,redact,raw    # ~$470, resumable
python3 tools/traj_read_report.py                              # every cut in the section above

python3 tools/choice_rank.py --stage a           # 1175 calls, ~$125, resumable
python3 tools/choice_rank.py --stage b --topk 6  # 4568 calls, ~$884, all 188 choice sets
python3 tools/choice_rank.py --stage b --topk 10 # 4086 more, ~$793, and worth ~0.0005
python3 tools/choice_rank.py --stage noid        # 1175 calls, ~$125, the run-id ablation
python3 tools/choice_rank_report.py              # the regret tables, free
```

`run.py` evaluates the shipped split first as a positive control and exits
non-zero unless it reproduces per-agent regret@3 = 0.0, Spearman = 0.7507 and
`agent_model` R² = 0.6632. If the control does not reproduce, nothing below it
means anything. Set `OMP_NUM_THREADS=4`; on a many-core box sklearn oversubscribes
badly on data this small.
