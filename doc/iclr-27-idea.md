# ICLR-27 idea: one learned outcome predictor, two head-to-head wins

**Status**: living document — append, do not rewrite history. **Opened** 2026-08-25.
**Companions**: `doc/spec/2026-08-24-repo-architecture.md` (how we run things),
`doc/reference/harness_facts/` (what upstream actually ships), `scope/*.yaml` (which tasks count).

Facts carry **VERIFIED** (checked against a paper, the repo, or a computation recorded here)
or **UNVERIFIED** (my claim, with the check that would settle it). Never promote a tag
without recording the evidence.

---

## 1. One-liner

> We're building a single learned outcome predictor for AI-research agents, and shipping it as
> two head-to-head wins: beating **DataMaster** on **PostTrainBench** and beating the
> **AI Research Preference Model (RPM)** on **AIRS-Bench**.

The long form:

> Both systems decide where to spend GPU budget with a mechanism that ignores outcomes it has
> already observed — RPM uses a frozen pairwise judge that sees only the current run, DataMaster
> uses a content-blind UCB scheduler — so we replace both with one model trained on prior run
> logs that predicts the performance delta of an unexecuted candidate, with calibrated
> uncertainty.

Showing that **one** model does both jobs is the claim. Beating either alone is not a paper.

## 2. The two targets and their shared blind spot

### 2.1 RPM — AI Research Preference Models (arXiv 2608.13940, FAIR at Meta, 14 Aug 2026)

Ranks unexecuted sibling candidates in AIRA-dojo's child-creation step so only the winner is
executed. **VERIFIED** from the paper:

| | |
|---|---|
| Form | Frozen pretrained LLM, **no task-specific training**, **pairwise** tournament knockout over 15 candidates |
| Why ordinal | "language models perform unreliably when forecasting absolute metrics or execution outcomes" — so they reformulated selection as relative ranking |
| Context | BFS from the parent over the **current** tree, non-buggy nodes only, each with its validation score |
| End-to-end (AIRS-Bench, 20 public text/tabular tasks, 24 h × 1 H200, 10 seeds, Qwen3.6-27B for both operator and RPM) | No-RPM 0.684 → inference-only **0.711** → agentic **0.729**; validation oracle 0.748, test oracle 0.759 |
| Efficiency | Reaches the baseline's 24 h score in 14.88 h / 15.50 h, <2/3 of the execution budget |
| Offline accuracy (1,000 sibling pairs, 40 unreleased image/video/audio tasks, random floor 50%) | single frontier model 64.66–67.44%; majority vote 68.04%; LLM-arbiter ensemble **69.35%** |
| Agentic offline accuracy by pilot budget | 5 min **78.52%**, 30 min 82.78%, 4 h 84.02% |
| Label | **subtree-max** — best test score anywhere in the candidate's subtree, chosen to reward "long-term fixability and extensibility" |
| Admitted limitations | offline data is off-policy (different backbones, different modalities), the subtree-max label inherits greedy-search bias, integration only at child creation, one backbone, one benchmark |

Two numbers do most of the work for us:

- **Frozen priors are worth 19 points over chance (69.35% vs 50%); five minutes of real
  execution is worth 28 (78.52%).** The paper's own data says empirical signal dominates
  in-weights priors.
- **5 min → 4 h buys only 5.5 more points.** The cheapest observation carries most of the
  information. Downscaled pilots are viable; long pilots are not.

### 2.2 DataMaster (arXiv 2605.10906) on PostTrainBench (arXiv 2603.08640)

**VERIFIED** from the papers:

- PostTrainBench: post-train a base LLM under a bounded budget (10 h / 1×H100 officially;
  DataMaster ran 12 h), **no training data provided**, agent must find data on the web. Seven
  capabilities: AIME25, ArenaHard-Writing, BFCL, GPQA, GSM8K, HealthBench-Easy, HumanEval.
  Frontier agents 23.2% vs official instruct models 51.1%. Records reward hacking: training on
  the test set, downloading existing instruct checkpoints instead of training, using found API
  keys for unauthorised synthetic data.
- DataMaster: DataTree (red nodes discover external data / black nodes exploit it) + shared Data
  Pool + Global Memory. Qwen3-1.7B-Base, 12 h, **GLM-5 backbone**. Avg 8.47% → **31.17%**
  (human reference = Qwen3-1.7B instruct, 46.91%). GSM8K 9.02 → **49.43** (human 86.95),
  HumanEval 8.53 → **43.29** (human 60.36). Leakage checked **for GPQA only** (their Fig. 4).
- The scheduler is a UCB variant over **branch reward statistics** — cumulative reward, visit
  counts, a decaying exploration coefficient. It never reads what a pending node proposes.

### 2.3 The shared blind spot

| | RPM | DataMaster |
|---|---|---|
| Trained on prior outcomes | no (frozen by design) | no (UCB has no parameters) |
| Sees other runs / seeds / tasks | no — one tree per judgment | no — Global Memory is within-run |
| Output | ordinal, pairwise | a priority number from ancestor statistics |
| Can it say "run nothing here"? | no — needs N candidates to compare | only via the exploration term, content-blind |

AIRS-Bench is 20 tasks × ≥10 seeds = ≥200 trees, and the tasks come in near-duplicate clusters
(four QM9 targets on one dataset). Every one of those trees is thrown away. PostTrainBench ships
**1,842 public trajectories, 62 agent configurations × 7 benchmarks** (see the README download
table) — an outcome corpus that no current method reads.

## 3. The model

$$p\big(\Delta \mid \text{context},\ \text{hypothesis},\ [\text{optional cheap observation}]\big)$$

- **Pointwise, not pairwise.** One executed run yields one label directly; no pairing required.
  A pairwise preference is recoverable by subtracting two predictions, so this strictly dominates
  RPM's form.
- **Cardinal with uncertainty, not ordinal.** Ordinal answers "which of these N", at a moment
  when N candidates already exist. Cardinal answers *whether to run anything*, *how much budget a
  branch deserves*, and *when to stop* — which is what a scheduler needs and what UCB
  approximates from history alone.
- **Δ, not the absolute score.** Absolute score is dominated by "how hard is this task" and "how
  strong is this base model"; a model that learns per-task means alone gets a flattering R².
  Predicting the change conditioned on context removes that free win.
- **Encoder, not regressor, for the LLM part.** Let the LLM embed each open-world artefact (a
  freshly discovered HuggingFace dataset, a code diff, a plan) and let a light structured head
  produce the numbers. RPM found LLMs unreliable at emitting metrics; the fix is to stop asking
  them to emit metrics, not to give up on cardinality.

### Three departures, each a separate ablation

1. **Learned on cross-run logs** vs frozen. Ablate: same architecture, no training.
2. **Cardinal + uncertainty** vs ordinal. Ablate: take our model's predictions, discard
   magnitude, keep only the sign of the difference.
3. **Pilot as a requested action** vs a fixed stage. With no observation the model returns a
   prior; a downscaled run (subsampled data, one fold, few epochs) conditions in as a partial
   observation and the posterior contracts. The model buys a pilot only when its own uncertainty
   exceeds the pilot's cost. RPM ships inference-only and agentic as two disjoint systems; here
   they are two conditioning sets of one model.

## 4. The hierarchical verifier tower

Proposed 2026-08-25 as the concrete mechanism for the empirical levels: candidates pass through
staged gates, losers exit at the cheapest level that can kill them.

```
  A    B    C
  │    │    │
  ▼    ▼    ▼
 [L0 correctness]   unit tests / gradient + shape checks    ~seconds
  │    │    ✗  C out: unit test fails
  ▼    ▼
 [L1 implementation] micro-benchmark / parity              ~minutes
  │    ✗  B out: 3× throughput regression
  ▼
 [L2 recipe]        short training + curve extrapolation   ~ten minutes
  │
  ▼   A survives, predicted ≈ 0.62
```

**What is right about it.** L0 is close to free money: RPM's prompt was optimised with MIPROv2
into an instruction that tells the judge to "remain tolerant of minor, fixable issues"
(**VERIFIED**, their Appendix A.1), i.e. the frozen judge is explicitly told *not* to reject on
correctness — while AIRA-dojo carries a separate Debug operator, so broken nodes are a real
population. A seconds-long import/shape/unit-test gate removes exactly the class the judge is
instructed to ignore. And every elimination is a labelled training example produced at the price
of the cheapest level that could produce it — **the tower is the data-collection apparatus for
§3, not an alternative to it.**

**Three objections that must be answered before building it.**

1. **The compute is not counted.** RPM's whole selling point is 0.66 h of inference across a 24 h
   run. AIRA-dojo generates 15 candidates per step; L0 (seconds × 15) is free, but L1 (minutes ×
   ~5 survivors) + L2 (ten minutes × ~2) is 30–40 min *per child-creation step*. Over a run with
   tens of steps, **the tower as drawn does not fit in the budget**. The agentic RPM caps its
   pilot at 5 min for exactly this reason and *still* slows early progress before overtaking.
   → Fix the per-step budget first (match 5 min), then size the levels to fit.
2. **A cascade can only reject, and the ground-truth label rewards recoverability.** L1 kills B
   for a 3× throughput regression; B's approach might have won after the Debug operator fixed it.
   RPM's label is deliberately subtree-max, rewarding fixability and extensibility — a tower that
   eliminates on immediate correctness and immediate throughput optimises the opposite quantity.
   → Measure the **per-level false-negative rate against subtree-max** on historical logs
   *before* writing any gate. This number decides whether the tower is admissible at all.
3. **The real opponent is agentic RPM, not the black box.** Agentic RPM already runs pilots — it
   is an unstaged, budget-unaware tower at 0.729. Beating inference-only (0.711) is free and
   meaningless. The testable claim is narrower: **at equal per-step compute, staged early-exit
   with extrapolation beats one undifferentiated 5-minute pilot agent.** There is evidence this
   is winnable: the paper reports its pilot agent schedules its budget too conservatively and
   needs two hacks — overstating the remaining time and a separate feedback model proposing the
   next experiment — to be productive. Staged scheduling is the principled fix.

Interpretability and fault localisation are real engineering virtues but win no benchmark points.
Pitch them as means (faster iteration, training data) and never as the deliverable.

**The synthesis.** Do not run the tower open-loop:

```
 predictor ──decides how far up the tower each candidate climbs──▶ tower
     ▲                                                              │
     └──────────────── each level's reading updates the posterior ──┘
```

Per-step cost becomes adaptive rather than fixed, which dissolves objection 1. Note where the
risk concentrates: L0/L1 remove broken nodes (hygiene), but **which of two working candidates is
better is decided entirely at L2**, so system accuracy ≈ L2 accuracy. Curve extrapolation is
plausible on the smooth QM9 MAE curves and much harder on the noisy short time-series tasks;
measure it per task, not in aggregate.

## 5. What our task set does to the claim

### 5.1 AIRS-Bench

**Scope discrepancy, resolve first.** `scope/airs.yaml` currently lists 4×QM9 + ZINC + Yelp +
DuoRC + SQuAD. The working list discussed on 2026-08-25 was 4×QM9 + ZINC + the three time-series
tasks (WebTraffic / Rideshare / SolarWeekly), which the scope header records as *backup*. Two
different 8-task sets are in circulation; the overlap is 5. Pick one and write it down before any
run, or the two tables will disagree silently.

**ZINC is a shortcut task.** From prior AIRS work: the ZINC graph-regression target is penalized
logP (`logP − SA − #cycles>6`), exactly computable with RDKit, no training required; in an
earlier experiment the arm that trained a real model *lost*, and ZINC together with APPS carried
81% of an apparent between-arm gap. **UNVERIFIED in this repo** — the check is to read
`third_party/airs-bench/.../GraphRegressionZincMae/{prepare.py,evaluate.py,metadata.yaml}` and
confirm the target column, then score a pure-RDKit submission. At 1/8 of a subset instead of 1/19
of the benchmark, its weight is inflated ~2.4×. A learned predictor trained on these logs will
correctly learn "the node that imports rdkit wins" — which measures shortcut discovery, not
research. **Drop it or report it as its own column.**

**The four QM9 targets are not four independent tasks. UNVERIFIED, cheap to settle.** Cv, G,
R²⟨abs⟩ and U0 are all *extensive* properties — they grow with molecule size and are largely
predictable from atom counts; QM9's genuinely hard *intensive* targets (HOMO, LUMO, gap, dipole)
are all absent. Worse, G = U0 + thermal corrections, and the two are very highly correlated
across QM9, so they are close to the same question asked twice. Half the subset's weight may rest
on one-and-a-bit problems. **The check: fit a linear regression on atom counts, submit it, record
the normalized score.** Same for the time-series tasks with a seasonal-naive / recent-median
baseline (the Kaggle Web Traffic competition was famously near-medal with a median rule). Run
these floors *before* committing the task set.

**Equal weight is not equal resolution.** The normalized score is
`NS = (φ(s) − φ(s_min)) / (φ(s_sota) − φ(s_min))` with `φ(s) = −log10(|s − s_opt|)`
(**VERIFIED**, `doc/reference/harness_facts/airs_bench.md` §168). Computed from
`scope/airs.yaml` on 2026-08-25:

| Task | denominator | NS per 10× MAE improvement | NS per +0.05 accuracy |
|---|---|---|---|
| QM9-Cv | 3.800 | 0.263 | — |
| QM9-G | 6.172 | 0.162 | — |
| QM9-R2Abs | 5.297 | 0.189 | — |
| QM9-U0 | 6.618 | 0.151 | — |
| ZINC | 2.756 | 0.363 | — |
| Yelp | 0.566 | — | 0.196 |
| DuoRC | 0.271 | — | 0.157 |
| SQuAD-EM | 0.848 | — | 0.222 |

Read the two extreme rows together: **a full order-of-magnitude MAE improvement on QM9-U0 (+0.151
NS) is worth slightly less than five accuracy points on DuoRC (+0.157 NS).** Under a flat mean,
the subset average will be driven by the QA/classification tasks and will barely register real
progress on the regression tasks. Report per-task and per-family scores; never lead with the
mean.

**You cannot compare against 0.711 / 0.729 directly.** `s_min` (= `estimated_worst_score`) *is*
shipped in each task's `metadata.yaml` (**VERIFIED**, confirmed by the maintainer in airs-bench
issue #6), but Meta's per-run raw scores are **not** released, and no
normalisation/aggregation code ships in the repo. So we can compute NS consistently for *our own*
runs but cannot reproduce theirs. **Beating RPM means reproducing RPM ourselves as a control
arm** under our anchors, our harness (Harbor + `hv/adapters/airs.py`) and our seeds. Citing their
published number as the baseline is not a delta.

**Backbone parity is mandatory.** RPM deliberately used Qwen3.6-27B for *both* the candidate
generator and the judge, so gains come from the framework rather than a stronger selector; using
a frontier model would win for free and mean nothing. Same for DataMaster: GLM-5 backbone,
Qwen3-1.7B-Base, 12 h. Match, or report both.

**Budget.** 8 tasks × 10 seeds × 24 h = 1,920 GPU·h per configuration. Three configurations
(no-selector baseline, RPM reproduction, ours) ≈ 5,760 GPU·h ≈ 30 node-days on an 8-GPU node.
Pin each cell to one card with `srun --exact --gres=gpu:1` — a per-cell `CUDA_VISIBLE_DEVICES` is
a default the agent can overwrite, and a cell has previously been measured across four cards, one
of them its partner arm's slot.

### 5.2 PostTrainBench — start with GSM8K and HumanEval

Right choice, and for a better reason than "DataMaster gained most there": **both are
verifiable.** GSM8K is exact-match on a numeric answer, HumanEval runs unit tests. That means
synthetic data can be filtered automatically *and* the predictor's training labels are clean —
ArenaHard-Writing's LLM judge would inject noise straight into the training signal. Headroom is
ample: GSM8K 49.43 → 86.95, HumanEval 43.29 → 60.36.

**They are also the two highest-contamination tasks in the suite.** GSM8K test items leak into
public math instruction sets and HumanEval into code instruction sets routinely, and DataMaster
audited leakage for GPQA only. Two consequences:

- A win here without a decontamination audit is not a result.
- More seriously, **the predictor will learn contamination as a winning strategy** if the labels
  permit it. This is not a risk, it is a certainty: the label is the score, and leaked data does
  raise the score. The annotation pipeline needs an independent overlap check (n-gram +
  embedding) and leaked configurations must become **explicit negative labels**, not merely be
  dropped. Teaching the model to recognise the hack is a result in its own right, and connects
  directly to the reward-hacking taxonomy PostTrainBench already documents.

## 6. What to measure first, cheapest first

Nothing below needs the model to exist.

1. **Trivial-baseline floors** on every candidate task: RDKit-direct for ZINC, atom-count linear
   regression for the four QM9 targets, seasonal-naive / recent-median for the three time-series
   tasks. Output: a "where is the floor" table. This decides the task set.
2. **U0 vs G correlation** across QM9. Decides whether they are two tasks or one.
3. **Per-level false-negative rate against subtree-max**, from existing logs. Decides whether the
   tower is admissible.
4. **Per-task validation↔test rank correlation** across nodes. RPM found near-chance
   override-win-rate on final-node selection because Hidden Consistent Evaluation makes
   validation track test well — but that is an average. Tasks with weak validation (SolarWeekly:
   137 series, horizon 5) are where a predictor can beat the validation oracle; tasks with strong
   validation leave pre-execution selection as the only lever. This map tells us where the
   headroom is.
5. **Contamination audit** of GSM8K and HumanEval against whatever data corpus we index.
6. **Learnability probe**: fit the baseline ladder — per-task mean Δ; constant-zero; GBDT on
   structured config features — on the 1,842 PostTrainBench trajectories. If the GBDT already
   matches an LLM encoder, the story is "post-training outcomes are highly predictable", which is
   still publishable but is a different paper. Establish this before building anything large.

**Metric warning for step 6**: most proposed changes have Δ ≈ 0, so a constant-zero predictor
wins on MSE while being useless. Report **top-k regret** (pick k by predicted value, execute,
measure how far the best result falls short of the oracle) as the primary metric, and report
**calibration** (regression slope, reliability curve) separately — a model that inflates every Δ
threefold has a perfect Spearman and ruins every budget decision. This failure mode does not
exist for an ordinal judge and is the price of going cardinal.

## 7. Open questions

- Where is the state/action boundary? A world model over *the search tree* has no learnable
  dynamics — the transition function is the agent's own LLM, which is why RPM degenerated to a
  comparator. A model over *the training run* (state = checkpoint + consumed mixture + step;
  action = add source X at ratio p, algorithm A, N steps; transition = actual SGD) has real
  structure: scaling laws, curve shapes, cross-model transfer. **Only the second is worth
  calling a world model.** Are we committing to that boundary?
- Do we need multi-step at all? Post-training is genuinely sequential (SFT → DPO → RL; staged
  mixtures), but a rich enough `context` may let a one-step evaluator internalise composition.
  Try that before imagining rollouts. Training a search policy inside the model invites model
  exploitation and needs pessimism plus periodic real-rollout recalibration — future work unless
  the one-step version is already solid.
- Does the predictor transfer across the two benchmarks, or are we training two models and
  claiming one architecture? The stronger claim needs a shared representation and a cross-target
  transfer experiment.
- Which capability trade-offs does it need to represent? Post-training's real structure is that
  capabilities trade off (DataMaster: GPQA 31.02 above the instruct model while AIME sits at
  3.33). A scalar target cannot express that; a multi-task curve target can.

## 8. Changelog

- **2026-08-25** — Opened. Positioning against RPM (2608.13940) and DataMaster (2605.10906) /
  PostTrainBench (2603.08640); the cardinal cross-run predictor; the verifier tower with three
  objections and the closed-loop synthesis; AIRS task-set findings including the computed
  denominator table; GSM8K/HumanEval choice and its contamination consequence; the measurement
  ladder.
