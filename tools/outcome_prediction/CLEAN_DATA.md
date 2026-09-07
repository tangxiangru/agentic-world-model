# Current labeled-only data policy

The current **training experiment** is
`data/analysis/wm_one_step_predictors/v1_bundle` + `v1_results`, documented in
`data/analysis/wm_one_step_predictors/README.md`. It uses the additive
`data/analysis/wm_delta_reference/01406da734fb_v1` dataset: 185 complete delta
examples (123 train / 62 test), combining 44 measured-parent examples with 141
explicitly sourced published-base references. Measured-parent-only training is
reported separately. No missing accuracy is imputed or used in any fit.

The current cleanup entry point is `wm_one_step_data.py`; its versioned output is
`data/analysis/wm_one_step/01406da734fb_v1`. See
`data/analysis/wm_one_step/README.md` for the resolved counts, remaining gaps, and
the supported loaders. Use `cohort="one_step"` (the default) for measured-parent
delta prediction; use `cohort="target"` explicitly for final-accuracy prediction.
Missing parent accuracy is never filled with zero or a training mean. The parent
score must be independently bound to an earlier official checkpoint; a dirty
parent recipe does not, by itself, invalidate that checkpoint's observed score.

For delta experiments, a missing or invalid target accuracy **or** parent/reference
accuracy makes the example ineligible for training, validation, and testing.
Exclude it before fitting any preprocessing, feature selection, or model; do not
mix target-only rows into delta training or replace missing scores with zero,
means, or model predictions. Genuine numeric zero scores remain valid. Audit
records may be retained without becoming training examples.

Verified published scores for a fixed base model may serve as explicitly sourced
references in a separately versioned cohort. The expansion above is additive;
the original frozen `wm_one_step/01406da734fb_v1` measured-parent dataset remains
unchanged at 44 examples.

The earlier predictor results below used a different, overly restrictive parent
join and mixed missing-reference rows into a nominal delta architecture. They
remain reproducibility artifacts, not validation of the intended known-parent
world model. Do not retrain from their old default cohort accidentally.

## Archived v2 policy and benchmark context

The previous data refresh was
`data/analysis/wm_clean/01406da734fb_v2`.
It remains unchanged for reproducibility, along with the broader frozen
small-predictor cohorts. The new cleanup reads its source-verified inventory and
preserves its whole-session split; it does not overwrite the old artifact.

Follow-up predictor experiments are now complete in
`data/analysis/wm_clean_predictors/README.md`. They refit only clean labeled
training rows and preserve this dataset/split; the data refresh itself remains
unchanged.

## Dataset

Hugging Face dataset `JerrrrryL/awm-gsm8k-trajectories`, pinned revision
`01406da734fb9016530bcdfeee027e3760587c6e`. All 4,336 recorder-arm files
(298,615,284 bytes) were downloaded and checked against upstream Git blob/LFS
hashes. Raw files remain in
`data/traj/raw/awm-gsm8k-trajectories-01406da734fb`.

| Arm | Official labels | Strictly screened labels | Train | Test |
| --- | ---: | ---: | ---: | ---: |
| GSM8K, original | 181 | 80 | 55 | 25 |
| AIME2025, original | 160 | 36 | 23 | 13 |
| AIME2025, new `aime2-r0` | 95 | 33 | 27 | 6 |
| Total | 436 | 149 | 105 | 44 |

There are 551 inventory entries: 149 retained, 287 labeled but quarantined,
and 115 missing/invalid labels. The old AIME inventory includes an orphan
metric without a versioned registration. The split is assigned by whole session,
stratified by benchmark/scientist, before filtering; 35 train and 14 test sessions
have retained targets. There are three genuine zero-score targets in training.
Zero and low scores are never an exclusion criterion.

## What “screened” means

Require a valid numeric official target label, a complete prospective first
registration, an unchanged first/final setup, and known literally matching
planned/final output paths (ignoring trailing slash). Preserve prior exclusions
and unresolved target-binding holds. Changed formerly rejected records need a
new review, rather than being automatically reinstated by ID. Source hashes
distinguish unchanged material from changed/unbound material.

This deliberately quarantines some good data: for example, `/ckpts/run` versus
`/ckpts/run/final` may be harmless, but is not automatically treated as equivalent.
Quarantined does **not** mean proven corruption. Code absence is unknown, not a
failed run; only timestamp-reconstructed pre-proposal code is used, never mutable
final snapshots. The screen does not prove which checkpoint bytes were executed
or independently graded.

The retained subset is not representative of the entire release. In particular,
all 24 labeled new Opus 4.6 records are quarantined: 23 have output-path
differences, and the remaining record fails registration checks (reasons overlap).
The 33 retained new AIME targets are all Opus 4.7. Do not interpret evaluation on
this subset as evaluation of all new scientists or all 95 new labels. A future
score-independent target-binding review may recover many quarantined examples.

## Training/evaluation entry point

```python
from tools.outcome_prediction.wm_clean_refresh import load_partition

bundle = "data/analysis/wm_clean/01406da734fb_v2"
train, train_labels = load_partition(bundle, "train")
test, test_labels = load_partition(bundle, "test")
```

Both loaders verify bundle hashes, label validity, clean eligibility, identities,
and session separation. Labels are separate from recipe material. Do not use the
private full inventory as a supervised training/evaluation table. It exists for
audits and later recovery only.

The exported inputs are current-proposal material for the existing fixed positive
numeric feature extractors, **not** unrestricted LLM/text-encoder input. Free text
and code can still contain implicit clues that numeric allowlists must ignore.
Generic prior observations and ancestor scores are not exported. This is not yet
a certified full-history/delta-model dataset: before joining a parent score, verify
the exact consumed checkpoint, same-session earlier ancestry, and clean context.
Do not substitute a producing card's score for a different intermediate checkpoint,
or invent a base checkpoint accuracy.

New models must be refitted on clean training rows; merely rescoring a model
trained on dirty rows does not implement the user's clean-only requirement.
Existing saved inference-only LLM forecasts lack new AIME2 rows and use different
training banks/folds. They are not a refreshed fair comparison. No new paid LLM
calls or predictor fits were made here.

To prepare a new immutable output from the pinned local download:

```sh
.venv/bin/python -m tools.outcome_prediction.wm_clean_refresh \
  --output data/analysis/wm_clean/01406da734fb_next
```

The builder refuses to overwrite existing bundles. `v1` is superseded: the final
review added the unresolved `aime-r0-20/exp-09` target-binding hold in `v2`.
Use `summary.json`, `decisions.json`, `exclusion_counts.json`, and `policy.json`
inside `v2` for counts, per-example reasons, and full pinned source provenance.

Validation: 159 targeted tests passed (102 new refresh tests plus existing dataset,
label-validity, and code-benchmark tests); Ruff passed. A separate audit checked all
download hashes, cohort membership, source binding, and split isolation.
