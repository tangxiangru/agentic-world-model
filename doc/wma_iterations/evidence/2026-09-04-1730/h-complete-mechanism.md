# H complete mechanism readout — w14r01..04

Bounded read-only review of the fully harvested H arm, 2026-09-04. This extends the completed [w14r04 review](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-1730/h-first-cell.md) with eligible-card and uptake checks for w14r01–03. No Opus 4.8 in-flight trace was read; no model, Slurm, or experiment mutation was made.

The authoritative result snapshot is [h-latest-results.json](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-1730/h-latest-results.json): **4/4 validator-clean** cells and no PTB judge flags. PTB accuracies are w14r01 **.7361637604**, w14r02 **.7073540561**, w14r03 **.8006065201**, w14r04 **.6846095527**; arithmetic mean **.7321834723** and sample SD **.05025711856** (73.218347% ± 5.025712 pp SD). These four cells are exploratory screening, not a score-gain or promotion test. [Preregistered screening limit](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-03-wma-round04-probe-selection.md:134)

## Frozen eligibility

H eligibility is fixed by the locked proposal: a named average of at least two checkpoints with identifiable prelaunch lineage; an evaluation-only soup card counts. Eligibility is independent of the outcome or the WMA label. The same preregistration requires at least three eligible soup **cards**, and actual post-verdict action for uptake. [Eligibility and uptake rule](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-03-wma-round04-probe-selection.md:90)

**Six eligible cards across three of four cells** (w14r01: 3; w14r02: 1; w14r03: 0; w14r04: 2):

| Eligible card | Frozen soup opportunity | Citation | Observed outcome |
|---|---|---|---|
| w14r01/exp-05 | Three-way exp-04 trajectory soup; dev-150 | [Frozen card](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r01/task/memory/cards/exp-05.yaml:44) | iterate; .7533 vs ingredient mean .7400 / best .7600 |
| w14r01/exp-06 | Evaluation of soup plus two named checkpoints; full n=1319 | [Frozen card](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r01/task/memory/cards/exp-06.yaml:48) | adopt soup; .7354 vs top checkpoint .7407; paired p=.603 |
| w14r01/exp-08 | Four-way soup across two stage-2 trajectories; full n=1319 | [Frozen card](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r01/task/memory/cards/exp-08.yaml:44) | reject; .7392 vs best ingredient .7407; p=.932 |
| w14r02/exp-06 | Four-way same-trajectory soup; dev-300 | [Frozen card](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r02/task/memory/cards/exp-06.yaml:35) | reject; .7067 vs best ingredient .7100 |
| w14r04/exp-05 | Three-way sequential-final soup; probe-300 + dev-150 | [Frozen card](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r04/task/memory/cards/exp-05.yaml:37) | adopt soup; .7333/.7067 vs best ingredients .7167/.6800 |
| w14r04/exp-06 | Two named wider soups: six-way A and four-way B; probe-300 | [Frozen card](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r04/task/memory/cards/exp-06.yaml:36) | reject both; A=B=.7233 vs incumbent soup .7333 |

I reproduced every eligible card's `plan_sha256` from its harvested YAML and every locked script SHA from its harvested bytes; all match the corresponding `.lock.json`. The list therefore comes from frozen prelaunch lineage. w14r01/exp-06 qualifies because its locked proposal explicitly evaluates the named exp-05 soup against identifiable checkpoints, as the preregistration says evaluation-only soup cards qualify. w14r04/exp-06 counts once even though it contains two soup variants. All other cards are individual training, decode, baseline, or delivery checks. w14r03 supplies no soup opportunity and cannot inform the H mechanism.

The minimum opportunity count is nominally met (**6 ≥ 3**), but opportunity is concentrated in three cells and repeated decisions reuse within-cell lineages and evaluation sets. There are four independent scientist cells for the PTB score distribution, three cells with an eligible H opportunity, six eligible card decisions, and more than six constructed/evaluated soup artifacts because two cards contain multiple variants. Those denominators are not interchangeable.

## Time-short-default adherence

**Positive endorsement of “when time is short, default to averaging”: 0/6 eligible cards. No endorsement: 6/6.** The preregistered target of zero positive endorsements is met on this text metric. None of the six eligible verdicts uses time pressure alone as the positive reason to average. All six return L3 `yes`, but their bases combine observed candidate geometry/history, an existing safe incumbent, concrete planned cost, remaining time, and a decision-specific measurement or guard. Examples include w14r01/exp-05’s flat historical C6 prior and explicit warning about selection noise, [verdict](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r01/task/memory/cards/exp-05.verdict.json:114); w14r01/exp-08’s statement that the likely outcome changes nothing and a precommitted adoption floor protects the incumbent, [verdict](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r01/task/memory/cards/exp-08.verdict.json:117); and w14r02/exp-06’s warning that one ingredient was unmeasured plus a recommendation to retain the incumbent on a tie, [verdict](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r02/task/memory/cards/exp-06.verdict.json:151). The w14r04 classification remains 0/2; see the prior review’s “Primary adherence numerator.”

This is adherence, not demonstrated utility. The absence of the prohibited positive prior does not prove that H caused better soup or ingredient choices, and it does not offset the failed scope guard.

## Bounded advice-to-action check

Only three actual decision traces are cited:

1. **w14r01 after exp-05 — advice matched a later measurement decision, attribution remains ambiguous.** After the exp-05 verdict returned, the scientist launched the already locked three-way soup unchanged and printed the verdict suggestions (decompressed [solve trace](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r01/solve_parsed.txt.gz) lines 16058–16064). The verdict suggested that the decisive comparison was a full n=1319 read of the soup and checkpoint-1484 rather than another n=150 selection, [exp-05 suggestion](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r01/task/memory/cards/exp-05.verdict.json:180). The next locked card performed that full comparison, including a third named checkpoint, [exp-06 frozen command](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r01/task/memory/cards/exp-06.yaml:80). This is an actual action after the returned advice, but the closed exp-05 card independently prescribes the same next step; it is not clean evidence of an H-induced change.

2. **w14r02/exp-06 — direct precondition uptake, no ingredient uptake.** The returned verdict’s first precondition required checking for 883 tensors, two shards, and about 8.6 GB before evaluation. The scientist then ran the original four-ingredient locked merge and, after it returned, printed **883 tensors / 8.6 GB** before launching evaluation (decompressed [solve trace](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r02/solve_parsed.txt.gz) lines 13073–13133). This is verified validity-check uptake. It did not adopt the suggested three-scored-ingredient variant or score the unmeasured checkpoint-170 first; ingredient selection stayed unchanged. [Suggestions](/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/w14r02/task/memory/cards/exp-06.verdict.json:157)

3. **w14r04/exp-06 — selection advice not taken.** The scientist launched the unchanged two-variant script immediately after the verdict, running both A and B; it did not take the B-only suggestion or restore `--verify`, and it verified the incumbent only after both probes. The variants failed the frozen gate and the incumbent remained. Evidence and exact lines are in [w14r04 review](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-1730/h-first-cell.md), “Actual actions after advice.”

Across the arm, these examples show one measurement action that matches advice but is independently motivated, one direct structural-check uptake without an ingredient change, and one unchanged selection. That is insufficient to claim behavioral benefit. Beneficial soups occurred (notably w14r04/exp-05), but observed benefit is not causal uptake.

## Ledger, cost, and preserved flags

The authoritative frozen H ledger has **25 final verdicts / 14 scored / 11 original scope flags**, `L2_coverage=.727` on 11 scorable verdicts, `L2_width_over_noise=4.9453`, `L3_hit=.6`, and retained-final cost **$49.117**. Per cell, the retained final verdict/flag/cost counts reproduce as:

| Cell | Finals | Original flagged verdicts | Retained-final cost |
|---|---:|---:|---:|
| w14r01 | 8 | 3 | $15.8318 |
| w14r02 | 5 | 2 | $9.6321 |
| w14r03 | 6 | 4 | $11.7509 |
| w14r04 | 6 | 2 | $11.9022 |
| **Total** | **25** | **11** | **$49.1170** |

The 11 existing flags are preserved exactly; none was cleared or semantically reclassified. They are verdict-level flags and may contain multiple operations, as documented for w14r04. Empty PTB judge-flag lists are a separate surface. The preregistration makes unchanged H scope discipline a guard: if it fails, H cannot be promoted. [Scope guard](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-03-wma-round04-probe-selection.md:127) With **11 original flags**, that guard fails regardless of the 0/6 time-short-default result.

## Mechanism conclusion

H removed positive reliance on the clock as a default in every eligible verdict observed (**0/6 positive endorsements**), across enough eligible cards to read that text metric. The arm does not establish behavioral benefit: opportunities are clustered, advice-to-action evidence is mixed and often redundant with frozen or scientist-authored rules, and no clean counterfactual attributes a soup, ingredient, or incumbent change to H. The unchanged scope discipline independently fails the preregistered guard. **No promotion claim is warranted or made.**
