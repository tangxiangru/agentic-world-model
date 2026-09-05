# w10 completed-baseline decision and probe audit — 2026-09-03

Read-only scope: completed `w10r01..04` in
`results/ptb/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2/` only. No running candidate outcome read; no repository or queue mutation; no additional model invocation. This is delegated diagnosis, not promotion evidence. Parent owns validator/ledger reproduction and frozen cohort gating.

All citations below use `B = /home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2/`. `solve_parsed.txt.gz:L` means line number after gzip decompression. JSON locators name exact fields/probe IDs and are stable under formatting.

## Findings and limitations

There are 30 final card verdicts (8+7+9+6), all L3=yes. The relock audit `/tmp/wma-relock-audit-20260903/verdict-sequences.json` has 43 requests (12+10+12+9), 42 printed verdict summaries, and one absent summary for w10r03/exp-01. All 42 recovered summaries say L3=yes; the missing request's final verdict also says yes@0.92 (`B/w10r03/task/memory/cards/exp-01.verdict.json`, issued_at 2026-09-03T07:35:06Z). Do not confuse 30 final verdicts with 43 requests or infer loss of all first-review evidence from overwritten final files.

**The two recovered L0/L1 negative cases led to useful repairs, not demonstrated harm.**

1. `w10r01/exp-02`: first L0=no@0.68, L1=yes@0.72, L3=yes@0.70 at `B/w10r01/solve_parsed.txt.gz:6485`; suggested memory smoke and batch reduction at lines 6501–6505. The scientist reduced bs16 to bs8 while preserving effective batch32, relocked, waited for L0=yes@0.86, then launched (6982, 6986, 6991). Final card result is 1.79 h and accuracy 0.7067 vs 0.0600 at n150 (+0.6467). The final reviewer itself notes the earlier smoke probably already exercised long sequences; the L0 no was conservative, not proof that original training would fail. Do not claim an avoided OOM as observed fact.
2. `w10r04/exp-03`: first L0=no@0.70, L1=no@0.62, L3=yes@0.58 at `B/w10r04/solve_parsed.txt.gz:8219`. The precondition identifies the parent GenerationConfig save trap (8238); scientist creates a training-safe parent (8259–8284), relocks and smoke-tests the save (8310–8313), obtains written weight shards (8344–8359), waits for returned positive verdict (8405), then launches main training (8432). Final card gives 0.6990 vs0.6876 at n1319, +0.0114, cost0.40h. No valid training was killed. This case supports preserving blocker discovery and the blocking relock boundary.

**Overoptimistic-L2-to-waste is not established by the prominent negative cards.**

- `w10r03/exp-05` initially looks harmful (1.44h, n150 0.7400→0.7267) but the card's `conclusion.superseded_by` explicitly records the reversal at n400: exp-05=0.735 vs exp-04=0.705; exp-05 is the submitted model. Its first rejection was wrong. Treat this as a counterexample to blindly killing flat-prior continuation, not a wasted-training exemplar. Sources: `B/w10r03/task/memory/cards/exp-05.yaml` result/conclusion; `exp-07.yaml` result.
- `w10r04/exp-06` is an observed no-gain training:0.72h;939/1319 vs941/1319;dev250 0.744→0.708; incumbent preserved. But L2=[-0.012,0.018] covers the -0.0015 delta. The verdict is flat, not a confident optimistic L2 miss. L3=yes@0.72 was justified by remaining3.1h and protected incumbent (`exp-06.verdict.json`, e11). It is a candidate opportunity-cost question, not identified causal harm from optimism.
- `w10r02/exp-05` is an actual L2 downside miss: repetition-penalty card interval[-0.01,0.018], result delta-0.024 at n500,0.25h, rejected. One miss at interval confidence0.72 is not enough to rewrite a prior, and this alone does not supply two independent harm cases.

## Repeated and low-marginal-value probe evidence

Across final verdicts:143 probes,20 `changed=none`,14.5 self-reported cpu_min attached to those20,165.8 self-reported cpu_min overall. Of the20, at least4 explicitly say not run/no dynamic probe. Therefore20/143 is not a waste rate, and14.5cpu_min is not measured GPU-idle time. History-labelled probes occur21 times with37.0 reported cpu_min, but many retrieve different change-type precedents and cannot all be called duplicates.

Direct cases for a narrower, defensible probe-policy experiment:

1. `B/w10r02/task/memory/cards/exp-01.verdict.json`, p2:1cpu_min verifying training prompt byte equality on an evaluation-only card; reviewer explicitly says it bears on later training cards, not this card's levels. p5:0.5cpu_min verifying data decontamination; explicitly C10 work for a later training card with no effect on this card. This is scope expansion while the current lock waits.
2. `B/w10r04/task/memory/cards/exp-06.verdict.json`, p5:1cpu_min recomputing the earlier111k-row corpus's prefix distribution for p1; reviewer marks none. Current-data format and shape were already checked in p1; repeated population detail had no stated level change. This is evidence for requesting an explicit discriminating outcome before an additional check, not proof all earlier-data rechecks are needless.
3. `B/w10r03/task/memory/cards/exp-05.verdict.json`, p4:0.5cpu_min rechecking the standing3% vs100% few-shot mismatch, marked none and described as a missed lever, not an L1 fault. This supports the same scope question, though a different observable outcome could potentially expose a new fault.
4. Do not suppress useful no-change probes categorically: `B/w10r02/.../exp-03.verdict.json`, p3 identifies save_total_limit=2 deleting the midrun checkpoint even though marked none. Likewise fresh disk checks are not automatically duplicate evidence, and history probes often change L2. `changed=none` is a post-probe description, not a safe pre-probe stopping rule.

## Preferred independent single-edit candidate: prospectively scoped probe stopping

Add one rule to WMA's Probes section, without changing review.py, the fence, runtime budgets, schema or scorer:

> Before each additional probe, name the unresolved question about the current card, the plausible observations that would change its verdict or a current-card precondition, and the probe's cost. If no plausible observation would change either, stop probing and write the verdict from checked evidence and calibrated priors. Do not spend the current blocking review on verification useful only to a hypothetical later card. A probe that was capable of changing the verdict but confirms the status quo remains legitimate and is recorded as changed:none.

This operationalizes the existing general “run only if result changes a level” requirement without forbidding legitimate confirming checks. It targets current-card scope and stopping, not new L0/L1 no thresholds(C), suggestion ordering(F), L3 economic thresholds(E), checkpoint policy(D), or L2 format/width(A/B). It does not promise to cache unavailable relock evidence, never accepts a changed dependency without checking it, and never relaxes the requirement to wait for a verdict before launch.

Evidence: two strongest independent citations are w10r02/exp-01 p2+p5 and w10r04/exp-06 p5, with w10r03/exp-05 p4 as sensitivity. Hypothesis: unnecessary current-lock work falls, leaving more scientist time for training/evaluation. Four-cell exploratory manifest on exactly the established runtime baseline; no treatment outcome currently read.

Suggested preregistration:
- Primary: measured request wall time per closed card including every relock, compared with matched baseline; preregister a20% reduction target as an experimental choice, not an empirical effect size.
- Mechanism: count probes explicitly serving only later-card work and probes lacking any stated current-card discriminating outcome; report confirming changed:none separately.
- Falsification: scope-irrelevant probing persists or measured wait fails to decline; any gain depends on dropped blocker detection, increased missing/invalid verdicts, or shortcutting the launch fence.
- Guards: leaks0; same PTB guard against baseline spread; actual cost<=1.5x; L0/L1 recall and save/load faults no deterioration; report relock count, outliers and idle fraction independently. Do not use self-reported cpu_min as the primary cost metric.
- Promotion requires the existing formal same-cohort/held-out gates. This is a design proposal, not an accepted candidate or evidence of score improvement.

## Secondary independent candidate: L3 refers to the as-written proposal

An alternative single edit in the L3 definition: when the **current as-written command** is predicted not to run or produce a valid candidate because an identified unresolved prerequisite, give L3=defer until that prerequisite is addressed; do not let hypothetical post-repair success turn current L3 into yes. Do not automatically convert speculative low-confidence L0 no into permanent rejection. A later returned verdict after repair can be yes normally.

Evidence: the two negative-then-repaired cases above (w10r01/exp-02 and w10r04/exp-03). This differs from E's resolvable-effect/no-cheap-discriminator economics and C's evidence threshold for L0/L1 no; it is internal conditioning consistency. However both scientists already repaired before main launch despite L3=yes, so existing traces give weak reason to expect score improvement and no demonstrated reduction in harm. Primary would be0 contradictions among grounded L0/L1 no vs L3yes and correct eventual repair/launch rate, with wrongly-killed hours and relock costs as guards. Rank below scoped-probe stopping unless the reviewer finds an additional behavioral channel. No need to launch this merely to fill the queue.
