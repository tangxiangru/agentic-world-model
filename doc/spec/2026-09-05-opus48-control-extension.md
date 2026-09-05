# Opus 4.8 control extension across task and base model

Status: preregistered for validation and asynchronous submission. This is a
control study; it changes neither the WMA skill nor the WMA runtime.

## Question and evidence

The first Opus4.8 wave cannot estimate a WMA effect: every completed GSM8K and
BFCL S/M/J cell is judge-flagged, and the current skill has no delivered
verdict. The usable no-WMA controls are also too thin for a stable effect:
GSM8K raw and protocol each have three clean cells with 12.06 and 10.40
percentage-point sample SD, while BFCL has four clean protocol cells but only
one clean raw cell. HumanEval has semantic-contamination and lifecycle risks.

This extension asks two narrower questions that do not depend on the failed
WMA path:

1. Does the experiment protocol change end-to-end score or clean-completion
   rate on the existing Gemma3 GSM8K and BFCL settings when four new concurrent
   replicates are added per arm?
2. Does the same raw-versus-protocol contrast generalize on BFCL to
   Qwen3-4B-Base and SmolLM3-3B-Base?

The second question tests a base-model dimension absent from the first wave.
All three model revisions already exist in the frozen local cache. BFCL is used
because the completed scores show materially lower dispersion than GSM8K and
because its evaluator is locally available. HealthBench and ArenaHardWriting
are not launched because their evaluator requires an unavailable
`OPENAI_API_KEY`; GPQA remains gated; HumanEval awaits its semantic guard; AIME
remains promotion-only.

## Frozen design

Scientist: Vertex `claude-opus-4-8`, high effort, 200k context, Claude Code
2.1.219. Each cell receives one H100, 16 CPU, 128 GiB memory, 400 GiB scratch,
a ten-hour scientist window, the official judge profile and the existing
validated containers. Each manifest has four independent replicates.

The eight four-cell manifests are:

| task | base | raw | protocol-only |
|---|---|---|---|
| GSM8K | Gemma3-4B-PT | c71r01–04 | c72r01–04 |
| BFCL | Gemma3-4B-PT | c73r01–04 | c74r01–04 |
| BFCL | Qwen3-4B-Base | c75r01–04 | c76r01–04 |
| BFCL | SmolLM3-3B-Base | c77r01–04 | c78r01–04 |

Raw cells install no AWM component. Protocol-only cells install the public
experiment protocol from `31b854bbc5e1f7f66685a8ec0d43845a6c2472f1`
in single-decision mode and attach no WMA. Therefore the strict verdict gate is
not bypassed: these cells make no WMA claim and require no WMA verdict.

The PostTrainBench execution worktree is
`9695e32f3a160b8c7e927cbb4de66727de7c72ad`; against the first-wave
`e62036f0c244995a6f45496522d3310b239383c6`, the only changed path is the
standalone `src/judges/smoke_claude_vertex.sh`, which is not reachable from
scientist launch, evaluation or official judging. New cells remain a distinct
runtime cohort. Pool with the first wave only as an explicitly labelled
sensitivity after verifying the frozen reachable diff again.

## Readout and falsification

The unit of analysis is the validator-clean cell. For each of the four
task/base strata report separately:

- clean-completion count out of four for raw and protocol;
- score mean, sample SD, range and all individual values;
- protocol-minus-raw mean difference with uncertainty;
- scientist, judge and WMA costs separately; allocated GPU-hours and sampled
  utilization separately;
- every incomplete or judge-flagged attempt in the denominator and in a
  sensitivity table, never as a synthetic zero.

The protocol-benefit hypothesis for a stratum is falsified if the clean
protocol mean is no higher than raw. If the difference is smaller than one
pooled within-arm SD, record it as unresolved rather than extending only the
apparently losing arm. No cross-task or cross-model pooling is allowed.

This control study has no WMA promotion outcome. It may establish a reusable
control substrate only after at least eight clean cells per compared arm when
the matched first-wave cells are eligible for the same cohort or prespecified
sensitivity. Four new replicates alone remain exploratory.

## Guards

- leak: zero judge-confirmed held-out/test use for training; use public
  task-relevant data only and preserve all contamination findings;
- cost: maximum 320 nominal H100-hours for 32 cells; report actual allocation,
  API spend and failed-cell cost;
- PTB: do not change evaluator, scorer, judge, templates or completion guards;
- lifecycle: preserve every background/end-turn failure and do not selectively
  retry it inside this cohort;
- WMA: no `wma:` block, no policy conclusion, and no failed-review `proceed`
  path;
- source: exact manifest, receipt, cell, commits and canonical result path must
  be retained for every outcome.

## Queue decision

The WMA subqueue reached zero pending jobs with only three running tails. These
32 independently specified controls restore a reserve above eight after normal
backfill and use currently idle GPUs while V1 verdict enforcement is developed
and validated. They do not depend on the remaining HumanEval tails, the S0
legacy policy comparison, or a new WMA skill decision.
