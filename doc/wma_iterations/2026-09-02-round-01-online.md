# Round 01 online — 2026-09-02 — skill v0.2 against no-WMA control

**Status:** running; this is an open record. No promotion or skill change is
allowed until the validator-complete comparison boundary below is met.

## Variants

| label | private WMA commit / skill hash | what differs from control |
|---|---|---|
| `wma-v0.2` | first wave `e8a8599`; fixed-runtime buffers `bf87dfb`; skill `176f0a464986` in both | private Claude Opus 5 / high online sidecar, train-only history and `cpu=10,gpu=0,wall=15,turns=40` |
| `ctl` | no private WMA checkout | same public checkout and protocol; the only difference is no `wma` manifest block |

`bf87dfb` is a registered measurement/runtime repair, not a skill variant: it
allows the schema's legal pre-launch `result.execution: not_run` sentinel.
`skills/wma/`, the prompt, verdict schema, scorer and history are unchanged.

## Cells

Task: `gsm8k` · base model:
`google/gemma-3-4b-pt@cc012e0a6d0787b4adcc0fa2c4da74402494554d` ·
scientist: `claude-opus-5[1m]`, high, 1M context · 10 h · one H100 per cell.

- Core: first-wave `w01r01..08` and `c01r01..08` (8 per arm).
- Pre-result precision extensions: fixed-runtime v2 `w02/c02` and `w03/c03`
  (16 more per arm). The v1 buffers were cancelled while all 32 jobs were
  PENDING; they never replace a result.
- Held-out task this round: none. `aime2025` remains promotion-only.
- Primary end-to-end view: WMA **as attached** versus control. Sensitivity:
  **as answered**, requiring at least one valid verdict in that WMA cell.

## Results

No cell is terminal or validator-complete as of 2026-09-02 14:32 UTC. Slurm
has 16 first-wave cells RUNNING and 32 fixed-runtime cells PENDING; therefore
there is no PTB score comparison or online ledger conclusion yet.

### Inflight WMA evidence

- All eight WMA scientists invoked `exp_protocol` as their first tool action
  and all eight sidecars started with the frozen Opus 5 / high contract.
- `w01r01`, `w01r02`, `w01r04`, `w01r07`, `w01r08` produced valid exp-01 and
  exp-02 transcripts. The first verdicts used 28–32 of 40 turns, 5.1–7.2 min
  of 15, and $1.46–$1.72 shadow cost. Isolation stayed within `/session`,
  train-side `/history`, private skill and scratch.
- `w01r03`, `w01r05`, `w01r06` pre-filled `result.execution: not_run`; the old
  runtime rejected exp-01 as post-hoc. This is infrastructure evidence, not a
  WMA miss. It caused `bf87dfb` and the v2 buffer replacement. These cells stay
  in the as-attached cohort and out of as-answered until a later card receives
  a verdict.

### First four known card outcomes (not terminal PTB results)

The exp-02 transcripts expose the already-closed exp-01 proposal cards for
`w01r01`, `w01r02`, `w01r04`, `w01r08`. Their dev-150 base-checkpoint
accuracies are `{0.0667, 0.0533, 0.0533, 0.0333}`. All four found the same
mechanism: the pretrained checkpoint often fails to emit `<end_of_turn>`,
continues into fabricated Q&A, and the end-anchored numeric grader reads a
number from the tail. None shows the feared `!!!!` corrupted-serving signature
at `--max-connections 16`.

- L0 `yes` is 4/4 correct.
- L1 `yes` is 4/4 correct; the invocation ran validly despite the stated
  concurrency risk.
- The WMA named the termination mechanism correctly but put too little mass on
  the near-zero mode.
- The pre-registered Window 02 absolute-accuracy-slot candidate is falsified:
  recorded absolute intervals were `[0.02,0.40]`, `[0.10,0.50]`,
  `[0.12,0.48]`; two of three miss low. The missing slot is not the current
  bottleneck—the prior is.

All four exp-02 plans independently converge on completion-only SFT over
GSM8K-derived OpenMathInstruct-2 data with `<end_of_turn>` targets. Their WMA
L2 delta intervals are 0.30–0.40 wide, about 10× the n=150 noise floor. Whether
that width is earned remains open until the measured exp-02 deltas arrive.

## Decision

Inconclusive. Runtime/isolation/procedure are supported for the five answered
cells; skill calibration and PTB effect do not yet meet the evidence boundary.
Keep skill v0.2 unchanged.

The end-to-end conclusion requires at least eight validator-complete cells per
arm. A skill candidate additionally requires closed-card ledger evidence and
the pre-registered falsification checks below.

## Change

None to `skills/wma/`.

Measurement-only changes already made and separately attributable:

- `bf87dfb`: accept `not_run` as pre-launch in the post-hoc guard.
- `2b094c7`: replace only unstarted v1 buffer cells with paired v2 manifests.

## Evidence

- Receipts and inflight snapshots under
  `results/ptb/wma-gsm8k-gemma4b-high-r01-*`.
- PR 23 Fable windows:
  [Window 02](https://github.com/tangxiangru/agentic-world-model/pull/23#issuecomment-5510983693),
  [Window 03](https://github.com/tangxiangru/agentic-world-model/pull/23#issuecomment-5511204993).
- Runtime diagnosis and replacement contract:
  `doc/spec/2026-09-02-wma-round01-online-gsm8k-gemma4b.md`.

## Next round

Primary proposed single change, not yet applied: add a C11/C12 manual prior
that a pretrained non-instruct checkpoint evaluated through a chat template
with end-anchored numeric grading should place its expected result at the
termination floor (currently `<= 0.07`), and demand the cap-hit / missing-answer
diagnostic.

Pre-registered falsification:

1. If matched control exp-01 values are not also at the floor, the observation
   is WMA-arm-specific and the prior is wrong.
2. If the v2 pre-change WMA exp-01 verdicts already centre on the floor, the
   entry adds no useful discipline.

Competing candidate: cap L2 interval width at a multiple of the noise floor
unless cited evidence earns more width. Choose it instead only if the exp-02
deltas cluster narrowly inside the current 0.30–0.40-wide predictions; if the
wide intervals are needed for coverage, the cap is falsified.
