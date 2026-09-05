# Uptake and score levers: bounded read of the w10/c10 matched cohort

Read-only follow-up to event `20260903T180444Z-79c8d29810`; 2026-09-03.
Read AGENTS.md, wma_meta, Round 02/03 records, the event's Claude report and
operator-review.md. No repository/queue mutation, additional model invocation,
or in-flight candidate reading. This report is input to preregistration, not
an acceptance or promotion decision.

## Scope and reproduction

Repository root: `/home/robtang_google_com/gangda_workspace/agentic-world-model`.
Only the event's validator-clean, judge-clean `w10r01..04` and `c10r01..03` were
read. The c10r04 tail, all extensions, and all A–F outcomes are excluded from
this frozen seven-cell diagnosis. No earlier runtime cohort was pooled.

Provenance roots (relative paths below resolve under the repository root):

- WMA receipt: `data/ptb/batches/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2/formal-2026-09-03T014918.962581+0000.json`.
- WMA manifest: `experiments/posttrainbench/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2.yaml`.
- Control receipt: `data/ptb/batches/wma-gsm8k-gemma4b-high-r02-ctl-x4-v2/formal-2026-09-03T014940.966939+0000.json`.
- Control manifest: `experiments/posttrainbench/wma-gsm8k-gemma4b-high-r02-ctl-x4-v2.yaml`.
- Both spec: `doc/spec/2026-09-02-wma-round01-online-gsm8k-gemma4b.md`.
- WMA result root B: `results/ptb/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2`.
- Control result root C: `results/ptb/wma-gsm8k-gemma4b-high-r02-ctl-x4-v2`.
- Each cited card: `B|C/CELL/task/memory/cards/exp-NN.yaml`; associated verdict
  is `exp-NN.verdict.json`, source trace is `B|C/CELL/solve_parsed.txt.gz`.
- Public runtime `ae46724`; WMA v0.2 hash `176f0a464986`. The frozen event
  `results.snapshot.json` is the validator/provenance authority; this bounded
  task did not independently rerun the expensive remote discovery.

Ran `tools/wma-rca/cells.py` and `uptake.py` on the two roots. Ran `timeline.py`
on w10r01–04 and c10r03 and manually cross-checked relevant actual command
calls rather than taking its launch inference literally. Outputs, decompressed
traces with original line numbers, and card summaries are in
`/tmp/wma-followup-20260903/uptake-levers/`.

## Core numbers and limits

| cell | official final | first-SFT card value | card-reported training h |
|---|---:|---:|---:|
| w10r01 | .694466 | .7067 | 3.26 |
| w10r02 | .717210 | .6733 | 4.43 |
| w10r03 | .719484 | .5800 | 4.83 |
| w10r04 | .722517 | .7267 | 4.50 |
| c10r01 | .815011 | .7333 | 5.31 |
| c10r02 | .767248 | .7200 | 5.03 |
| c10r03 | .702047 | .6867 | 4.61 |

WMA mean .713419, sample SD .012821 (n=4); control mean .761435, SD .056706
(n=3). This is provisional matched-runtime evidence. The control high .815011
is a score record inside this window, not evidence of a skill promotion.

Raw `cells.py` correlations across these seven cells: first-SFT .607,
training-wall .821, max-data-rows .857, epoch-rows .571. These are descriptive
and confounded by choices, not treatment effects. In particular w10r03's first
SFT .580 was sampled; its same-weight greedy exp-03 is .6933. Thus the raw
first-SFT feature mixes training and decoding and should not be presented as a
clean recipe-quality measurement. The data extractor also omits some `derived:`
training descriptions and is unsuitable for a precise total-token-budget claim.

The automated uptake output is NOT a reliable causal denominator: 30/30
terminal cards have requests/verdicts and lock state `delivered`, but its
`12/29 before launch`, `26 adopted / 1 ignored / 3 post-hoc`, and C5
`suggested/acted=22/17` use broad command matching over later calls. For example
w10r03's timeline assigns the exp-03 launch to 10:34, whereas the actual
`run_eval.sh ... exp-03_greedy_dev150 greedy 150` call is at 10:41:24 after the
10:40 verdict. Later card-text writes and other evaluations can be counted as
launches/actions. Do not turn those heuristic counts into compliance or
causal-uptake claims.

## Hand-verified checkpoint uptake and its limitation

- **w10r01** really scored intermediate checkpoints and then compared a
  restricted four-candidate set at n=500. exp-06's result has .694/.692/.680/.666;
  trace lines 15816–15860 prepare checkpoints 350/381, 16253 gives the WMA
  paired-comparison precondition, 16587 and 16700 record the executed comparison
  and paired counts. Its later soup omitted the weaker candidates; see exp-07.
- **w10r02** changed checkpoint saving after advice, but retention invalidated
  part of the plan: exp-03 `result.training_summary.notes` and trace 8622 say
  save_total_limit=2 deleted checkpoints 450/900, leaving 1350/1757. This is
  partial uptake with an execution defect, not simply advice ignored. exp-02
  did score ck650 (.680) vs final (.6733), and exp-03 scored ck1350 (.6867) vs
  final (.660). WMA's exp-04 two-ingredient soup suggestion is directly adopted
  below. Later exp-06 advice to score two old final checkpoints before training
  is not shown as performed; the cell instead trained two epochs and improved.
- **w10r03** changed save_total_limit from 2 to 8 following the first exp-02
  review (trace 5548–5556), retained epoch-1 ck1318, and actually evaluated it
  through `run_eval.sh ... checkpoint-1318 ... inherit 150` (line 7525). Card
  exp-02 records .5267 epoch 1 vs .580 final. This is genuine saving/scoring
  uptake, not proof that the early checkpoint should win.
- **w10r04** scored epoch-1 checkpoint660 on its local train-derived dev250
  (`local_eval.py` call line 6967) and later retained/scored exp-04 epoch-1 on
  the full official protocol (exp-04 measurements .710 vs .7134 final).
- **Control background**: c10r03 exp-02 scored epoch1 .6067 vs final .6867;
  c10r02 exp-08 used an intermediate checkpoint as an ingredient but did not
  establish a stand-alone official C5 sweep; c10r01's main comparisons were
  stage endpoints. Therefore the raw C5 label 4/4 WMA vs 2/3 control must not be
  equated with an official, matched-ruler intermediate-checkpoint sweep rate.

All four WMA cells show some checkpoint action, so the event's unfinished
claim that content/uptake did not improve is unsupported. The action need not
increase score. Relocks bought real edits in several cells; their latency is
not automatically waste. Use the separate operator-review response-duration
proxy (80.11/58.76/69.58/59.66 min) with its caveat, not a >1.5h measured-idle
claim. Reducing or bypassing blocking review is not proposed here.

## Candidate mechanism 1: C6 ingredient selection before expanding a soup

**Two direct WMA examples, plus a control cross-check:**

1. **B/w10r02/exp-04** is the strongest uptake evidence. The locked proposal
   was the four-way soup of exp02 ck650/1293 and exp03 ck1350/1757. The terminal
   verdict's `suggestions.cheaper_variants` explicitly asks for the two-way
   soup ck650+ck1350 as an additional lower-risk comparison. In the source
   trace, line 9233 delivers that advice; 9299 executes the four-way n=500 eval;
   9300 constructs the two-way soup; 9392 executes its n=500 eval; 9626–9639
   records four-way .694, two-way .710, best measured ingredient .704. Both
   used the same `run_eval.sh ... 500` wrapper and prepped greedy generation
   config; line 9249 checks soup config identity to the comparator. This is an
   adopted, promising action worth making consistent. The +.006 versus best
   ingredient is below the n=500 floor; no significant quality win is claimed.
2. **B/w10r03/exp-06** averages exp02 and exp04 from separate training runs
   off the same base. On matched greedy n=150, exp02 .6933, exp04 .7400,
   soup .6933. Card diagnostic: it recovers 6 of the 13 items the weaker
   ingredient alone got right, breaks 14 of exp04's correct items, and fixes
   only 1 of 26 shared failures. Same base/tokenizer is not proof of useful
   averaging; a weak ingredient's potential complementary wins must be priced
   against regressions. Its terminal WMA L2 was already cautious [-.06,.02];
   the proposed edit targets actionable alternative selection, not changing
   this interval after the fact.
3. **C/c10r03/exp-05** broadens an existing two-way soup with weaker epoch-1
   checkpoints. On the same train-derived n=500 probe (not official GSM8K
   accuracy), baseline soup .764, three-way .748, four-way .720. exp-04's
   two-way soup did improve the separate official n=400 read .7025→.735.
   This is independent supporting mechanism evidence, not poolable scoring.
   **C/c10r02/exp-08** also drops .7733→.760 on official n=150 after a third
   near-duplicate trajectory point is added, though that margin is small.

**Proposed single edit to C6/Suggestions (draft, to preregister):**

> For a C6 proposal, compare ingredient quality on the same evaluation and
> decode settings and note whether its ingredients are adjacent snapshots or
> distinct trajectories. Same base and tokenizer establish compatibility, not
> benefit. When a proposed soup adds clearly weaker or near-duplicate
> ingredients, offer a two-ingredient subset already supported by same-ruler
> scores as the cheaper variant, price the merge plus paired evaluation, and
> state what result would justify expanding it. Do not reject an ingredient
> solely for a lower point score: complementary errors can still help.

This is distinct from D (checkpoint saving), E (training L3 defer), F (which
precondition is first), A/B (L2 floor/width), and C (probe before L0/L1 no).
Do not stack this with any of them during a baseline comparison.

Primary: uptake of an explicitly proposed cheaper C6 subset on eligible
multi-ingredient C6 cards, reported alongside paired same-ruler score delta
and merge/eval time versus the proposed full soup. Also report eligible-card
count; a wave with no eligible C6 proposals is uninformative. Falsify if advice
adds evaluations without reducing costly rejected merges or improving the
chosen-model result; retain zero-leak, <=1.5x baseline cost, and unchanged PTB
spread guards. The evidence does not support a hard rule that two ingredients
always win. Final quality and ingredient diversity are both relevant.

## Candidate mechanism 2: distinguish prefix coverage from repeated-run noise

**Two concrete ranking reversals:**

1. **B/w10r01/exp-06 and exp-08**: initial n=150 favored exp04 ck250 (.740)
   over the eventual soup (.7133). At n=500 the same candidates were .666 and
   .700; at n=1319 .6740 and .6983. exp-08 trace 17785–17822 states the competing
   prefix rankings before the comparison; 18322–18433 records the full read
   and paired 102/70 discordant counts (p=.0178). The id-set difference of
   the 819 not previously scored items agrees in direction (.6972 soup vs
   .6764 ck250); it is not an independent held-out promotion test.
2. **B/w10r03/exp-07**: official n=150 ranks exp04 .740 above exp05 .7267;
   official n=400 reverses them to .705 vs .735. Commands actually observed:
   12:32:36 `run_eval.sh .../exp-04/final exp-04_final_greedy_dev150 greedy 150`,
   14:24:21 corresponding exp05 call, 14:36:01 exp04 n400 call, 14:38:59 exp05
   n400 call. Trace lines 10680–10685 give same ids=True (400 each), 282/294
   correct, 19/31 discordant (p=.1189). Lines 10739–10740 show the 250 new
   item ids alone agree (.704 exp04 vs .732 exp05). Card result and conclusion
   record a suggestion, not conclusive evidence of a 3pp population gain.

**What was verified about decoding and repetition:**

- w10r03 `package_final.py` source in trace 4963–4967 removes top_k/top_p and
  sets temperature=0.0 for the greedy branch. The executed wrapper source at
  5900–5917 passes `--max-connections 16`, `--gpu-memory-utilization .85`, the
  chosen `--limit`, and the staged checkpoint to the unmodified evaluate.py.
  Actual calls above share that wrapper/settings. The exp03 verification at
  7720–7759 also reports one server log receipt that HF config overrode
  sampling parameters. No observed source mutation of the wrapper separates
  the cited n150 and n400 calls.
- This is evidence of matched intended greedy evaluation, **not proof of
  bit-identical inference or deterministic repeated outputs**. The two n400
  inspect-config summaries expose matching limit and model args but null
  max_connections/max_tokens fields; wrapper calls provide the latter
  requested parameters. A direct per-run engine readback was not recovered
  for every one of these calls.
- No same-weights/same-settings repeated n150 study was run as part of this
  analysis, and no claim that such repetitions yielded zero variance is
  supported. w10r03 exp08's same-weight comparison changed concurrency 16→2
  and changed 8/150 correctness labels; it is not a C18 identical-protocol
  repeat. c10r02 exp07 reproduced the same aggregate score with some item
  churn, further reason not to equate greedy with bit determinism.
- The supported mechanism is narrower: repeating the same fixed item ids
  cannot acquire evidence about unseen items, even if it estimates runtime
  variability. The observed larger-item comparisons addressed an uncertainty
  that merely repeating a short prefix cannot remove.

**Proposed single edit replacing the categorical C18 advice (draft):**

> When a candidate-selection margin is smaller than the ruler can resolve,
> name the uncertainty before suggesting another evaluation. Repeating
> byte-identical weights under identical effective settings estimates
> run-to-run variability; it does not add item coverage when --limit selects
> the same prefix. If existing reads show prefix-sensitive ranking, offer a
> paired comparison of the same candidate shortlist on a larger common item
> set through the unchanged official evaluator, priced from this session's
> measured throughput. Use C18 repeats when stochastic/runtime variability is
> the unresolved mechanism, and do not assume greedy execution is bit exact.

This changes WMA measurement advice only. It must not change the scorer,
noise-floor rule, formal promotion set, or existing candidate guards. The
current manual says C18 is the only correct move below the floor; replacing
that categorical sentence is more precise than changing the floor to bless a
small difference. It overlaps none of A–F's single intervention mechanisms.

Primary: rate of eligible uncertain selection cards whose suggestions
explicitly separate repeated-run noise from item coverage and whose scientist
executes a matched larger-common-N comparison when prefix sensitivity is
present; report follow-through, runtime and actual ranking stability. A
four-cell wave can test uptake, not prove a PTB advantage. Falsify if the
larger-N advice fails to change action, chooses mismatched decoding/item sets,
or increases cost beyond 1.5x without a more stable selection. Preserve
zero-leak/PTB guards and bounded candidate shortlist. Do not automatically
mandate full evaluation on every card or every checkpoint.

## Ranked conclusion

1. These two narrow suggestion mechanisms have direct outcome/uptake evidence
   and can be tested without waiting for unrelated tails. They are not a
   claim that baseline C6/C18 never works: the positive actions already occur
   in some v0.2 traces, so the test is whether precise text makes them reliable.
2. First-SFT/data-volume observations explain where large absolute scores
   originate but remain too confounded here for a new numeric prior or hard
   training recipe; no extra data/epoch rule is proposed.
3. Checkpoint advice is visibly adopted; retention and subset-selection
   quality remain consequential. D already targets the checkpoint-save
   precondition, so do not duplicate it to pad pending counts.
4. Preserve all formal comparison, leak and cost gates. The event's request
   to weaken measurement fences, bypass cheap-card waiting, or defer all new
   experiments until every extension completes is not adopted.
