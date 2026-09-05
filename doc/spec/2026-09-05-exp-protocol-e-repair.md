# E repair discovery candidate

Reference: frozen E `dcfa742`. This candidate is a new multi-component treatment,
not a relabel of the four completed Opus4.8 E sessions. No baseline promotion,
GPU launch, queue mutation or new scientist recipe is authorized by this file.

## Evidence and components

1. **Phase readiness before engine construction.** e03g02/92134 L5856–6081,
   2026-09-05 01:52–06:03 UTC: missing future training data was overridden at
   lock, then rejected by both guarded run and sampling after engine creation;
   timeout consumed about 4h10. e03g04/92136 L5250, 01:43:10 explicitly splits
   sampling, data persistence and a new training card, and succeeds. Preserve
   strict live source/input hashes; expose their CPU check before the factory
   and document the existing supported phase separation. No future-input bypass.
2. **Native sampling API and bounded failure observation.** e03g01/92133
   L5975–6207, 01:41–02:32 and e03g03/92135 L5213–5864, 01:42–03:04:
   unsupported `prompt_token_ids` keyword and delayed failure cost about .9h
   and 1.4h. Reuse native adapter, durable raw capture and owned execution
   wrapper. Show vLLM0.11 TokensPrompt calling convention and stage deadlines;
   do not introduce global process management or silently alter batching/RNG.
3. **Serving decode evidence.** e03g02 L6641/07:19 and e03g03 L8918/08:24:
   actual submitted JSON has only `do_sample:false`, but both claim greedy.
   Actual API requests omit temperature; vLLM0.11 reads sampling fields rather
   than that boolean. Freeze selected JSON and supported request fields;
   distinguish intent, file, native defaults, native request parameters and
   unknown effective execution. No automatic temperature edits or recipe choice.

Raw provenance and independent reports:
`doc/exp_protocol_iterations/analysis-2026-09-05-opus48-wave1/e-review/`
in the operator checkout (jobs92133–92136). Historical same-runtime source:
`results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-x16-v3/p00r05/solve_parsed.txt.gz`
L2988–3059. These are process facts, not estimates of recoverable accuracy.

## Outcomes and decision

Planner may use four discovery cells within the separately frozen next-wave
manifest. Measure official quality, unsupported/false blocks, producer failures
and wall time, actual feature adoption, and whether declared decode matches its
evidence. Four repeats cannot isolate component effects or establish equivalence.
Preserve e03g02's flagged failure as cost evidence; it is not a clean denominator.
GPU spend, placement, task coverage and held-buffer decisions remain with the
planner and receipt-backed manifests. Held-out confirmation remains necessary
before promotion. Do not install MetaMath/RFT/few-shot/precision recommendations
as obligations; those remain conditional scientist strategy observations.

## Validation boundaries

CPU tests must cover stale/missing live inputs before factory invocation, native
entrypoint checks without model construction, raw retention on failure, immutable
decode snapshots, do_sample-only ambiguity, and native-object type/provenance
boundaries. Existing execution tests cover owned child signals and exit records.
No CPU test claims model loadability, GPU shutdown behavior, performance or score.
Independent forward review and native GPU validation remain separate evidence.
