# GPQA Main onboarding: independent software work

2026-09-04. This is part of the approved40-cell study, not additional experiments
or task admission. Official data access still returns403; do not supply fake
rows/reference hashes/counts as real data, accept terms, or substitute a mirror.
Synthetic CPU validation of the existing scorer/loader interfaces remains
possible without obtaining benchmark questions, so the overall goal is not yet
at a complete software-work impasse.

## Preserve the actual existing contract

Source: PTB09c90b6 `src/eval/tasks/gpqamain/evaluate.py` (unchanged from dcf5da0).

| Identity | Existing value |
|---|---|
| PTB/receipt task directory | gpqamain |
| Native registered task function | gpqa_main |
| Dataset/config/split | Idavidrein/gpqa / gpqa_main / train |
| Solver/scorer | multiple_choice(cot=True) / choice |
| Epochs |1|
| Developer limit / formal selection |50 / full task|
| Default concurrency / GPU memory fraction |6 /0.8|
| Token cap phases |16000×4 attempts,12000×3,8000×2|

The mapper uses Question, Correct Answer, Incorrect Answer1–3 (actual column
spelling includes the spaces before numbers), and Record ID. Correct answer is
initially A, then native dataset choice shuffling changes presentation and the
target letter. Preserve the declared randomization contract; do not introduce a
silent fixed seed or assume matching Record IDs imply matching inputs/targets.

## Next bounded implementation

1. Isolate a pinned local CSV-to-Sample route using the existing mapper's exact
   columns and native solver/scorer. Fail on missing/unverified data rather than
   reverting to an unpinned network fetch. Actual source SHA, population, unique
   typed IDs and contamination reference remain unfrozen until lawful data can
   be inspected by the trusted metadata-only preparation path.
2. Test choice presentation/target mapping with invented rows, including repeated
   answer text. Preserve actual option provenance, not a string-only join that
   can confuse duplicate choices. Read and exercise the native shuffle source
   before selecting a recording mechanism; any common contract change must be
   explicit and frozen for every arm, not hidden in an adapter.
3. Add GPQA-specific durable per-attempt Inspect JSON and validation of native
   choice scores, exact observed n/typed IDs, actual question/options/target,
   model/source/serving/attempt identity and finite recomputed accuracy/SEM.
   Do not reuse HumanEval's verify/code-execution checks under a renamed task.
4. Exercise the native multiple-choice/choice route with a mock model and
   invented samples, then synthetic complete/partial/error/stale result bundles
   through validator and AWM consumers. These are CPU mechanism tests only.
5. Once lawful data and runtime/source/site prerequisites are verified, freeze
   data/reference identity and the common PTB source, then prepare/admit the
   three four-repeat GPQA arms. No receipt or held-floor contribution beforehand.

Physical launch still needs native two-node isolation, ownership, exact job
placement and eight useful held cells. Keep hourly monitoring live. The lack
of GPU/data authority blocks execution, but does not justify claiming that
the independent GPQA software portion is already complete.

## Completed CPU checkpoint

The independent implementation/review above completed at PTB60df491, including
native presentation/parser differential tests, complete synthetic AWM evidence
flow and the reviewer's output-to-score correction. See the
[checkpoint and raw evidence](../exp_protocol_iterations/analysis-2026-09-04-opus48-onboarding/gpqa-runtime-checkpoint.md).
This supersedes the earlier pending-software status, not actual-data/node gates.
Official access remains403; no real data profile/reference or GPQA manifest/
receipt has been fabricated. Next admission requires external prerequisites,
not another synthetic implementation cycle.
