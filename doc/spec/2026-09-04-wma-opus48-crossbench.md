# WMA Opus 4.8 cross-benchmark study

Status: implementation complete; production acceptance and immutable launch
preparation. The user requested other benchmarks, Opus 4.8, four repeats per
setting, a raw baseline and the newest method variants; then explicitly asked
to wait for implementation completion before enqueuing. Completion is recorded
in `doc/wma_iterations/2026-09-04-decision-review-implementation.md`.

## First-wave contract

The first wave is **GSM8K, BFCL and HumanEval**, five arms each, four independent
cells per arm: 60 scientific cells, ten hours per agent, one H100/16 CPU/128GiB/
400GiB scratch each. This is 600 nominal GPU-hours for agent windows; preparation
and grading add to allocated time. GPQA-main remains prepared but blocked: the
existing Hugging Face account has not been granted Idavidrein/gpqa access. No
mirror or alternative credentials are used to bypass that restriction.

Scientist is exactly `claude-opus-4-8`, high, 200000 context in every arm. The
same model/effort is used for WMA in S/M/J. A host call verified that route and
context metadata; actual container/model acceptance is a separate gate. The
trainable base stays `google/gemma-3-4b-pt` at
`cc012e0a6d0787b4adcc0fa2c4da74402494554d`. Scientist/evaluator container
digests and Opus5/high official judges remain pinned as in the existing PTB
contract. The container name `opus_5` does not select the scientist model.

All work belongs to gangda_wma_evolve on slurm2-a3nodesetondem-[2-3]. Receipts,
including technical validation jobs, must be registered and routed while held
before release. AIME2025 stays promotion-only. New Opus4.8 cells are never
pooled with the old Opus5 experiments.

| Arm | Public decision mode | Single-card WMA | Joint WMA | Contrast |
|---|---|---|---|---|
| R raw PTB | no AWM installation | no | no | practical bare baseline |
| P protocol only | single | no | no | P-R: protocol workflow |
| S scoped single-card | single | yes | no | S-P: single-card WMA treatment |
| M multi-candidate self choice | multi-self | yes | no | M-S: explicit candidate generation |
| J joint comparison | multi-joint | yes | yes, once per decision | J-M: joint comparison increment |

M and J share real candidate-brief requirements, initial scientist preference,
formal-card review and total budget. In M, `compare` freezes the inputs and
returns **not_requested**, never fake not_attached, timeout or failure. P/S
do not fabricate singleton candidate pools. All protocol arms retain current
lock/action/fingerprint checks. Mode identity enters the request/action/launch
chain so switching mode cannot reuse old authorization. The configuration is
explicit in every new manifest's sandbox setup and the harvested setup record.

## Readout and evidence

Primary comparisons are final official score J-R and J-M **within each task**;
P-R, S-P and M-S explain the components. Show all four scores, mean, sample SD,
uncertainty and incomplete/flagged cells. Do not average raw scores across
benchmarks, select only the highest run, or attribute the whole redesign to
one skill sentence. Four repeats are an exploratory screen, not promotion.

Report the pre/post-advice choice, actual executed version, advice disposition,
review and comparison costs, waits, failures and complete request counts.
Unexecuted endpoints stay unknown. Single-card v1 ledger remains a distinct
historical measurement; it does not score joint comparisons or duplicated
immutable verdict copies. Semantic access audits remain separate from original
scanner flags and PTB validator/judge status. No old flag is cleared.

The WMA skill in S/M/J is the new evidence-scope policy, with the same private
runtime and fixed read-only history across these arms. The existing curated
train history is retained; no target GPQA/BFCL/HumanEval test material is added
to it. The broker exports bounded selected text, not caches, raw evaluation
files or model weights. Filesystem isolation does not sanitize already
contaminated authorized text; manual semantic audit remains necessary.

Old F's observed 75.834% mean (n=4) is not a promoted best method; it still has
original access flags and uncertain counterfactual outcomes. G/H are ongoing
and are not selected from unfinished scores. An optional legacy-F comparison
would be a separate four-cell GSM8K setting under this new common runtime,
not a pooled continuation of old F. It is not part of the first wave.

## Acceptance and source freeze

1. CPU tests cover joint/no-joint behavior, mode changes, stale proposals,
   current proceed and lock integrity, scanner compatibility, real kernel
   canaries, process-group cleanup and isolated MCP transport. Stage all
   intended source changes only after review; freeze AWM and PTB commits.
2. A registered held `context-smoke` validates the exact Opus4.8/200k/high
   profile and image. Its record must verify the actual returned model, not
   merely the requested alias and reported context size. These jobs are
   validation-only and never harvested as scientific cells.
3. The same production image must pass synthetic OS canaries plus a real
   model-to-broker joint comparison and a matching blocking review. Persist
   source/model/UID/kernel/node/job and isolation evidence. No real benchmark
   or training outcome is produced by this acceptance.
4. WMA formal manifests require the successful production-acceptance path,
   matching private SHA/model/effort; the receipt freezes its digest. Raw R/P
   depend only on their own context/scaffold/evaluator gates and can be queued
   while the private WMA gate is still running. Do not delay an independently
   valid batch for unrelated tails or for the final candidate group.

The shared-UID failure found during validation was real: an absolute NPROC=128
prevented the shell from forking on a UID with thousands of threads. The new
common broker preserves inherited NPROC and supervises only its non-escaping
probe process group at 50ms intervals, killing it above 128 tasks; a transient
overshoot is possible, so this is not an atomic cgroup pids quota. File/network
restrictions and the other resource limits remain enforced, with no unisolated
fallback. This compatibility change is shared infrastructure, not a skill win.

GPQA access and any other unmet task gate are recorded as blocked settings,
not filled with an unvalidated substitute. HumanEval was moved forward from
the proposed next extension on availability grounds before any study result.
The fixed contamination-check asset inventories are in
`doc/wma_iterations/evidence/2026-09-04-crossbench/decontamination-assets.json`.

After a candidate is preferred, promotion still requires the existing
independent confirmation and held-out gates. The first-wave data are not an
excuse to weaken those guards or repeatedly tune on AIME outcomes.
