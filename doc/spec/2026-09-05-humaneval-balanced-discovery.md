# Balanced HumanEval discovery after node acceptance

Continue the user-approved HumanEval portion of the cross-benchmark study,
skipping GPQA for this round. Four independent10h scientist sessions per arm:
protocol-free, process-knowledge, and the repaired E package. This is12 sessions,
not an expansion of the approved per-arm repetition count. AIME2025 stays reserved.

## Treatments and common contract

- Plain `claude_vertex_high`, no protocol or WMA.
- Process-knowledge359de271b889f616995968097ddda2e2cf1741b0,
  protocol tree0baf88005fa85d62bf3cef6a953a0a7e4fc317b2.
- E-repair854464c677bdcc2bcf31fc504798f316e7dff8f7,
  protocol treede552c0555c42c50615c69dd28403da734ef08d4.

E-repair explicitly replaces the unrun old-E HumanEval draft; the old draft and
historical E results keep their original meaning. Compare complete packages,
not a purported pure tool effect. Keep Gemma-3-4B-PT revision, Opus4.8/high/1M,
images,10h, one H100/16CPU/128G, evaluator and actual node placement matched.

## Interleaved two-cell operational blocks

Each arm has block A (replicates1–2) and block B (replicates3–4). Six immutable
two-cell manifests collectively implement the same four-replicate arm contract.
Their optional manifest-local replication-shape assertion is omitted because
each is a partial arm; the planner checks the union is exactly1,2,3,4 per arm.
Batch aliases never reset spend or become new hypotheses.

Submit/release order is none-A, knowledge-A, E-repair-A, then the corresponding
B blocks. With two GSM8K E-repair sessions on ondem0, the first six available
slots can therefore start two of each HumanEval arm rather than all controls
first. Queue remaining valid work for backfill; record actual start windows.
Do not infer exact concurrency from this ordering before observing allocations.

## Admission and evidence

Every scientific manifest must carry an actual tracked environment receipt and
its SHA256. Submission and release revalidate the terminal probe's raw native
evidence, source/image/data/runtime/UID bindings and accepted node coverage.
The first wave is explicitly requested on ondem0, within the unchanged0–1 site;
no unaccepted node is allowed by any placement consumer. All arms use the same
PTB commit as the accepted probe. No fabricated receipt/hash or boolean readiness.

The approved source data has164 HumanEval examples. It is evaluation/reference
data, not a new scientist training corpus. Native single-epoch verify semantics,
isolated code execution and official/raw evidence retention are the common
contract. Failed generation, infrastructure and ordinary incorrect programs
retain their distinct outcomes.

## Analysis and follow-up

Retain all four outcomes per arm, official numerator/denominator, completion,
judge and placement status, time to useful model, actual sampling/serving tools
used, launch/lock coverage, and execution/repair costs. Report block windows and
whole-arm distributions; do not pool unlike benchmark percentages.

The first balanced block or eight new clean cells can trigger review. No
automatic repeats, no old-E rerun, no recipe mandates, and no baseline promotion.
Use actual native results to decide whether the new process transfers to code.
