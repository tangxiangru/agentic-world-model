# Cross-benchmark comparison: approved 2026-09-04

The latest user discussion requests experiments beyond GSM8K, a scientist-model
change to Opus4.8, several leading/current protocol variants and a genuinely
protocol-free baseline, **four independent replicates per arm**, plus a GSM8K
comparison of the latest method. Four supersedes the earlier two-cell default
for this study. The subsequent “好的你去做吧” approves the forty-cell matrix
below; historical frozen runs/receipts remain unchanged. The execution contract
is [the cross-benchmark spec](../../doc/spec/2026-09-04-exp-protocol-opus48-cross-benchmark.md).

Treat Opus4.8 as the scientist agent, not the open model being post-trained.
Keep base-model revision, scientist effort/context, GPU/time budget, runtime,
official scorer and data contract matched within each benchmark. When changing
the scientist model, old Opus5 controls are historical context, not matched
Opus4.8 controls. Four independent scientist sessions are not four invocations
of the same final-model evaluator or necessarily four fixed-recipe training seeds.

Distinguish candidates with observed scientific evidence from newly engineered
ones. Guard has historical scores, including a single82.79% record; it is not
a proven stable winner. Integrated E is CPU/forward accepted but unrun on GPU.
The concurrent current-checkout process-knowledge skill (`process_checks.md`,
updated skill/template/pitfalls/hook guidance) is a distinct lightweight
candidate, not the same E source and not yet a measured winner. Preserve and
freeze its owner's changes separately; do not overwrite them with E or claim
they already contain E's new execution/rendered/export capabilities.
L/P remain designs, not ready methods or measured winners. The protocol-free
arm omits exp_protocol/WMA treatment while preserving the common PTB task,
runtime, official scoring and safety/contamination contract.

The approved tasks are GSM8K as a regression anchor, GPQA Main for science
reasoning and HumanEval for code; AIME2026 is not in this wave. AIME2025 remains the
previously reserved held-out task unless the user explicitly changes its role.
The approved first matrix has protocol-free, process-knowledge and E arms on
all three tasks (four each), plus four old-guard GSM8K runs as a matched Opus4.8
historical bridge:40 full-budget cells. Approval is not evidence of submitted
work. At10h each this is400 nominal scientist GPU-hours, excluding
evaluation/judge/scheduler overhead. Do not silently replace the discussion
with a larger factorial or omit matched Opus4.8 controls.
Do not quietly use it for repeated discovery or choose tasks after seeing
favorable treatment outcomes. A new-benchmark exploration is not held-out
promotion evidence once it informs further method changes.

Readiness requires more than an evaluator file: the pinned PTB tree has these
tasks, but `awm.ptb_experiments.APPROVED_TASKS` currently permits only GSM8K and
AIME2025, and the approved scientist profiles currently name Opus5. Validate
the actual Opus4.8 route/context, task data access, evaluator metric/schema,
judge/contamination coverage and any local code-execution isolation before
freezing manifests. An allowlist edit alone is not task acceptance.

Report all four outcomes, mean/spread, failures, time to usable incumbent,
execution/repair cost and protocol violations per task. Do not infer a stable
gain from a maximum or pool unlike benchmark percentages as one accuracy.
Retain matched item evidence when available; do not feed test items into
training/watch sets. No automatic extra repeats or post-hoc task selection.

The new scientific plan does not waive ownership, frozen node placement,
native two-node isolation or the useful held floor. No new-study receipt or
release has been made. This update is planning memory, not a launch declaration.

Official model-route reference checked2026-09-04:
[Google Cloud Opus4.8](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/opus-4-8)
documents `claude-opus-4-8`, a1M input context and the global endpoint; this does
not establish this project's current entitlement or the pinned CLI's behavior.
