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

Readiness requires more than an evaluator file. New task/model admission must
be explicit; the verified Opus4.8 high/1M profiles do not automatically admit
GPQA or HumanEval. Validate
the actual Opus4.8 route/context, task data access, evaluator metric/schema,
judge/contamination coverage and any local code-execution isolation before
freezing manifests. An allowlist edit alone is not task acceptance.

Report all four outcomes, mean/spread, failures, time to usable incumbent,
execution/repair cost and protocol violations per task. Do not infer a stable
gain from a maximum or pool unlike benchmark percentages as one accuracy.
Retain matched item evidence when available; do not feed test items into
training/watch sets. No automatic extra repeats or post-hoc task selection.

The new scientific plan does not waive ownership, frozen node placement,
native two-node isolation or the useful held floor. Receipts establish scheduler
registration, not execution. Consult the [operator state](../../doc/exp_protocol_iterations/operator-state.md)
for the actual held/running inventory; do not infer launch from this policy.

Official model-route reference checked2026-09-04:
[Google Cloud Opus4.8](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/opus-4-8)
documents `claude-opus-4-8`, a1M input context and the global endpoint; this does
not establish this project's current entitlement or the pinned CLI's behavior.

## Evaluator onboarding lessons

Distinguish host/library unit acceptance, the actual image and compute-node
deployment. Equal Python version strings do not establish binary equality.
A nested bind can fail because the outer image mount is unbindable: inspect
the source mount with a minimal public-file probe before changing isolation.
A hash-checked public closure transport can preserve original image identity
and isolation; a passing import does not justify a broad host/root bind. Record
transport identity separately so session-specific scratch paths do not become
an apparent treatment. The [HumanEval checkpoint](../../doc/exp_protocol_iterations/analysis-2026-09-04-opus48-onboarding/humaneval-runtime-checkpoint.md)
records the exact evidence and remaining deployment gates.

Preserve outcome semantics from executable scorer code, not README claims or
untrusted generated stderr. A bad admitted program's error text is not proof
the evaluator failed. Proven bootstrap/policy/cleanup failure requires a failed
attempt; ordinary exits and scorer-defined timeouts follow the declared mapping.
Establish public dependency completeness independently before launch, and freeze
the changed common execution contract across all method arms.

Do not assume custom exception attributes or in-memory backend records reach
the trace. Test success, ordinary failure, timeout and evaluator failure through
the persisted sample log. Formal publication must bind complete selection/counts,
input/target/test metadata, actual scorer, executed code/runtime/limits, model
and source identities. Recompute archive hashes and outcomes: matching saved
hash labels alone does not validate bytes. Make verified raw bytes durable
before atomic metrics publication, and preserve failed attempts. CPU fixtures
never enter the scientist score table or clean-cell review trigger.

Carry the expected task from the immutable receipt job/cell or specified
manifest through every consumer, including manual harvest and queue failure
resolution. A result's self-reported task must not choose a weaker validator;
do not retry a rejected strict-task check under a legacy interface. An unscoped
directory audit is diagnostic, not completion authority. Missing judge files
can yield an empty flag list: only a completed validated result may be labelled
judge-clean. Administrative cancellations remain unverified/null-score evidence,
as in the [old-block retirement](../../doc/exp_protocol_iterations/2026-09-04-opus48-legacy-queue-retirement.md).

For multiple-choice evidence, preserve option identity through positions rather
than answer-text joins: duplicate text can represent distinct correct/incorrect
source options. Record native production randomization honestly; deterministic
CPU seams must not silently become formal fixed seeds. Bind actual generation
events and retained output to the native answer parser and score, not just score
labels and aggregate arithmetic. A host parser projection needs differential
checks against the pinned runtime, including case, first-match, empty and
structured-content behavior. Do not change native parsing to make a test pass.
The [GPQA checkpoint](../../doc/exp_protocol_iterations/analysis-2026-09-04-opus48-onboarding/gpqa-runtime-checkpoint.md)
records the informed review's demonstrated output/score gap and its correction.

Keep development snapshot compatibility separate from formal final artifacts:
HF file symlinks may be hashed for developer evaluation while formal final-model
files remain strict. A provided frozen base-model identifier must not become an
unpinned download fallback. When data or deployment is unavailable, finish
independent synthetic integration first, then stop at the actual authority/access
gate; do not fabricate data profiles, task admission, filler experiments or GPU
occupancy. A live external-event detector can remain running while the planning
goal is blocked on required user action.
