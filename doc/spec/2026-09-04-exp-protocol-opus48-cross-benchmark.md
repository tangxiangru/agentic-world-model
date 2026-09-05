# Opus4.8 cross-benchmark protocol comparison

2026-09-04. User approved the proposed40-cell matrix with “好的你去做吧”.
This is the experiment specification before onboarding implementation and
immutable manifests. Approval covers these comparisons, not an exception to
ownership/native-isolation gates or a new held-out task role.

## Matrix and purpose

| Arm | GSM8K | GPQA Main (`gpqamain`) | HumanEval (`humaneval`) |
|---|---:|---:|---:|
| protocol-free PTB baseline |4|4|4|
| current lightweight process-knowledge skill |4|4|4|
| integrated E tools/skill package |4|4|4|
| old guard historical bridge |4|0|0|

Forty independent scientist sessions; each receives10h on one H100 with the
existing16-CPU/128G/400GB-scratch outer resource contract. Nominal scientist
budget400 GPU-hours excludes final evaluation, judges and scheduler overhead.
Four repetitions supersede the prior two-cell proposal. No automatic additional
replicates, winner-to-eight expansion or unapproved factorial extension.

Primary questions: do the two current packages improve research outcomes over
an otherwise matched protocol-free scientist on science reasoning and code,
and do the latest packages improve GSM8K relative to both no protocol and the
old guard? Report package effects, not component attribution. This is not a
fixed-recipe study: the scientist's recipes remain observed choices.

AIME2025 remains reserved for later independent promotion confirmation. Neither
its questions/results nor AIME2026 enter this approved discovery wave.

## Frozen treatments and matched outer contract

- Scientist: Claude Opus4.8, high effort,1M context, Vertex `sercan-v1/global`.
  Resolve and validate the exact CLI model spelling using the pinned runtime
  before admitting it. The expected explicit-context spelling is
  `claude-opus-4-8[1m]`; provider/CLI actual resolution, canonical model,
  context and response success must be observed rather than inferred from
  the requested string. Public availability is not project entitlement.
- Trained base: `google/gemma-3-4b-pt`, revision
  `cc012e0a6d0787b4adcc0fa2c4da74402494554d`, unchanged within all cells.
- Initial PTB pin: `dcf5da031435c54e3680b6ec3f63e7e317efc13e`.
  Initial scientist image `opus_5.sif`, SHA256
  `35f287e7b17d62ab44cd95db26dfeeac166943daed5f7b557b008bae51acc759`;
  CLI2.1.219, automatic updates disabled. The image name is not the scientist
  model choice. Evaluation image `vllm_debug.sif`, SHA256
  `72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8`.
- Keep task prompt, task data/scorer, evaluator settings, official judge profile,
  model revision, runtime and resource budget matched across arms for each task.
  A necessary shared runtime/evaluator repair must be tested and frozen for all
  arms before launch; do not silently retain different PTB pins by method.
- Protocol-free uses the otherwise matching `claude_vertex_high` scaffold,
  without AWM protocol setup, skill or WMA. It still follows the common PTB
  task/safety/contamination/evaluation contract. Method arms use the matching
  `_awm` scaffold and exact six `EXP_PROTOCOL_SHIP` paths, no meta/WMA shipment.
- Guard: source `4ae3d87c446bbda9732537a72b2f0fb3f96ac35a`, protocol tree
  `189319d63d301d64d96f8f41d051795404679f37`. Its82.79% record is a single old
  Opus5 session, not a proven stable winner or a matched Opus4.8 control.
- E: source `dcfa742dbc8813970192efe3fbf2bd30dfc38ea9`, protocol tree
  `b33422364c70f4ea3c08ff83c97009a41438caa6`. CPU/independent-forward accepted;
  unrun on GPU. Its source includes the complete E1–E8 package, not L/P.
- Process-knowledge: freeze the current user-authored runtime skill changes
  separately, including `process_checks.md`; do not merge in E or discard
  those changes to make a clean worktree. Record exact source/protocol tree
  and hashes once frozen. Unrelated user analysis/meta drafts remain untouched.

## Task and provider acceptance before receipts

1. Add only the requested high/1M Opus4.8 profiles and GPQA Main/HumanEval
   task contracts. Preserve old manifest meanings and existing profiles.
2. Exercise the actual pinned CLI/provider with a bounded harmless response,
   no research context, tools or GPU/model workload. Retain original stream,
   exit, model/context/effort/CLI/image identity and hashes. Record API failure
   honestly; no automatic fallback to another model. A provider probe is not a
   scientist result or one of the40 cells.
3. Verify tracked task assets and dataset/cache/access identity without feeding
   benchmark items/gold into planning, training, examples or watch sets.
   Validate whole-task evaluation versus developer limits and actual metric
   names/counts/reduction. Do not relabel a non-accuracy metric as accuracy.
4. Trace formal evaluate→metrics→validator→results→judge/harvest consumers for
   both new tasks. Add native/synthetic CPU tests for their real metric/schema
   paths, errors and missing evidence; an allowlist edit alone is insufficient.
5. Audit HumanEval's generated-code execution boundary: filesystem mounts,
   privileges, credentials, network, timeout and process cleanup. Do not run
   generated code with unreviewed access to shared experiment data or secrets.
   Fix or establish the supported isolation before its jobs become runnable.
6. Independently forward-check the new skill on non-math synthetic card/data
   scenarios and retain honest unsupported behavior; do not invent task data,
   comparators or claims that an unavailable automatic check ran.
7. Full manifest/source/site validation before each independent held block.
   Readiness of one task does not depend on unrelated slower task onboarding.

## Outcomes, order and interpretation

For every attempt retain official task metric/denominator or explicit failure,
validator and judge status, placement eligibility, producing receipt/cell,
time to usable incumbent, execution/repair/idle costs and evidence coverage.
Report all four sessions per arm, mean/range and individual scores; missing/
failed attempts are not hidden or assigned a fabricated score. Valid-only
accuracy is conditional on completion and must accompany failure coverage.

Within each task interleave arms/repeats in scheduler order where gates allow,
so a method is not deliberately run wholly before another method. Keep task
metrics separate; do not average unlike percentages into a headline score.
Paired-item evidence, where available, supplements session-level variation.
Four runs do not prove a small stable gain or equivalence. Selection/adaptive
diagnostics remain exploration, not held-out confirmation. No promotion here.

Use whole predeclared block completion or eight new validator-clean cells to
trigger deep review, following current meta/local-Claude workflow. Preserve
the one-hour external detector and count new clean cells separately from
terminal jobs. Do not add filler just to meet an analysis threshold.

## Queue transition

Submit validated independently specified blocks as `PENDING(JobHeldUser)` via
the existing immutable manifest/queue/reconcile workflow. No handcrafted sbatch.
Authority stays `gangda_exp-protocol-evolve`, nodes ondemand0–1 only. Native
reservation still had11 nodes at11:55 UTC; no release exception has been granted.
Release requires ownership, exact frozen per-job ReqNodeList and restored native
two-node isolation. Maintain at least eight scientifically justified held cells.

After new useful held receipts exist, re-audit/withdraw obsolete wholly
unstarted old blocks through exact immutable receipt IDs only. Never cancel
running or foreign work. Preserve and harvest all terminal attempts. Forty
planned cells are not forty receipts until the scheduler/registry proves them.
