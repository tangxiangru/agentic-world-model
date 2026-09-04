# Approved Opus4.8 study: initial onboarding evidence

User approved40 cells in the [spec](../../spec/2026-09-04-exp-protocol-opus48-cross-benchmark.md).
The independent16-cell GSM8K wave is registered held as92125–92140, with
receipts committed inc45c944. This does not say GPQA/HumanEval are ready or
that any job has been released.

## Frozen methods and model route

- Process-knowledge source359de271b889f616995968097ddda2e2cf1741b0,
  protocol tree0baf88005fa85d62bf3cef6a953a0a7e4fc317b2. Main froze the six
  user-authored runtime skill files unchanged; unrelated meta/analysis drafts
  were not staged. The independent [cross-task forward report](process-skill-forward.md)
  exercised GPQA-style and code-only synthetic inputs using the legacy CLI.
- E source dcfa742dbc8813970192efe3fbf2bd30dfc38ea9,
  protocol treeb33422364c70f4ea3c08ff83c97009a41438caa6. Separate from the
  knowledge-only treatment; no L/P or new research-recipe prescription.
- Old guard source4ae3d87c446bbda9732537a72b2f0fb3f96ac35a,
  protocol tree189319d63d301d64d96f8f41d051795404679f37.
- Baseline uses matching plain `claude_vertex_high`, no AWM/protocol/WMA.

The actual provider [record](context/record.json) and [raw stream](context/stream.json)
resolve `claude-opus-4-8[1m]` to canonical `claude-opus-4-8`,1M context, high
effort, Vertex sercan-v1/global, CLI2.1.219. The assistant itself returned OK;
no tool use occurred. Exact CLI bytes were compared against the SHA256-verified
opus_5 image. The call used extracted CLI bytes plus host libraries in read-only
bwrap, **not full Apptainer execution**, and used no GPU. This is provider/CLI
context evidence, not a new compute-node, performance or training acceptance.
The original record is retained separately; the archive record relocates its
raw-trace pointer while preserving the original path and raw hash.

The first existing `context_probe.sh` attempt stopped before an API call because
bare apptainer was absent from PATH. Its verified:false record/error are preserved. It
was not interpreted as model rejection or replaced by fabricated success.
The bounded fallback tool/tests live in `tools/ptb_opus48_context_probe.py` and
its dedicated test. The actual success costs about$0.020 in the raw stream.

## Readiness and limits

Supplementary [full-container probe](full-container-record.json) also passed,
using the existing configured Apptainer binary/library paths. Its
[raw stream](full-container-stream.json) confirms the same CLI/model/context;
no tool call or model experiment occurred. This supplements, not rewrites, the
already-receipted extracted-CLI record. Initial failure was PATH, not an absent
installation or model rejection.

All four GSM8K manifests (none/knowledge/tools/guard, each4) passed the full
`awm ptb check` with0 issues, including actual image hashes, frozen method
sources/protocol trees, provider context and site configuration. They are
marked held, without a release override. Final operator/manifest/probe tests:
110 passed in9.03s; Ruff and diff checks passed.

The [new-task audit](new-task-eval-audit.md) found missing tracked contamination
references, unpinned dataset contracts/count validation and unsafe HumanEval
same-user local code execution. The actual metrics are accuracy/stderr, with
choice/verify scorers and one epoch, not a presumed pass@5 route. Do not add
those tasks to the ready set merely because evaluator files exist.

Metadata/download check:

- GPQA Main repo Idavidrein/gpqa at633f5ee89ab8ad4522a9f850766b73f62147ffdd,
  gated auto, CC-BY-4.0. Downloading gpqa_main.csv using existing HF credentials
  returned403 (account not authorized). User has been asked to obtain official
  access or supply an already legally authorized local path; no mirror bypass
  or new terms acceptance was attempted.
- HumanEval repo openai/openai_humaneval at7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544,
  MIT, metadata164 test examples. The pinned parquet was downloaded to the
  dedicated temporary cache:83,920 bytes, SHA256
  2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6.
  No questions/gold were displayed or used in planning. Actual data normalization,
  reference freezing and safe executor/runtime acceptance remain to finish.

Full independent process-forward materials were copied byte-identically under
`process-forward-raw/` (verified with diff -qr); original `/tmp/process-cross-task-forward.qLgkKz`
is retained. The original evaluator-audit folder is `/tmp/ptb-new-task-eval-audit.zmakED`.
These checks add zero validator-clean scientist results.

At11:55 UTC ownership was OK and22 old held cells remained,0/16 GPUs allocated.
Reservation still names11 nodes. The original hourly monitor PID2612586 was
confirmed alive through a properly escalated host read: the managed shell's
missing PID was a visibility difference, not a terminal process or restart trigger.
Native isolation/release authority is separate from these onboarding checks.
