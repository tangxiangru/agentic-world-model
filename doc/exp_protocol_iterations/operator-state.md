# exp-protocol operator state and dependencies

Mutable handoff, updated2026-09-04 after the real monitor handoff and blocking audit.
Receipt source remains immutable apart from appended cancellation records.
Latest release/access checks16:19–16:21; new monitor first tick16:16:40.
Scope remains only
`gangda_exp-protocol-evolve`, `slurm2-a3nodesetondem-[0-1]`; never AWM full.

## Approved study and actual scheduler state

User approved the [40-cell specification](../spec/2026-09-04-exp-protocol-opus48-cross-benchmark.md):
GSM8K16, GPQA Main12, HumanEval12; scientist Opus4.8/high/1M, Gemma-3-4B-PT,
10h, four independent sessions per included arm. AIME2025 remains held-out.
[Current meta policy](../../skills/exp_protocol_meta/cross_benchmark_policy.md)
supersedes earlier two-cell preparation. No unrequested L/P arm or repeat is funded.

| GSM8K arm | immutable receipt-backed jobs | current result |
|---|---|---|
| plain PTB, no protocol/WMA |92125–92128 / n03g01–04|4 JobHeldUser; no model results|
| process-knowledge skill |92129–92132 / k03g01–04|4 JobHeldUser; no model results|
| complete E tools/skill |92133–92136 / e03g01–04|4 JobHeldUser; no model results|
| old guard bridge |92137–92140 / g03g01–04|4 JobHeldUser; no model results|

Four manifests are `experiments/posttrainbench/exp-protocol-opus48-gsm8k-{none,knowledge,tools,guard}-x4-v1.yaml`.
Receipts are committed under the corresponding `results/ptb/<batch>/formal-*.json`
in c45c944, pushed. Each freezes clean top75310e08fa9e75e454324b0116275a278cf9d6ef,
PTBdcf5da031435c54e3680b6ec3f63e7e317efc13e and exact ondem0–1 site scope.
All16 jobs were individually checked: PENDING, Reason=JobHeldUser,
ReqNodeList=slurm2-a3nodesetondem-[0-1], one GPU/16 CPUs. OWNERSHIP OK.
**0/16 GPUs allocated,0 running. Submitted held does not mean running.**

There are **21 physical held jobs**:16 newly specified/validated study cells plus
5 untouched mixed-baseline holds. The17 other old jobs were retired and harvested
under the [whole-block decision](2026-09-04-opus48-legacy-queue-retirement.md).
Only the16 new cells currently have an affirmative scientific purpose under
this approved wave. Do not release more than the justified buffer permits.
With only16 justified held, releasing all16 would violate the eight-held floor.

## Legacy holds, retirements and completed evidence

| Legacy block | exact receipt IDs | disposition |
|---|---|---|
| strict baseline tail |90826–90830|mixed receipt with90823–90825 completed; explicit remaining-tail disposition needed|
| old strict control repair |91965|CANCELLED/harvested; stop old strict8 completion, retain strict7+sensitivity|
| A v2 |91046–91049|CANCELLED/harvested; new agenda retires old configuration, not proof E covers A|
| B v2 |91050–91053|CANCELLED/harvested; E6 carries engineering, no standalone attribution spend|
| drift A |91058–91059|CANCELLED/harvested with old companion wave|
| H |91068–91071|CANCELLED/harvested; applicability engineering enters E2|
| drift B |91072–91073|CANCELLED/harvested with old companion wave|

The17 administrative retirements are saved in4cd1dbd: exact receipts, no starts,
zero runtime, no result directories, incomplete/null-score bundles. They add no
clean result or scientist-failure observation. No running/foreign work was touched.
The old strict-baseline receipt has3 completed plus5 held; its mixed-state
release refusal is documented in [boundary audit](2026-09-04-mixed-receipt-release-boundary.md).
Do not rewrite receipt membership or release arbitrary IDs to bypass it.

Previously C91054–91057,D91060–91063,E1 91064–91067 were wholly unstarted,
withdrawn and harvested as administrative incomplete/null-score evidence.
All spillovers/terminated attempts still require harvest; no clean result is
created by cancellation, a CPU test or a Slurm terminal label.

The original seven cohorts retain57 validator-complete results:56 eligible
(55 clean,1 general-anomaly flagged),1 placement-quarantined;2 scientist failures
p00r08/p00r16. Cohorts: v3 baseline16(14 clean),null8(7 clean+1 flagged),
null-B8(8 clean),old guard8(8 clean),strict guard8(8 clean),strict control8
(7 clean+90820 quarantined),strict baseline8(3 clean+5 held).
Highest clean official remains **g01r03/90649=1092/1319=82.79%**, not a stable
method effect; same old-guard8 mean73.76%. [p00r16](2026-09-03-p00r16-scorer-failure.md)
has no official score; do not substitute its developer metric.

## Release gate and next dependencies

At16:19 the reservation `robtang-ptb-a3` still named11 nodes, not an exact
native two-node subqueue reservation. No infrastructure change or new release
exception has been authorized. The old09-03 exception only covered90791–90798;
it does not authorize92125–92140. Ownership and frozen ReqNodeList passed for
the new jobs; native isolation has not. No releases occurred.

```text
approved40
  GSM8K16: registered + held + ownership/node verified; native release gate unmet
  GPQA12: existing HF credentials receive403; official access or lawful local source needed
  HumanEval12: reference data prepared; isolated executor and formal pipeline not admitted

release: scientific readiness + ownership + frozen nodes + native isolation
         + at least8 scientifically justified held after release
```

Do not invent filler or count checked/unsubmitted cells as scheduler inventory.
GSM8K onboarding did not wait for unrelated task construction. GPQA/HumanEval
remaining24 are real required work, not replaced by the completed GSM8K portion.

## Frozen methods, provider and source workspaces

- Knowledge-only:359de271b889f616995968097ddda2e2cf1741b0,
  protocol tree0baf88005fa85d62bf3cef6a953a0a7e4fc317b2. This freezes the
  user-authored process guide/skill/template/example/hook/pitfalls without E.
  Main independently ran46 compatibility tests; non-math forward evidence is
  [archived](analysis-2026-09-04-opus48-onboarding/process-skill-forward.md).
- E:dcfa742dbc8813970192efe3fbf2bd30dfc38ea9,
  protocol treeb33422364c70f4ea3c08ff83c97009a41438caa6; source at
  `/rmeng_data/robtang/exp-protocol-bundle-work-5iV6EzGB/repo`,
  branch `codex/exp-protocol-bundles-20260904`, pushed. Full E1–E8 is CPU/forward
  accepted:328 pinned tests,464 original-env passes/82 explicit native skips.
  No GPU method improvement/promotion follows from these tests.
- Guard:4ae3d87c446bbda9732537a72b2f0fb3f96ac35a /189319d6.
- Plain baseline uses `claude_vertex_high`, no AWM block, not an old Opus5 control.

Provider/CLI checks actually returned Opus4.8/high/1M via Vertex sercan-v1/global.
The manifest's frozen [record](analysis-2026-09-04-opus48-onboarding/context/record.json)
uses the exact image CLI in read-only bwrap and explicitly reports its scope.
A separate **full Apptainer-image** probe subsequently succeeded too:
[record](analysis-2026-09-04-opus48-onboarding/full-container-record.json),
[raw stream](analysis-2026-09-04-opus48-onboarding/full-container-stream.json).
Both use CLI2.1.219 and the pinned opus_5 image. No tools/model training or
benchmark scoring occurred. Do not replace the already-receipted context file.

The initial bare-apptainer failure was PATH resolution, not absence of an
installation: configured binary is `/rmeng_data/robtang/tools/apt-root/usr/bin/apptainer`
and its library path is in the existing PTB configuration. The original failed
record remains. A restricted shell also cannot see the host PID/controller;
use approved host reads, not a duplicate monitor or fabricated terminal state.

The shared clean submitting clone is
`/rmeng_data/robtang/exp-protocol-opus48-submit-ob4GmM8N/repo`, currently4cd1dbd.
It has independent Git objects, correct fork/upstream remotes, the shared data
link and the same600-mode non-secret site configuration (path references, not
copied auth-file contents). Auto-review initially rejected unknown .env copying;
after content-class/ownership checks it approved the copy. No bypass was used.
The earlier `/tmp/exp-protocol-opus48-submit-HDEYY15Q/repo` is not the worker
submission source. Root's unrelated user meta/analysis drafts remain untouched.

## GPQA and HumanEval onboarding

See [audit](analysis-2026-09-04-opus48-onboarding/new-task-eval-audit.md).
Only GSM8K/AIME2025 currently pass the task allowlist; Opus4.8 high/1M profiles
are admitted. The136-test operator/task-identity suite passes; all four GSM8K
full source/site checks returned0 issues. No new-task allowlist bypass exists.

GPQA Main:Idavidrein/gpqa revision633f5ee89ab8ad4522a9f850766b73f62147ffdd,
official gpqa_main.csv download returned403 despite an existing HF token.
User was asked asynchronously to authorize the account or supply a legally
authorized local path; never ask for a token in chat, accept terms implicitly
or substitute a mirror. No GPQA reference/manifest/receipt exists.

HumanEval:openai/openai_humaneval revision7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544.
Pinned parquet83920 bytes/SHA2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6
was normalized by a trusted offline CPU converter, without displaying or
executing dataset code. Actual164 rows/164 IDs; reference SHA
85d6c3ab3a76590160695424d75d79e90049d5ceaaa809c9771d292054ad68a5.
Files are in `/tmp/ptb-opus48-onboarding-oGLZFUf4/repo/src/eval/tasks/humaneval/`.
The PTB working branch `codex/opus48-cross-task-runtime` is now clean at
09c90b63ad4f9daa2259ab0137ac323e0e345605, pushed to the designated PTB fork.
This is a construction checkpoint, not the operator's adopted PTB pin or task
admission. See the [full checkpoint/evidence](analysis-2026-09-04-opus48-onboarding/humaneval-runtime-checkpoint.md).

Frozen helper94f41494 fixes initial registration/output loss, stderr-only
misclassification and actual-image unbindable mounts without weakening isolation.
49 native backend tests passed. Actual vllm_debug image CPU original-scorer
C/I/import-I/timeout-I plus native sample-log/publication/revalidation passed;
policy failure retained stdout/report and published no score. Host regression
104 passed/22 native skips. Pinned data, one-epoch verify/count bindings, durable
raw evidence, runtime/model/source/limit checks and formal retry/validator wiring
are implemented. Reviews were **informed, not blind**. Zero model results.

Actual scientist opus_5 image CPU original-scorer/log/private-visibility tests
now pass too (five invented programs,52s). AWM independent expected-task checks
are wired through all consumers including manual harvest;136 focused tests pass.
Old guard8 and strict-control7+one quarantine remain unchanged in actual re-audits.
Shared offline parquet is now provisioned and read/hash-checked in both images.
Both local and shared-operator ignored PTB .env files contain the same public
bwrap path, with no scheduler/identity changes.12 matched four-repeat
drafts are in `experiments/posttrainbench/*humaneval*.draft.yaml`, explicitly
unqueued and still rejected by the task allowlist. They contribute no held cells.
Remaining: compute-node/GPU-enabled/outer-timeout acceptance, common PTB pin
adoption and full three-arm admission. On ondem0, trusted host-key verification,
OS Login username and existing key-pair/profile matches are established, but the
node still refuses that identity. The other configured cloud identity cannot
refresh its metadata credential. No IAM, key or login-policy changes occurred.
See [preparation evidence](analysis-2026-09-04-opus48-onboarding/shared-data-preparation/README.md).
Authorized node access is needed. No HumanEval receipt exists.

## Monitoring and trace review

Current monitor **PID3684437**, verified live; first tick16:16:40 shows0/21
terminal, next17:16:40. Args:90826–90830 plus92125–92140, threshold8,poll3600s.
Log `/tmp/exp-protocol-admin-handoff.sTGNfhBs/new-monitor.log`.
Old3564003 naturally completed with17 known administrative terminals; helper
3676924 rechecked their harvested receipt/status evidence and then started the
new detector. Both old/helper processes are now finished. No early kill/reset
occurred and no clean result was generated. The old ready state, completed
handoff and source/tests are [archived](analysis-2026-09-04-opus48-onboarding/monitor-admin-handoff/README.md).
The previous22→38 expansion record remains separate and unchanged.

New Opus4.8 clean count0. Old c01s05 and p00s03 remain separately buffered legacy
evidence, not new-study samples. Slurm terminals only wake the loop: harvest,
validate, filter eligibility/judges and then invoke local Claude Opus5[1m]/max
after eight NEW clean or a completed predeclared comparison block. Scientist
Opus4.8 and analyst Opus5 are different roles. The hourly detector remains live.

Launch blockers persist across at least three goal turns: native isolation/change
approval and authorized node access remain unavailable, GPQA remains403. However,
GPQA's synthetic CPU scorer/loader/evidence integration is independent remaining
work, so the full goal stays active rather than being marked blocked prematurely.
Proceed with [that bounded work](../spec/2026-09-04-exp-protocol-gpqa-onboarding.md),
then re-audit the actual remaining authority/access gates. The study is incomplete;
keep the approved40-cell objective and do not count12 unsubmitted drafts as inventory.

[Window04](2026-09-04-round-02-window04-decision.md) is closed at14 NEW clean,
all traces/syntheses/focused audits read; do not launch a duplicate synthesis.
L/P designs remain in the E branch, not implementations or funded new arms.
E2/J/K old standalone drafts are historical; J had no receipt and its queue
entry was removed. Never use the old763701c J submission clone.

Earlier E5/E6/E7/E8 native/forward evidence and full persistent raw archives are
recorded in the candidate's `2026-09-04-bundle-construction.md` and
`data/ptb/review-artifacts/`. The extracted CPU runtime remains
`/tmp/exp-protocol-save-runtime.JEZlHo`. Preserve all original traces, receipts,
failed probes and user drafts; no artifact or running Slurm work was deleted.
