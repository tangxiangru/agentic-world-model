# Final local validation of the completed WMA redesign

Read-only verification of the implementation-complete state on 2026-09-04. Implementation record read: `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-04-decision-review-implementation.md`. No source/test edit, privilege change, limit relaxation, model invocation, Slurm action or cancellation was performed.

## Exact result

**The ordinary comparison and immutable-record flows now pass, but the new isolation runtime still fails its real kernel self-check in this actual tool environment. It is not production-ready on the strength of the “354 passed” record alone.**

| Test file | Passed | Failed | Errors | Skipped |
|---|---:|---:|---:|---:|
| `tests/test_wma_isolation.py` | 3 | 3 | 6 | 0 |
| `tests/test_wma_comparison_flow.py` | 9 | 0 | 0 | 0 |
| `tests/test_wma_record_integrity.py` | 6 | 0 | 0 | 0 |
| Total | **18** | **3** | **6** | **0** |

Exit code **1**. JUnit timestamp `2026-09-04T12:39:10.223587+00:00`, duration **2.545 seconds**, hostname `slurm2-a3nodeset0-3`.

Every failure/error is the same isolated Python self-check failure:

```text
awm/wma/isolation.py:418
IsolationError: isolated probe self-check failed: /bin/sh: 1: Cannot fork
```

The command-backend integration test wraps the same error as `BackendError: claude: input isolation unavailable`. The nine affected tests do not reach their intended positive/negative broker or kernel-canary assertions. They are **not skipped**, and this run does not demonstrate those canaries passing. The three passing isolation tests cover rejection/selection paths that do not require successful normal isolated child execution.

The earlier comparison transcript-naming failure is fixed: `comparison-flow` now passes **9/9**, including the standalone comparison/measurement test. Record integrity passes **6/6** including path mapping, write-once atomic publication, stale-reply protection, candidate binding/abandonment, and nested private harvest without ledger double count.

## Exact command and evidence

Working directory:
`/home/robtang_google_com/gangda_workspace/agentic-world-model`

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest -q -p no:cacheprovider --tb=short -ra --junitxml=/tmp/wma-crossbench-20260904/final-local-tests.xml tests/test_wma_isolation.py tests/test_wma_comparison_flow.py tests/test_wma_record_integrity.py > /tmp/wma-crossbench-20260904/final-local-tests.log 2>&1
```

`PYTHONDONTWRITEBYTECODE` is process-local and only prevents repository bytecode writes. No global environment, ulimit, UID, capability, seccomp/Landlock setting or test implementation was changed to make a pass. The tests themselves execute the implementation's unchanged restricted child policy.

Artifacts:

- `/tmp/wma-crossbench-20260904/final-local-tests.xml`: exact per-test results.
- `/tmp/wma-crossbench-20260904/final-local-tests.log`: full short-trace pytest output.
- `/tmp/wma-crossbench-20260904/final-local-pretest.json`: actual UID/load/kernel and source hashes before the run.
- `/tmp/wma-crossbench-20260904/final-local-posttest.json`: matching hashes and environment after the run, plus parsed result totals.

All eight tracked audit source/test hashes are unchanged between pre-test and post-test. The isolation source is also byte-identical to the previous readiness audit.

## Why the result differs from the implementation record's 354 passed

The current code still sets `_MAX_UID_TASKS = 128` at `awm/wma/isolation.py:52` and applies it with `_cap_resource(resource.RLIMIT_NPROC, _MAX_UID_TASKS)` at line195. Linux's NPROC limit is scoped to a real UID and counts its tasks/threads; it is not limited to the new probe. The shell's normal child creation therefore fails when the real UID already owns substantially more than128 tasks. The control process inherits unlimited NPROC, but the implementation deliberately lowers the restricted child's own limit before asking `/bin/sh` to start the Python self-check.

This task ran unprivileged, **UID=EUID708035571**:

| Snapshot | Same-UID processes | Same-UID tasks | Parent NPROC soft/hard |
|---|---:|---:|---|
| Before, 12:31:50 UTC | 220 | 8072 | unlimited/unlimited |
| After, 12:45:51 UTC | 181 | 5309 | unlimited/unlimited |

Host/kernel: `slurm2-a3nodeset0-3`, `6.8.0-1027-gcp-tcpx`. `/proc` counts used ownership and task-directory metadata only; no process command lines, environment, credentials or workload contents were read.

This reproduces the prior audit (214 processes /6417 tasks, same UID and kernel) with the **same isolation source hash**, so “the implementation is complete” has not removed this deployment-sensitive limit. The normal comparison and record tests passing is compatible with isolation failing because those flows largely use test backends, while the isolation tests exercise the kernel/broker path.

The implementation record gives a count of354 and its claimed coverage, but does **not** preserve the exact test command, effective/real UID, same-UID task load, capabilities or source hashes of that run. Accordingly, the exact historical cause of354 passing cannot be established from that record. A lower-load/dedicated UID, root or relevant capabilities can make this UID-scoped limit behave differently; these are plausible explanations, **not verified claims about the other run**. It would be inaccurate to say that354 passed was disproved everywhere, or that this actual high-load non-root context is validated by it.

## Precise production acceptance conclusion

**Proceed with experiment design/profile preparation, but do not treat this current local environment as an accepted isolated WMA runtime.** The safety boundary currently fails closed before any paid WMA API call; the scientist protocol is allowed to record WMA unavailable and proceed. Such cells would test degraded/unavailable advice, not the intended new treatment, even if final PTB validation succeeds.

Acceptance must be executed using the exact frozen shipping SHA, production SIF, Python and Claude CLI, normal worker UID/capabilities and expected concurrent workload on the planned nodes. Record UID/EUID, same-UID task counts, inherited limits, kernel/ABI, image identity, exact command and source hash with the canary results. Require successful isolated self-check, actual indirect file/cache/symlink/network/process canaries and normal inherited child execution without skipped tests or an unisolated fallback.

If the production execution context has more than128 accounted tasks for the nonprivileged real UID, this unchanged hard cap remains a concrete blocker. Choose a reviewed runtime resource-accounting fix or a genuinely isolated/dedicated worker identity/resource boundary, then rerun acceptance under that declared design. Do not cancel unrelated tasks, elevate this audit, raise limits behind the tests, or mark the canaries passed because an ordinary comparison stub passes. Root-worker tests alone also do not validate a128-task per-probe quota, because NPROC does not provide that guarantee for privileged workers.

Even if a correctly identified production context passes these CPU gates, the separately documented **real Claude/Opus4.8 MCP/broker round trip** remains untested here. Both a joint comparison and an ordinary blocking review must show the intended broker-only tools, correct result delivery and measured cost in the production image before scientifically interpreting the new WMA treatment. No such model call was made by this read-only validation.

## Source/test SHA256 at this run

- `awm/wma/isolation.py`: `0d25f4fcb59abfc2ecd540defb756fd63a4214b5946d1c33c3bfc1678f6ccd6d`
- `awm/wma/backends.py`: `eb4e3e9814456f0711c584956dec785c9a3dd1a5f4dcf2daa63e47897b09f94b`
- `awm/wma/compare.py`: `ccd9ddd506e34c767a9b0fbe5b627a2b97feff89b0dd9b5a2b38b0ebbd4afdc9`
- `awm/wma/sidecar.py`: `d4c522c5e6fdb9d54a27b1d7ba299357c4de015fb473b6bcb562c3bd84bc5457`
- `awm/exp_protocol/decisions.py`: `07e7817ca1457c66ef9633a16e86d68ffaf5a2c477d5bbe14162f2acde0c11ee`
- `tests/test_wma_isolation.py`: `4af3acee4000828104ae4eb65782ce77806982c94d89ce20373cb856c3cf47e6`
- `tests/test_wma_comparison_flow.py`: `1929823e3468626191737655defe68664ef925d62722a7ab9654e9484022ffcb`
- `tests/test_wma_record_integrity.py`: `014b0264719d0afea7b88aa1471051966bd3e6f655f7614430e054e2d22abfe6`
