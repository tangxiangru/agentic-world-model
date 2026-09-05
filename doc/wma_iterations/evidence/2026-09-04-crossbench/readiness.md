# 2026-09-04 WMA redesign readiness audit

Read-only bounded review; observed working tree around 11:50–11:58 UTC. Source hashes: `readiness-source-hashes.json`. The working tree is changing concurrently; no repository file was edited by this audit. No real model, Slurm job, container, commit or submission was started. Tests used process-local settings and temporary files only.

## Decision

**The architecture is substantially wired up but is NOT ready for a scored launch in its current form.** There is no missing production import caused by the new top-level `awm/wma_decisions.py`: it is only a compatibility alias, and the real contract now lives in the already shipped `awm/exp_protocol/decisions.py`. Public shipped-code acceptance passed, and an independent private-shipped-tree import succeeded.

The immediate concrete blocker is the real isolated-probe self-check: it fails under this shared Unix UID with `/bin/sh: 1: Cannot fork`. The broker caps `RLIMIT_NPROC` at 128, a count for the whole real UID, while this host had **214 UID processes / 6417 UID tasks** at the read-only check. One additional test failure is an obsolete comparison-transcript glob, not evidence of a missing transcript. The production Apptainer/kernel/CLI/API combination has not been accepted. `apptainer` is not on this task's PATH, so that gate was not exercised here.

## Blocking / acceptance items

1. **P1: A shared-user process limit prevents all built-in Claude reviews before the API call.** `awm/wma/isolation.py:_child` applies `_cap_resource(RLIMIT_NPROC, 128)` before executing `/bin/sh`; `isolated_tools` then self-checks by asking that shell to run Python. The observed test failure is `isolated probe self-check failed: /bin/sh: 1: Cannot fork`. Nine tests cannot reach their intended positive/negative canaries for this reason (3 failure bodies, 6 fixture errors). The behavior is fail-closed for WMA, which is correct for the I/O boundary, but operationally this can reduce the treatment to unavailable WMA. A fix needs a real per-probe process accounting approach or a deliberately justified compatible resource policy; simply count fewer nominal jobs, cancel unrelated processes, skip the canaries, or retry unisolated is not a remedy. Re-run the full canaries after a fix, in the actual worker UID/context.
2. **Production container and real model/broker round trip remain a gate.** The launcher currently starts `apptainer exec --containall --cleanenv --writable-tmpfs`, with the same image as the scientist, a read-only session plus writable card/queue overlays, private artifacts and private history. It does not synchronously establish Landlock/seccomp or exercise broker tools at sidecar startup; this happens only when a review arrives. Both future allocated nodes should demonstrate Landlock ABI >=3, seccomp, Python stdlib/shared-library allowlist viability, writable private scratch, and the real same-image UID task behavior before GPU scientific time is committed. Then test the deployed Claude flags (`--bare --disable-slash-commands --tools '' --strict-mcp-config --mcp-config ...`) and one real Opus 4.8 comparison + one ordinary review through `list_inputs/read_file/run/write_result`, preserving transcript/tool inventory/cost. This audit intentionally did not start a model.
3. **Test suite is not green.** The selected 52-test run had **42 passed, 4 failed, 6 errors**. Three failures and six errors are item 1. `tests/test_wma_comparison_flow.py:237` expects `*.transcript.*.jsonl`, but `backends.transcript_path` deliberately writes `comparison.transcript.jsonl`; the comparison itself completed with correct schema/model/$0.025 cost. Actual artifact is `/tmp/pytest-of-robtang_google_com/pytest-726/test_command_backend_accepts_s0/private/output/comparisons/awm-wma-compare-1fgoeecf/comparison.transcript.jsonl`. Align the intended naming assertion (or filename contract), then rerun the relevant test. Do not confuse this test mismatch with a failed joint-comparison call.
4. **Freeze and document the deployed treatment before launch.** New modules and policy files are still untracked/modified. Git-archive shipping from an existing commit cannot include these edits until the user-authorized review/commit is completed. The design spec still calls itself a draft and says no skill/prompt/harness has been modified, which no longer describes this working tree. Update implementation status, separate common isolation/action-record infrastructure from policy and joint-comparison treatments, and pin exact public/private AWM and PTB submodule SHAs. Four-replicate manifests should be preregistered after these identities exist. Do not mutate previously frozen historical lists to import this new runtime into old cohorts.

## Shipping closure

| File / feature | Public scientist ship | Private WMA ship | Evidence |
|---|---|---|---|
| `awm/exp_protocol/decisions.py` | Included by `awm/exp_protocol` directory | Included by same directory | Production imports consistently use `.exp_protocol.decisions` / `awm.exp_protocol.decisions`. |
| `awm/exp_protocol/run.py` | Included by directory | Included by directory, although WMA does not execute it | New public launch wrapper resolves within shipped code. |
| `awm/wma/compare.py`, `isolation.py` | Excluded intentionally | Included by `awm/wma` directory | Private runtime closure imported successfully from a copied ship tree. |
| `awm/wma_client.py` | Explicitly included | Not included and not required by sidecar/compare import closure | Public CLI routes candidate/action commands through client. |
| `awm/wma_decisions.py` | Not listed | Not listed | Convenience alias only; production callers no longer require it. Do not report it as a missing dependency. Tests/ergonomic root imports may still use the alias in the full checkout. |
| `skills/exp_protocol/proposal.example.json` | Included by `skills/exp_protocol` directory | Not needed | Public template accompanies the modified protocol skill. |
| WMA skill/manual/example | Excluded intentionally | Included by `skills/wma` | Scientist/WMA role boundary retained. |

Actual lists inspected: `awm/ptb_experiments.py:EXP_PROTOCOL_SHIP` and `WMA_PRIVATE_SHIP`.

Independent private closure test copied exactly `WMA_PRIVATE_SHIP` from the working tree to `/tmp/wma-crossbench-20260904/ship-private-sn_0l0u8`, imported sidecar/compare/isolation/backends/decisions under `python -I -B`, and asserted all 12 loaded `awm*` module paths came from that copied root. Exit 0; machine-readable result `ship-private-import.json`.

Public closure passed both `tests/test_ptb_sandbox_acceptance.py` tests. These copy the working tree shipping list, disable user-site imports, and exercise the public CLI. This proves current closure, not git-archive completeness at an older commit or production container compatibility.

## What is already connected

- `awm wma propose/compare/choose/act` is routed through the public thin client. A proposal carries 1–3 real candidates, shared situation/evidence and the scientist's initial preference. A singleton requires a reason. Comparison is a single `Backend.run` over a detached frozen proposal, not per-candidate L3 sorting.
- `compare.py` validates candidate-comparison outputs (complete ranking, adjacent-pair reasons, candidate assessments and scoped suggestions); it stores unique private proposal/output/failure artifacts and replaces model-authored cost/access/isolation claims with runtime measurements. Legacy single-card ledger files remain separate.
- Sidecar dispatches the new comparison request schema and retains the old review schema. Normal review now archives request-scoped card/lock/verdict, uses a private card snapshot, binds a recomputed fingerprint before/after review, and publishes a current request-bound verdict only after archive creation.
- Decision/action files use write-once atomic publication. `exp_protocol run` verifies the lock, current pinned inputs, request-bound verdict when delivered, a current `proceed` action, and selected training candidate binding; a single proceed action cannot start twice. `not_attached`, `failed`, `timeout`, `skipped` are explicit recorded states that can permit scientist progress. Therefore a completed scientist cell **does not prove that the isolated WMA ran successfully**.
- Built-in Claude uses the broker unconditionally (`isolated=True`), strips legacy directory grants and exposes no built-in tools. Broker `run` children receive Landlock/seccomp and a cleared environment, with no GPU/network/ML packages. Broker direct reads are limited to immutable copied exported inputs. No unisolated fallback occurs when the self-check fails.
- Built-in Codex is configured isolated but has no supported broker adapter and **fails closed**. It is not currently a runnable cross-backend comparison arm. Heuristic remains a single-card floor and is explicitly rejected as a joint-comparison backend.

## Boundaries and incomplete portions that matter for experimental interpretation

- The new safety architecture intentionally changes the available evidence and capabilities. It is **CPU/static only**, not feature-equivalent to the prior online scratch environment. No trusted GPU/evaluation-summary broker has been implemented; that part of the design remains future work. Missing ML/GPU capability must remain “untested”, never an automatic scientific failure.
- `collect_review_inputs` exports a current frozen card/lock/preflight, the **one** pinned `lock['script']` if its suffix and hash pass, earlier closed cards (max128), historical card/index files (max512), skill files and explicitly trusted `probe_files`. It does **not** automatically export every separately pinned `lock['configs']` entry, all proposal-linked scripts, the session monitor, live raw logs/evals, model checkpoints, training data or evaluator internals. The documentation phrase “locked script/config file” means that one script slot may hold an approved suffix; it is not an all-configs export implementation.
- For a **decision comparison**, `card_id` is `decision-NN` and the proposal is JSON, so the exp-card branch of `collect_review_inputs` does not run: current-session earlier closed cards are not automatically supplied. The actual joint context is the frozen proposal's embedded evidence, bounded curated history, skill, and any explicit trusted exports (none added by `compare.py`). Arbitrary evidence path strings in a proposal do not make the referred files readable or hash their contents. This is safe by default but needs to be understood when judging comparison quality. Do not claim it reuses all live session evidence.
- The model API/CLI process remains credentialed and outside Landlock; the guarantee relies on disabling native tools and using the broker as its only model-controlled I/O surface. Every broker probe child has the OS boundary; the model process itself is not an isolated network-off process. Real CLI feature compatibility is thus a substantive acceptance test, not a packaging detail.
- Exported text is not semantically decontaminated. Prior cards can already contain summaries copied from a prohibited evaluation; file isolation cannot remove that. Keep legacy leak flags/exclusions and freeze the new permitted evidence policy symmetrically. Do not claim zero old scanner flags prove zero semantic leakage.
- `scan_transcript` is still the legacy built-in tool/path scanner; broker call names are different. The recorded isolation method, exact input inventory and real canaries provide new boundary evidence, while the old `access.outside` count alone is no longer a sufficient health metric. Do not silently treat a changed scanner surface as a skill gain.
- The `run` wrapper is a protocol-enforced launch path, not OS-wide prevention of a scientist manually running another shell command. The skill instructs compliance, and the wrapper proves checked launches only. Outcome collection and raw trajectory audit still need to distinguish actual use from merely having these commands installed.
- `RLIMIT_NPROC=128` is not a per-review process cap and does not constrain privileged/root workers; the 60s CPU/wall and512MiB caps are per probe, not a separately enforced aggregate review CPU budget. Scheduler/container controls remain material.

## Runnable comparison architecture after gates

For cross-benchmark Opus 4.8 ×4 repeats, the smallest interpretable two-arm design is:

1. Shared redesigned public scientist protocol, same real-candidate briefs, initial preference and action/launch records, same total scientific budget and evaluator; scientist selects without WMA in control (`not_attached` recorded).
2. Identical public protocol and budget, new isolated Claude/Opus4.8 WMA with one joint comparison at the decision point **plus** the selected formal card's separate blocking review. Count both API costs, wait and all unsuccessful calls. Retain scientist final choice.

That contrast estimates the whole added WMA treatment under the new common protocol. It does not separately identify the evidence-scope skill change versus joint comparison. If the question is the incremental benefit of joint comparison, a third matched isolated single-card-only WMA arm (same corrected policy, same candidate preparation/records) is needed, with a deliberately defined way to omit joint calls while still recording the shared choice contract. The present built-in client automatically attempts joint comparison whenever a sidecar is attached; no manifest treatment switch for “attached but single-card-only” was found in the audited files. Do not infer that third arm exists merely because legacy multi-card review remains.

Freeze benchmark-specific allowed evidence/evaluator and scale only after real review/comparison broker smoke passes. Four repeats are a mechanism screen; promotion/held-out gates remain independent. Do not fill a failed isolated-WMA treatment with silent no-WMA cells and report it as a healthy loop.

## Validation run

Command (process-local settings, no source edits):

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest -q -p no:cacheprovider tests/test_wma_isolation.py tests/test_wma_compare.py tests/test_wma_comparison_flow.py tests/test_wma_sidecar.py tests/test_exp_protocol_run.py tests/test_ptb_sandbox_acceptance.py
```

52 collected: isolation12, comparison10, comparison-flow9, sidecar10, run9, public ship2. 42 passed, 4 failed, 6 errors as itemized above. Kernel was `6.8.0-1027-gcp-tcpx`; UID708035571 inherited unlimited NPROC. `/proc` count was metadata only, no command lines or process environments read. No cap, UID setting, kernel policy or global environment was changed by this audit.
