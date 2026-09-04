# Claude advisory provenance and wrapper-failure verification

Read-only operator verification, 2026-09-04. Advisory reviewed: `/home/robtang_google_com/.claude/plans/ultracode-you-are-the-snazzy-liskov.md`, especially lines 85–116 and 461–463. Event: `20260904T164313Z-792dc7f482`.

## Findings

1. **Correct the deployed flushing claim.** The top-commit diff does introduce `say(..., flush=True)`, but none of the 20 targeted cells was configured to import that newer client. Both halves of w10, both halves of c10, and G bind the same materialized public checkout from `ae4672487cc16f1d75921dcaab85498d7adc938e`. Its `awm/wma_client.py` has `out=print`, no `say()` and no `flush=True`. This is verified from the materialized bytes, not inferred from `status.awm_sha` alone.
2. **The G package is a skill-only change against the baseline package.** The private ship list excludes `awm/wma_client.py`. All 22 materialized private source/skill files match their frozen archives; baseline-to-G differences consist only of `skills/wma/SKILL.md` (13 added lines). Their checkout marker JSON also differs, as expected, and is not executable runtime content.
3. **The error-message counts are correct for the exact seven event cells.** Four G jobs plus c10r08 have the same line-240 `fi` syntax-error message; c10r05 and c10r07 have the same stale-file-handle message. All seven retained stdout tails end `PREFLIGHT PASSED`.
4. **Qualify “two causes in the sbatch epilogue.”** These are two observed failure signatures, not two established root causes. Frozen PTB `62203e49` contains a 212-line `single_task.sbatch`, ending at the `bash src/run_task.sh` invocation. It has no line-240 epilogue. The error names the live repository entrypoint. The exact wrapper bytes/inode read at failure have not been recovered here.
5. **No n=8 cohort exclusion or score recalculation follows from this flushing allegation.** Keep the 4+4 submission strata for provenance and sensitivity reporting, but withdraw the claim that w10r01–04 used a different deployed client from w10r05–08 or G. The matched public runtime, baseline private runtime and frozen PTB pipeline are the same across both w10 halves. This does not prove absence of every temporal/environmental difference or establish any causal score effect.

## Frozen receipts and deployed packages

The following receipts are the job-to-cell authority for this verification. All are on `gangda_wma_evolve`, and all record PTB `62203e498d1007875b32a3415c11caa95ccf4a2e`.

| Cells | Jobs | source.top_commit | Public SHA | Private SHA |
|---|---|---|---|---|
| w10r01–04 | 90982–90985 | 76d75dec263a5b57a25fd0b6984a0775e7984bd3 | ae46724… | ae46724… |
| w10r05–08 | 91425–91428 | 6f6a5755bced28e009d1478e66337168ed096ef5 | ae46724… | ae46724… |
| c10r01–04 | 90998–91001 | 76d75dec263a5b57a25fd0b6984a0775e7984bd3 | ae46724… | None |
| c10r05–08 | 91429–91432 | 6f6a5755bced28e009d1478e66337168ed096ef5 | ae46724… | None |
| w13r01–04 (G) | 91441–91444 | 487483842337a51057f0e45f1933d9bc7fcbfd06 | ae46724… | 125a434… |

The three materialized directories were checked against `git archive <sha> -- <receipt paths>` in memory. Every source file matched byte-for-byte; the existing tree-digest algorithm was independently recomputed with the marker excluded.

| Package | Source/skill files | Exact frozen-archive match | Recomputed digest equals receipt/marker |
|---|---:|---|---|
| Common public ae46724 | 19 | Yes | `12d559c2ef232f37a234ad6adee565c631fa1a2d839a9f458a72225d8a48207c` |
| Baseline private ae46724 | 22 | Yes | `8c4ed1dcbe5693eaeacb97c6da65bc7457cbcc7cdb19287131f04d76f1a628c0` |
| G private 125a434 | 22 | Yes | `ea08c45498d2d9b83da9711b45198dab04c882bfa2fbb99bd03b7d4bd15a5918` |

Full directories:

- Public: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/awm-checkouts/ae4672487cc16f1d75921dcaab85498d7adc938e-160709abf33b`.
- Baseline private: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/wma-private-checkouts/ae4672487cc16f1d75921dcaab85498d7adc938e-06ebd6026b1f`.
- G private: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/wma-private-checkouts/125a434e6d73d067427911332663060fe2dce558-06ebd6026b1f`.

Every receipt's `awm_checkouts[cell]` names the identical public directory, path list, SHA and digest. The public marker says it was materialized at `2026-09-03T01:49:18.030308+00:00`, before the first w10 batch. All 20 retained `status.json` files agree on `awm_sha`; the retained `runtime_provenance.json` files agree with the receipt top/PTB/private identities. The control provenance correctly records `wma_runtime.enabled=false`.

The common public client SHA-256 is `860fd9ade85478cd6fc6310311ac1e7af0508d47af393eb34013bfe1c47b50ce`. Its `wait_for_verdict` line 140 and `review_and_wait` line 173 default `out` to `print`. The public protocol CLI imports the sibling `wma_client` at line 160 and calls `review_and_wait` at line 174 without overriding `out`.

The private ship list in the frozen submission code and receipts is exactly:

```
awm/__init__.py
awm/paths.py
awm/exp_protocol
awm/wma
skills/wma
```

Neither private checkout contains `awm/wma_client.py`. The non-skill private diff `git diff ae46724 125a434 -- awm/__init__.py awm/paths.py awm/exp_protocol awm/wma` is empty. The complete private shipped-path diff is `skills/wma/SKILL.md | 13 +++++++++++++`. Thus the advisory's “one edit is true of the skill but not of the shipped archive” is wrong if “shipped archive” means the scientist and sidecar packages actually mounted in these cells. The full outer repository tree is not the shipped package.

## Reachable launch wiring

The following references are frozen Git objects, not assertions about the current working tree:

- Outer `48748384:awm/ptb_experiments.py`, lines 246–285: `materialize_awm_checkout(sha, shipped)` archives the explicitly supplied SHA/path list. Lines 643–659: public checkout comes from `cell.awm.sha`, binds read-only to `/home/ben/awm`, and private checkout comes from `cell.wma.sha`. `awm/ptb_experiments.py` has no diff across the three top commits in the table.
- PTB `62203e49:src/commit_utils/slurm/single_task.sbatch`, lines 119–137: materializes **PTB** `POST_TRAIN_BENCH_FROZEN_PTB_COMMIT` into job scratch and runs from it. Lines 205–212 invoke `src/run_task.sh`.
- PTB `62203e49:src/run_task.sh`, lines 388–406 and 433: passes the explicit public read-only bind to the scientist container. Lines 305–331: the sidecar uses a separate clean container, `PYTHONPATH=/opt/awm`, private checkout bound read-only at `/opt/awm`, and `python3 -m awm.wma.sidecar`.
- PTB `62203e49:agents/claude_vertex_high_awm/solve.sh`, lines 28–46: selects `/home/ben/awm`, prepends it to `PYTHONPATH`, writes the scientist's `awm` command as `exec python3 -m awm.cli`, then runs setup using the mounted SHA. The associated `env_passthrough.txt` forwards `AWM_SANDBOX_SETUP` and `AWM_CHECKOUT_SHA`.
- PTB `62203e49:src/utils/record_runtime_provenance.py`, lines 177–190: `source.top_commit` is recorded from `POST_TRAIN_BENCH_FROZEN_TOP_COMMIT`. It is metadata, not the Python module source selector. Lines 212–215 separately record the private mounted checkout SHA/digest.

The `76d75dec`→`6f6a5755` top-tree diff really adds `say()` and changes both defaults to it. But the launch wiring above has no route from that newer outer `awm/wma_client.py` to these cells. The evidence supports “same client source bytes”; it does not claim that every scientist chose identical shell redirection, PTY settings or optional Python buffering environment. A separate trajectory census would be needed to assess those process-level choices.

## Exact retained wrapper failures

These seven jobs alone were checked for this claim. Every retained stderr tail is a single line. Using the common prefix

```
/home/robtang_google_com/gangda_workspace/agentic-world-model/third_party/PostTrainBench/src/commit_utils/slurm/single_task.sbatch
```

the two complete suffixes are:

```
: line 240: syntax error near unexpected token `fi'
: error reading input file: Stale file handle
```

| Cell | Job | Retained state | stderr signature | stderr file SHA-256 |
|---|---|---|---|---|
| c10r05 | 91429 | FAILED | Stale file handle | `a67ea049d0f6c01424356cf9b0ddcc5d1ea1c307339d7ee98563ba14d6f9c5e2` |
| c10r07 | 91431 | FAILED | Stale file handle | `a67ea049d0f6c01424356cf9b0ddcc5d1ea1c307339d7ee98563ba14d6f9c5e2` |
| c10r08 | 91432 | FAILED | line 240 syntax error | `53cbef946dcfbe4b470f24a61aebd83c3c2204ea4e67b088cf2daf13c49fb8ed` |
| w13r01 | 91441 | FAILED | line 240 syntax error | `53cbef946dcfbe4b470f24a61aebd83c3c2204ea4e67b088cf2daf13c49fb8ed` |
| w13r02 | 91442 | FAILED | line 240 syntax error | `53cbef946dcfbe4b470f24a61aebd83c3c2204ea4e67b088cf2daf13c49fb8ed` |
| w13r03 | 91443 | FAILED | line 240 syntax error | `53cbef946dcfbe4b470f24a61aebd83c3c2204ea4e67b088cf2daf13c49fb8ed` |
| w13r04 | 91444 | FAILED | line 240 syntax error | `53cbef946dcfbe4b470f24a61aebd83c3c2204ea4e67b088cf2daf13c49fb8ed` |

All seven stdout tails contain 31 lines and end `PREFLIGHT PASSED`. This is expected from the logging arrangement: frozen `src/run_task.sh` redirects stdout/stderr to the result directory's `output.log` / `error.log` at lines 52–53. An sbatch stdout tail ending at preflight does not mean the scientist or evaluation stopped at preflight.

All seven exact raw `output.log` files end:

```
PTB COMPLETE FLOW PASSED: final model, canonical official verdicts, full-eval metrics/log, trace, monitor, and provenance are valid.
```

The frozen event snapshot independently records each as `complete=true`, `issues=[]`, `judge_flags=[]`. Together these support the operational interpretation that the wrapper failed after a scientifically complete pipeline. This verification relied on the frozen validator result and pipeline completion evidence; it did not rerun evaluation or judges.

**Why the precise “epilogue” root-cause attribution is not established:** frozen `single_task.sbatch` at PTB `62203e49` ends on line 212. PTB `62203e49:src/commit_utils/slurm/submit.sh` exports the repository path as `POST_TRAIN_BENCH_SLURM_ENTRYPOINT` at line 98. The sbatch root handoff at lines 21–26 executes that live repository entrypoint with `/bin/bash` under the scientist Unix user. The wrapper itself is therefore distinct from the frozen PTB `src/run_task.sh` pipeline. The error pathname names this live entrypoint; it does not establish that a syntax error exists in the frozen 62203e49 script. Script replacement or modification while Bash retained an input stream is a plausible explanation, particularly given the stale-file-handle siblings, but this report has no contemporaneous inode/change history or captured wrapper bytes to prove it. The two message classes may share an underlying cause.

The statement “FAILED carries no information about experimental validity” should be narrowed: these seven FAILED states do not invalidate their independently validated completed artifacts. FAILED still conveys an operational failure worth recording and investigating. The advisory's broader “12 of 20 cells” count was outside this exact-tail check and is not confirmed here.

## Cohort matching disposition

Keep w10r01–08 as the existing eight-cell baseline for the current descriptive pooled comparison; do not drop its first four cells or match G only against extension4 on account of the proposed client flush change. The corresponding c10r01–08 control pool likewise needs no provenance-based score revision from this claim. Preserve first4/extension4 labels because the batches were launched at different times under different outer source commits, and use them if a sensitivity analysis is otherwise called for. The wrapper failure uncertainty should stay in the operational audit and does not establish altered scientist client code.

G's configured and materialized difference relative to both baseline halves is the 13-line private skill edit. This verifies the shipped source contrast; it does not establish causal attribution for scores, erase the n=4 G limitation, certify unrestricted statistical pooling, or adjudicate the advisory's other scientific claims.

## Artifact index

Receipt → manifest → spec, then retained per-cell evidence and raw result directory:

- **wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2**
  - Receipt: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/batches/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2/formal-2026-09-03T014918.962581+0000.json`
  - Manifest: `/home/robtang_google_com/gangda_workspace/agentic-world-model/experiments/posttrainbench/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2.yaml`
  - Spec: `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-02-wma-round01-online-gsm8k-gemma4b.md`
  - Retained evidence: `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2/<cell>/status.json` and `runtime_provenance.json`; cells: w10r01, w10r02, w10r03, w10r04.
- **wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1**
  - Receipt: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/batches/wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1/formal-2026-09-03T180842.687711+0000.json`
  - Manifest: `/home/robtang_google_com/gangda_workspace/agentic-world-model/experiments/posttrainbench/wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1.yaml`
  - Spec: `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-02-wma-round01-online-gsm8k-gemma4b.md`
  - Retained evidence: `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1/<cell>/status.json` and `runtime_provenance.json`; cells: w10r05, w10r06, w10r07, w10r08.
- **wma-gsm8k-gemma4b-high-r02-ctl-x4-v2**
  - Receipt: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/batches/wma-gsm8k-gemma4b-high-r02-ctl-x4-v2/formal-2026-09-03T014940.966939+0000.json`
  - Manifest: `/home/robtang_google_com/gangda_workspace/agentic-world-model/experiments/posttrainbench/wma-gsm8k-gemma4b-high-r02-ctl-x4-v2.yaml`
  - Spec: `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-02-wma-round01-online-gsm8k-gemma4b.md`
  - Retained evidence: `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r02-ctl-x4-v2/<cell>/status.json` and `runtime_provenance.json`; cells: c10r01, c10r02, c10r03, c10r04.
- **wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1**
  - Receipt: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/batches/wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1/formal-2026-09-03T180905.303723+0000.json`
  - Manifest: `/home/robtang_google_com/gangda_workspace/agentic-world-model/experiments/posttrainbench/wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1.yaml`
  - Spec: `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-02-wma-round01-online-gsm8k-gemma4b.md`
  - Retained evidence: `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1/<cell>/status.json` and `runtime_provenance.json`; cells: c10r05, c10r06, c10r07, c10r08.
- **wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4**
  - Receipt: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/batches/wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4/formal-2026-09-03T192916.640188+0000.json`
  - Manifest: `/home/robtang_google_com/gangda_workspace/agentic-world-model/experiments/posttrainbench/wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4.yaml`
  - Spec: `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-03-wma-round04-probe-selection.md`
  - Retained evidence: `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4/<cell>/status.json` and `runtime_provenance.json`; cells: w13r01, w13r02, w13r03, w13r04.

Exact seven error/output-tail and raw completion evidence directories:

- **c10r05 / 91429**: retained `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1/c10r05` (`slurm.err.tail`, `slurm.out.tail`, `status.json`, `runtime_provenance.json`); raw `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1_c10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91429` (`output.log`).
- **c10r07 / 91431**: retained `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1/c10r07` (`slurm.err.tail`, `slurm.out.tail`, `status.json`, `runtime_provenance.json`); raw `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1_c10r07_formal_r10/gsm8k_google_gemma-3-4b-pt_91431` (`output.log`).
- **c10r08 / 91432**: retained `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1/c10r08` (`slurm.err.tail`, `slurm.out.tail`, `status.json`, `runtime_provenance.json`); raw `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1_c10r08_formal_r10/gsm8k_google_gemma-3-4b-pt_91432` (`output.log`).
- **w13r01 / 91441**: retained `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4/w13r01` (`slurm.err.tail`, `slurm.out.tail`, `status.json`, `runtime_provenance.json`); raw `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4_w13r01_formal_r13/gsm8k_google_gemma-3-4b-pt_91441` (`output.log`).
- **w13r02 / 91442**: retained `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4/w13r02` (`slurm.err.tail`, `slurm.out.tail`, `status.json`, `runtime_provenance.json`); raw `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4_w13r02_formal_r13/gsm8k_google_gemma-3-4b-pt_91442` (`output.log`).
- **w13r03 / 91443**: retained `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4/w13r03` (`slurm.err.tail`, `slurm.out.tail`, `status.json`, `runtime_provenance.json`); raw `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4_w13r03_formal_r13/gsm8k_google_gemma-3-4b-pt_91443` (`output.log`).
- **w13r04 / 91444**: retained `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4/w13r04` (`slurm.err.tail`, `slurm.out.tail`, `status.json`, `runtime_provenance.json`); raw `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4_w13r04_formal_r13/gsm8k_google_gemma-3-4b-pt_91444` (`output.log`).

Frozen event payload: `/rmeng_data/robtang/wma-evolve-hook/gangda_wma_evolve/events/20260904T164313Z-792dc7f482/payload.json`. Frozen validator snapshot: `/rmeng_data/robtang/wma-evolve-hook/gangda_wma_evolve/events/20260904T164313Z-792dc7f482/results.snapshot.json`.

Scope: no models, Slurm actions, cancellations, submissions, repository edits or shared-event mutations. No H or new Opus 4.8 result artifacts were inspected. The only written artifact is this temporary verification report.
