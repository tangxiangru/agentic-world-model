# WMA online Round 03 — L3 and uptake extension

Status: launched; 8/8 cells safely PENDING; no candidate outcome read.

## Basis

Round 01 root-cause analysis is complete and independent of Round 02 outcomes:
L3 was `yes` on 22/22 delivered v0.2 verdicts, so no GPU hour could be saved;
only 4/22 verdicts produced an observable action, and C5 advice was acted on in
2/8 cells, no better than control. The lock-gated protocol now surfaces the
first precondition before launch, creating a channel for both edits to act.

These two candidates are queued because safe PENDING fell to 17. They are not
selected from Round 02 evidence and do not change its denominator.

## Candidates

| candidate | commit | skill hash | single edit | primary metric | falsification |
|---|---|---|---|---|---|
| E `l3-evidence` | `5c266f2327c116a4e63aca48395141276ad9345d` | `20c8837ab7a4` | C3/C4 L3 `yes` requires resolvable effect and no cheaper C2/C5 discriminator; otherwise defer | nontrivial L3 distribution, gpu_h_saved vs wrongly_killed, PTB guard | all-yes unchanged, or wrongly-killed hours / PTB loss exceed baseline spread |
| F `decisive-precondition` | `a7e4d0fb547044f2641fe7da952ec7565b676e49` | `959fbdd5bbb2` | first precondition is one highest-value pass/fail action tied to a changed level | first-precondition uptake and downstream action rate vs w10 | uptake does not beat v0.2 spread or cost exceeds 1.5× |

Both use public lock-gated `awm.sha=ae46724`, four cells each, the frozen
Gemma/GSM8K/scientist/judge/history/resource contract, and compare against the
same-wave w10 v0.2 baseline. Compliance (`lock.wma.state`, verdict-before-launch
≥0.8) is checked before effects. Leaks must be zero; PTB is directional and a
guard at n=4. At most one edit can be promoted, and only after the earlier
Round 02 candidates are read on their own preregistered metrics.

## Launch (2026-09-03 08:52–08:54 UTC)

- E jobs: `91173..91176`.
- F jobs: `91177..91180`.

Both manifests passed the held-route gate. Post-release audit found all eight
jobs PENDING with `ReqNodeList=slurm2-a3nodesetondem-[2-3]`. Owned allocation
remained 16/16 and safe PENDING rose from 17 to 25. No cancellation or other
queue action was part of this launch.

## Planner audit (2026-09-03 09:0x UTC)

Chain and hashes verified from `git archive`: `5c266f2` → `20c8837ab7a4`,
`a7e4d0f` → `959fbdd5bbb2`, both v0.2 (`176f0a464986`) plus one paragraph; F
removes E's paragraph before adding its own. Manifests pin public
`awm.sha=ae46724` (the running wave's), so w10 stays the shared baseline.

One readout added for E, from the Round 01 levers: the first SFT of a run
(C3 from the format floor) is the score-dominant card (first-SFT dev ρ +0.60,
training hours +0.48), and v0.2 over-predicted C2 decode-config effects near
the floor three times. If E's "no unspent C2/C5 discriminator" clause defers
first-SFT cards toward decode probing, it could cost the lever that matters.
Read for E, per cell, against w10: time to the first SFT launch, its dev
accuracy, and the number of C2 cards locked before it. E is falsified on this
channel if its first SFT lands later than w10's spread and no cell shows a
GPU hour saved.

Round 03 shares the Round 02 in-flight finding that a relock costs a second
full wait (~6 min); the cost line (1.5 h, ×1.5) applies to E and F unchanged.

## Completion-trigger amendment (2026-09-03 09:5x UTC)

The analysis dependency on a future PR #23 comment is removed. A singleton
hourly `tools/wma-evolve-hook` now reads registry-backed manifests and PTB
validator output, acknowledges the 32 already-analyzed clean cells as its
starting watermark, and freezes a new event after eight new clean-complete
cells (or a four-cell tail after six hours). Each event invokes read-only
Claude Code with explicit `claude-opus-5`, effort `max`, and the `ultracode`
workflow for ledger/trajectory/uptake/harm/compliance analysis and next-wave
design. The report is advisory; Codex verifies it and retains sole authority
for skill edits, experiments, queue actions, commits and pushes.

The same hourly pass checks WMA queue ownership and exact pending routes. It
raises a replenishment event below 24 safe PENDING jobs so the one-hour cadence
does not approach the hard `>8` floor. It never mutates Slurm. Shared state and
event artifacts are under
`/rmeng_data/robtang/wma-evolve-hook/gangda_wma_evolve/`; the operating
contract is `doc/reference/wma_evolve_hook.md`.
