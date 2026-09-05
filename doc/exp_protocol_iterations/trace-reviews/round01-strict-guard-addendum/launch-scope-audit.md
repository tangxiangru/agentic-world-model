# Launch-order denominator audit — 2026-09-03

Planner read-only follow-up while full strict-cohort synthesis runs. This does not change any frozen scientist tree, manifest, receipt, PTB completion verdict or score.

## What the counter covers

At operator commit `cd0fe00`, `tools/exp_protocol_cell_read.py:153–191` considers only cards in `TRAIN_FAMILIES` with a recorded lock. It matches a Python script basename and the card's output-directory basename, excludes commands containing `--dry-run`/`--help`, takes the **first** matching launch and omits cards whose launch was not found from the printed ratio. Consequently `lock_before_launch=3/3` means three matched card-launch checks passed. It is not an exhaustive count of all training, evaluation, retries or probes.

## Two raw-trace counterexamples to the broad claim

Both cells belong to strict manifest `experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2.yaml`, immutable receipt `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/formal-2026-09-02T204221.237369+0000.json`, and Round01 session-guard spec. Bundle root below is `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/`; line numbers refer to each cell's uncompressed `solve_parsed.txt.gz`.

| cell / job | GPU-training evidence before the training card | later first lock |
|---|---|---|
| g01s01 / 90791 | exp-01 was closed at 10:01:18 (L3378–3380); at **10:01:24**, `scripts/train_sft.py --out .../ckpts/smoke --limit 6000 --max-steps 25` launches (L3389–3394); CUDA OOM and step1/25 appear at L3448–3450. exp-02 is only created at 10:19:29 (L4458–4462). | **10:21:01**, exp-02 lock history and L4798; later relock 10:35:47 is a separate event |
| g01s07 / 90797 | at **10:04:01**, `train_sft.py --out ckpts/smoke --limit 4000 --max-steps 12 --grad-ckpt 0` launches (L3617–3622); Trainer reaches loss computation and CUDA OOM at 10:05:25 (L3650–3682). This is actual GPU execution, not a CPU dry run. | **10:20:05**, persisted exp-02 lock |

The later exp-02 cards retain smoke descriptions, but retrospective `situation.smoke_runs` entries do not prove a pre-launch lock. The already-locked exp-01 cards are baseline evaluations, not declarations of these training commands. Both reports correctly distinguish smoke timestamps from the first main training launch, yet their aggregate 3/3 labels can be overread.

## Instruction-scope conflict and next decision

The frozen `4ae3d87` card template calls `smoke_runs` “trial runs that are not experiments”; rule 8 asks scientists to record prior smoke failures. Those are explanatory context for the behavior, not an exemption from the repository/user requirement that training and evaluation follow card creation, checks and lock. The template's wording and deployed launch guidance need an explicit scope resolution before claiming or relying on universal launch-order compliance.

Separate three questions:

1. **PTB scientific completion:** the existing validator-clean status/official scores are unchanged; its validator is not an all-command ordering audit.
2. **Round01 guard-specific safety:** absence of work lost at session end, false blocks, locked-open cards and its score guardrail remain the predeclared questions. This audit does not silently replace them with a different post-hoc outcome.
3. **Overall runtime compliance / next launch:** card-matched training counts cannot prove the user-required all-training/evaluation invariant. Read the complete launch inventory, including GPU smoke/probes and evaluation-only commands. Resolve the conflicting runtime guidance through a separately specified, baseline-relative change; do not quietly exempt small GPU jobs or rewrite held manifests in place.

This is ledger direction #26, not a built candidate or permission to release. Ownership/native-isolation gates also remain closed.
