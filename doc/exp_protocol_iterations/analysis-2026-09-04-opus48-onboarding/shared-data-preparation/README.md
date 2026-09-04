# Shared HumanEval preparation — 2026-09-04

No experiment submitted, admitted or scored in this step. The approved study
still has16 GSM8K held receipts and24 not-yet-admitted GPQA/HumanEval cells.

## Public data and actual image checks

The already verified public parquet was copied with no-clobber semantics to:

`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/hf/hub/datasets--openai--openai_humaneval/snapshots/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval/test-00000-of-00001.parquet`

83,920 bytes, mode600, SHA256
`2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6`.
The local PTB configuration's HF_HOME alias resolves to this shared cache.
Both actual opus_5/vllm_debug images read this exact path offline and confirmed
164 rows and full selection SHA
`7aeeca4daf4f06680efd86ea58b1d2233a100c2912e2618285fde80072f5494d`.
The probe maps data only to check content bindings; no question/gold is printed,
no dataset code or model runs, and no network download occurs. This does not
establish compute-node permissions or GPU-enabled execution acceptance.

The local operator PTB `.env` received only
`POST_TRAIN_BENCH_PYTHON_BWRAP=/home/robtang_google_com/.local/bin/bwrap`, the
already tested public binary. No scheduler/identity settings changed. Its SHA is
`d78807229d616606e339c5988392b9e0ab4a6a6998fa51e4590837f426a12fca`.
This ignored local config is not shipped in git, and the separate shared
submitting clone has not yet received the key; configure it explicitly when
HumanEval is admitted. Existing GSM8K frozen jobs do not use this key.

## Match the actual concurrency, not a stale comment

The synthetic executor tests explicitly used max_samples=1; formal evaluate.py
sets only max_connections=1. Main checked the exact native implementation and
then executed its semaphore constructor in both real images: omitted
max_samples resolves to1, not the max_connections+1 described in a stale
source comment. Backend default_concurrency=1 additionally limits sandbox
lifecycle concurrency. No runtime or experimental setting was changed to make
the checks match. Native run.py SHA in both images:
`0627f2455a4f16d566f173f48aee1cdbf0481acbcff680f92f415e7982b01dab`.
An additional reviewer slot was unavailable; this was a main-agent source and
native execution check, not a new independent/blind review.

## Twelve explicit unsubmitted drafts

`experiments/posttrainbench/exp-protocol-opus48-humaneval-{none,knowledge,tools}-x4-v1.draft.yaml`
contains four independent sessions for each approved arm, cells n03h01–04,
k03h01–04 and e03h01–04. Structural comparison against the corresponding frozen
GSM8K manifests confirms identical common budgets/model/effort/context/images,
method sources and context records, changing task/cell identity only (plus
batch/description). Common PTB09c90b6 is an explicit prerequisite, not the
operator's currently adopted PTB pin. All three still fail current task admission
with the expected unsupported-HumanEval message. They have **no queue entries,
receipts or held-floor contribution**. Do not remove that gate for a green check.

## Existing node identities exhausted without changing authority

The node and project enable OS Login. Project SSH metadata therefore does not
establish a usable login. Current cloud credentials map to an existing service
account POSIX identity; the existing google_compute_engine public key is already
in its OS Login profile, and derivation confirms the private/public files match.
Strict SSH server-key verification succeeds, but the matching account still
receives Permission denied(publickey). No cause beyond that rejection is claimed.
No explicit MFA/certificate override was present in the inspected metadata.
The only other configured cloud identity could not refresh its existing metadata
credential (404); no login or new credential was obtained. Global account
selection, IAM, OS Login keys, node configuration and Slurm reservations were
not changed. Authorized node access remains required; GPQA access is unchanged.

## Preserve the pending detector handoff

One-shot helper `/tmp/exp-protocol-admin-handoff.sTGNfhBs/handoff.py`, PID3676924,
was verified live waiting on a pidfd for the actual old detector3564003.
It never kills the old process or queries Slurm early. Only after a ready event
containing exactly the17 already-harvested CANCELLED jobs will it archive the old
state and start the unchanged provided detector on the21 remaining jobs, hourly.
Any extra terminal, changed owner/state or unverified harvest stops the handoff
without consuming the event. Eight tests passed, including a real pidfd/process
handoff using synthetic detectors that never query Slurm or real monitor state.
Completion is
not claimed until completed.json and the new live monitor/first tick are observed.
