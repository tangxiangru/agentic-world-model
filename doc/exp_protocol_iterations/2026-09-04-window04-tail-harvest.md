# Window04 tail harvest — 2026-09-04

The score-status request exposed three new completions before the hourly tail detector's next tick. All three were harvested with `awm ptb reconcile --apply` at05:01:57–05:02:00 UTC. The dry run contained only these three harvests: no submission, release or cancellation occurred. The live hourly detector PID2446155 was preserved, not restarted or reset after this early harvest. It then completed normally3/3 at05:06:32 (`window04-local/tail-trigger.json`). A new hourly detector PID2579442 now watches21 relevant held IDs, with threshold6 and2 buffered clean cells; exact settings and verified liveness are in the current operator view.

| cell / job | official accuracy | PTB validator | primary eligibility |
|---|---:|---|---|
| c01s05 /90817 |0.7255496588324488|complete, no judge flags|eligible|
| c01s08 /90820 |0.7717968157695224|complete, no judge flags|quarantined: runtime node `slurm2-a3nodeset-1` outside frozen ondem0–1|
| p00s03 /90825 |0.7103866565579985|complete, no judge flags|eligible|

The exact receipt→manifest→spec→raw result mappings remain in the harvested status/receipt bundles:

- [Strict control receipt](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1/formal-2026-09-02T210446.182614+0000.json), [manifest](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8.yaml), [spec](../spec/2026-09-02-exp-protocol-round00-null-control.md), [c01s05 status](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1/c01s05/status.json), [c01s08 status](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1/c01s08/status.json).
- [Strict baseline receipt](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/formal-2026-09-02T210823.064737+0000.json), [manifest](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8.yaml), [spec](../spec/2026-09-02-exp-protocol-round00-gsm8k-baseline.md), [p00s03 status](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s03/status.json). Each status records the original absolute result directory and retained/skipped files.

Fresh canonical `awm ptb results MANIFEST --json` gives strict control primary n7 mean0.7477526264486082, placement-sensitivity n8 mean0.7507581501137225; strict baseline n3 mean0.7525903462218851. Seven formal cohorts now total57 validator-complete results:56 eligible (55 clean,1 flagged) and1 placement-quarantined. Two older failed/incomplete attempts remain outside completion counts.

These are **two NEW clean cells after the frozen Window04 fourteen**, not a new eight-cell window. Preserve c01s08 as sensitivity evidence and do not move it into primary merely because its Slurm job completed. Strict control still lacks its eighth primary repeat; any required replacement needs a new immutable manifest/receipt, not reuse of90820. Tail trace review remains due; it must not silently change Window04's frozen means or reviewer assignment.

At05:04:23 UTC the current checkout's registry-aware queue check reports OWNERSHIP OK,0/16 GPUs allocated, zero running,29 actual PENDING(JobHeldUser), and no unknown/name/placement/capacity violations.90820 ended naturally; no running job was cancelled. The reservation still covers11 nodes, so **native isolation is not restored and no release is authorized**. Existing09:39 authorization is only for completed strict-guard jobs90791–90798.

E2's full manifest check returns0 issues on this snapshot. This does not waive its documented registration/release boundary or the native-isolation gate. Before any registration, reassess the source/spec gate separately from release; no unregistered manifest counts as a held cell. Old E remains unstarted pending its replacement decision. D's unconditional parent-config predicate is now under focused CPU scope audit; it must not be released merely because it was previously first-wave priority. The other21 held cells remain available behind the operational gate even if D4/oldE4 are excluded from the scientifically usable-buffer count.
