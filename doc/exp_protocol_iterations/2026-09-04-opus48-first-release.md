# Opus4.8 GSM8K first release — authorization and scope

At22:02 UTC the user explicitly authorized proceeding after the operator had
reported three gates: dedicated ondem0–1 isolation, existing-account node access
and official GPQA access. Fresh read-only inspection found all21 watched jobs
still PENDING(JobHeldUser), OWNERSHIP OK,0/16 allocated and monitor3684437 live.
All four new Opus4.8 manifests independently pass the full PTB check with0 issues.

Creating an exact new reservation would not make the existing jobs native to it:
their immutable receipts freeze `robtang-ptb-a3`. Retargeting would require a
replacement submit/cancel cycle for all16 jobs and could disturb shared
reservation infrastructure. This release therefore uses the repository's
smaller, per-receipt, user-authorized shared-reservation exception. It changes
neither reservation nor receipt identity. Release still requires OWNERSHIP OK,
all jobs PENDING(JobHeldUser), and each live ReqNodeList exactly matching frozen
`slurm2-a3nodesetondem-[0-1]`. The authorization, actual11-node reservation set
and frozen two-node set are written into each released receipt.

Release exactly these two whole receipts:

- protocol-free `n03g01–04`, jobs92125–92128;
- complete E `e03g01–04`, jobs92133–92136.

This provides a complete matched baseline/E block rather than outcome-selecting
individual cells. Keep process-knowledge jobs92129–92132 and guard jobs92137–92140
held: eight independently specified, validated useful cells remain. The five
legacy mixed-receipt holds90826–90830 are untouched and are not needed for the
new-wave floor. No running job is cancelled, no WMA/AWM-full node is borrowed,
and no GPQA/HumanEval result or admission is implied.

After the queue commit reaches the clean operator clone, run dry reconciliation,
then apply only the two planned releases. Re-read both tracked/source receipts,
the registry, Slurm state and exact placement. If either release fails, leave
that block held and investigate rather than issuing direct `scontrol release`.
Once jobs allocate, the existing hourly terminal detector remains the completion
wake-up; separately sample GPU process utilization without cancelling a job on
a transient zero. Release the second eight-cell wave only after capacity opens,
its block remains scientifically required, and the useful held floor is first
replenished with admitted GPQA/HumanEval work or another authorized valid block.

GPQA still returns403 and HumanEval still lacks compute-node acceptance. The
user's authorization permits addressing those gates but is not evidence they
already passed; it does not permit accepting data terms, fabricating task assets,
uploading credentials, or weakening isolation.
