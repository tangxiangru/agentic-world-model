# Historical cancelled-job evidence backfill — 2026-09-03

State-normalization fix `68253ee` exposed30 historical terminal jobs previously skipped by reconcile because Slurm returned `CANCELLED by 0`. The operator validated the full plan as **harvest-only**, checked every job/cell against its tracked receipt and then used the existing harvest implementation. **No job was newly cancelled, submitted or released by this backfill.**

| batch | jobs | classification |
|---|---|---|
| baseline-b | 90500–90507 | 8 incomplete startup/spillover bundles with placement quarantine |
| nullctl-c | 90508–90515 | 8 terminal records; no result directory |
| Round02 A v1 | 90845–90848 | 4 terminal records; no result directory |
| Round02 B v1 | 90849–90852 | 4 terminal records; no result directory |
| Round02 C v1 | 90853–90856 | 4 terminal records; no result directory |
| old drift pair | 90857–90858 | 2 terminal records; no result directory |

The [machine-readable audit](../../results/ptb/cancelled-backfill-20260903.json) retains exact batch/cell/job, receipt, status and raw-result paths. All30 have `complete:false`, `eligible:false`, null accuracy. They add **zero validator-clean cells**, do not change any score pool and do not trigger a new clean trace window. Full reconcile reports no remaining actions after harvest.

For baseline-b, the retained evidence is initial `runtime_provenance.json` and Slurm tails, not a completed scientist trace: all eight lack parsed trace, model, metrics, evaluation, judge files and finalized provenance. Their recorded node is `slurm2-a3nodeset1-2`, outside frozen ondem0–1. Each stderr tail records cancellation **due to job requeue at2026-09-02 21:49:33**, preceding the final held-block withdrawal. Preserve these as incomplete placement/startup evidence, **not placement-only validator-complete sensitivity results**. There is no scientist trace to invent or send for recipe analysis.

An operator line ending in `clean` only means no flags were listed; it cannot override `complete:false`, missing judges or quarantine. Existing baseline-strict repeats and the active hourly monitor continue; no automatic retry, new capacity claim or new scientific result follows from this historical backfill.
