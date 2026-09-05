# Spillover pre-requeue audit snapshot

At `2026-09-02T21:49:40Z`, the 30 receipt-owned spillover jobs were externally
requeued (`Restarts=1`) with their original frozen
`ReqNodeList=slurm2-a3nodesetondem-[0-1]` restored and a delayed eligible time of
`2026-09-02T21:59:41Z`. They had previously run outside those nodes since about
18:58 UTC.

PTB reuses the same result directory for the same Slurm job ID and opens some
top-level logs with truncation, so the partial spillover attempt could be
overwritten by the restarted strict-site attempt. Before restart, the operator
saved a minimal, weight-free audit snapshot for all 30 jobs:

```text
/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/audit/pre-requeue-20260902T214940Z
```

Contents per job where present: `runtime_provenance.json`, metrics/time/CLI
metadata, gzipped solve/output/error/system-monitor traces, and task memory,
report, and sandbox metadata. Checkpoints and model weights were not copied.

- jobs: `90485–90498`, `90500–90507`, `90647–90654`
- index rows: 30
- files: 122
- bytes: 6,968,785
- `index.tsv` SHA-256:
  `24bb319d2c2a26be3bbf77ec08c3c2df184d50b2261374a1861ebb774dbc8e54`
- `SHA256SUMS` SHA-256:
  `b0b18c0868777f42310fd062e575f89ce4a5a44c27c8ef88447de70934895919`

The restarted attempt remains the same receipt/job identity but is not the same
execution attempt. Analysis must retain `Restarts=1` and must not merge the
pre-requeue partial trace with the restarted strict-site trace as if it were one
continuous scientist session.
