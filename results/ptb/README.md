# results/ptb

What the operator commits from the cluster, one directory per batch:

```
results/ptb/<batch_id>/
  pilot-<ts>.json, formal-<ts>.json     receipts, copied from the launcher; cancellations appended
  blocked.md                            why a submission was refused, if it was
  <cell>/status.json                    job, Slurm state, validator verdict, judge flags, awm sha, what was not copied
  <cell>/metrics.json, runtime_provenance.json, judgement_*.json, time_taken.txt, cli_version.txt
  <cell>/solve_parsed.txt.gz            the parsed trajectory; solve_out.txt stays on the cluster
  <cell>/slurm.out.tail, slurm.err.tail the last 200 lines of the Slurm logs
  <cell>/task/                          the task directory minus weights, binaries and files over 2 MB
  <cell>.j<job>/                        an earlier attempt of the same cell, kept when a retry lands
results/ptb/ops-log.md                  one line per operator action
```

`awm exp_protocol collect results/ptb/<batch>/*/task --csv` works on this
layout unchanged. Only the operator writes here; the planner reads.
