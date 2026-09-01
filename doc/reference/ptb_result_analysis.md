# PTB completed-result analysis workflow

Queue state and scientific completion are different facts. Slurm `COMPLETED` only says the batch
script exited successfully. A PTB result is `COMPLETE` only when the repository validator accepts
the final model, metrics, trace, monitor, provenance, evaluation log, and required judge files.

## Find completed results

Use any committed experiment manifest; discovery follows `runtime_provenance.json`, not job-ID or
directory-name conventions.

```bash
uv run awm ptb results experiments/posttrainbench/EXPERIMENT.yaml
uv run awm ptb results experiments/posttrainbench/EXPERIMENT.yaml --all
uv run awm ptb results experiments/posttrainbench/EXPERIMENT.yaml --task gsm8k
uv run awm ptb results experiments/posttrainbench/EXPERIMENT.yaml --cell g02 --json
```

The default prints only validated completed cells. `--all` additionally prints the latest
incomplete attempt and its missing evidence. If a cell has several attempts, the report keeps the
latest validated completion for analysis while separately retaining the latest attempt.

## Resolve one job or cell

```bash
gangda-slurm-queue show JOB_ID
```

This joins live/history Slurm identity to the receipt, cell configuration, manifest, spec, frozen
commits, result directory, accuracy, completion verdict, and judge flags. The receipt is the
job-to-cell authority; the manifest defines the configuration; the spec explains why the cell
exists.

## Inspect result evidence

Inside the reported result directory, the main files are:

- `metrics.json`: official accuracy and standard error;
- `runtime_provenance.json`: batch/cell, model revision, agent route, commits, containers, Slurm;
- `solve_out.txt` and `solve_parsed.txt`: raw and parsed agent trace;
- `system_monitor.log`: runtime resource timeline;
- `judgement_gpt5_4*.json`: contamination and model-identity verdicts;
- `judgement_api*.json`: disallowed external API verdict;
- `judgement_ptb_lookup*.json`: benchmark-lookup verdict;
- `judgement_general*.json`: broad anomaly requiring human interpretation;
- `final_model/`, `final_eval_*.txt`, and `metrics.json`: submitted artifact and official scoring.

Prefer a nonempty `_rerun` judgement over the original canonical filename when both exist. A
`general_anomaly` is a review flag, not automatically a bad score; contamination, disallowed
model/API, and PTB lookup flags are stronger validity failures.

## Analysis order

1. Establish coverage: complete cells / intended cells and which cells remain incomplete.
2. Separate clean completed results from judge-flagged completed results.
3. Compare scores only within the same benchmark and official evaluation contract.
4. For AIME, report both `correct/30` and accuracy; a one-question change is 3.33 points.
5. Use matched model/profile/context contrasts before interpreting isolated score maxima.
6. Preserve receipt, manifest, spec, and result paths in every derived report so conclusions can
   be replayed without relying on a transient queue display.

This workflow is intentionally manifest-driven: it works for Batch 1, Batch 2, retries, and future
experiments without embedding their job IDs in analysis code.
