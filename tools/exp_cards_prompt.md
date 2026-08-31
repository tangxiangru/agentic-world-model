You are reconstructing experiment cards from one finished PostTrainBench run. Read these two files first and follow them exactly:

- results/exp-cards/gsm8k-gemma-holdout-v1/card.template.yaml  (the schema — a frozen copy; do not read doc/exp-card-specs/experiment-card.template.yaml)
- doc/exp-card-specs/extraction-protocol.md          (the rules, the section-by-section fill table, and the `provenance` block you must add)

The worked example doc/exp-card-specs/example-card.yaml shows a fully filled card; yours will have many nulls, and that is correct.

## Your run

- run_ref: {run_ref}
- base model being post-trained: {trained_model}
- benchmark: gsm8k, 10 h budget, one H100
- digest (the event stream, filtered; blocks headed `--- [i] turn=T t=+H.HHh act ---`, where `t=` is hours since the run started and is present only when the harness recorded timestamps — copy it into `elapsed_h`, else leave `elapsed_h: null`): {digest}
- workspace snapshot (the agent's scripts and its own eval outputs; symlinks): {task_dir}
  files: {task_files}

Read the whole digest. Then open the workspace files a card cites (training scripts for argparse defaults, `eval_*.json` / `baseline_results.json` for measured values, `prepare_data*.py` for data construction). Do not read anything outside these paths.

## Output

Write one YAML file per launch to {out_dir}/exp-NN.yaml (NN zero-padded, launch order), then {out_dir}/index.md with one line per card:

`| exp-NN | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |`

## Hard rules

1. A card exists only for a launch you can point at: `provenance.launch_i` is the event index and `setup.command.argv` is the command as it appears there. Training runs, merges, packaging (copy to final_model), and decode/config changes that produced a candidate all count. Scripts written but never run do not. **Smoke tests and dry runs** (a deliberately truncated run — a handful of steps, a tiny subset — whose purpose is to check that the pipeline works, not to produce a candidate) are NOT cards: list them on the next real launch's card as `provenance.smoke_runs: [{launch_i, outcome}]` (outcome one line, e.g. "crashed: SFTConfig has no evaluation_strategy" or "passed"). A run the agent intended as real and then killed or that crashed IS a card.
2. If the stream does not state it, leave it empty (`null`, `[]`). Never fill a hyper-parameter from an argparse default unless the script text is in the workspace AND the launch did not override it — and then say so in `hyperparams.other`.
3. `problem` and `hypothesis` are the agent's words from BEFORE the launch. If the agent launched without saying why, set `problem.statement: null`, `hypothesis.claim: null`, and `provenance.stated_by_agent.{problem,hypothesis}: false`. Do not invent a hypothesis. Do not fabricate `failure_examples`; only include items the agent actually printed.
4. Every number in `result.measurements` is the agent's own eval, with the workspace file that holds it when one exists. Never write `outcome.official_accuracy`; leave the key with `null`.
5. Never write the agent's model name, the harness/CLI name, or any experiment/run identifier anywhere. If the agent refers to itself by name in the stream, do not carry it over.
6. `conclusion.decision`: `adopt` if the output became final_model / the submission or the parent of a later card; `reject` if evaluated and dropped; `abandon_line` if killed or never used. `verdict` is `supported`/`contradicted` only when the agent measured this output against the comparator under the same `--limit`; otherwise `inconclusive`.
7. Chain cards: `setup.parent_checkpoint.origin` is the earlier card whose `result.output_checkpoint` the launch loaded, else `base_model`.
8. Record in `provenance.unresolved` anything the stream leaves open (e.g. which checkpoint was actually submitted).

When done, reply with exactly: the number of cards written, which one (if any) is the adopted/submitted card, and any run-level problem (e.g. digest truncated, no launches found). Nothing else.
