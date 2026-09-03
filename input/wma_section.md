## Experiment log
Every experiment you run must be registered, by command, at two moments. The template is `exp-card.template.yaml` in the current directory; keep your filled cards under `memory/cards/exp-NN.yaml`, numbered in launch order.

**Before you run an experiment** — after you have implemented it, before you launch it — fill sections 1–4 of the card and register it:

```bash
awm wm submit memory/cards/exp-01.yaml
```

The recipe must carry every field another agent would need to rerun it from the base model: the parent checkpoint and its origin (`base_model`, or the `exp-NN` that produced it), the training data with its source, selection rule, example count and build command, the method with framework versions and the **effective** hyperparameters (defaults included — read them off your script), and the exact launch argv with the script it names. The command snapshots your scripts, so the card stays true even after you edit them. It prints `missing` — the fields still needed; fill them and submit again before launching. Unknown is `null` plus a fix, never a guess.

**After the experiment ends** — completed, crashed, or killed — fill sections 5–6 (execution status, the eval command and n, every measurement with its output-file path, and `result.output_checkpoint` for a completed run) and submit the same file again. On a completed run the command archives a copy of the checkpoint under `wm/`, so do not delete or overwrite a checkpoint directory before its result is submitted — archived checkpoints are evaluated after the run, and one lost is an experiment wasted. A changed script, changed data, or changed hyperparameters is a new card, never an edit to one already run.

When you finish, record what you shipped: `awm wm outcome --card exp-NN --final <accuracy> --shipped <checkpoint>`.

`wm/` is the run's durable record. Read it freely; never create, edit, or delete anything there yourself — it is written only by `awm wm` commands. Registration is bookkeeping, not user interaction, and never blocks your work; but an unregistered experiment counts for nothing, however well it scores.
