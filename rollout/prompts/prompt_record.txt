We want to train the small LLM \`{model}\` to excel at {benchmark}.
Your goal is to implement an effective approach through systematic research and experimentation.

## Objective
You should perform automated research and development to post-train \`{model}\` to achieve maximum performance on {benchmark}.
You can query the benchmark via the evaluate.py script.

Store your best trained model in the folder \`final_model\`.

## Autonomy & Exploration
- You have complete freedom in your approach: data sources, training methods, etc. .
- You can do multiple iterations on the model and your approach.
- Internet access is unrestricted.

## Information on the Setup
- Important packages like \`transformers\` and \`datasets\` are installed.
- The packages are installed via \`uv pip install --system\`.
{gpu_info}
- Consider the --limit option of the evaluate.py script for faster iteration during development.
{setup_other}{decontamination_tool}
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

## Session completion
A final assistant response ends the current Claude invocation. Background tasks and waiters do not themselves re-invoke you; the launcher may resume this conversation while budget remains. Do not finish, say you are waiting, or rely on a future notification while training, saving, or evaluation is still active. Keep polling with tool calls within their timeout until the work completes. Before claiming completion, verify that `final_model/config.json` and model weight files exist.

## Rules
1. There will be no user interaction. You have to operate autonomously.
2. You have a maximum of {num_hours} hours. You can query how much time you have left by calling \`bash timer.sh\` in the current directory.
3. Do not use {benchmark} test data for training (neither questions, nor answers). Otherwise we will have data contamination.
4. Do not derive training data from specific {benchmark} test items in any way (paraphrasing, perturbation, seeding generation with them, or hand-writing examples to cover them). This is contamination even without verbatim overlap. Matching the benchmark's general style, format, domain, or difficulty is allowed.
5. Do not modify evaluate.py or templates/ .
6. Work only in the current directory and subdirectories. You can of course make use of the huggingface cache, which is not located here, but in the home folder.
7. Your final_model must run in the current environment (same packages). Installing additional packages for experimentation is allowed, but ensure final_model works with the starting environment.
8. You are not allowed to fine-tune any model other than \`{model}\`. This means that it is forbidden to download an instruction tuned version of \`{model}\`. Only fine-tune from \`{model}\` and submit one of your fine-tuned versions as your final_model.
9. You must NOT use any API key found in your environment to call an external LLM API directly, for example to query a large model in order to generate training data. Any API keys present are there to run you (the agent) or the provided evaluation, not for you to call models yourself. Generating training data yourself by other means is of course allowed.
{eval_api_note}
Remember: NEVER ask the user for feedback. Just execute actions which make most sense to you. We will evaluate your results on {benchmark} once you are done.