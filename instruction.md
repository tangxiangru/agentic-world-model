# GSM8K × Gemma-3-4B agent instruction

## Parameters

- `dir`: `{dir}` — the absolute path of this session's private agent directory.
- `submission`: `{submission}` — the absolute path of the submission directory inside `{dir}`. It
  must resolve to the selected model checkpoint, also inside `{dir}`, and be directly loadable by
  the official evaluator. It may be the checkpoint directory itself or a directory link to it.
- `model`: `{model}`.
- `benchmark`: `{benchmark}`.
- `time_limit`: `{time_limit}`.
- `gpu`: `{gpu}`.

Post-train `{model}` to achieve the highest possible accuracy on `{benchmark}` through systematic
research and experimentation. Your final submission is the directory `{submission}`.

You have `{time_limit}` and `{gpu}`. Check the remaining time with `bash timer.sh`. Reserve enough
time at the end to select, save, and verify `{submission}`; when fewer than 60 minutes remain, stop
launching exploratory experiments and finalize the best candidate.

## Environment

- This instruction describes one isolated agent session. Your assigned agent directory is
  `{dir}`. Treat it as the root of your session and keep all writable session state inside it.
- A node may run up to eight independent sessions on its eight GPUs. Your session has one assigned
  H100, exposed through the sandbox. Do not access another session's directory, processes, or GPU,
  and do not use shared paths to coordinate with other sessions.
- `{dir}` contains this instruction, `evaluate.py`, `templates/`, and the benchmark utilities. Put
  all code, derived data, logs, checkpoints, experiment outputs, `memory`, and the submission under
  `{dir}`.
- Important post-training packages, including `transformers` and `datasets`, are installed. You may
  install packages for experimentation, but `{submission}` must run in the original environment.
- Internet access is unrestricted.
- `evaluate.py` runs the `{benchmark}` evaluation and supports `--limit` for faster checks. Read it
  and the official template before training so that your saved model is compatible with the
  evaluator. Do not modify `evaluate.py` or `templates/`.
- `{model}` is a multimodal pretrained checkpoint, while this task is text-only. Decide
  how to handle that distinction and verify early that a saved checkpoint loads through the
  official evaluation path.
- A copy of the `{benchmark}` test set and a contamination checker are available. The test copy is
  input to the checker only; never use its questions or answers as research or training material.

## Research

Operate autonomously. Establish a working end-to-end baseline, then improve it through repeated
hypothesis, experiment, observation, and revision. You decide:

- what data to use or construct;
- how to clean, format, and combine it;
- what training methods and hyperparameters to try;
- how to evaluate intermediate models;
- what evidence is sufficient to continue, stop, or revise an experiment;
- how to allocate the remaining time between exploration and improving the incumbent.

Base decisions on observed evidence. Preserve a valid incumbent while exploring alternatives, and
avoid repeating work whose outcome is already known.

## Measurements

Define and maintain your own evaluation and measurement strategy. You may construct development,
probe, or diagnostic sets from the permitted `{benchmark}` training data. Keep comparisons
consistent enough to distinguish real improvements from evaluation noise, and record the evidence
needed to interpret experiment outcomes.

The official PostTrainBench evaluation determines the final score. Do not train on, derive data
from, or otherwise exploit individual `{benchmark}` test questions or answers.

## Memory

Create and maintain an external, persistent, and queryable object called `memory` inside `{dir}`. It
is external to your model context but private to this session. Use it to retain the past
observations, current state, and other information you need to conduct the research effectively
across a long run and possible context loss.

You decide the representation, contents, organization, and update policy for `memory`. Keep it
current and consult it when making research decisions. It should contain enough information for a
fresh instance of you to resume the work effectively.

## Finalization

Before the run ends:

- choose the strongest valid candidate supported by your evidence;
- make `{submission}` a directory path that resolves to that checkpoint;
- verify that `{submission}` loads and runs through the original evaluation environment;
- leave the workspace and `memory` in a state that explains the final choice.

## Rules

1. There is no user interaction. Operate autonomously and never ask for feedback.
2. You have a maximum of `{time_limit}`. Use `bash timer.sh` to check the remaining time.
3. Do not use `{benchmark}` test questions or answers for training.
4. Do not derive training data from specific `{benchmark}` test items by paraphrasing, perturbing,
   seeding generation with them, or manually constructing examples around them.
5. Do not modify `evaluate.py` or `templates/`.
6. Write only inside `{dir}`. You may read permitted shared model and dataset caches and use
   the supplied contamination checker, but do not place session artifacts in shared locations or
   access another session's sandbox.
7. `{submission}` must be a directory resolving to a complete model checkpoint that runs in the
   original environment.
8. Fine-tune only `{model}`. Do not substitute an instruction-tuned or otherwise
   disallowed model.
9. Do not use environment API keys to call an external language model for training-data generation
   or any purpose other than the benchmark functionality for which a key was provided.
