# X extraction: instructions for the extractor agent

You are reconstructing, for one scientist session ("cell"), the exact recipe that produced each
archived checkpoint, from the session's trace. The benchmark this feeds is defined in
`doc/experiments/wm_benchmark_spec.md`; the part you produce is **C** (what was launched: code,
arguments, environment, chained back to the base model) and **Δ** (post-processing between a
training output and the archived checkpoint). You never look at evaluation outcomes; they are
not your concern and must not be recorded.

## Inputs (all paths absolute)

- `TARGETS`: `/home/kalorona/awm-data/benchmark/targets/<cell>.json` — the checkpoints to
  reconstruct, with *hints* (submit events, candidate launches, the yaml's output_checkpoint).
  Hints are deterministic guesses; confirm each against the trace.
- Timeline: `/home/kalorona/awm-data/timeline/<cell>/`
  - `events.jsonl` — one JSON per line, in order, field `seq`. Kinds: `init`, `compact`, `text`,
    `tool_use` (with `tool` = Bash/Write/Edit/Read/..., full `command` for Bash, `path`+`sha256`
    for Write, `path`+`old_string`+`new_string` for Edit), `tool_result` (`for_seq` points at
    the tool_use; `text` is the first 1,500 chars).
  - `results/<seq>.txt` — the full text of every tool_result.
  - `fs.jsonl` — replay of Write/Edit: after each op, `path` → `sha256` of the content.
  - `_files/<sha256>` (in `/home/kalorona/awm-data/timeline/_files/`) — the content.
- The raw trace, if the timeline is not enough: `/home/kalorona/awm-data/hf-mirror/cells/<cell>/solve_out_sanitized.txt`.
- Use `grep -n` on events.jsonl and `sed -n` on results to move around; do not load whole files.

## What to produce

For each target checkpoint, one JSON file
`/home/kalorona/awm-data/benchmark/x_raw/<cell>/<checkpoint_id>.json` conforming to
`tools/wm_benchmark/launch_record.schema.json` (read it first). The essentials:

1. **archived_from_dir** and **archive_submit_seq** — the workspace directory the recorder
   archived as this checkpoint, and the moment it did so. The recorder (`awm wm submit`) copies
   `result.output_checkpoint` from the yaml **once**: at the first submit where the yaml has
   `result.execution: completed` and that directory exists. Its result prints `"archived": "<path>"`.
   Every later submit of the same card prints the same path **without copying again**. So the
   archived checkpoint is the directory's content at that first archiving submit; anything the
   scientist did to the directory afterwards (re-soups, config rewrites, copying a different
   model into it) is **not** in the checkpoint and must not be in the chain. `TARGETS` gives
   `archive_submit_seq` and `later_submits_after_archive`; confirm them. Find the yaml version
   current at that submit (Write/Edit in `fs.jsonl`, or a heredoc in a Bash command — many
   scientists wrote the yaml with `cat > memory/cards/exp-NN.yaml <<EOF`) and read its
   `result.output_checkpoint`.
2. **steps** — the chain from the declared base model to that directory, in execution order.
   Each step is one process the scientist ran: a training launch, a data build, a weight
   average, picking a `checkpoint-<step>` subdirectory, editing a saved config. For each step:
   - the exact Bash event (`seq`) and the full command; split out the argv of the process that
     produced the output; record inline env assignments and cwd;
   - which earlier attempts of the same launch were killed/failed and superseded (list them);
   - the **launch-time** content of every file the process reads as code or config: the
     entrypoint, modules it imports from the workspace, config files, and the builders that
     produced its data files. Use `fs.jsonl` to find the version current at the launch seq
     (the last Write/Edit of that path with seq < launch seq). If a file was created by a Bash
     heredoc or `python - <<EOF`, extract that text from the command and save it under
     `/home/kalorona/awm-data/benchmark/x_raw/<cell>/files/<basename>@<seq>`; cite the seq.
     If the content cannot be determined, say so (`unavailable:<reason>`), never guess.
   - the parent model: `base_model` (the cell's declared base), or a workspace dir; if that
     dir was produced by an earlier step in the same cell, name the step and, if it was archived,
     its checkpoint id. A merge/average with several model inputs puts the first in
     `parent_model` and the rest in `additional_parents`.
   - if the trace shows which HF snapshot of the base model was loaded (a `snapshots/<sha>`
     path in an `ls` or download output), record it in `base_model_snapshot`.
   - data files consumed and the build step that produced them; the data files' contents are
     NOT copied, only their builder code and the build command.
3. **environment** — the container is `standard.def` (PostTrainBench); list every
   `pip install` / `uv pip install` / `npm` / `apt` the scientist ran before the last launch,
   and the versions the scientist printed (e.g. `python -c "import torch; ..."`), with seqs.
4. **confidence** and **open_issues** — anything you could not settle, with the seqs you looked at.

## Rules

- Evidence first: every field that names an event carries its `seq`. A reader must be able to
  open the trace at that event and see what you saw.
- Launch-time only. A file edited *after* the launch keeps its pre-edit content in that step.
- Do not record scores, accuracies, or the scientist's conclusions anywhere in the record.
  The card yamls (`memory/cards/exp-NN.yaml`) embed scores: never copy their content into
  `files/`, and when a Bash `command` you quote contains a card heredoc, replace the heredoc
  body with `<<card yaml redacted>>` in the record. Use the yaml only to read
  `result.output_checkpoint` / `setup.output_dir` as the join key.
- Sessions were sometimes resumed and compacted (`init` / `compact` events); file state
  carries across those boundaries.
- The same output directory may have been launched several times (OOM, wrong flags, killed).
  The producing launch is the last one that completed before the closing submit; earlier ones
  go in `superseded_attempts`.
- Some scientists overwrote a directory with a later run; then the archived copy is whatever
  the directory held at the closing submit. Say which launch that was and note the overwrite.
- If a target's checkpoint was produced by averaging or copying other directories, those
  directories' own chains are part of this record (as earlier steps), back to the base model.
- Write the JSON files yourself (Write tool). When done, return a short summary: for each
  checkpoint, `archived_from_dir`, the producing launch seq, the number of steps, confidence,
  and open issues. Nothing else.
