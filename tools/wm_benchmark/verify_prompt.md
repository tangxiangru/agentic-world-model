# X verification: instructions for the verifier agent

An extractor agent has reconstructed, from a scientist session's trace, the recipe that produced
each archived checkpoint of one cell. Your job is to **try to break** those records against the
same trace. You are independent: do not trust the record's citations until you have opened the
cited events yourself.

Inputs are the same as the extractor's (`tools/wm_benchmark/extract_prompt.md` describes the
timeline layout under `/home/kalorona/awm-data/timeline/<cell>/`), plus the records under
`/home/kalorona/awm-data/benchmark/x_raw/<cell>/<checkpoint_id>.json` (schema:
`tools/wm_benchmark/launch_record.schema.json`).

For every record, check and report on each of these, citing seqs:

1. **Archive mapping.** Is `archived_from_dir` really the directory the recorder archived for
   this card, and is `archive_submit_seq` the **first** submit whose result printed `archived`?
   The recorder copies once and never again (later submits return the same path without
   copying), so the chain must end at the directory's state at that seq; steps that changed the
   directory after it (re-soups, config rewrites, copies into it) are errors if included as
   producing steps. Open the yaml version current at that submit and the submit result.
2. **Producing launch.** Is the cited launch the *last completed* launch that wrote that
   directory before the closing submit? Look for later launches into the same directory, kills,
   OOM restarts, `rm -rf` of the directory, or a copy/soup that replaced it.
3. **Launch-time files.** For every file in `files`: does the cited sha/seq give the content
   current at the launch seq? Look for Write/Edit events on that path between the cited version
   and the launch, and for Bash edits (`sed -i`, `cat >`, `python - <<EOF` writing the file).
   Is any file the entrypoint imports or reads (workspace modules, config yaml/json, prompt
   templates) missing from the list?
4. **Chain.** Does the parent model of each step match what the launch argv/script actually
   loaded (`--model`, `--resume`, a path inside the script)? Does the chain reach the declared
   base model without gaps? If a parent directory came from an earlier launch, is that launch
   in the record as a step?
5. **Δ.** Are averaging, `checkpoint-<step>` selection, and post-training config edits (e.g.
   rewriting `generation_config.json` or the tokenizer files in the output dir) captured as
   steps? Was anything applied to the directory *after* the closing submit (which would not be
   in the archive) wrongly included?
6. **Environment.** Any `pip install`/`uv pip install` before the last launch that the record
   omits, or one listed that actually failed?
7. **Leakage.** Does the record contain any score, accuracy, or the scientist's conclusion?
   (It must not.) Card-yaml heredocs quoted inside a `command` must be redacted.

Never delete anything: do not `rm` or clear any directory, and do not modify the extractor's
records; you only add verdict files. Write your verdicts to `/home/kalorona/awm-data/benchmark/x_verify/<cell>/<checkpoint_id>.json`:

```json
{"checkpoint_id": "...", "verdict": "confirmed" | "needs_fix" | "cannot_verify",
 "checks": {"archive_mapping": "ok|wrong|unclear", "producing_launch": "...", "files": "...",
            "chain": "...", "delta": "...", "environment": "...", "leakage": "ok|violation"},
 "findings": [{"check": "files", "severity": "high|medium|low", "detail": "...", "seqs": [..],
               "proposed_fix": "..."}],
 "seqs_examined": [...]}
```

Return only a per-checkpoint summary line: checkpoint id, verdict, number of high/medium findings.
