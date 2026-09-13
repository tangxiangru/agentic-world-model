# World-model benchmark v2 — reproduction runbook

Everything below is reproducible from the pinned HF revision; the construction log
(`doc/experiments/wm_benchmark_construction.md`) records decisions D1–D9 and the counts each step
produced on 2026-09-12/13. `DATA` is the working data root (`data/` in the repo is a symlink to it;
on the CPU box `/home/kalorona/awm-data`). Token: `MY_HF_TOKEN` in the environment (read access to
the private dataset `JerrrrryL/awm-gsm8k-trajectories`); never write it into a file.

| step | command (from the repo root) | output | expected (2026-09-13) |
|---|---|---|---|
| 0 mirror | `cd $DATA && python tools/wm_benchmark/mirror_download.py` then `python tools/wm_benchmark/mirror_fetch.py hf_tree_07132f15.txt hf-mirror 8` (second pass: size-checked fetch of every file in the revision's tree; rerun until `DONE … failed=0`) | `$DATA/hf-mirror/` | 25,177 files, 141 GB |
| 1 timeline | `python tools/wm_benchmark/trace_timeline.py $DATA/hf-mirror/cells $DATA/timeline` | `$DATA/timeline/<cell>/{events,fs}.jsonl, results/, summary.json`, `_files/<sha>` | 160 cells, 71,456 events, 2,983 file versions |
| 2 labels | `python tools/wm_benchmark/build_labels.py $DATA/hf-mirror $DATA/benchmark` | `labels.jsonl`, `labels_summary.json` | 4,182 rows: 3,000 matrix valid, 1,039 native valid, 143 native invalid |
| 3a targets | `python tools/wm_benchmark/prepare_targets.py $DATA/benchmark $DATA/timeline` | `targets/<cell>.json` | 124 cells, 582 checkpoints |
| 3 X (agents) | Claude Code `Workflow` with `workflow_extract_verify.js`, args `{already_extracted: [...], remaining: [...]}` (cells); agents read `extract_prompt.md` / `verify_prompt.md` / `launch_record.schema.json` | `x_raw/<cell>/<ckpt>.json` (+ `files/`), `x_verify/<cell>/<ckpt>.json` | 582 records, 582 confirmed |
| 3b check | `python tools/wm_benchmark/check_records.py $DATA/benchmark $DATA/timeline [--json out]` | findings per record | no defect class; see log §5 for what the residual lines mean |
| 5 Z | `python tools/wm_benchmark/normalize_z.py $DATA/hf-mirror $DATA/benchmark --workers 24` | `z/<example>/{samples.jsonl.gz,log_ref.json}` | 4,182; 4,172 `log_is_record_of_label`, 10 not (D6) |
| 4 assemble | `python tools/wm_benchmark/assemble.py $DATA/benchmark $DATA/timeline` | `examples.jsonl`, `splits.json`, `x/<ckpt>/`, `assemble_summary.json` | 3,544 eligible (2,972 matrix + 572 native), 28 aliases, 0 records with problems |
| 6 verify | `python tools/wm_benchmark/verify.py $DATA/hf-mirror $DATA/benchmark` | `verify_report.json`, exit code | exit 0, no hard failures, 3,544 eligible clean |

Step 3 is the only non-deterministic step: it is performed by extractor/verifier agents whose
records cite every trace event (`seq`) and file version (sha256) they rely on, so a rerun is checked
against the trace rather than against a previous run. `workflow_reverify_cell.js` runs one more
independent verifier round on a single cell. Agent-independent review tools: `check_records.py`
(programmatic tests of every citation) and the log's §5 "how to inspect one checkpoint by hand".

Order of steps 4 and 5 does not matter (assemble reads `z/` if present); rerun 4 and 6 after any
change to `x_raw/` or `x_verify/`.
