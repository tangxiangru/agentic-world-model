# Bootstrap tool observations (before capture.py existed)

- Read skill-creator, repository AGENTS, current scientist SKILL and process_checks;
  reread selected instruction files where combined output was truncated.
- Read current card.template.yaml and pitfalls.yaml, then ran root
  `.venv/bin/awm exp_protocol --help`: return0; supported commands are
  new/check/preflight/lock/close/index/chain/collect/install, no E run API.
- `mktemp -d /tmp/process-cross-task-forward.XXXXXX` returned this owned root.
- The initial `awm exp_protocol new --dir ROOT` returned0 and created exp-01
  with created_at2026-09-04T11:56:31Z; subsequent new/check/lock/close and CPU
  execution commands are captured automatically in transcript.jsonl.
- Only local tokenizer metadata/assets were resolved; no benchmark items were read.
