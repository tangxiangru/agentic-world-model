# SYNTHETIC focused target-count follow-up

Read current SKILL.md, deferred-comparator.md and card.template.yaml before this focused pass. No specs, prior audits or repository test files were read. No repository edits, model execution, benchmark runs, network or Slurm operations.

Session: /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup
Repo: /home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator

Used the current .venv/bin/awm with PYTHONPATH explicitly set to the repo. Installed the current scientist skill into this synthetic session, then index/check/lock before running the declared Python fixture producer. Comparator and target files both contain eight synthetic rows' summarized measurements: comparator accuracy=0.5, target accuracy=0.75. No real evaluator was run.

Commands:
- PYTHONPATH=REPO .venv/bin/awm exp_protocol install --target SESSION --tool claude
- PYTHONPATH=REPO .venv/bin/awm exp_protocol index --dir SESSION
- PYTHONPATH=REPO .venv/bin/awm exp_protocol check --dir SESSION exp-01
- PYTHONPATH=REPO .venv/bin/awm exp_protocol lock --dir SESSION exp-01
- /usr/bin/python3 SESSION/synthetic_pair.py (cwd SESSION)
- After recording result/conclusion, PYTHONPATH=REPO .venv/bin/awm exp_protocol close --dir SESSION exp-01
- PYTHONPATH=REPO .venv/bin/awm exp_protocol index --dir SESSION
- PYTHONPATH=REPO .venv/bin/awm exp_protocol collect SESSION
- /usr/bin/python3 -S SESSION/skills/exp_protocol/hooks/stop_open_cards.py with stdin {"cwd":"SESSION"}

## n=150 target card with valid n=8 comparator

close exited 1 with:
ERROR result.measurements: deferred comparison needs a recorded target measurement at the locked metric and n

No comparator completion receipt was created. Index marked unverified and offered no starting point. collect recorded n_deferred_unverified=1, verified=0. Installed standalone -S Stop hook blocked.

The hook message no longer claimed filling result/conclusion by hand closes this opted-in card. It explicitly stated a valid close receipt is required and filling conclusion alone is not verification. The hook's generic opening "Locked cards without a conclusion" is still imprecise for an already-filled but unverified conclusion; this is cosmetic and did not affect behavior.

The n=150 card snapshot is preserved at /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/card-n150.SYNTHETIC.yaml.

## Corrected n=8 target record

Changed only the postrun measurement n to 8 and the synthetic summary wording. Plan and comparator declaration were unchanged. close exited 0, wrote a verified receipt, index offered the checkpoint starting point, collect recorded n_deferred_verified=1 / unverified=0, and the installed -S Stop hook returned {}.

No remaining functional defect found in this focused pass. The updated deferred-comparator guidance clearly explains raw versus verified counters, historical inspection versus re-closure, and that target-file contents are not independently audited. This test only checks target-record n consistency, not exhaustive non-finite inputs or target-file integrity.

## Actual outputs

### Lock

```text
wrote /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/skills/exp_protocol
wrote /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/.claude/skills/exp_protocol
wrote /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/CLAUDE.md
wrote /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/memory/index.md
WARNING situation.alternatives_rejected: no alternatives recorded; was this the only option?
ok (1 warning, advisory)
PASS  data_files_exist — every setup.data[].path is a file: 1 file(s) present
PASS  data_n_examples_match — n_examples equals the line count of each jsonl file: 1 file(s) match (whole-file line count)
PASS  command_resolves — cwd is a directory, the script is a file, and argv names it: cwd and script resolve; argv names the script
PASS  output_dir_creatable — output_dir exists or its parent does: /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/output
WARN  stop_token_consistent — training targets end with the declared stop token: setup.method.stop_token not declared; the eos-mismatch pitfall cannot be checked
WARN  answer_marker_single — each target contains the answer marker exactly once: setup.method.answer_marker not declared; the double-format pitfall cannot be checked
WARN  max_seq_len_headroom — rows fit in max_seq_len (chars/4 estimate): hyperparams.max_seq_len not declared; truncation cannot be estimated
WARN  comparator_same_protocol — the comparator's eval file exists and used the same n: /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/comparator.json: measurement deferred; close must verify actual n and accuracy
PASS  parent_checkpoint_loadable — a local parent checkpoint has a config.json: /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/parent has config.json
-- 5 pass, 4 warn, 0 fail, 0 skip
-- not checkable by machine; check yourself:
   * template_unreachable: Embed the exact template the grader uses (byte-for-byte, hash-checked) in your training code and render one example both ways before launching.
   * final_model_not_loadable: Merge adapters, save the tokenizer alongside the weights, and load final_model/ once on CPU with transformers before the deadline. The grader loads final_model/ with vLLM from a fresh process.
   * datasets_version_drift: Use the datasets version the harness ships (PostTrainBench container: datasets==4.5.0) and pin dataset revisions in setup.data[].source.
   * run_dies_with_the_session: Never end the turn while a run is alive. Wait for it in the foreground - `sleep 900; tail -n 3 <log>` repeated, or `while [ ! -f <out>/config.json ]; do sleep 300; done` with a long Bash timeout - then evaluate, fill sections 5-6 and close the card. Launch long runs with `setsid nohup <cmd> > <log> 2>&1 < /dev/null &` so a tool timeout cannot take them down. The Stop hook (hooks/stop_open_cards.py) blocks the end of the turn while a locked card is open and says so.
locked exp-01 at 2026-09-03T22:40:31Z (plan 762540d07675)

```

### n=150 close (exit 1)

```text
ERROR   result.measurements: deferred comparison needs a recorded target measurement at the locked metric and n
WARNING evaluation.comparator: comparator.json: actual n=8, accuracy=0.5; remaining protocol identity needs manual verification
not closed

```

### n=150 consumers

```text
wrote /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/memory/index.md
session=target_count_followup  accuracy=  hours_used=  n_cards=1  n_unreadable=0  n_closed=1  n_locked=1  n_locked_open=0  n_relocked=0  n_overrides=0  preflight_fail=0  pitfalls_hit=0  pitfalls_cost_h=0.0  adopted=1  fields_filled=1.0  n_deferred=1  n_deferred_verified=0  n_deferred_failed_closed=0  n_deferred_unverified=1
{"decision": "block", "reason": "Locked cards without a conclusion: exp-01. Ending this turn ENDS THE SESSION: there is no next turn, and every background process you started dies with it, a training run included. If a run is still going, wait for it in the foreground (`sleep 900; tail -n 3 <log>` repeated, or `while [ ! -f <out>/config.json ]; do sleep 300; done`; give the Bash call a long timeout), then evaluate, fill sections 5-6 and run `awm exp_protocol close --dir <dir> <card>`. If the run is dead, record it (result.execution: failed or killed) and close the card. (block 1 of 12) Deferred comparator cards need a valid close receipt; filling conclusion alone is not verification. Run close after producing valid evidence, or record an inconclusive, non-adopted failed experiment and close it."}
# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0 | other | base_model | unverified | yes | supported | adopt | accuracy=0.75 | /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/output |

```

### n=8 close (exit 0)

```text
WARNING evaluation.comparator: comparator.json: actual n=8, accuracy=0.5; remaining protocol identity needs manual verification
ok (1 warning, advisory)
deferred comparator receipt: verified
closed exp-01; index at /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/memory/index.md

```

### n=8 consumers, receipt and source hashes

```text
wrote /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/memory/index.md
  start from exp-01 [checkpoint] accuracy=0.75 /tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/output
session=target_count_followup  accuracy=  hours_used=  n_cards=1  n_unreadable=0  n_closed=1  n_locked=1  n_locked_open=0  n_relocked=0  n_overrides=0  preflight_fail=0  pitfalls_hit=0  pitfalls_cost_h=0.0  adopted=1  fields_filled=1.0  n_deferred=1  n_deferred_verified=1  n_deferred_failed_closed=0  n_deferred_unverified=0
{}
{
  "schema_version": "awm-deferred-comparator-v1",
  "card_id": "exp-01",
  "verified_at": "2026-09-03T22:41:11Z",
  "locked_at": "2026-09-03T22:40:31Z",
  "plan_sha256": "762540d07675c9792567bffd2f5d6b5306a4ff4b4350c4995ba2d7443eabeef0",
  "card_sha256": "11bbea5df824b2718249137bbe59eb8d8be70d5419fb367dd3964936c64f0e12",
  "declaration": {
    "path": "/tmp/synthetic-deferred-forward.wqpGJc/target_count_followup/comparator.json",
    "n": 8,
    "metric": "accuracy",
    "session_dir": "/tmp/synthetic-deferred-forward.wqpGJc/target_count_followup"
  },
  "outcome": "verified",
  "observation": {
    "status": "pass",
    "detail": "comparator.json: actual n=8, accuracy=0.5; remaining protocol identity needs manual verification",
    "n": 8,
    "metric": "accuracy",
    "value": 0.5,
    "sha256": "4e379e9d408ce02f13eb6eeb21e0489a7c3059cb94839219bc0a0059a3ad43ff"
  }
}
b2171992d91ec52cdeb2693f0fcfe509e4dda87a7863ab84a2c37b9f3dfe8b6b  awm/exp_protocol/schema.py
03b075288ad9eca898627432e06927ef0031b38f10adc5063924887dc6098d9c  skills/exp_protocol/hooks/stop_open_cards.py
de32a29d530e1adebb0998f508e5380625fc1027700ef12d4609a1a6afbbaf0f  skills/exp_protocol/deferred-comparator.md

```
