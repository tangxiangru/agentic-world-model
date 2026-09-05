# SYNTHETIC deferred-comparator independent forward test

All artifacts under /tmp/synthetic-deferred-forward.wqpGJc are SYNTHETIC protocol fixtures, not model results or scientific evidence. No model execution, benchmark runs, network, Slurm, or repository/git mutations were performed.

## Inputs and method

Read skills/exp_protocol/SKILL.md, deferred-comparator.md, and card.template.yaml completely before exercising the workflow. Did not read specs, historical audits, or the deferred-comparator test file. Inspected implementation only to explain actual consumer behavior. Used /home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/.venv/bin/awm, confirmed awm.__file__ points to this checkout, and explicitly set PYTHONPATH to this checkout for final consumer checks.

All cards used evaluation.protocol.n=8, hypothesis.expected_effect.metric=accuracy, evaluation.comparator.value=null, and defer_validation=true. Scripts only emitted hard-coded, explicitly SYNTHETIC JSON fixtures or exited with a SYNTHETIC failure; parent/output directories contain dummy placeholders, not models. Data contains eight synthetic JSONL rows.

The first success card used:
- awm exp_protocol install --target /tmp/synthetic-deferred-forward.wqpGJc/success --tool claude
- awm exp_protocol index --dir /tmp/synthetic-deferred-forward.wqpGJc/success
- awm exp_protocol new --dir /tmp/synthetic-deferred-forward.wqpGJc/success
- Filled minimal 0-4 fields.
- awm exp_protocol check --dir /tmp/synthetic-deferred-forward.wqpGJc/success exp-01
- awm exp_protocol lock --dir /tmp/synthetic-deferred-forward.wqpGJc/success exp-01
- /usr/bin/python3 /tmp/synthetic-deferred-forward.wqpGJc/success/synthetic_pair.py, cwd=/tmp/synthetic-deferred-forward.wqpGJc/success
- Filled 5-6 without altering the planned null comparator value or true defer flag.
- awm exp_protocol close --dir /tmp/synthetic-deferred-forward.wqpGJc/success exp-01

Initial fixture setup mistakes were corrected: parent needed a dummy config.json for preflight; result closure required conclusion.mechanism_verdict, set to not_tested. Successful lock occurred before comparator.json existed and before the fixture script ran.

## Outcomes

| Case/session basename | close | Index / starting point | Deferred collection counters | Standalone installed Stop hook |
|---|---|---|---|---|
| original success | 0, verified receipt | closed / checkpoint | verified=1 | {} |
| no_actual_n | 1, actual sample count absent | unverified / absent | unverified=1 | block |
| partial | 1, samples=4 not 8 | unverified / absent | unverified=1 | block |
| failed_report | 1, evaluator not successfully completed | unverified / absent | unverified=1 | block |
| failed_experiment, no report | 0, failed receipt; observation null | closed / absent | failed_closed=1 | {} |
| mutate_card | inspected existing close after measurement 0.75 -> 0.9 | unverified / absent | unverified=1 | block |
| mutate_receipt | inspected existing close after receipt value 0.5 -> 0.4 | unverified / absent | unverified=1 | block |
| mutate_evidence | inspected existing close after report value 0.5 -> 0.4 | unverified / absent | unverified=1 | block |
| portable_with_evidence | existing successful receipt | closed / recipe | verified=1 | {} |
| portable_receipt_only | existing successful receipt | closed / recipe | verified=1 | {} |

The successful Inspect-shaped report contains status=success, eval.dataset.samples=1319, eval.config.limit=8, results.total_samples=8, results.completed_samples=8, an accuracy=0.5 score, and eight per-sample score records. Its intervention summary is synthetic n=8, accuracy=0.75. The dataset population is correctly not used as evaluated n.

no_actual_n contains accuracy/stderr plus only dataset.samples=1319 and requested limit=8. partial contains four sample rows and completed_samples=4. failed_report retains eight rows but status=error. All three completed/adopted conclusions were correctly prevented from serving as starting points.

The no-report failure is result.execution=failed, reasoned failure, verdict=inconclusive, decision=abandon_line, no measurements. It closed without pretending to verify a comparator.

## Mutations and portability

Copies were made using cp -a after successful closure. Mutations only changed the respective isolated copy.

The original success session was moved, not deleted, to /tmp/synthetic-deferred-forward.wqpGJc/original_success_preserved. Therefore the frozen original absolute paths are absent. The portable_receipt_only copy's comparator file was preserved as comparator.saved-synthetic.json rather than deleted, leaving both recognized original and relocated report paths absent.

Matching relocated evidence is hashed and accepted. When neither recognized evidence path exists, completion_state explicitly reports evidence_available=false and "historical verification receipt; original evidence unavailable after relocation". Index/collect/Stop still accept historical verification and index offers a recipe because the original checkpoint is gone.

Re-running:
  awm exp_protocol close --dir /tmp/synthetic-deferred-forward.wqpGJc/portable_with_evidence exp-01
returns 1 because it checks the original absolute comparator path; it also warns that the original script/data paths no longer exist. Historical consumer acceptance is therefore not the same as portable re-closure.

## Usability caveats, not established runtime defects

1. collect retains raw conclusion counters: rejected or stale opted-in cards still show n_closed=1, n_locked_open=0, adopted=1, alongside n_deferred_unverified=1 and n_deferred_verified=0. This looked inconsistent from scientist guidance alone. Parent clarified the raw/new-counter split is intentional for compatibility; guidance should explain which counters establish verified completion. Both text and --csv expose the deferred counters.
2. Scientist guidance did not explain historical-receipt portability or that relocated evidence is accepted for inspection but not for re-close at the old absolute path.
3. Stop's generic reason says manually filling conclusion "closes it too", then its deferred-specific suffix correctly says filling conclusion alone is not verification. The latter is operationally enforced; the combined prose could be clearer.

No acceptance of an unverified starting point or stale edited receipt was found in the requested forward cases.

## Revalidation

After receiving notice of runtime refinements, reinstalled the current skill into every tested consumer session and reran index, collect, and standalone /usr/bin/python3 -S .../hooks/stop_open_cards.py with JSON stdin {"cwd":"SESSION"}. PYTHONPATH explicitly selected this repo. All table outcomes persisted.

Final source hashes:
- awm/exp_protocol/comparator.py: 140b33928b7955ee4ce34345ae48e8e4d72eee1cdadfdf50607c8bd32604d727
- awm/exp_protocol/collect.py: 73a2c33f31ca48950a637446e23cc16269f0275cc0c43abb025ca89284163290
- awm/exp_protocol/lineage.py: df24d8492a5962b6f8c8a2ec59f0fe5cbde7f61244ef90ae9afc020d691762db
- skills/exp_protocol/hooks/comparator_receipt.py: 9b9f5b00cd7535d5ce28141e056345c33a2f8960800f4af16c01c211add3f10f
- skills/exp_protocol/hooks/stop_open_cards.py: 685c056d6db9fd08262bfd0d16ddd71fc664e6daff2d0ead46b4875768b4dac0
- skills/exp_protocol/deferred-comparator.md: 50744a4fe90c2f0f3a58323795e3cf596f4d55fb009e35197232c9eedf434c03

## Limitations

Synthetic reports, not actual Inspect evaluator execution; not an exhaustive malformed-input or race/security audit. No proof of dev-set/seed/model/decoder identity, no PTB validation, no real checkpoint load, no twelve-block exhaustion test, and no ordinary non-deferred regression suite. YAML was formatted conventionally for installed Stop-hook checks.

## Actual close outputs

```json
[
  {
    "case": "no_actual_n",
    "chunk_id": "155681",
    "wall_time_seconds": 0.088541536,
    "exit_code": 1,
    "original_token_count": 36,
    "output": "ERROR   evaluation.comparator: comparator.json: actual sample count is absent; a requested limit or stderr estimate is not evidence\nnot closed\n"
  },
  {
    "case": "partial",
    "chunk_id": "278ca3",
    "wall_time_seconds": 0.115861311,
    "exit_code": 1,
    "original_token_count": 23,
    "output": "ERROR   evaluation.comparator: comparator.json: samples=4, expected actual n=8\nnot closed\n"
  },
  {
    "case": "failed_report",
    "chunk_id": "7ffb62",
    "wall_time_seconds": 0.143684912,
    "exit_code": 1,
    "original_token_count": 27,
    "output": "ERROR   evaluation.comparator: comparator.json: evaluator did not report successful completion\nnot closed\n"
  },
  {
    "case": "failed_experiment",
    "chunk_id": "e3f57f",
    "wall_time_seconds": 0.000006273,
    "exit_code": 0,
    "original_token_count": 63,
    "output": "WARNING evaluation.comparator: failed/unrun experiment: no verified comparator measurement\nok (1 warning, advisory)\ndeferred comparator receipt: failed\nclosed exp-01; index at /tmp/synthetic-deferred-forward.wqpGJc/failed_experiment/memory/index.md\n"
  }
]
```

## Final consumer outputs

```json
[
  {
    "case": "no_actual_n",
    "chunk_id": "0ab7da",
    "wall_time_seconds": 0.543475847,
    "exit_code": 0,
    "original_token_count": 379,
    "output": "wrote /tmp/synthetic-deferred-forward.wqpGJc/no_actual_n/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/no_actual_n/.claude/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/no_actual_n/CLAUDE.md\nwrote /tmp/synthetic-deferred-forward.wqpGJc/no_actual_n/memory/index.md\nsession=no_actual_n  accuracy=  hours_used=  n_cards=1  n_unreadable=0  n_closed=1  n_locked=1  n_locked_open=0  n_relocked=0  n_overrides=0  preflight_fail=0  pitfalls_hit=0  pitfalls_cost_h=0.0  adopted=1  fields_filled=1.0  n_deferred=1  n_deferred_verified=0  n_deferred_failed_closed=0  n_deferred_unverified=1\n{\"decision\": \"block\", \"reason\": \"Locked cards without a conclusion: exp-01. Ending this turn ENDS THE SESSION: there is no next turn, and every background process you started dies with it, a training run included. If a run is still going, wait for it in the foreground (`sleep 900; tail -n 3 <log>` repeated, or `while [ ! -f <out>/config.json ]; do sleep 300; done`; give the Bash call a long timeout), then evaluate, fill sections 5-6 and run `awm exp_protocol close --dir <dir> <card>`. If the run is dead, record it (result.execution: failed or killed) and close the card. If the CLI is broken, fill result and conclusion in the YAML by hand; that closes it too. (block 2 of 12) Deferred comparator cards need a valid close receipt; filling conclusion alone is not verification. Run close after producing valid evidence, or record an inconclusive, non-adopted failed experiment and close it.\"}\n"
  },
  {
    "case": "partial",
    "chunk_id": "63ed4c",
    "wall_time_seconds": 0.316817684,
    "exit_code": 0,
    "original_token_count": 374,
    "output": "wrote /tmp/synthetic-deferred-forward.wqpGJc/partial/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/partial/.claude/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/partial/CLAUDE.md\nwrote /tmp/synthetic-deferred-forward.wqpGJc/partial/memory/index.md\nsession=partial  accuracy=  hours_used=  n_cards=1  n_unreadable=0  n_closed=1  n_locked=1  n_locked_open=0  n_relocked=0  n_overrides=0  preflight_fail=0  pitfalls_hit=0  pitfalls_cost_h=0.0  adopted=1  fields_filled=1.0  n_deferred=1  n_deferred_verified=0  n_deferred_failed_closed=0  n_deferred_unverified=1\n{\"decision\": \"block\", \"reason\": \"Locked cards without a conclusion: exp-01. Ending this turn ENDS THE SESSION: there is no next turn, and every background process you started dies with it, a training run included. If a run is still going, wait for it in the foreground (`sleep 900; tail -n 3 <log>` repeated, or `while [ ! -f <out>/config.json ]; do sleep 300; done`; give the Bash call a long timeout), then evaluate, fill sections 5-6 and run `awm exp_protocol close --dir <dir> <card>`. If the run is dead, record it (result.execution: failed or killed) and close the card. If the CLI is broken, fill result and conclusion in the YAML by hand; that closes it too. (block 2 of 12) Deferred comparator cards need a valid close receipt; filling conclusion alone is not verification. Run close after producing valid evidence, or record an inconclusive, non-adopted failed experiment and close it.\"}\n"
  },
  {
    "case": "failed_report",
    "chunk_id": "53d680",
    "wall_time_seconds": 0.386094091,
    "exit_code": 0,
    "original_token_count": 382,
    "output": "wrote /tmp/synthetic-deferred-forward.wqpGJc/failed_report/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/failed_report/.claude/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/failed_report/CLAUDE.md\nwrote /tmp/synthetic-deferred-forward.wqpGJc/failed_report/memory/index.md\nsession=failed_report  accuracy=  hours_used=  n_cards=1  n_unreadable=0  n_closed=1  n_locked=1  n_locked_open=0  n_relocked=0  n_overrides=0  preflight_fail=0  pitfalls_hit=0  pitfalls_cost_h=0.0  adopted=1  fields_filled=1.0  n_deferred=1  n_deferred_verified=0  n_deferred_failed_closed=0  n_deferred_unverified=1\n{\"decision\": \"block\", \"reason\": \"Locked cards without a conclusion: exp-01. Ending this turn ENDS THE SESSION: there is no next turn, and every background process you started dies with it, a training run included. If a run is still going, wait for it in the foreground (`sleep 900; tail -n 3 <log>` repeated, or `while [ ! -f <out>/config.json ]; do sleep 300; done`; give the Bash call a long timeout), then evaluate, fill sections 5-6 and run `awm exp_protocol close --dir <dir> <card>`. If the run is dead, record it (result.execution: failed or killed) and close the card. If the CLI is broken, fill result and conclusion in the YAML by hand; that closes it too. (block 2 of 12) Deferred comparator cards need a valid close receipt; filling conclusion alone is not verification. Run close after producing valid evidence, or record an inconclusive, non-adopted failed experiment and close it.\"}\n"
  },
  {
    "case": "failed_experiment",
    "chunk_id": "d8d7c7",
    "wall_time_seconds": 0.363917151,
    "exit_code": 0,
    "original_token_count": 163,
    "output": "wrote /tmp/synthetic-deferred-forward.wqpGJc/failed_experiment/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/failed_experiment/.claude/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/failed_experiment/CLAUDE.md\nwrote /tmp/synthetic-deferred-forward.wqpGJc/failed_experiment/memory/index.md\nsession=failed_experiment  accuracy=  hours_used=  n_cards=1  n_unreadable=0  n_closed=1  n_locked=1  n_locked_open=0  n_relocked=0  n_overrides=0  preflight_fail=0  pitfalls_hit=0  pitfalls_cost_h=0.0  adopted=0  fields_filled=1.0  n_deferred=1  n_deferred_verified=0  n_deferred_failed_closed=1  n_deferred_unverified=0\n{}\n"
  },
  {
    "case": "mutate_card",
    "chunk_id": "777347",
    "wall_time_seconds": 0.609563542,
    "exit_code": 0,
    "original_token_count": 379,
    "output": "wrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_card/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_card/.claude/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_card/CLAUDE.md\nwrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_card/memory/index.md\nsession=mutate_card  accuracy=  hours_used=  n_cards=1  n_unreadable=0  n_closed=1  n_locked=1  n_locked_open=0  n_relocked=0  n_overrides=0  preflight_fail=0  pitfalls_hit=0  pitfalls_cost_h=0.0  adopted=1  fields_filled=1.0  n_deferred=1  n_deferred_verified=0  n_deferred_failed_closed=0  n_deferred_unverified=1\n{\"decision\": \"block\", \"reason\": \"Locked cards without a conclusion: exp-01. Ending this turn ENDS THE SESSION: there is no next turn, and every background process you started dies with it, a training run included. If a run is still going, wait for it in the foreground (`sleep 900; tail -n 3 <log>` repeated, or `while [ ! -f <out>/config.json ]; do sleep 300; done`; give the Bash call a long timeout), then evaluate, fill sections 5-6 and run `awm exp_protocol close --dir <dir> <card>`. If the run is dead, record it (result.execution: failed or killed) and close the card. If the CLI is broken, fill result and conclusion in the YAML by hand; that closes it too. (block 2 of 12) Deferred comparator cards need a valid close receipt; filling conclusion alone is not verification. Run close after producing valid evidence, or record an inconclusive, non-adopted failed experiment and close it.\"}\n"
  },
  {
    "case": "mutate_receipt",
    "chunk_id": "4e9583",
    "wall_time_seconds": 0.331731548,
    "exit_code": 0,
    "original_token_count": 383,
    "output": "wrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_receipt/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_receipt/.claude/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_receipt/CLAUDE.md\nwrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_receipt/memory/index.md\nsession=mutate_receipt  accuracy=  hours_used=  n_cards=1  n_unreadable=0  n_closed=1  n_locked=1  n_locked_open=0  n_relocked=0  n_overrides=0  preflight_fail=0  pitfalls_hit=0  pitfalls_cost_h=0.0  adopted=1  fields_filled=1.0  n_deferred=1  n_deferred_verified=0  n_deferred_failed_closed=0  n_deferred_unverified=1\n{\"decision\": \"block\", \"reason\": \"Locked cards without a conclusion: exp-01. Ending this turn ENDS THE SESSION: there is no next turn, and every background process you started dies with it, a training run included. If a run is still going, wait for it in the foreground (`sleep 900; tail -n 3 <log>` repeated, or `while [ ! -f <out>/config.json ]; do sleep 300; done`; give the Bash call a long timeout), then evaluate, fill sections 5-6 and run `awm exp_protocol close --dir <dir> <card>`. If the run is dead, record it (result.execution: failed or killed) and close the card. If the CLI is broken, fill result and conclusion in the YAML by hand; that closes it too. (block 2 of 12) Deferred comparator cards need a valid close receipt; filling conclusion alone is not verification. Run close after producing valid evidence, or record an inconclusive, non-adopted failed experiment and close it.\"}\n"
  },
  {
    "case": "mutate_evidence",
    "chunk_id": "9df72e",
    "wall_time_seconds": 0.386684583,
    "exit_code": 0,
    "original_token_count": 384,
    "output": "wrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_evidence/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_evidence/.claude/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_evidence/CLAUDE.md\nwrote /tmp/synthetic-deferred-forward.wqpGJc/mutate_evidence/memory/index.md\nsession=mutate_evidence  accuracy=  hours_used=  n_cards=1  n_unreadable=0  n_closed=1  n_locked=1  n_locked_open=0  n_relocked=0  n_overrides=0  preflight_fail=0  pitfalls_hit=0  pitfalls_cost_h=0.0  adopted=1  fields_filled=1.0  n_deferred=1  n_deferred_verified=0  n_deferred_failed_closed=0  n_deferred_unverified=1\n{\"decision\": \"block\", \"reason\": \"Locked cards without a conclusion: exp-01. Ending this turn ENDS THE SESSION: there is no next turn, and every background process you started dies with it, a training run included. If a run is still going, wait for it in the foreground (`sleep 900; tail -n 3 <log>` repeated, or `while [ ! -f <out>/config.json ]; do sleep 300; done`; give the Bash call a long timeout), then evaluate, fill sections 5-6 and run `awm exp_protocol close --dir <dir> <card>`. If the run is dead, record it (result.execution: failed or killed) and close the card. If the CLI is broken, fill result and conclusion in the YAML by hand; that closes it too. (block 2 of 12) Deferred comparator cards need a valid close receipt; filling conclusion alone is not verification. Run close after producing valid evidence, or record an inconclusive, non-adopted failed experiment and close it.\"}\n"
  },
  {
    "case": "portable_with_evidence",
    "chunk_id": "7dadd2",
    "wall_time_seconds": 0.361622488,
    "exit_code": 0,
    "original_token_count": 192,
    "output": "wrote /tmp/synthetic-deferred-forward.wqpGJc/portable_with_evidence/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/portable_with_evidence/.claude/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/portable_with_evidence/CLAUDE.md\nwrote /tmp/synthetic-deferred-forward.wqpGJc/portable_with_evidence/memory/index.md\n  start from exp-01 [recipe] accuracy=0.75 (checkpoint gone: rerun exp-01 <- base_model)\nsession=portable_with_evidence  accuracy=  hours_used=  n_cards=1  n_unreadable=0  n_closed=1  n_locked=1  n_locked_open=0  n_relocked=0  n_overrides=0  preflight_fail=0  pitfalls_hit=0  pitfalls_cost_h=0.0  adopted=1  fields_filled=1.0  n_deferred=1  n_deferred_verified=1  n_deferred_failed_closed=0  n_deferred_unverified=0\n{}\n"
  },
  {
    "case": "portable_receipt_only",
    "chunk_id": "bd3c07",
    "wall_time_seconds": 0.333195736,
    "exit_code": 0,
    "original_token_count": 190,
    "output": "wrote /tmp/synthetic-deferred-forward.wqpGJc/portable_receipt_only/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/portable_receipt_only/.claude/skills/exp_protocol\nwrote /tmp/synthetic-deferred-forward.wqpGJc/portable_receipt_only/CLAUDE.md\nwrote /tmp/synthetic-deferred-forward.wqpGJc/portable_receipt_only/memory/index.md\n  start from exp-01 [recipe] accuracy=0.75 (checkpoint gone: rerun exp-01 <- base_model)\nsession=portable_receipt_only  accuracy=  hours_used=  n_cards=1  n_unreadable=0  n_closed=1  n_locked=1  n_locked_open=0  n_relocked=0  n_overrides=0  preflight_fail=0  pitfalls_hit=0  pitfalls_cost_h=0.0  adopted=1  fields_filled=1.0  n_deferred=1  n_deferred_verified=1  n_deferred_failed_closed=0  n_deferred_unverified=0\n{}\n"
  }
]
```

## Direct receipt-state explanations

```text
{
  "mutate_card": {
    "valid": false,
    "outcome": "unverified",
    "detail": "card, plan or lock changed since comparator closure"
  },
  "mutate_receipt": {
    "valid": false,
    "outcome": "unverified",
    "detail": "comparator receipt is not the close record sealed by this lock"
  },
  "mutate_evidence": {
    "valid": false,
    "outcome": "unverified",
    "detail": "comparator evidence changed after closure"
  },
  "portable_with_evidence": {
    "valid": true,
    "outcome": "verified",
    "evidence_available": true,
    "detail": "verified comparator receipt and evidence hash"
  },
  "portable_receipt_only": {
    "valid": true,
    "outcome": "verified",
    "evidence_available": false,
    "detail": "historical verification receipt; original evidence unavailable after relocation"
  }
}
```
