# Window04 focused audit: frozen candidate D's save check applies to eval-only cards

Audit date: 2026-09-04. Read-only repository/trace inspection plus an isolated CPU-only preflight fixture. No model load, training, evaluation, Slurm query/action, repository edit, manifest change, or git commit was performed. This is mechanism/scope evidence, not a screen metric/design decision or new score claim.

## Finding

Frozen D has a real scope defect: a parent JSON's incompatibility with Transformers **saving** is made a blocking condition for **every** card, including two trace-confirmed evaluator-only cards whose commands never save a model. D would counterfactually fail g01r03 exp-08 and baseline-strict p00s02 exp-07. The latter is exp-07, **not exp-06**: p00s02 exp-06 is real SFT.

The relevant distinction is not simply `family in TRAINING_FAMILIES`. g01r03 exp-07 is `merge`, yet its original soup really did call `model.save_pretrained` and failed on precisely this invalid inherited configuration. That is a rightful D hit. A blanket exception for every non-training family would remove genuine save protection.

There is an additional, adjacent precision limit: D does not see an in-code reset of `model.generation_config` before saving. p00s02 exp-06 and the repaired g01r03 soup retain a greedy parent file, but replace the in-memory config before save and successfully save. D still rejects their parent file. This audit does not choose how to represent or verify such repairs, and does not propose weakening save validation.

## Exact frozen surface

All line numbers in this section refer to **git object 8332917**, not current-worktree line numbers.

- Commit: `833291799b37991f0ca0e22a5d6a42679916b167`.
- Protocol tree: `7160d360ee6ca50b8b377efb029dc7d0361c3704`.
- `awm/exp_protocol/preflight.py` Git blob: `7b10b0421770486877a1cc4568df30b1dee6ceb8`; SHA-256 of raw frozen source: `25496ebb778b107e4789bb5e8bc6b9e768a69916ae786bb2c1c130d7598b8fac`.
- `awm/exp_protocol/schema.py` SHA-256 of raw frozen source: `4280b4c482ccac697ef3aecf8f6d32cad80a6d11dccb53e3a3829313032f15e8`.
- `preflight.py:308`: `GREEDY_INCOMPATIBLE = {"temperature": 1.0, "top_k": 50, "top_p": 1.0, "typical_p": 1.0, "min_p": None}`.
- `preflight.py:311–334`: `parent_generation_config_valid` reads only `setup.parent_checkpoint.path`, checks the local `generation_config.json`, skips nonlocal/absent config, and fails when `do_sample` is false and an explicitly present non-None key differs from the constant's default. There is **no** family, argv, framework, checkpoint cadence, or save-operation condition.
- The failure text (`preflight.py:330–333`) says the run "will die at its first checkpoint" and recommends either a valid parent copy or setting `model.generation_config` before the first save. Those future model mutations are not inspected by the predicate.
- `preflight.py:368–383`, especially `371`, runs every registered check for every card.
- `schema.py:29–31` accepts `sft`, `rft`, `dpo`, `grpo`, `distill`, `merge`, `decode-config`, `other`; only the first five are `TRAINING_FAMILIES`. There is no `eval-only` family. `schema.py:296–308` uses the training-family distinction only to require stop token/max sequence length. It does not scope D or declare a save operation.
- `cli.py:125–140`: schema validation precedes full preflight, and any unoverridden failure prevents the lock. A documented reasoned `--override CHECK=REASON` exists (`98–112`, `132–139`): this is a false-block/override burden, not an unavoidable inability to proceed.
- `pitfalls.yaml:66–71` frames D as the greedy-parent first-save failure. Existing D tests (`tests/test_exp_protocol_preflight.py:203–241`) cover invalid config, three valid configs, missing config, hub path, but do not cover evaluator-only scope or in-code repair.

Re-read exact source with `git show 8332917:awm/exp_protocol/preflight.py | nl -ba` (and the corresponding schema/cli/test paths).

## CPU reproduction

Reproducible runner: [reproduce.py](/tmp/window04-d-scope.gadLFh/reproduce.py). It loads complete frozen schema and preflight source directly from git objects into isolated modules in memory. Only the unused catalogue fallback `awm.paths.REPO_ROOT` is supplied as a stub; all calls pass `pitfalls=[]`. The predicate and `run_preflight` source are unmodified. Five fixture parent directories contain only small JSON files. No Transformers/model/accelerator code is imported.

```bash
cd /home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/window04-d-scope.gadLFh/reproduce.py
```

Observed exit code: **0**. Final output: `PASS: 5 real-card replays; 8 schema-valid family preflights; 5 config controls; 2 static evaluator checks. No training/evaluation/model/Slurm calls.`

For real-card replays, the original YAML is read and passes frozen `validate_plan`; only `setup.parent_checkpoint.path` is remapped in a deep copy to the CPU fixture. The test calls the actual frozen check; it does not claim full historical files were present or a historical D check occurred. Parent generation-config artifacts are omitted from these harvested bundles, so JSON is reconstructed from the retained trace/code evidence below (the temperature-only fixture carries the decisive fields, not a byte-for-byte whole-artifact copy).

| Real card / actual operation | Fixture configuration | Frozen D result |
|---|---|---|
| g01r03 exp-08 / `evaluate.py`, family `other` | `do_sample:false, temperature:0.0` | `fail`, offender `temperature=0.0` |
| p00s02 exp-07 / `evaluate.py`, family `other` | `do_sample:false, temperature:0.0, top_k:0, top_p:1.0` | `fail`, offenders `temperature=0.0, top_k=0` |
| p00s01 exp-06 / Trainer RFT | temperature-only unsafe parent | `fail` |
| g01r03 exp-07 / model soup and `save_pretrained`, family `merge` | temperature-only unsafe parent | `fail` |
| p00s02 exp-06 / SFT with in-code valid-config reset | p00s02 unsafe parent | `fail` |

All five pass the same check when the parent fixture is changed to `{"do_sample":false,"eos_token_id":[1,106]}`. Separate controls: stock sampled config **pass**, valid greedy config **pass**, missing `generation_config.json` **skip**. This does not alter the unsafe-save predicate.

Full frozen `run_preflight` was also exercised on a schema-valid fixture derived from the frozen example card. For **each of all eight method families**, it returned exactly `9 pass, 0 warn, 1 fail, 0 skip`, the sole failure being `parent_generation_config_valid`. This rules out an external family-scoping gate in `run_preflight` and avoids confounding with missing historical `/home/ben/task` paths. The fixture command is merely metadata and is never run.

AST inspection of each complete retained evaluator finds `inspect_eval` at line 74 and no `.train`, `.save_model`, or `.save_pretrained` call. This is static command-path evidence, not a rerun of evaluation or proof about every transitive library operation. The retained successful serving logs below independently show that these were not Trainer-save failures.

## Named real evidence

All trace line references below are lines in decompressed `solve_parsed.txt.gz`; times are UTC. Retained scripts can reflect repairs, so original failure behavior is tied to the corresponding earlier traceback rather than inferred from the final script alone.

### g01r03 exp-08: pure final-artifact evaluation

Bundle: `/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r03/`.

- [exp-08.yaml](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r03/task/memory/cards/exp-08.yaml:38): parent `/home/ben/task/ckpts/exp-07_soup` at 39–42; `family: other` at 54; evaluator-only argv/script at 61–66; `no training` at 94.
- [evaluate.py](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r03/task/evaluate.py:59): calls Inspect with `model=vllm/{args.model_path}` at 74–86; the only output write in `main` is metrics JSON at 88–96. Default model path is `final_model` at 20. No Trainer or model save.
- Trace L8838–8848 (result at **02:43:10**) prints the soup parent's config, including `do_sample:false` and `temperature:0.0`. L8919 (**02:43:41**) copies `ckpts/exp-07_soup/*` to final_model. L8965–8975 (**02:44:46**) prints that serving copy's same config.
- Trace L9147–9153: exp-08 checked/locked at **02:45:49** under its actual earlier protocol, not D. L9158 launches only `python evaluate.py --json-output-file ...`.
- [exp-08 result](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r03/task/memory/cards/exp-08.yaml:87) records successful n=150 evaluation. [retained eval log](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r03/task/logs/exp-08_final_model_default.log:263) ends in metrics/normal server shutdown, not a model-save exception. No accuracy mechanism or deterministic-repeat claim is needed for this audit.

### baseline-strict p00s02 exp-07: pure final-artifact evaluation (correct card ID)

Bundle: `/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s02/`.

- [exp-07.yaml](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s02/task/memory/cards/exp-07.yaml:42): parent `ckpts/exp-05/final` at 43–46; no-training declaration at 54; `family: other` at 58; `python evaluate.py --model-path final_model --limit 500 ...` at 66; no-training result at 96. Prior `finalize.py` is recorded as a smoke run at 21; it is not this card's launch command.
- [evaluate.py](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s02/task/evaluate.py:59) has the same Inspect-only path as above.
- Trace L7641–7645 (**01:39:25**) explicitly applies `set_decode.py --model ckpts/exp-05/final --mode greedy`. [set_decode.py:17–23](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s02/task/set_decode.py:17) writes `do_sample:false, temperature:0.0, top_p:1.0, top_k:0`. The exp-06 continuation resets an in-memory model, not this parent file.
- [finalize.py:25–36](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s02/task/finalize.py:25) copies the parent to final_model and applies greedy JSON there. Trace L7981–7984 (**01:46:53**) prints the loaded final_model config with `temperature:0.0` and `top_k:0`; this is serving-copy confirmation, not a fresh read of the parent at exp-07 lock.
- Trace L8396–8405 locks exp-07 at **02:57:08**. L8410 (**02:57:12**) launches the evaluator. L8443–8447 (**03:00:12**) shows the vLLM config-override message, 500 served items, and successful result. D's unsafe-parent match here is code/trace-derived, not an observed D override.

### baseline-strict p00s01 exp-06: genuine inherited-config Trainer save failure

Bundle: `/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s01/`.

- [exp-06.yaml](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s01/task/memory/cards/exp-06.yaml:65): parent `ckpts/exp-05/final` at 67, `rft` at 81, training argv `--save-steps 150` at 89. Result at 123 explicitly preserves the first-attempt failure despite the eventually completed card.
- [train_sft.py:118–120](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s01/task/train_sft.py:118) loads `args.model`; 142 configures save cadence; 154–160 constructs/runs Trainer; 165 saves the final model. This path is not inferred from `family:rft` alone.
- Trace L8407: stopped at **150/306**, elapsed **15:12**. L8413–8453, discovered **01:18:44**, shows the complete actual call chain: `trainer.train()` → `_save_checkpoint` → `save_model` → `model.save_pretrained` → `generation_config.save_pretrained` → `validate(strict=True)`, rejecting `do_sample=False` plus `temperature=0.0`.
- Trace L8467–8473 (**01:19:13**) removes only the parent's temperature key and validates the config; L8481–8483 (**01:19:15**) prints the valid config and relaunch. The final retained training log is the overwritten repaired attempt, so the original traceback must be cited from the trace, not inferred absent from that log.
- Frozen D's rejection is appropriate for this original unsafe save. No new training was run to establish this fact.

### g01r03 exp-07: non-training soup still genuinely saves a model

- [exp-07.yaml](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r03/task/memory/cards/exp-07.yaml:39): first parent `ckpts/exp-04/final`, `family:merge` at 55, `scripts/soup.py --models exp-04/final exp-05/final` at 63.
- Trace L8658 (**02:24:08**) launches the original soup. L8700–8726, discovered **02:29:11**, is the original traceback: `soup.py:43 model.save_pretrained(args.out)` → `generation_config.save_pretrained` → strict validation → `temperature:0.0` incompatible with `do_sample:false`.
- Trace L8739–8752 (**02:29:25**) inserts a valid in-memory config **before** save, re-locks and reruns. The retained [soup.py:39–60](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r03/task/scripts/soup.py:39) is the repaired script: load first parent, assign weights, set `do_sample=True` and `temperature=None`, call `save_pretrained` at 48, then write greedy JSON at 54–60 for vLLM.
- D is appropriate for the **original** soup. Its unchanged parent-only predicate still fails the **repaired** soup: a separate limitation, not evidence that merge itself is exempt from save validation.

### p00s02 exp-06: actual SFT with a pre-save in-code repair

- [exp-06.yaml](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s02/task/memory/cards/exp-06.yaml:42) names the greedy exp-05 parent, family `sft` at 66, and `train_sft.py` argv at 74. It is not a final-selection/eval-only card.
- [train_sft.py:128–138](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s02/task/train_sft.py:128) loads the parent, then assigns a fresh valid `GenerationConfig(do_sample=True, top_k=64, top_p=0.95, ...)` before constructing Trainer. This code was introduced at trace L7486–7504 after exp-05's earlier failed attempt; it is already present for exp-06.
- [exp-06.log:40–41](/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8-v1/p00s02/task/logs/exp-06.log:40) records train runtime 3561.4003 seconds and successful `saved /home/ben/task/ckpts/exp-06/final`. Frozen D still reads the unchanged parent JSON, so it cannot distinguish this successfully repaired path.

## Decision boundary

Confirmed: two concrete eval-only false blocks, one true Trainer-save hit, one true non-training model-save hit, and parent-only blindness to verified in-code repair. These are counterfactual D replays over historical non-D cells, not candidate-screen outcomes. No D2 predicate, schema addition, acceptance metric, guardrail, promotion, release, withdrawal, or queue action is selected here. Preserve save safety and route design/metric choices to the planner's explicit prelaunch decision.
