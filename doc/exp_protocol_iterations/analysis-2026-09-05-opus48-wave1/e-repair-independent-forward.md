# Prepared workflow and actual outcome

This is a CPU-only preparation. No model, tokenizer, weights, dev set, or measured comparator was supplied. The model-metadata directory contains only generation_config.json; it is not a parent checkpoint. Both experiment cards remain unlocked drafts. No generated or combined dataset was fabricated.

1. Complete exp-01 with the chosen local sampler checkpoint, tokenizer/template and exact sampling script/argv. It currently names only the two existing input rows. The proposed recipe is two additional draws per prompt, temperature 0.7, seed 0, maximum 128 generated tokens per draw. These are planned choices, not observed outputs.
2. CPU-prepare actual prompt tokens using the selected template and add_special_tokens=False; resolve <end_of_turn> through that tokenizer. Do not infer stop IDs from supplied serving metadata. Call sampling_ready only once its real prerequisites exist. Check and successfully lock this sampling card before constructing the engine or invoking inference. Use record_vllm_from_factory within that locked command and a budgeted stage timeout with bounded cleanup. In this exercise the factory is unsupported because weights/tokenizer are unavailable; runtime compatibility remains unchecked. The factory was not invoked.
3. After sampling's actual successful exit, verify current-invocation request/raw/capture evidence. Preserve all four requested draws, including failed parsing, before filtering. Only retain nonempty, structurally valid outputs with one ANSWER marker and the declared terminator; this is not evidence of answer correctness. Deduplicate exact prompt/completion pairs, retain source ID/draw provenance in a sidecar, and persist original plus accepted generated rows as data/combined.jsonl. Actual counts and hashes must come from that artifact. Failed or truncated sampling is not automatically an acceptable training dependency.
4. Complete exp-02 only after that file exists. Select the model parent independently of the data producer; a generated dataset is never the parent checkpoint. With an actual tokenizer, prepare a RenderedTrainingBundle and set setup.data to its actual token artifact/data_entry, including the rendered receipt declaration. Verify masking, stop tokens, answer-marker counts and lengths. The tentative SFT recipe is one epoch, learning rate 2e-5, batch size 1, seed 0, max length 512. Fill the exact implementation/command, retained-checkpoint policy and held-out evaluation protocol; measure the parent under that same protocol. The current missing dev set/comparator is not replaced by the training examples.
5. Check and lock exp-02 against its actual inputs before model construction/training. A future supported native implementation should use SaveSafeTrainer and GenerationSaveContract, check the actual model after repairs, protect all saves and preserve selected serving JSON separately. Unsupported model/save routes remain unverified. Keep the final checkpoint as promised. Retain exit and artifact evidence, then evaluate under the locked protocol and close with actual results.

## Decode audit

The requested intent is greedy. The supplied file declares do_sample:false, eos_token_id:[106,1,106], pad_token_id:0, and no temperature. The request root contains only max_tokens:4000. In the documented pinned vLLM route, do_sample alone is not mode evidence. Neither supplied layer establishes greedy decoding. Native model defaults, native request resolution and actual engine decoding remain unknown. Repeated EOS IDs were preserved exactly; their correctness cannot be determined without the actual tokenizer/template. No metadata was changed.

freeze_decode_evidence with request_pointer="" succeeded, and verify_decode_evidence returned unchanged_evidence. This verifies the frozen metadata/request identity, not model loadability, serialization safety, deterministic execution or quality. A future explicit greedy request requires a deliberate decoder configuration and a matching locked evaluation; no such evaluation was executed.

## Actual checks

The original two rows have unique IDs, nonempty prompts, the declared stop suffix and exactly one answer marker each. Raw checks cannot prove token-level supervision or actual model consumption.

Sampling preflight: exit 0, four PASS, four WARN, two SKIP. Check and lock refused incomplete fields. Training preflight: exit 1, with data_files_exist FAIL for the absent combined dataset. Its check and lock also refused incomplete fields. No override was requested. The null optional watch_set object initially triggered a schema error; it was replaced by null and checks repeated, with the genuine blockers retained.

Full actual CLI argv, stdout, stderr and exit codes are in evidence/commands.json. The preparation process itself exited 0; this is not an experiment exit. Its exact command was:

```bash
PYTHONPATH=/tmp/exp-protocol-e-repair-20260905 PYTHONDONTWRITEBYTECODE=1 /home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/.venv/bin/python scripts/prepare_workflow.py > evidence/preparation.stdout 2> evidence/preparation.stderr
```

Bootstrap: python -m awm exp_protocol --help failed because awm has no __main__. An initial CLI help invocation used PYTHONPATH=/tmp/exp-protocol-e-repair-20260905/src and resolved the workspace installation. This was corrected to the candidate root before card creation; importing awm printed /tmp/exp-protocol-e-repair-20260905/awm/__init__.py. All card and audit operations used that candidate root. No optional WMA review was attempted because model/network activity is forbidden here.

Instruction ambiguity: unknown nested optional watch-set fields are rejected by check, so omit/null the entire unused watch_set instead. Genuine missing prerequisites were kept null, not guessed. No construction, forward pass, inference, training, benchmark execution, network call or scheduler operation occurred.
