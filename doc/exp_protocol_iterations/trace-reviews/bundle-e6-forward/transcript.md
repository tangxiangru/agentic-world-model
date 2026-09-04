# E6 independent CPU transcript (selected exact observations)

Worktree: /rmeng_data/robtang/exp-protocol-bundle-work-5iV6EzGB/repo
Runner: bash /tmp/e6-forward-sampling.ntRxXE/inside.sh [Python args]
No implementation/tests were read; cases were frozen in cases-before-implementation.md.

1. import_probe.py, 2026-09-04 10:16 UTC:
   vllm 0.11.0; transformers 4.57.3; torch 2.8.0+cu129.
   Native LLM/SamplingParams/RequestOutput/CompletionOutput imported.
   Local tokenizer preparation returned PreparedPrompt and resolved EOT to [106].
   No platform detected and missing libcuda.so.1 warnings were expected in CPU/no-device isolation.
   Incidental probe `from awm.cli import app` failed (no app export); `-m awm.cli` worked.
2. `-m awm.cli exp_protocol check --dir /tmp/e6-forward-sampling.ntRxXE exp-01`:
   `ok (1 warning, advisory)`; no alternatives recorded.
3. `-m awm.cli exp_protocol lock --dir /tmp/e6-forward-sampling.ntRxXE exp-01`:
   5 pass,4 warn,0 fail,1 skip; locked at 2026-09-04T10:19:19Z, plan d204b707c26a.
   Warnings were absent optional rendered/stop/answer/max-length declarations on an `other` CPU fixture.
4. First workflows.py run, 10:19 UTC:
   double BOS, unknown stop and multi-token stop rejected by SamplingEvidenceError.
   Four capture attempts raised FileNotFoundError because recordings/ parent did not exist.
   engine_calls []; original_params_unchanged true.
5. Created owned recordings/.keep with apply_patch; same locked script rerun, 10:20 UTC:
   capture_valid: status captured;3 requests,6 completions,34 tokens,raw_durable true.
   parse_numeric: count6,parsed5,parser_errors1,all_parsed false.
   parse_interrupted: KeyboardInterrupt: synthetic parser interruption; no inference.
   parse_recovery: count6,parsed6,parser_errors0,all_parsed true.
   raw_unchanged_numeric/interrupted/recovery: true.
   reuse: FileExistsError; existing recording unchanged true.
   capture_missing_draw: SamplingEvidenceError: request is unfinished or completion count differs from requested n.
   capture_wrong_order: SamplingEvidenceError: returned prompt tokens/order differ from the actual submitted input.
   capture_aborted: SamplingEvidenceError: completion text or native finish reason is unsupported/incomplete.
   Each invalid capture retained request.json,raw.jsonl,capture-failure.json; no capture.json.
   parse_recording on each invalid capture: FileNotFoundError for missing capture.json; no certification.
   original_params_unchanged true; engine_calls [valid,missing_draw,wrong_order,aborted].
6. Native captures/parse summaries and failures were read structurally after the run.
   Interruption: zero-byte parse prefix + .failure.json(status interrupted_or_failed), no .summary.json.
   Numeric overflow row: status parser_error; ValueError: numeric value is not finite.
   Original raw remains SHA256 508a69f820513aac07e4507bb90fd1e7ff6ad597bd44bc48b2b19cb40c4fca3a.
