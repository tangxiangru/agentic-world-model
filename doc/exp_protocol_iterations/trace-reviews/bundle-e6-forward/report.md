# E6 independent sampling-evidence forward review

## Outcome

**No false block on the supported valid native-output workflow, no returned-draw loss, and no false capture success on the tested count/order/abort failures.** Recovery behaved as documented. One small guide/setup gap: the parent of the new recording directory must already exist; the guide does not show that prerequisite.

This is **CPU integration, not real inference**. No LLM constructor, weight loading, forward, sampling kernel, training, evaluator, GPU, network, install, repository edit, index or commit was used. Native vLLM types/source/version checks stayed active; **only `sampling._call_engine` was intercepted**. Engine-call timings measure the inert fixture callable and must not be interpreted as inference throughput or GPU savings.

## Independence and environment

Cases were chosen from `sampling-evidence.md`, runtime `SKILL.md`, `save-safety.md`, and card examples before implementation behavior was tested. No implementation/test source was read, and no case was copied from tests. API signatures and returned objects were inspected; implementation bytes were hashed only.

Worktree: `/rmeng_data/robtang/exp-protocol-bundle-work-5iV6EzGB/repo`.
Owned artifacts: `/tmp/e6-forward-sampling.ntRxXE/`.
Runner `inside.sh` uses the documented extracted runtime `/tmp/exp-protocol-save-runtime.JEZlHo/rootfs`, read-only host/runtime bindings, `--unshare-net`, fresh `/dev` without GPU devices, and only the owned directory writable.
Imports verified **vLLM0.11.0 / Transformers4.57.3 / Torch2.8.0+cu129**. Expected no-platform/libcuda warnings did not prevent native CPU object use.
Tokenizer-only source: `/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/data/ptb/hf/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d`, `local_files_only=True`.
The fixture uses `LLM.__new__(LLM)` without `__init__`, with an inert `llm_engine.get_tokenizer` provider. Intercepted returns are actual native `RequestOutput`/`CompletionOutput`, not duck-typed stand-ins.

`memory/cards/exp-01.yaml` describes synthetic CPU evidence review, not a scientific model experiment. Native CLI `check` and `lock` succeeded before capture. Its parent is explicit metadata-only fixture JSON, with no model/weights. The sampler was not told to skip lock/source/native validation.

## Cases and observed artifacts

Prepared three rendered prompts with typed/repeated source IDs **[7,"7",7]**, ordinals0–2, first two texts identical. Actual Gemma tokenization used `add_special_tokens=False`, single leading BOS and vocabulary-resolved EOT106. Native requested settings: **n2, temperature.7, top_p.91, max_tokens32, seed17, stop_token_ids[106], additional stop string END_FIXTURE**.

| Case | Observation | Evidence below owned root |
|---|---|---|
| Valid native records | 3 requests /6 draws /34 returned tokens, captured and raw_durable true; integer/string/repeated IDs retained separately via ordinal. Requested settings and caller SamplingParams remained unchanged. | `recordings/valid/{request.json,raw.jsonl,capture.json}` |
| EOS with null stop_reason | A native stop completion ending at EOS1 with stop_reason:null is accepted; no literal-text-suffix requirement. | valid raw ordinal0/completion1 |
| Developer overflow parse | Synthetic `ANSWER:1e999` becomes **parser_error**, not a numeric zero/correct answer:5 parsed +1 error,all_parsed:false. | `parse-5a91b9ae4bdd4f52bf515e619ccb15cc.jsonl` + summary under valid/ |
| Interrupted parse | KeyboardInterrupt propagates; zero-byte prefix and failure record retained; **no success summary**. Raw hash unchanged. | `parse-994347290c854c87965269bcbcfd09d0.{jsonl,failure.json}` |
| Recovery parse | New pass sees all6 raw draws; callback explicitly returns `status:nonfinite_answer,value:null` for overflow.6 callback outputs,all_parsed:true,official_score:false. | `parse-e4f42d9da0e74b40a9ce52ab43cce696.jsonl` + summary |
| Missing returned draw | Capture rejected:3 returned requests/5 draws/29 tokens **all retained**, raw_durable:true; no capture.json. | `recordings/missing_draw/` |
| Nonidentical prompt order swap | Capture rejected by token/order check; all3 requests/6 draws preserved; no capture.json. | `recordings/wrong_order/` |
| Abort completion | Capture rejected for unsupported/incomplete native finish reason; all3 requests/6 draws preserved. | `recordings/aborted/` |
| Parse invalid capture | Parsing refused in all three cases; generic FileNotFoundError for absent capture.json, no manufactured valid comparison. | transcript and capture-failure.json files |
| Reuse recording directory | FileExistsError before overwrite; every existing file/hash unchanged. | `transcript.md`, valid/ |
| Static token mistakes | Immediate double BOS, unknown stop spelling and multi-token required stop each rejected by SamplingEvidenceError before any intercepted call. | `transcript.md` |

`raw.jsonl` is request-granularity JSONL (3 records), not one line per draw. Structural counts above count nested completions. Every failed post-return capture has `capture-failure.json`, `scientific_validation:not_performed`, and counts matching its durable returned evidence.

All three parser passes leave raw SHA256 unchanged: **`508a69f820513aac07e4507bb90fd1e7ff6ad597bd44bc48b2b19cb40c4fca3a`**. Parser provenance retains source name/module/path/hash and explicitly marks closure/dependencies unverified. `all_parsed` indicates callback completion, **not valid finite answers or accuracy**; the recovered nonfinite-answer row remains explicitly unusable numerically.

## Gaps and scope boundaries

- **Small usability gap:** `record_vllm(..., /new_parent/leaf, ...)` raised FileNotFoundError when `/new_parent` was absent, before any engine call. Creating the owned parent fixed it without script/lock changes. Clarify “create the parent; leave the new recording leaf absent.” This is not raw loss or a scientific false verdict.
- Invalid-capture parsing fails safely but via missing-file error rather than pointing to capture-failure.json; a user-facing diagnostic could be clearer. No implementation change is required by this review.
- Identical prompts remain distinguishable only through the documented native input-order contract; this test confirmed typed IDs/ordinals are preserved, **not** that an order-violating engine could be detected when prompt token sequences are identical.
- Metadata explicitly leaves resolved engine configuration unverified. Actual checkpoint weight identity, resolved EOS defaults, RNG/kernel behavior, real generate ordering, process teardown, in-engine partial-output recovery, disk faults, live-source tampering, and unsupported objects were **not tested**. No claim about real inference completeness/performance follows.
- Native validations were left active; this is not a malicious-bypass/security audit. The locked command/source checks passing with the intentional inert call do not certify real model execution.

## Source identity and reproduction

All four worktree hashes were unchanged between pre/post review:

| Worktree file | SHA256 |
|---|---|
| `skills/exp_protocol/sampling-evidence.md` | `029d9de2bf16363b47e688b617c58394211818df2085e0df9a1ce023783747cb` |
| `skills/exp_protocol/SKILL.md` | `c5e57bb1061caf7fdf0d50832cc16f5f96feba4c222d3264eb12c8ad8fd385fa` |
| `skills/exp_protocol/save-safety.md` | `d915e5044aba9e0e3a7ceee51ddcacbdc01124c571248a097f90bbddfc2c4857` |
| `awm/exp_protocol/sampling.py` | `7f867012d93650980254fc3dad72ecf991ab60d16ab519a416795c1de3d4624e` |

Owned `workflows.py` SHA256 **`81737a766713466c35a1b59fcecdb144dce98b1cf1b7a2ef76182aaa2fd17dba`**; frozen cases **`72b054ef654637967cb14e9a5134c7a39230b835b30e3a3dba0284ae06e56534`**; lock **`b97ee2780f67bc2a8ff46658b8faeb686e4de5d8a1fdafeefeaa6f0107cdaaad`**. Native source hashes for `entrypoints/llm.py`, `outputs.py`, `sampling_params.py` are preserved in each `request.json.native.sources`.

Commands used: `bash inside.sh import_probe.py`; `bash inside.sh -m awm.cli exp_protocol check --dir /tmp/e6-forward-sampling.ntRxXE exp-01`; corresponding `lock`; `bash inside.sh /tmp/e6-forward-sampling.ntRxXE/workflows.py`. Paths are absolute in the real `inside.sh`. Existing recording directories are intentionally immutable: **do not expect rerunning over these paths to overwrite evidence**; prepare a new owned fixture and matching card/lock for a fresh run.

No core/runtime or E7 file was modified. This bounded forward review supports the documented raw-first/parser-recovery behavior on native-shaped CPU evidence only; it is not a promotion, GPU-launch authorization or scientific completion claim.
