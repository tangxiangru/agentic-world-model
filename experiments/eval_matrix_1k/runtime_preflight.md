# Historical runtime preflight: temperature-zero files do not prove resolved greedy mode

Status: **launch gate remains open / not passed**. Read-only inspection of two existing local logs; no GPU evaluation, API inference call, or download. The policy inventory JSON is unchanged.

## What is established

The pinned uploaded generation files for `r0-29-exp-02` and `aime-r0-04-exp-04` both contain `temperature: 0` and `do_sample: false`. However, every saved client request in their ten-pass logs **omits temperature, seed, top_p, top_k, and extra_body**. The client delegated sampling defaults to the server.

Both eval scripts likewise provide max_tokens, concurrency, timeouts, model path, GPU-memory utilization, and a selected chat-template path—not explicit sampling settings. The template helper only supplies chat_template. Actual logged generation configurations and generate-step parameters agree with that code.

This removes an explicit client-side temperature/seed override as an explanation **in the recorded calls**. It does not expose or verify the server's resolved SamplingParams, model-default import, startup environment overrides, exact loaded generation-file hash, or primary tokenizer EOS.

## Variation with unchanged request messages

The scan hashed complete request-message JSON separately for every problem and repeat; it did not use textual answer contents in this report.

| Checkpoint | Samples | Problems | Request cap | Max connections | Problems with changing request messages | Problems with changing completion text | Problems changing correctness | Mean pass@1 | Epoch SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r0-29-exp-02 | 13190 | 1319 | 4000 | 64 | 0 | 668 | 92 | 68.802% | 0.228 pp |
| aime-r0-04-exp-04 | 300 | 30 | 16000 | 48 | 0 | 30 | 10 | 15.000% | 5.821 pp |

Each log has exactly one model event per sample and ten full epochs. “Epoch SD” is the population standard deviation of the ten complete-run accuracies; it is not a binomial standard error.

The GSM8K prompt includes ten few-shot examples with fewshot_seed=42 and shuffle_fewshot=true. Despite that task option, **the actual request messages were identical across the ten epochs for every individual problem**. AIME request messages were also identical per problem.

The AIME scorer is `aime_scorer`, not `match`; treating a missing `match` key as incorrect would incorrectly report zero accuracy. This audit explicitly used each task's actual scorer.

## Exact metadata checked

### r0-29-exp-02

- Source: [saved Inspect log](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/trajectories/r0-29-exp-02.json.gz), SHA256 `f0dd65d9cd2037e5bd6b98cd12562a8ab05f0acc24245a0809ae712a7c156b83`.
- Eval timestamp: `2026-09-06T20:41:34+00:00`.
- Model: `vllm//mnt/localssd/wm-eval/rescore10/r0-29-exp-02/model`.
- Recorded model_args: `{"gpu_memory_utilization":0.85,"chat_template":"../../templates/gemma3.jinja"}`.
- Recorded model_generate_config: `{"timeout":18000000,"attempt_timeout":18000000,"max_connections":64,"max_tokens":4000}`.
- Recorded package: `inspect_ai=0.3.150`; the log's packages map does not list Transformers/vLLM.
- All 13190 model events have the same parameter signature: `{"max_tokens":4000,"model":"/mnt/localssd/wm-eval/rescore10/r0-29-exp-02/model","tool_choice":null}`, excluding messages and per-call request-ID headers.
- Requests containing temperature, seed, top_p, top_k, or extra_body: **0 for each field**.
- Plan's generate step parameters: `{}`; its generation config matches the listed model_generate_config.
- Scorer: `match`. Recomputed per-epoch correctness matches the corresponding rescore results JSON exactly.
- Finish reasons: `{"stop":13143,"length":47}`.


### aime-r0-04-exp-04

- Source: [saved Inspect log](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/trajectories/aime-r0-04-exp-04.json.gz), SHA256 `11ef09ee39a49bebe3bfc3276377be17ae0a6303ee1d9f3037e9ca489206643f`.
- Eval timestamp: `2026-09-06T23:26:00+00:00`.
- Model: `vllm//mnt/localssd/wm-eval/rescore10/aime-r0-04-exp-04/model`.
- Recorded model_args: `{"gpu_memory_utilization":0.85,"chat_template":"../../templates/qwen3.jinja"}`.
- Recorded model_generate_config: `{"timeout":18000000,"attempt_timeout":18000000,"max_connections":48,"max_tokens":16000}`.
- Recorded package: `inspect_ai=0.3.150`; the log's packages map does not list Transformers/vLLM.
- All 300 model events have the same parameter signature: `{"max_tokens":16000,"model":"/mnt/localssd/wm-eval/rescore10/aime-r0-04-exp-04/model","tool_choice":null}`, excluding messages and per-call request-ID headers.
- Requests containing temperature, seed, top_p, top_k, or extra_body: **0 for each field**.
- Plan's generate step parameters: `{}`; its generation config matches the listed model_generate_config.
- Scorer: `aime_scorer`. Recomputed per-epoch correctness matches the corresponding rescore results JSON exactly.
- Finish reasons: `{"stop":48,"length":252}`.


## Interpretation: unresolved, not “sampling proven”

Observed text variation at nominal T=0 is not by itself proof that sampling was enabled. Candidates include a different loaded/default configuration, unrecorded server-level overrides or artifact mismatch, and runtime/batching/numerical nondeterminism even with greedy selection. This bounded audit does **not** identify which cause applies.

No server-resolved generation parameters appear in the inspected model-event config or saved request fields. Therefore retain the corpus categories as **source-derived nominal policies**, not fully verified historical effective modes. The first policy audit's conditional source-code interpretation remains conditional.

The actual-fleet [README](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/trajectories/README.md) identifies vLLM0.11.0/Transformers4.57.3. These two logs independently record Inspect0.3.150, but do not themselves establish the other installed package versions.

## Required checks before spending the 1,000-cell budget

1. Record the launched server command, relevant override environment, exact installed package versions, model/weight digest, tokenizer digest, generation-config file digest, and explicit chat-template/tokenization settings.
2. Avoid inherited ambiguity: explicitly set the complete candidate sampling/cap/stop policy. When using `generation_config=vllm`, restore the planned stop_token_ids explicitly because the generation-file EOS contribution is disabled.
3. Capture the **server-resolved** SamplingParams, including sampling mode, temperature, top-k/top-p/min-p, repetition penalty, seed, max_tokens, stop IDs, ignore_eos, primary tokenizer EOS, and generation-config/override provenance. Client request JSON alone is insufficient.
4. Verify actual per-prompt output caps after request, inherited model defaults, context-length, and platform limits. A request cap alone is not proof of the effective cap.
5. Require resolved-policy assertions to pass before a full benchmark launch. If a small approved pilot is needed to assess T=0 repeatability, compare fixed prompts under controlled batching and record the policy; do not infer mode solely from completion differences. Pilot allocation must be explicit within the budget.
6. Reuse historical labels only when full artifact/protocol equivalence can be established. These two metadata checks do not grant that equivalence.

## Local code inspected

- [GSM8K evaluate_epochs.py](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/eval/tasks/gsm8k/evaluate_epochs.py): argument defaults, main call, and template_kwargs.
- [AIME evaluate_epochs.py](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/eval/tasks/aime2025/evaluate_epochs.py): corresponding argument/main/template code.
- The two saved Inspect logs and their matching result JSON files were read locally; no trajectory answer text is copied into this artifact.
