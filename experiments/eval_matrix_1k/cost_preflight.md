# Cost preflight: seven illustrative saved runs

Read-only, proposal-stage check. No GPU run, inference API call, download, or full-corpus decompression was performed. Only top-level metadata from seven already-local Inspect logs was read.

## Exact completion budget

The proposed allocation is **480 GSM8K cells + 520 AIME cells** (AIME: 160 each of A01/A02/A03, plus 20 each of A04/A05), each with ten full benchmark passes:

- GSM8K: 480 × 1,319 × 10 = **6,331,200 completions**.
- AIME: 520 × 30 × 10 = **156,000 completions**.
- Total: **6,487,200 completions**.

A GSM8K cell has about 44 times as many completions as an AIME cell. This does **not** establish its relative GPU cost.

## Observed tokens from individual representatives

These logs expose `stats.model_usage.<model>.input_tokens/output_tokens/total_tokens`. Their `results.total_samples` and `completed_samples` match the full ten-pass denominators used below.

| Nominal policy | Representative checkpoint | Completions | Input tokens, whole cell | Output tokens, whole cell | Mean input tokens/completion | Mean output tokens/completion |
|---|---|---:|---:|---:|---:|---:|
| G01 | [gsm2-r0-01-exp-01](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/trajectories/gsm2-r0-01-exp-01.json.gz) | 13,190 | 29,105,870 | 1,761,247 | 2206.7 | 133.5 |
| G02 | [r0-29-exp-02](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/trajectories/r0-29-exp-02.json.gz) | 13,190 | 29,105,870 | 1,931,309 | 2206.7 | 146.4 |
| A01 | [aime-r0-07-exp-01](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/trajectories/aime-r0-07-exp-01.json.gz) | 300 | 83,020 | 3,665,406 | 276.7 | 12218.0 |
| A02 | [aime-r0-01-exp-01](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/trajectories/aime-r0-01-exp-01.json.gz) | 300 | 83,020 | 3,754,036 | 276.7 | 12513.5 |
| A03 | [aime-r0-07-exp-02](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/trajectories/aime-r0-07-exp-02.json.gz) | 300 | 83,020 | 613,274 | 276.7 | 2044.2 |
| A04 | [aime-r0-04-exp-02](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/trajectories/aime-r0-04-exp-02.json.gz) | 300 | 83,020 | 3,329,539 | 276.7 | 11098.5 |
| A05 | [aime-r0-04-exp-04](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/trajectories/aime-r0-04-exp-04.json.gz) | 300 | 83,020 | 4,196,098 | 276.7 | 13987.0 |

G01/G02 here have about **134–146 output tokens per completion**. The long-budget AIME examples have about **11,098–13,987**, and the short-budget A03 example about **2,044**. These are observations from **one checkpoint per listed policy**, not estimates of the corpus mean or causal policy effects.

Despite far fewer questions, these long-AIME examples produce **3.3–4.2 million output tokens per cell**, compared with **1.76–1.93 million** for the two GSM examples. GSM has much more input-token volume: approximately **29.1 million per cell**, versus **83,020** for these AIME cells. Prefill versus decoding work, prompt caching, concurrency, sequence length, and checkpoint behavior prevent a token-total ratio from being a GPU-hour estimate.

## Historical policy uncertainty still applies

These policy labels are the source-derived config categories from [config_audit.md](config_audit.md), not independently verified server-resolved settings.

All seven client logs request max_tokens=16,000 for AIME or 4,000 for GSM8K. Notably, A03's client request is **16,000**, although its generation file has max_new_tokens=2,048 and its mean observed output length is 2,044.2. This is consistent with inherited capping, but does not substitute for capturing actual resolved max_tokens.

The two nominal temperature-zero cases still have the unresolved repeat-variation issue documented in [runtime_preflight.md](runtime_preflight.md). Their observed costs cannot validate a future explicit-policy implementation.

## What the proposed 100-cell pilot must measure

Treat the pilot as **part of**, not additional to, the 1,000-cell budget. Cover all scheduled policy types and diverse checkpoints; do not make it a convenience sample of short or already-fast runs.

First pass the resolved-policy/tokenizer/artifact launch gate. Then measure actual input/output token distributions, stop-versus-length termination, errors/retries, loading/warm-up time, sustained throughput, peak memory, concurrency/cache behavior, and wall-clock/device utilization on the intended hardware. Record costs by checkpoint and policy rather than assuming every cell is interchangeable.

Use those measurements to estimate the remaining workload and identify unexpectedly long or pathological cells **before releasing the remaining 900**. A pilot comprising 10% of cells need not consume 10% of compute. This small historical sample supplies no defensible GPU-hour or dollar forecast.
