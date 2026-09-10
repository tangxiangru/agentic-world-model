# Corpus-derived generation-policy audit for the 1,000-cell design

Proposal only: no GPU evaluation, weight download, or source trajectory modification was performed.

## Outcome

Use **two dominant GSM8K policies** and **three dominant AIME policies** as the cheapest support-grounded common core. Those sampling/cap signatures occur in **491 of 579 eligible checkpoint records (84.8%)**. A broader seven-policy set covers **527/579 (91.0%)**. These are source-derived categories, **not behaviorally proven distinct performance groups**.

The five-policy core is G01/G02 and A01/A02/A03 below. G03 and A04 are supported extensions. A05 is a lower-prevalence, mechanism-driven greedy probe. This audit does not choose checkpoints or commit to a fixed allocation; the final 1,000-cell manifest is separate.

All frequencies below count checkpoint records, **not deduplicated weight hashes**. Shared weights and within-session dependence can reduce independent support.

## Source and eligibility

- Pinned metadata: HF dataset `JerrrrryL/awm-gsm8k-trajectories`, revision `446127629d7b271d537390e69bfb2d960a3aa515`.
- Labels: external v6 labels `data/analysis/wm_exp_designs/prefix_recipes_v6/labels.jsonl`, SHA256 `d16638674fd76590f2939fb06700a319b8cc8409bc3761e9f79bbf01fd7b862b`.
- Retain `output.status == complete`, then exclude the documented quarantines `r0-25-exp-02`, `aime-r0-11-exp-02`, and `aime2-r0-12-exp-05`.
- Result: **579 = 326 GSM8K + 253 AIME**. All 579 generation files were present and valid JSON. No accuracy values were used for this prevalence-based selection.
- This provisional eligibility is not a substitute for exact weight, tokenizer, and recipe/checkpoint binding checks.

[config_audit.json](config_audit.json) includes every ranked signature, all supporting exp_ids, trajectory counts, raw field-value counts, complete raw-config prevalence, and representative raw JSON with source path/hash.

## Candidate policies

Every listed policy has min_p=0. Unfiltered means top_k=0 and top_p=1. Each cell runs **ten complete benchmark passes**, reporting mean pass@1, not pass@10.

| policy_id | Benchmark | Temperature | top_k | top_p | Repetition penalty | Nominal token cap | Existing sampling/cap support | Trajectories | Role | Representative raw file |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| G01 | GSM8K | 1 | 64 | 0.95 | 1 | 4000 | 192/326 (58.9%) | 47 | dominant core | [gsm2-r0-01-exp-01](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/blob/446127629d7b271d537390e69bfb2d960a3aa515/checkpoints_meta/gsm2-r0-01-exp-01/generation_config.json) |
| G02 | GSM8K | 0 | 0 | 1 | 1 | 4000 | 110/326 (33.7%) | 24 | dominant core | [gsm2-r0-14-exp-03](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/blob/446127629d7b271d537390e69bfb2d960a3aa515/checkpoints_meta/gsm2-r0-14-exp-03/generation_config.json) |
| G03 | GSM8K | 1 | 0 | 1 | 1 | 4000 | 14/326 (4.3%) | 5 | supported extension | [gsm2-r0-18-exp-01](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/blob/446127629d7b271d537390e69bfb2d960a3aa515/checkpoints_meta/gsm2-r0-18-exp-01/generation_config.json) |
| A01 | AIME | 1 | 0 | 1 | 1 | 16000 | 72/253 (28.5%) | 23 | dominant core | [aime-r0-07-exp-01](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/blob/446127629d7b271d537390e69bfb2d960a3aa515/checkpoints_meta/aime-r0-07-exp-01/generation_config.json) |
| A02 | AIME | 0.6 | 20 | 0.95 | 1 | 16000 | 63/253 (24.9%) | 25 | dominant core | [aime-r0-01-exp-01](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/blob/446127629d7b271d537390e69bfb2d960a3aa515/checkpoints_meta/aime-r0-01-exp-01/generation_config.json) |
| A03 | AIME | 1 | 0 | 1 | 1 | 2048 | 54/253 (21.3%) | 20 | dominant core | [aime-r0-07-exp-02](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/blob/446127629d7b271d537390e69bfb2d960a3aa515/checkpoints_meta/aime-r0-07-exp-02/generation_config.json) |
| A04 | AIME | 0.6 | 20 | 0.95 | 1.05 | 16000 | 22/253 (8.7%) | 9 | supported extension | [aime-r0-04-exp-02](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/blob/446127629d7b271d537390e69bfb2d960a3aa515/checkpoints_meta/aime-r0-04-exp-02/generation_config.json) |
| A05 | AIME | 0 | 0 | 1 | 1 | 16000 | 8/253 (3.2%) | 4 | mechanism probe | [aime-r0-04-exp-04](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/blob/446127629d7b271d537390e69bfb2d960a3aa515/checkpoints_meta/aime-r0-04-exp-04/generation_config.json) |

The representatives establish the observed sampling/cap signatures, **not** exact equivalence to the planned standardized EOS policy.

### Why these policies

The dominant-core screen is at least **50 records and 20 trajectories**; it lands at the observed prevalence breaks. GSM support falls from 110 to 14 after its top two; AIME falls from 54 to 22 after its top three. The broader screen is at least **10 records and three trajectories**. These are transparent design thresholds, not statistical significance tests.

- **G01 versus G02** tests the prevalent Gemma sampling policy versus greedy across the same weights.
- **A01 versus A03** isolates the common 16,000-versus-2,048 token-cap difference.
- **A01 versus A02** compares two common complete sampling policies; it does not isolate temperature from top-k/top-p.
- **A02 versus A04** isolates the best-supported non-default repetition penalty, 1.05.
- **A05** adds greedy AIME coverage on a selected subset without pretending it is a high-prevalence policy.

Do not substitute the earlier illustrative grid without recognizing its lack of exact support: GSM T=.3/k64/p=.95 has **zero** eligible matches; AIME T=.6/k20/p=.95/cap2048 has **zero**. All 54 common AIME short-cap records use **T=1 with no filtering**. AIME T=.6/k20/p=.95/rep1.1 has only **three** records, versus **22** for rep1.05.

## Coverage and concentration

| Benchmark | Raw JSON configs | Five-field sampling signatures | Sampling + nominal cap | Plus normalized config-EOS contribution |
|---|---:|---:|---:|---:|
| GSM8K, 326 records | 29 | 11 | 11 | 14 |
| AIME, 253 records | 75 | 18 | 20 | 29 |

There are **104 distinct raw JSON configs** overall. The rightmost column still is not a full effective-serving count: tokenizer EOS and prompt-dependent limits are not resolved here.

| Benchmark | Top policies | Covered records | Coverage |
|---|---:|---:|---:|
| GSM8K | 2 | 302/326 | 92.6% |
| GSM8K | 3 | 316/326 | 96.9% |
| AIME | 3 | 189/253 | 74.7% |
| AIME | 4 | 211/253 | 83.4% |
| AIME | 6 | 227/253 | 89.7% |
| AIME | 7 | 231/253 | 91.3% |

The seventh-ranked AIME variant is rep1.12, supported by only one trajectory. Chasing an arbitrary 90% cutoff therefore introduces a weakly distributed tail policy; it is not automatically better than the smaller common core. Full cumulative curves are in the JSON.

Raw files can look different without different imported sampling settings. The two largest GSM raw variants have **116** and **73** records and differ only by an extra duplicate EOS ID (`[1,106]` versus `[1,1,106]`). Both map to G01. Complete raw-config counts and representative IDs are retained in the JSON rather than discarded.

## Standardize the full serving policy

Use one **intended effective stop-token set per benchmark**:

- GSM8K: `[1,106]`.
- AIME: `[151643,151645]`.

These union sets occur directly in **321/326** GSM and **173/253** AIME generation files after sorting/deduplication. Other config-only sets are GSM `[106]` (5), AIME `[151643]` (41), and AIME `[151645]` (39). A tokenizer's primary EOS may already make some of these equivalent; this audit does not assume which ones.

The point of fixing stops is to remove a hidden serving difference from the predictor PoC. Standardizing a historical single-EOS configuration to the union is an explicit **new intervention**, not an exact historical replay. Freezing each checkpoint's own differing EOS would fail to make the serving policies common.

Before launch, pin the tokenizer artifact/hash and explicit benchmark prompt/chat template, verify vocabulary/token-ID compatibility with each checkpoint, and require the primary tokenizer EOS to be inside the intended union. Do not silently swap an incompatible tokenizer. BOS/padding/template tokenization also needs checking.

Each JSON policy entry supplies both an intended generation-config payload and a complete OpenAI-compatible request payload. Preferred execution: disable inherited generation-config loading with `generation_config=vllm`, then explicitly supply sampling settings, request max_tokens, stop_token_ids, and ignore_eos=false. Disabling generation-config loading also removes its EOS contribution, which is why explicit stop IDs are required.

An explicit request cap of 16,000 **does not defeat an inherited generation-config cap of 2,048**. Either disable inheritance or fully replace the generation configuration, and record the actual request/context cap. vLLM takes the minimum of request, model-default, context-capacity, and platform limits. [vLLM cap handling](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/utils.py)

For predictor feasibility under this fixed protocol, do not add an EOS sweep merely to cover raw-file diversity. If the eventual target includes varying stop policies, that becomes a separately declared generalization test.

## Normalization evidence and caveats

The actual-fleet [trajectory README](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories/resolve/cc2ac9d884a7d962a6024ab0d5cd8ed3370070de/rescore10/trajectories/README.md) reports **vLLM 0.11.0, Transformers 4.57.3, Inspect 0.3.150**; the kit's Transformers 5.14.1 is not the scored runtime.

Inspect 0.3.150 starts `vllm serve` through its OpenAI-compatible provider, and sends temperature/top_p only when explicitly set. The eval scripts set the benchmark token cap but do not explicitly set sampling fields. This interpretation assumes no unrecorded environment/server overrides. [Inspect provider](https://github.com/UKGovernmentBEIS/inspect_ai/blob/0.3.150/src/inspect_ai/model/_providers/vllm.py), [request construction](https://github.com/UKGovernmentBEIS/inspect_ai/blob/0.3.150/src/inspect_ai/model/_openai.py)

The importer reads five sampling parameters plus max_new_tokens, after Transformers `to_diff_dict()`; EOS is applied separately. Missing/null fields use vLLM defaults T=1, k=0, p=1, min_p=0, repetition penalty=1. **do_sample is not imported as a sampling switch.** In Transformers 4.57.3, explicit top_k=50 equals the HF default and disappears from the diff, leaving vLLM's k=0 fallback. [vLLM import](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/config/model.py), [HF normalization](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/generation/configuration_utils.py)

Temperature zero neutralizes top-k/top-p/min-p; positive temperatures below .01 clamp upward before the greedy check. top_k=-1 and 0 both disable filtering. Config EOS is combined with primary tokenizer EOS, so config-only EOS counts are deliberately separate from fully effective stops. [Sampling and EOS handling](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/sampling_params.py), [tokenizer EOS source](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/inputs/preprocess.py)

Other observed generation-JSON keys not imported by this sampling/EOS path are bos_token_id, pad_token_id, cache_implementation, max_length, min_new_tokens, min_tokens, and transformers_version. This does not make independently loaded tokenizer/model metadata irrelevant.

## Historical labels and expensive replay cells

The dominant core has 491 nominal historical sampling/cap matches; the broader set has 527. **Neither count is a verified reusable-label count.** Exact weights, tokenizer/effective stops, prompt/context protocol, runtime/server overrides, and ten-pass evaluation must all match before reuse.

Do not spend cells automatically replaying every baseline. Verify reuse eligibility first; use a small representative protocol-check subset where needed. A genuinely new or unverified replay consumes one checkpoint/config cell and receives a new result. Keep old and new labels distinguishable.

The fixed budget is **1,000 checkpoint/config cells × ten benchmark passes**. GSM cells require 13,190 answers versus AIME's 300, before output-length differences. Existing valid labels should be retained as evidence without disguising them as newly run cells. Checkpoint choice, family/trajectory-held-out splits, deduplication, and the final allocation remain the parent design's responsibility.
