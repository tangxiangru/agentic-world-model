# Opus4.8 GSM8K control review, 2026-09-05

Reviewer: Codex direct raw audit. The requested local Claude 2.1.260 background helper (`claude-opus-5[1m] --effort max`, plan, Read/Grep/Glob/Bash only) did **not** start: initial invocation returned EROFS creating `~/.claude/jobs/1a854a7b`; elevated attempt and the one permitted retry both timed out in automatic approval review before execution. No active session ID and no claim of Claude file reads. Do not repeat dispatch based only on these attempts. Review proceeded by selectively reading actual traces, retained scripts, JSONL row counts, structured Inspect logs, metrics, train logs and final configs.

Canonical receipt: `results/ptb/exp-protocol-opus48-gsm8k-none-x4-v1/formal-2026-09-04T125911.714295+0000.json`. Harvested bundles read from `/rmeng_data/robtang/exp-protocol-opus48-submit-ob4GmM8N/repo/results/ptb/exp-protocol-opus48-gsm8k-none-x4-v1/{cell}` (operator harvest commit 7704ce8 per planner). Raw source prefix: `data/ptb/results/claude_vertex_high_claude-opus-4-8_1m__10h_gangda_exp_protocol_evolve_exp-protocol-opus48-gsm8k-none-x4-v1_{cell}_formal_r1/gsm8k_google_gemma-3-4b-pt_{job}`. Every report's L references refer to that cell's **raw solve_parsed.txt**; times UTC.

| Cell/job | Official score | Session wall hours | Validity |
|---|---:|---:|---|
| n03g01 / 92125 | 59.5148% | 8.91 | clean complete |
| n03g02 / 92126 | 59.4390% | 8.48 | clean complete |
| n03g03 / 92127 | none | 0.790 | failed/truncated, missing final_model |
| n03g04 / 92128 | 70.4321% | 6.52 | clean complete |

Clean-session mean **63.1286%**, n=3, not n=4 and not number of inner training branches. Failure rate 1/4 separately. Scores are official metrics.json; intermediate values in reports are local diagnostic/selection evaluations, not official replacements.

## Corrections to generated facts and judge prose

Facts/timeline outputs are preserved unchanged. `first_eval` and `final_model_written` are regex hints that match read source and TODOs: the latter is **false in all four generated timelines**. Actual first fallback copies: n03g01 23:18:21, n03g02 05:27:03, n03g03 never, n03g04 01:08:12. The timeline regex misses n03g04's `train.py`; real smoke starts 22:46:12, first full train 22:47:57. Its detected first_rl is also not proof of RL: **none of these cells ran policy-gradient RL**. Waiting hours include productive background training and generation, and cannot be counted as idle GPU hours. n03g04 judge's opening '8.5h' contradicts both trace 22:36→05:07 and time_taken 06:31:16; use 6.52h.

## Decision-relevant findings

All three complete controls independently discover few-shot continuation/stopping failures and improve by changing training prompt context. All ship temperature=0, do_sample=false; recorded evaluator requests omit temperature, leaving model defaults to supply it. n03g01 and n03g04 measure ~9.3 points decoding benefit with same weights on n=150. Repeated final scores still vary slightly, so do not promise bitwise deterministic accuracy from greedy config.

The lower scores versus historical Opus5 cannot be assigned causally to scientist model. Inner recipes differ radically: the two low cells use small GSM8K/RFT corpora, eager attention and BF16-loaded multimodal model paths; the higher cell uses explicit text-only Gemma3, flash attention, 67,473 rows of teacher/gold, and higher LR. All three load BF16 weights, so precision is **not** isolated by the 59%/70% split. Effective parameter/optimizer update diagnostics remain a good short controlled test, not an explanation already proved. The first two sessions finish with 1.1–1.5h unused; n03g04 finishes 3.5h early and incorrectly calls ~70% a capability ceiling.

Matched-ID opportunity counts (scores reconstructed from structured `samples[].id`, not array offsets):

| Cell comparison | Common n | A correct | B correct | A-only | B-only |
|---|---:|---:|---:|---:|---:|
| n03g01 v3 / v5 | 250 | 154 | 154 | 18 | 18 |
| n03g02 exp4 / exp7 | 500 | 262 | 304 | 47 | 89 |
| n03g04 iter2 / iter3 | 1319 | 918 | 922 | 127 | 131 |
| n03g04 iter3 / RFT | 1319 | 922 | 893 | 148 | 119 |

These reveal complementary errors, **not realizable ensemble gains**. Test IDs/answers must not become a selector training set. Full-test reuse makes these discovery observations, not independent confirmation.

## Minimal next work that can change decisions

1. CPU scaffold lifecycle regression with a fake long background task: entering an assistant end_turn while an active scientist child exists must not silently discard it and declare success. Also check final artifact readiness before accepting completion. This is directly tied to n03g03, and need not consume a GPU research repeat. Preserve its failed receipt; replacement gets new provenance.
2. A short fixed-data precision diagnostic (same initialization/batches/LR, BF16 direct versus FP32 parameters with BF16 compute), measure weight deltas, zero-update fraction and fixed diagnostic loss before spending full sessions. All current controls use BF16; no claim that FP32 will win.
3. Within a fixed teacher/gold recipe, vary few-shot training context while holding token exposure and training work as closely matched as possible. Record actual rendered prompt, labels, terminal token and stopping. The current +8.4-point exp4→exp7 observation also changes corpus and length, so it is not a pure context estimate.
4. For complementary RFT/teacher routes, compare actual fixed mixtures or continuation branches on fresh prespecified validation data, including generation time. Do not mandate RFT or model soup in protocol. n03g04 RFT adds 119 successes but loses 148; a useful branch must beat a cost/integrity or realizable quality criterion.

Process transfer: actual effective decode and stopping checks; scoped task lifecycle ownership and valid deliverable check; incremental generation persistence; report full-test reuse and sample IDs. Strategy precedents: few-shot format matching, teacher/gold diversity, RFT complementarity remain conditional options. Existing E package coverage must be checked against its frozen implementation by planner before proposing duplicate checks.
