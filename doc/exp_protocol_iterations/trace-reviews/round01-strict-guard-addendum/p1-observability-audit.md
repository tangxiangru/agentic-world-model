# P1 predicate and early-observability audit — 2026-09-03

Additional read-only reviewer `/root/strict_guard_audit`; original Opus synthesis remains unchanged. Scope: strict g01s03 exp-05 and g01s08 exp-06, with g01s06 exp-03 as a boundary case. These are evidence for investigating P1, **not two validated, immediately executable step-20 stops**.

All bundles are under `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/`, joined by the formal receipt `formal-2026-09-02T204221.237369+0000.json`, strict manifest of the same batch, and Round01 session-guard spec. Trace L-lines below refer to each cell's uncompressed `solve_parsed.txt.gz`.

## Literal frozen text, not a reconstructed predicate

`git show a4c4954:skills/exp_protocol/pitfalls.yaml` describes a first-20-step loss starting within **about 0.05** of the parent's final loss and not moving. Neither tolerance is numerically defined. The entry does **not** literally exclude all replay mixtures. Matched n≥500 outcomes appear in the retrospective symptom, not as information available before a step-20 stop. Its first-round sentence describes successful historical examples, not an explicit blanket exception; both claimed positive cases are themselves first RFT rounds from SFT-only parents.

The synthesis's “full frozen predicate” adds sharper exclusions than the shipped text contains. Do not use those additions to claim that the original stop policy is safe.

## Retained evidence

| field | g01s03 exp-05 / job90793 | g01s08 exp-06 / job90798 |
|---|---|---|
| sampled/trained parent | `ckpts/exp-04/final`; sampler L6071, first launch L6349, repaired relaunch L6614 | `ckpts/exp-04/final`; sampler L6557, train L6893 |
| targets | 24,990 accepted self targets; replay rejected; 20% few-shot prompts | 40,000 selected accepted self targets; about35% prompt re-rendering, not teacher-target replay |
| parent terminal / whole-run mean loss | **0.2481** at step3580/3582 / 0.2767496861 | **0.2515** at step370/375 / 0.2528312098 |
| child step10 / step20 loss | **0.2043 / 0.1952** | **0.2241 / 0.2200** |
| step10 / step20 grad_norm | **3.421875 / 2.46875** | **3.859375 / 1.8828125** |
| matched n=500 parent / child | **359 / 360 correct** | **374 / 360 correct** |

Sources include g03 `task/logs/exp-04_train.log:364`, `exp-05_train.log:7`, `task/memory/cards/exp-05.yaml`; g08 `task/logs/exp-04_train.log:41`, `exp-06_train.log:4`, `task/memory/cards/exp-06.yaml:44`, and `task/mix_prefix.py:54` (completion retained, prompt labels masked).

Both trainers log every10 optimizer steps. The synthesis's third/fourth losses are **steps30/40**, outside the specified first20; step1 is not retained. The decreases of0.0091 and0.0041 may be roughly flat, but no frozen operational tolerance decides that. Nonzero gradients refute a literal zero-gradient account; low loss alone does not establish useless updates.

Both matched comparisons used greedy decoding and `max_connections=16`. g03's successful comparisons were sequential after a failed concurrent attempt (`task/scripts/run_eval500.sh:3`). g08's parent is exp-04 through a greedy symlink (L6222–6230), with the same decode config copied to its child (L7064). **+1/500 does not prove equivalence; −14/500 is an adverse point estimate, not independently established significant regression.** Neither proves performance-neutral stopping.

## The first20 records were not necessarily available at step20

| successful run | step20 elapsed | train_runtime | ideal remaining loop time |
|---|---:|---:|---:|
| g03 repaired exp-05 | 78s | 3779.5775s | **1.0282h** |
| g08 exp-06 | 154s | 5596.1398s | **1.5117h** |

The ideal total is **2.5399h**, not2.60h, and excludes initialization, earlier sampling, g03's failed attempt/repair, final save and post-exit idle. It also assumes timely observations, which these logs do not demonstrate.

The merged logs show their first loss blocks after progress **781/1562 [31:35]** (g03) and **870/1250 [1:05:22]** (g08), consistent with stdout buffering. In g08 the trace corroborates the delay: a loss grep after30min (L6943) returns no losses despite reaching step439 (L6973–6978). At those apparent first flushes only approximately **0.5235h + 0.4650h = 0.9885h** of training remains, before reaction latency. These are timing bounds, not demonstrated safe savings. An early-stop screen must establish when its metric actually became observable, not merely what optimizer step an eventual line labels.

## Profitable boundary case: g01s06 exp-03

This first RFT round sampled/trained exp-02, with 57,845 self rows +25,000 replay rows (79,792 rows after length filtering). Parent terminal loss **0.2691**, mean **0.2911505855**; child steps10/20 **0.2109/0.2126** give gaps **0.0582/0.0565**, close to the fuzzy “about0.05” wording. Matched n=150 rises **0.6333→0.7000** (24 fixed/14 broken). Later0.733@n1000 is absolute, not a matched parent gain: no parent@n1000 was found.

This is a false-stop boundary worth preserving, not a proven falsifier or something literally “excluded twice.” Replay exclusion and an exact0.05 cutoff were added by the synthesis. P1 remains unregistered; its observable timing, stop criterion and result semantics need calibration before testing the frozen v1 as written.
