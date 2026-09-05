# Round 01 base-check snapshot — 2026-09-02 18:57 UTC

This is an inflight provenance snapshot, not a terminal PTB result. While all
16 first-wave jobs were still RUNNING, the operator used a read-only Slurm
overlap step on each job's allocated node and read:

```text
/home/ben/task/memory/cards/exp-01.yaml
```

For every card, the table records the SHA-256 of the complete current card and
the frozen `result.measurements` entry. The receipt-backed job id is the
locator; `gangda-slurm-queue show JOB_ID` resolves it to the batch, cell,
manifest, spec, and result directory. Terminal harvesting must reproduce these
hashes or explain any later card mutation before this snapshot is used as
frozen ledger truth.

| arm | cell | job | n | accuracy | complete-card SHA-256 | diagnostic locator / observed termination failure |
|---|---|---:|---:|---:|---|---|
| WMA | `w01r01` | 90556 | 150 | 0.0667 | `2bfb00098713c475e1434c535a478db7426d5df4930fe1acf7089a5d3611c621` | `exp-01_base_dev150.json`; 70/150 hit max tokens; other completions continue into fabricated questions |
| WMA | `w01r02` | 90557 | 150 | 0.05333 | `60d6238d7387d52ac9b13d9915f31101d9b04599b37d8da6d35552f7d7585db8` | `base_dev150.json`; 72/150 hit max tokens; 128/150 contain the answer marker |
| WMA | `w01r03` | 90558 | 150 | 0.0733 | `947393d89d99176ca1e89ecef2c65aaa246179c60fc5fe3552b15004194fead0` | `exp-01_base_dev150.json`; 74/150 hit max tokens and invent later Q/A blocks |
| WMA | `w01r04` | 90559 | 150 | 0.0533 | `bf405e5d4e6a033d874289f9650a223bb6d1892cf2f2ff7ab10ee8b3364f4699` | `exp-01_base_dev150.json`; 92/150 hit the 1024-token cap; only 36/150 end in an answer line |
| WMA | `w01r05` | 90560 | 150 | 0.0867 | `3b7de8b67ea747cb834b624ef648c77e1a06b4528e0e57754f890af576f0c08a` | `exp-01_base_dev150.json`; 75/150 hit max tokens; first-answer accuracy is 0.1267 |
| WMA | `w01r06` | 90561 | 150 | 0.0400 | `e216cbb5b2686ca8d5da312854a1b01daa1d7a8730ba4220b4b900554510bcf7` | `base_dev150.json`; 78/150 hit max tokens; 133/150 contain an answer line but continue |
| WMA | `w01r07` | 90562 | 150 | 0.0600 stock; 0.0400 greedy | `91bbcfe6404ce7ced8160a6bab6c21a341af7b6f3b3957e9051c96c40f5fd993` | `base_stock_dev150.json`, `base_greedy_dev150.json`; greedy terminates 0/150 although first-answer accuracy is 0.293 |
| WMA | `w01r08` | 90563 | 150 | 0.0333 | `a2a56f9359890396d38f8092dcb43f50fa8beb6431a013ba6f07ccc2c178fa89` | `base_dev150.json`; 76/150 hit max tokens; only 28/150 end in an answer line |
| control | `c01r01` | 90564 | 150 | 0.0667 | `03160f6e60ad47522e0c68583bbbd9f601e0ac6a174a3c9b3540147e660823f1` | `exp-01_base_dev150.json`; 63/150 hit max tokens and the median completion is far longer than the reference |
| control | `c01r02` | 90565 | 150 | 0.0400 | `95e7d176bb65012e76e151737623fc171f7d963f769c7e2db097b767cfae19d9` | `exp-01_base_dev150.json`; 42.7% hit the cap and 20% have no answer line |
| control | `c01r03` | 90566 | 150 | 0.0800 | `3fc079d43f8a79395b3e46703d3d94782431a24d341230320ff3565fb2379b4e` | `exp-01_base_dev150.json`; 75/150 hit max tokens and 25/150 have no answer line |
| control | `c01r04` | 90567 | 150 | 0.0400 | `8a300763deaef8965103a5b5cc248775285c32ff34883572bc56fd2e38e74971` | `exp-01_base_dev150.json`; 65/150 hit max tokens; a separate greedy probe stops 0/250 |
| control | `c01r05` | 90568 | 200 | 0.0650 | `41fa7c1f033d926bf37131327c3b080cd0171df4a804a9a6a188b503661a2caf` | `exp-01_base_dev200.json`; 105/200 hit the 1024-token cap; first-answer accuracy is 0.125 |
| control | `c01r06` | 90569 | 150 | 0.0333 | `4f199fce4ae4588c2e4ae49cf9b319b7f6b3af7b3fc795854e656611a0a933ae` | `exp-01_base_dev150.json`; 90/150 hit max tokens; 133/150 contain the marker but continue |
| control | `c01r07` | 90570 | 150 | 0.07333 | `76515c6dc55c57e7b7893eb39e121ddfb516b48101d2fa6a3a81b140ac276e95` | `base_dev150.json`; 79/139 failures hit max tokens and 23 have no answer line |
| control | `c01r08` | 90571 | 150 | 0.05333 | `9ec4c49d12776232c9fa65e43ac844354a8e3b500eed69d78f506b77922b2b87` | `exp-01_base_dev150.json`; 73/150 hit max tokens and 10 stop without an answer line |

All 16 cards—and all 17 measurements because `w01r07` has stock and greedy
arms—fall in 0.0333–0.0867. Fifteen measurements use n=150; `c01r05` uses
n=200. Every card independently records the same dominant mechanism: missing
termination, token-cap hits, or an answer followed by fabricated extra Q/A
whose tail is consumed by the end-anchored grader. This supports a
mechanism-gated *format-floor* prior; it does not establish a universal numeric
constant for other checkpoints, templates, graders, or tasks.

