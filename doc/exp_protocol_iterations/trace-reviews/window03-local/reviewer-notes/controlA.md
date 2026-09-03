I'll start by reading the brief.
I have everything I need. Here are the three reports.

---

## Shared reading notes (apply to all three cells)

- **Thinking blocks are empty in all three traces** (`"thinking": ""` + signature only; 70/76/70 blocks, 0 with content). Every "the scientist reasoned…" claim below therefore rests on visible prose turns, Bash `description` fields, code comments, `RESULTS.md`/`README.md`, and the harness `result` footer — never on internal reasoning. Where I cannot see the reasoning I say so.
- **Two generated-fact fields needed correction.** `largest_eval_n` from the inspect-log ÷ 44 KB heuristic is wrong for c01r05 (reported 353; the exact stderr-implied n is 200) and slightly off for c01r04 (590 → 600) and c01r06 (1330 → 1319). I used the stderr-implied n, which is exact. `final_model_written` is a false positive for c01r05 (07:25Z) and c01r06 (07:15Z) — at both timestamps the string only appears inside a `TaskCreate` subject, not a write.
- **E was measured from the GPU monitor, not from wall-clock waiting.** `task/system_monitor.log` samples GPU memory every 60 s; I bracketed each interval where the producing process had demonstrably exited (memory → 0 MiB, or `ps`/`nvidia-smi` evidence in-trace) and ran to the next useful command. GPU-allocated share of session: c01r04 6.67/8.71 h (77%), c01r05 8.18/9.06 h (90%), c01r06 5.30/8.51 h (62%).
- No `exp_protocol`, `memory/cards`, lock, check, close, preflight, or Stop-hook string occurs anywhere in the three traces (grep = 0 hits). Cards 0/0/0, overrides 0, relocks 0, `pitfalls_cost_h` 0.00 — all three are pure controls. Q6 is answered in its control form.

---



---



---



---

## Cross-cell notes for synthesis and pending-block triage

**Candidate coverage of every ≥0.1 h loss in the block (my three cells, 12 losses, 9.1 h total):**

| candidate | cells | hours | note |
|---|---|---|---|
| **D** `parent_generation_config_valid` | c01r04 (1.57 h), c01r06 (1.20 h) | **2.77 h** | Both are the exact mechanism D guards: greedy config written into a dir later used as `--init`, `ValueError` at the Trainer's first save, all trained weights lost. c01r05 is the clean counterexample — same greedy intent, implemented as a sibling symlink dir + `export_final.py`, zero cost. This is the strongest single-mechanism signal in my block and it is already a frozen first-wave candidate. |
| **E** wait-on-process | c01r04 (1.27 h), c01r06 (2.45 h); c01r05 passes (0.25 h, max episode 0.15 h) | **3.72 h** | **2/3 of my controls fail the <0.15 h target, badly.** This is control-arm evidence about the baseline pool, not about the Round 01 guard, and I am explicitly not inferring guard saturation from it. But it does contradict the premise that the E target has no movable baseline left: Fable's window-02 finding of 5/5 baseline cells under 0.15 h does not replicate here. The discriminator is mechanical and visible: c01r05 ran evals in the foreground; c01r04 and c01r06 backgrounded runs and polled with fixed `sleep 660–1800`. c01r06 episode #2 also shows a GPU-memory heuristic is insufficient — a zombie engine held 66 GB while the trainer was dead. |
| **B** vLLM offline sampling | orphan engine in all three (c01r04 08:14, c01r05 10:03 + 10:38, c01r06 12:08 → 0.49 h); parser-inf in c01r05 (0.54 h); stop-ids-with-n>1 diagnosed in c01r05 | **≈1.1 h** | B v2's three named mechanisms all appear. One mechanism is **not** covered: `limit_mm_per_prompt={"image": 0}` on a multimodal Gemma-3 checkpoint silently produces gibberish with no error (c01r05, 0.96 h across two passes). Single cell → **observation**, not a proposal. |
| **A** decode / grader observable | c01r04 (missed the line in its own log), c01r05 (found it pre-training), c01r06 (clamp line worth +2.75 pp) | 0.4 h + 2.75 pp | A v2 is frozen; this is screen-observable evidence, not a re-cut. Two refinements the screen should read: (i) the exact string is *"Default sampling parameters have been overridden by the model's Hugging Face generation config"* — c01r04 grepped `default_sampling_params` and `Using default` and missed a line that was present in `eval_r1_greedy.log`; (ii) that line alone is **not** proof of greedy — it appears identically under sampling. The discriminating observable is the clamp warning *"temperature X is less than 0.01 … maxed it out to 0.01"* (400 occurrences per clamped eval in c01r06). |
| **P2** Gemma-3 logits OOM / 64 MB overlay | c01r04 (0.55 h), c01r05 (0.02 h), c01r06 (0.02 h) | 0.59 h | Three cells, three outcomes. c01r04 installed liger into the 64 MB overlay and later lost 0.52 h to the corrupted install; c01r06 hit the same wall, ran `df -h` in 25 s, uninstalled, and wrote its own sparse-head trainer. The recovery c01r04 eventually used (`uv pip install --target /home/ben/task/pylibs` + `PYTHONPATH`) is the one-line fix. |
| **P3** checkpoint not vLLM-loadable | c01r04 paid 0.012 h; c01r06 pre-empted it (`finalize_ckpt.py`, 08:05:06Z) | <0.05 h | Below threshold this window. |
| **C** eval n | c01r06 measured the prefix bias at 3.32 pp (78.0@400 → 74.68@1319); c01r04 had a real 300→600 reversal that changed the shipped model; c01r05 shipped on a +1-item margin at n=200 | — | c01r05 is important counterevidence for C v2 as written: it **did** compute paired discordant counts (9 vs 8) and shipped anyway. The rule produced the number and not the decision. |
| **uncovered** | unpriced offline-sampling job size: c01r04 (74,730 prompts, ETA 2:58, killed at 0.37 h), c01r05 (116,838 prompts, ETA 1:53–7:14, killed at 0.12 h) | 0.49 h | **Two cells, single allowed item** → eligible as a proposal (below). |
| **uncovered** | own arithmetic-dedup filter hangs on `**` expressions (c01r05, ~0.23 h) | 0.23 h | One cell → observation. |

**Proposals (single allowed item, ≥2 cells).** Only one new item clears the bar from my three cells:

- *`pitfalls.yaml` entry, `offline_sampling_unpriced`* (check: null). Source cells c01r04 (07:52:59–08:14:53, ETA visible at 08:03:20 and not acted on for 10 min) and c01r05 (09:55:25–10:02:30). Target metric a 4-cell screen could read: hours of offline-sampling GPU time on a pass that was later killed or discarded for throughput reasons, target <0.15 h/cell; guardrail: ≥1/4 cells still attempts offline sampling (do not suppress the method). I flag that the held pool is already at 28 and P1–P4 are ahead of this in the ledger — this is queue material, not a request for a wave slot.

Everything else from my block is either already a frozen candidate (D, E, A, B, C) or a single-cell observation.

**Two facts that bear directly on decisions already on the table.**
1. **The E block-withdrawal question.** The round-02 spec and the 09-03 08:45 decision make E's fate conditional on the Round 01 guard showing ≥7/8 cells already under 0.15 h. My three controls run the same wait wording and 2/3 fail it by 8× and 16×, with 3.72 h at stake. If the guard reads as saturated, that will be a statement about three cells, not about the wait idiom — the controls in the same window say the idiom is still expensive. Reading the guard block remains the pre-registered test; I am only flagging that a "saturated" verdict from n=3 guard cells would sit against 2/3 fresh baseline cells that fail.
2. **`pitfalls_cost_h` does not order accuracy in this block.** c01r04 lost the most (2.46 h attributed, 2.04 h idle) and scored highest (0.7923); c01r05 lost the least idle time (0.87 h, 90% GPU-allocated) and scored lowest (0.7240). Ranked by total waste: c01r06 ≈50%, c01r04 ≈36%, c01r05 ≈32% — no relation to 0.7566 / 0.7923 / 0.7240. Within this window the score tracks landed training tokens (173.1M / ~160M token-passes on fewer unique rows / 96.6M) and the n behind the final selection (600 / 200 / 1319), not the protocol's own KPI.

