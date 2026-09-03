# wma-rca — the trace-reading kit for WMA iteration rounds

Scripts the `wma_meta` online loop hands to its subagents. Each reads harvested PTB cells
(`results/ptb/<batch>/<cell>/`) and writes JSON + markdown under `--out`; none writes into the
repository or reads scientist identity. Run from the repo root with `.venv/bin/python`.

| script | reads | writes | answers |
|---|---|---|---|
| `cells.py <batch>… --out D` | `metrics.json`, cards, `time_taken.txt` | `cells.{json,md}` | per-cell table (final accuracy, cards, first-SFT dev score, gain after it, training hours, epoch-rows, checkpoints kept, C5 / soup / full-set selection, max eval n, hours used, packaged card) and Spearman correlations of each feature with the final score, plus arm means — *what moved the score* |
| `uptake.py <batch>… --out D` | cards, locks, verdicts, `task/.wma/`, `solve_parsed.txt.gz` | `uptake.{json,md}` | per-card funnel: requested → response → delivered → **verdict before launch?** → read (lag) → uptake class; aggregates per cell and overall; since the blocking lock also `lock_wma_state` / `waited_s` from the lock file — *was the verdict in the loop* |
| `verdicts.py <batch>… [--inflight] --out D` | verdicts (+ `.rejected`), cards, `wma_private/*.transcript.jsonl.gz` | `verdicts.{json,md}` | per-verdict rows and aggregates **by skill hash**: L0–L3 answers, interval width / noise floor, raw L2 coverage on closed cards with misses above/below, probes and `changed`, history citations on C3/C4, checkpoint suggestions, turns and cost — *calibration and cost per candidate* |
| `timeline.py <cell>` | the same | stdout | one cell's lock → request → response → verdict → launch → read → run-end lines, for the hand-reading step |

`rcalib.py` holds the readers (cards, locks, verdicts, queue records, the scientist transcript
parser, launch/read detection, noise floor, Spearman).

## Definitions

- **verdict_before_launch**: `verdict.issued_at` < the first Bash call after the lock that launches a
  training/evaluation naming the card or its output dir. Round 01 baseline: 1/22.
- **uptake class** (mechanical first pass, an *upper bound* on real uptake): `never-read` (no tool
  call touches `exp-NN.verdict` after issue), `post-hoc` (first read after lock + `result.wall_h`),
  `adopted` (a Bash call between the read and the next card's lock matches a suggestion the
  verdict made: save_steps / scoring an intermediate checkpoint / generation_config / concurrency /
  full-set read), otherwise `ignored`. Hand-read the timelines of the `adopted` cells before
  quoting the number: in Round 01 the mechanical pass said 8, the hand count was 4.
- **noise floor** (manual §5): n ≥ 1000 → 0.01; 500–999 → 0.02; 150–499 → 0.03; 50–149 → 0.07;
  < 50 → 0.12; never below 1/n.
- **raw L2 coverage** counts every closed card whose `delta_vs_comparator` falls inside the interval.
  The harness ledger (`awm wma ledger … --by type`) is authoritative for scored levels: it applies
  the access fence (leak-suspected verdicts excluded), the self-measurement rule and the L0/L1
  truth rules. Use `verdicts.py` for the by-hash breakdown and the in-flight view, the ledger for
  the numbers that go into a round record.

## The subagent recipe (from `skills/wma_meta/SKILL.md`, online loop)

One subagent per question, in parallel, each writing a file under the round directory and
returning numbers with citations (`w01r05/exp-05`); the main agent reads the files, not the traces:

1. ledger by skill hash — `awm wma ledger` + `verdicts.py`;
2. uptake funnel and timing — `uptake.py`, then `timeline.py` on five cells by hand (three misses, two hits);
3. score levers — `cells.py`, correlations and arm differences;
4. harm cases — a `no`/`defer` that was heeded on a card that would have worked; an over-optimistic
   L2 that led to a wasted run (read the timelines of every non-`yes` L3 and every L0/L1 `no`);
5. protocol compliance since the blocking lock — `lock_wma_state` distribution, verdict-before-launch
   rate, wait time, timeouts and `--no-wma-wait` reasons (`uptake.py` aggregates).

Round 01 (2026-09-02, 8+8 cells) reproduced with this kit: verdict before launch 1/22; requests
without a verdict 11/33; L3 `yes` 22/22; checkpoint scoring suggested in 15/22 verdicts; training
hours with a verdict 23.3 of 47.7; Spearman with the final score: first-SFT dev +0.60, epoch-rows
+0.54, training hours +0.48, max eval n +0.49.
