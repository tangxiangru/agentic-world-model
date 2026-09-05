Write and ExitPlanMode aren't available in this session — the tool set is read-only, matching the brief's override ("read-only tools and final-output reports"). Here is the report as final output.

---

# P5 supplemental serving-contract audit — g01r01 / g01s02 / g01s07

Read-only supplement to the closed Window03 / strict-guard reviews. No cells counted, no candidate registered, frozen evaluator contract untouched. Existing per-cell reports used only as locators; every claim below is re-verified against the raw trace or a retained artifact path.

## Contract facts verified once (all three cells)

| item | evidence | verdict |
|---|---|---|
| evaluator code | `task/evaluate.py` md5 `2490079a39a1e242ab1b5286a0087137` in all three bundles | **same** |
| evaluator defaults | `evaluate.py:44-56` — `--max-connections 2`, `--max-tokens 4000`, `--gpu-memory-utilization 0.3` | requested-vs-default baseline |
| official command | `third_party/PostTrainBench/src/run_task.sh:637-642` — `--model-path $EVAL_DIR/final_model --templates-dir ../../../../src/eval/templates --limit -1`, no connection/memory flags; `:670` `run_evaluation_with_retry 4 ""` ⇒ no max-tokens override | official = defaults |
| official resolved line | each `final_eval_1.txt`: `max_connections: 2, max_tokens: 4000, dataset: openai/gsm8k`, `gsm8k (1,319 samples)` | **known** |
| chat template | `templates/gemma3.jinja` md5 `acabb12fa812ef3ab334ea6b817562f3` in all three bundles **and** in `src/eval/templates/` (copied at `run_task.sh:94`) | **same** |
| prompt identity | input tokens **exactly 2,910,587** in every n=1319 read, developer and official, all three cells | **same** |
| artifact transfer | `run_task.sh:506` `cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"` — byte copy, no re-save | **same artifact** |
| scorer | retained Inspect log `eval.scorers[0]` = `match {numeric: true}` | **same** |
| seed | evaluate.py passes none; `eval.model_args` = `{gpu_memory_utilization, chat_template}` only | **unset both sides; resolved value unknown** |
| dtype | artifact bf16 (g01s07 L7982/18:35:10Z `dtype torch.bfloat16`); no dtype arg either side | same artifact, **resolved server dtype unlogged** |
| process reuse | every developer eval log has exactly one `Starting new server for`; official reaps GPU procs then starts one (`run_task.sh:611-612`) | **fresh process both sides** |
| stack identity | dev `eval.packages.inspect_ai = 0.1.dev3780+g64db0afdd`; official version **not retained**. Both emit `vllm.py:119`/`vllm.py:135` (weak agreement). Official runs in `vllm_debug.sif`, agent in a different image (`run_task.sh:358` vs `:637`) | **unknown / unverified** |

n=1319 confirmed for all eleven reads by the metrics.md algebra `p(1-p)/SE²+1` (returns exactly 1319.00 each).

## 1. g01r01 — official 937/1319 (.7103866565579985)

Shipped artifact: `final_model` rebuilt from `ckpts/soup27` by `scripts/finalize.py`, **L9630/14:12:13Z** (12 files listed L9675-9681/14:12:27Z). Two earlier rebuilds exist — L7665/12:31:50Z from exp-04, L8940/13:36:19Z from exp-07 — so `final_model` names three different artifacts across the session.

| read | artifact evidence | n | items | mc | gmu | max_tok |
|---|---|---|---|---|---|---|
| dev, shipped | `eval/FINAL_full.json`; `logs/eval_FINAL_full.log:84` `vllm//home/ben/task/final_model`, launched L9686/14:12:33Z | 1319 | **948** | 32 | 0.85 | 4000 |
| official | `final_eval_1.txt`, `metrics.json` | 1319 | **937** | 2 | 0.3 | 4000 |
| dev repeat, soup27 ×4 | `eval/exp09_soup27_full_r1..r4.json`, log path `ckpts/soup27` | 1319 | 948, 948, 949, 950 | 32 | 0.85 | 4000 |

**Trap.** `eval/final_model_verify_full.json` (.7172100, 946) is *not* the shipped artifact: it read `final_model` while it held exp-07's weights, and the scientist's own grouping files it under the `'exp-07'` arm (L9527/14:11:31Z). A "final_model" log name does not identify the final artifact.

**Within-config repeat spread** (same settings, same n, fresh server each time, item counts): exp-07 ×6 = 946–952 (range 6); exp-04/`final_model` ×5 = **930–949 (range 19)**; exp-02_greedy ×4 = 928–943 (15); soup24 ×4 = 928–947 (19); soup27 ×4 = 948–950 (2). The scientist's own table (L9569-9574/14:11:44Z) gives per-arm sd 0.0007–0.0070.

**Identical-correctness ≠ byte-identical:** `exp09_soup27_full_r1` and `r2` score *exactly* 0.7187263078089462 with mean completion 553 vs 562 chars (`analysis/diag_exp09_soup27_full_r1.txt`, `_r2.txt`); `FINAL_full` scores the same value at 570 chars.

Gap 948→937 = **11 items (0.83 pp)**, against **two** changed settings (mc 32→2, gmu 0.85→0.3) and a same-cell repeat range reaching 19 items. Official output tokens 254,948 lie inside the dev band (244,929–264,500). **Not isolated.**

## 2. g01s02 — official 956/1319 (.7247915087187263)

Shipped artifact: `rm -rf final_model && cp -rL ckpts/exp-10/final_greedy/*` minus `training_args.bin`, **L9342/19:08:13Z**; greedy config read back L9385-9394/19:09:01Z (`do_sample false, temperature 0.0, eos [1,106]`).

| read | artifact evidence | n | items | mc | gmu |
|---|---|---|---|---|---|
| dev, shipped | `eval/exp-10_final_model_test1319.json`; `logs/exp-10_fm.log:84-85` `vllm/final_model`, cmd L9343/19:08:13Z | 1319 | **965** | 32 | 0.85 |
| official | `final_eval_1.txt`, `metrics.json` | 1319 | **956** | 2 | 0.3 |
| dev repeat (same weights, other path) | `eval/exp-10_test1319.json`, `ckpts/exp-10/final_greedy`, cmd L9256/19:00:41Z | 1319 | 964 | 32 | 0.85 |
| dev repeat, 13 min apart | `ckpts/exp-05/final_greedy` = 924 (`exp-07_eval.log`) vs its byte copy `final_model` = 911 (`exp-08.log`; copy at L7907/16:35:11Z) | 1319 | **924 → 911** | 32 | 0.85 |

**Paired disagreement floor, measured in-cell:** L8246-8247/16:45:25Z — `n 1319 identical completions 685 (0.519)` and `score disagreements 65 A_only 39 B_only 26`; retained at `task/analysis/exp-08_diag.json`. Two runs of identical weights under identical settings disagree on **65 items** and net 13.

Gap 965→956 = **9 items (0.68 pp)** — smaller than the cell's own 13-item same-config repeat and far inside its 65-item paired disagreement. Two settings differ. Official output tokens 264,172 lie inside the dev band (257,701–274,455). **Not isolated.**

## 3. g01s07 — official 970/1319 (.7354056103108415)

Shipped artifact: `install_final.py --src ckpts/soup-0205 --dst final_model --verify`, **L7286/17:11:20Z**; both shards md5-match `ckpts/soup-0205`, dtype bf16 (L7980-7982/18:35:10Z).

| read | artifact evidence | n | items | mc | gmu |
|---|---|---|---|---|---|
| dev, shipped | `eval/final_model_full.json`; `logs/final_full_eval.log:84-85` `vllm/final_model`, cmd L7382/17:29:12Z | 1319 | **991** | 16 | **0.3 (default)** |
| official | `final_eval_1.txt`, `metrics.json` | 1319 | **970** | 2 | 0.3 |
| dev repeat (same weights, other path), n=500 | `eval/soup_dev500.json` 388/500 (`exp-06_eval.log`, `ckpts/soup-0205`) vs `eval/final_model_dev500.json` 385/500 (`final_model_eval.log`) | 500 | 388 → 385 | 16 | 0.3 |

`--gpu-memory-utilization` **never appears in a command in this trace** — the only two matches are source reads at L616/L633/09:49:55Z — and the retained Inspect log `task/logs/2026-09-03T10-18-54+00-00_gsm8k_c9nAXCMAt5uogCsgH5DczH.json` confirms `eval.model_args = {"gpu_memory_utilization": 0.3, …}`. **This is the only cell where concurrency is the single differing requested setting.**

But there is **no n=1319 repeat at either concurrency**, so the 21-item gap (991→970, 1.59 pp) is one draw against one draw; the only repeat is the 3/500 pair above. Official output tokens 286,524 are *below* both dev full reads (302,336 on `final_model`; 317,366 on exp-02) — a direction a truncation story does not explain. **Suggestive, not isolating.**

## Cross-cell conclusion

**The existing evidence does not isolate serving concurrency. It shows confounded and non-repeatable reads.**

1. **No dose-response.** Gap vs developer concurrency: 11 items @32, 9 items @32, 21 items @16. The largest gap is at the lowest developer concurrency.
2. **Memory was not matched in 2 of 3 cells** (gmu 0.85 vs the frozen default 0.3), so those comparisons are two-factor. The one cell with memory matched has the largest gap — the opposite of a memory explanation — and it has no repeat to calibrate against.
3. **Two of three gaps sit inside the same cell's own same-config repeat spread** (g01r01: 11 vs a 19-item range; g01s02: 9 vs a 13-item repeat and a 65-item paired disagreement).
4. **The serving path is not reproducible with everything held fixed** — 51.9 % byte-identical completions across two identical-configuration runs (g01s02); three g01r01 reads with byte-equal scores at different mean completion lengths.
5. **The prompt side is clean**: identical evaluate.py, template md5, 2,910,587 input tokens, scorer, max_tokens, byte-copied artifact. Whatever moves the number lives in the serving/scoring path — exactly where the retained evidence stops.

No significance is claimed from these unpaired aggregates; no greedy determinism assumed.

## Exact missing evidence

1. **Official Inspect per-sample JSON**, named inside each bundle but not harvested: `logs/2026-09-03T14-30-14+00-00_gsm8k_i9Mrck25NHAAJk8nXJ5nmi.json` (g01r01), `…19-29-35…_L2khJCLGEvysotR3v6s4P4.json` (g01s02), `…18-47-04…_A2JvBEndjXK5ik6kzRRGmX.json` (g01s07). Carries resolved `eval.model_args`, `eval.packages.inspect_ai`, and per-sample ids/responses/scores. Without it there is no paired item table on any cell.
2. **Developer per-sample logs for the shipped-artifact reads.** g01r01's `run_eval.sh` wrote `analysis/samples_<tag>.json` but only the 137-byte `analysis/diag_*.txt` survived; g01s02 retains no gsm8k Inspect JSON; g01s07 retains only the 8-item smoke log.
3. **Resolved server metadata on both sides** — vLLM/torch version, dtype, KV-cache blocks, chunked prefill, seed. The Inspect vLLM provider echoes no server banner into any captured stdout.
4. **A one-factor read.** No developer read at (mc 2, gmu 0.3) and no official read at developer settings exists in any bundle.

Settling it would require, on one already-harvested artifact: ≥3 reads at (mc 2, gmu 0.3) and ≥3 at (mc 32, gmu 0.85) with per-sample logs kept, plus the official per-sample log, compared item by item.

## Protocol-surface proposal: none

- The only single-factor cell has no repeat on either side; the two cells with repeats show gaps inside their own repeat spread. Two cells do **not** establish a distinct mechanism — they establish that the reads are not repeatable.
- What remains ("average ≥2 reads; greedy vLLM is not reproducible") is C-style uncertainty. C v2's unstarted block was withdrawn because the baseline already meets its threshold (`doc/exp_protocol_iterations/2026-09-03-round-01-strict-guard.md:45`), and these cells confirm saturation: all three measured repeat variability unprompted (g01r01's 4–6 replicates per arm, g01s02's exp-08 diagnostic, g01s07's jitter probe). A 4-cell screen could not read movement.
- A `pitfalls.yaml` entry naming concurrency as the cause would assert a mechanism this evidence does not support; the frozen file has 8 entries and none on evaluation reproducibility.
- Ledger #24's judgement stands, with one correction: **memory utilisation, not only concurrency, differs — in 2 of 3 cells** — so its "matched … memory" precondition is not met. Recommend closing #24 as no-candidate.

Reported as an observation for the operator, **not** as a candidate: harvesting the official Inspect JSON next to `final_eval_1.txt` would make every future dev-vs-official question answerable at item level for zero GPU cost. That is a harvest change, outside the protocol surface.

---

**Note on delivery:** this session has no write tool, so the report exists only in this output. If you want it persisted, `doc/exp_protocol_iterations/trace-reviews/p5-serving-audit/` already exists (untracked) and `report.md` is the natural filename.
