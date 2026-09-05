# PTB Batch 2 evaluation status

Updated 2026-09-02 14:02 UTC. Batch:
`gsm8k-aime2025-opus5-selected16x2-batch2-v5`.

## Bottom line

- Intended cells: 32 (16 GSM8K, 16 AIME2025; two repeats of eight selected settings).
- Cells with a final model, trace, monitor, full-evaluation log, and `metrics.json`: **32/32**.
- Validator-complete cells: **32/32**.
- Clean/flagged complete cells: **30/2**. `g05r2` and `g13r2` are complete but carry the canonical
  `general_anomaly` flag; all other cells are judge-clean.
- Active Batch 2 Slurm jobs: none.
- Every result now contains all four canonical verdicts: `judgement_gpt5_4`, `judgement_api`,
  `judgement_ptb_lookup`, and `judgement_general` (canonical names retained after the official
  runtime changed to Claude Opus 5 high).

The scores below are validator-accepted full-evaluation metrics from frozen result directories.
Any aggregation that requires a clean-only cohort must exclude or separately report `g05r2` and
`g13r2`; the paired tables keep them visible and marked by this documented boundary.

## Provenance

- Manifest: `experiments/posttrainbench/gsm8k-aime2025-opus5-selected16x2-batch2.yaml`.
- Spec: `doc/spec/2026-09-01-ptb-gsm8k-aime2025-batch2-replication.md`.
- Main retry receipt:
  `data/ptb/batches/gsm8k-aime2025-opus5-selected16x2-batch2-v5/formal-retry1-2026-09-01T215322.199428+0000.json`.
  It freezes top commit `006a5854` and PTB commit `a3d62de`.
- `g13r1` retry receipt:
  `data/ptb/batches/gsm8k-aime2025-opus5-selected16x2-batch2-v5/formal-retry2-2026-09-02T025456.128841+0000.json`.
  It freezes top commit `cf140104` and PTB commit `a3d62de`.
- Successful recovery canary receipt:
  `data/ptb/batches/gsm8k-aime2025-opus5-selected16x2-batch2-v5/official-judge-recovery-2026-09-02T132948.255039+0000.json`
  (job `90520`, `g01r1`).
- Successful retry1 recovery receipt:
  `data/ptb/batches/gsm8k-aime2025-opus5-selected16x2-batch2-v5/official-judge-recovery-2026-09-02T133720.364398+0000.json`
  (jobs `90523–90552`, the other 30 retry1 results).
- Successful retry2 recovery receipt:
  `data/ptb/batches/gsm8k-aime2025-opus5-selected16x2-batch2-v5/official-judge-recovery-2026-09-02T133735.511086+0000.json`
  (job `90553`, `g13r1`).

All three freeze top commit `e8a8599`, PTB commit `2af3ccd`, model
`claude-opus-5[1m]`, effort `high`, Vertex auth, and the `opus_5.sif` digest.

The authoritative discovery command is:

```bash
uv run awm ptb results \
  experiments/posttrainbench/gsm8k-aime2025-opus5-selected16x2-batch2.yaml \
  --all --json
```

At this snapshot it reports `total=32`, `complete=32`, `clean_complete=30`,
`flagged_complete=2`, and an empty `incomplete_cells` list.

## GSM8K validator-complete score pairs

Scores are full official-test accuracy. `Spread` is the absolute difference between the two
independent repeats, in percentage points.

| Setting | Base model | r1 | r2 | Pair mean | Spread |
|---|---|---:|---:|---:|---:|
| max / 1M (`g01`) | Qwen3-1.7B | 0.8537 | 0.8666 | **0.8601** | 1.29 |
| max / 1M (`g02`) | Qwen3-4B | 0.9128 | 0.8999 | **0.9064** | 1.29 |
| xhigh / 1M (`g05`) | Qwen3-1.7B | 0.7657 | 0.8355 | **0.8006** | 6.97 |
| xhigh / 1M (`g06`) | Qwen3-4B | 0.9158 | 0.8870 | **0.9014** | 2.88 |
| high / 1M (`g10`) | Qwen3-4B | 0.9098 | 0.9083 | **0.9090** | 0.15 |
| high / 1M (`g11`) | SmolLM3-3B | 0.8567 | 0.8704 | **0.8635** | 1.36 |
| max / 200K (`g13`) | Qwen3-1.7B | 0.8544 | 0.8234 | **0.8389** | 3.11 |
| max / 200K (`g15`) | SmolLM3-3B | 0.8294 | 0.8029 | **0.8161** | 2.65 |

Across the 16 validator-complete GSM8K results, the unweighted mean is 0.8620 and the observed range is
0.7657–0.9158. The best pair mean is high/1M with Qwen3-4B (0.9090), and it is also the most stable
pair (0.15-point spread). The highest single cell is xhigh/1M with Qwen3-4B, `g06r1` at 0.9158,
but its repeat is 2.88 points lower. The xhigh/1M Qwen3-1.7B pair has a 6.97-point spread; this is
direct evidence that a single run is inadequate for that setting.

## AIME2025 validator-complete score pairs

AIME has only 30 questions; counts are primary and percentages are included in parentheses.

| Setting | Base model | r1 | r2 | Pair mean |
|---|---|---:|---:|---:|
| max / 1M (`a02`) | Qwen3-4B | 7/30 | 6/30 | **6.5/30 (21.67%)** |
| max / 1M (`a03`) | SmolLM3-3B | 4/30 | 3/30 | **3.5/30 (11.67%)** |
| xhigh / 1M (`a06`) | Qwen3-4B | 7/30 | 4/30 | **5.5/30 (18.33%)** |
| xhigh / 1M (`a07`) | SmolLM3-3B | 6/30 | 3/30 | **4.5/30 (15.00%)** |
| high / 1M (`a10`) | Qwen3-4B | 7/30 | 6/30 | **6.5/30 (21.67%)** |
| high / 1M (`a11`) | SmolLM3-3B | 3/30 | 6/30 | **4.5/30 (15.00%)** |
| max / 200K (`a14`) | Qwen3-4B | 4/30 | 6/30 | **5.0/30 (16.67%)** |
| max / 200K (`a15`) | SmolLM3-3B | 3/30 | 4/30 | **3.5/30 (11.67%)** |

Across all 16 AIME results the mean is 4.94/30 (16.46%), with a cell range of 3–7
correct. Max/1M and high/1M on Qwen3-4B tie for the best pair mean at 6.5/30. The observed
one-to-three-question repeat differences are large relative to the 30-question denominator and
do not support a fine-grained effort ordering.

## Descriptive comparison with Batch 1

Against the corresponding single Batch 1 cells, Batch 2 pair means move in both directions:

- GSM8K improves most for max/1M Qwen3-1.7B (about +4.7 points) and high/1M SmolLM3 (about
  +2.5), while max/200K SmolLM3 falls about 5.9 points and xhigh/1M Qwen3-1.7B falls about 3.0.
- GSM8K high/1M Qwen3-4B is almost unchanged (about +0.2 points) and has very low repeat spread.
- AIME Qwen3-4B remains the stronger family in the selected cells. Pair means differ from Batch 1
  by only zero to 1.5 questions for most settings; that is within the coarse resolution and
  run-to-run variation visible here.

These comparisons are descriptive. Batch 1 used one run per setting and Batch 2 has only two;
the two flagged GSM8K cells must remain explicit in any clean-only comparison.

## Recovery outcome

The recovery history exposed three distinct infrastructure failures before the successful repair:

1. Retry1 jobs `89589–89620` produced metrics for 31 cells but used the old official Codex judge
   path. Revoked ChatGPT OAuth caused all four judges to produce no verdict. `g13r1` additionally
   lacked a model/metric and was separately retried.
2. Retry2 job `89926` produced the missing `g13r1` model and metric (0.8544) but froze the same old
   PTB source, so its Codex judges hit the same revoked OAuth failure.
3. The first Claude recovery receipt (`90397–90427`) failed before execution because the root-owned
   allocation did not receive the run-as identity. The second (`90428–90458`) fixed that and all
   31 jobs entered the real Claude Opus 5 high judge.

For the second failed recovery:

- 31/31 result directories contain `judge_output_gpt5_4_rerun.json`, its parsed text, and metadata.
- All 31 metadata records agree on `profile=official`, `backend=claude`, `auth_mode=vertex`,
  requested/resolved `claude-opus-5[1m]`, effort `high`, and `opus_5.sif`.
- All 31 raw traces contain a completed write of `task/judgement.json`; parsing those writes for
  diagnosis gives 31 provisional `contamination=false, disallowed_model=false` verdicts.
- These are **not canonical evidence**: there are zero `judgement_gpt5_4_rerun.json` files and zero
  API/lookup/general verdicts.

The common failure occurs immediately after parsing the first judge trace:

```text
.env file not found at .../official-judge-<job>/source/.env
```

`parse_trace.py` calls `sanitize_trace.load_api_key_secrets()`, whose default path is the frozen
git-archive source's `.env`. Git archives correctly contain no secret `.env`, and this sanitizer
path ignores the separately configured site env file. The parser exits before
`collect_judge_output` copies `task/judgement.json` into the result directory; the remaining three
judges never run. This is why successful Claude reasoning appears in the raw trace while formal
coverage remains zero.

The repair made `sanitize_trace.py` honor the explicit site env path, added a frozen-archive
regression test, and pinned PTB commit `2af3ccd`. Job `90520` first proved one complete four-judge
canary. Jobs `90523–90552` then repaired the remaining retry1 cells, and `90553` repaired retry2
`g13r1`; all 31 completed successfully. No training or evaluation was repeated.

The recovery boundary is therefore closed: all four canonical `_rerun` verdicts exist for every
cell and the formal validator accepts 32/32. The only remaining interpretation boundary is the
real `general_anomaly` verdict on `g05r2` and `g13r2`, which must not be conflated with the repaired
infrastructure failure.

## Interpretation limits

- All 32 Batch 2 numbers are validator-accepted PTB results; 30 are clean and 2 are flagged.
- The canonical files were produced by receipt-backed judge recovery. Earlier raw judge writes
  remain diagnostic evidence only and were not renamed or manually promoted.
- Two repeats reveal instability but are insufficient for precise variance estimates.
- AIME changes in units of 1/30 = 3.33 points; always report counts.
- Do not average GSM8K and AIME or infer a monotone effort effect from this selected subset.
