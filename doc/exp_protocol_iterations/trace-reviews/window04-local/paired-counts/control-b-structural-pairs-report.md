# Window04 control-b: structural denominator and paired-count correction

Read-only follow-up on c01s04/c01s07. Original reports remain unedited. No training, evaluation, network, scheduler, git, or repository mutation was performed. This is an explicit correction for any synthesis that already read their approximate regex pair tables; it does not decide a candidate.

## Main correction

**Several old pair tables are materially misaligned, not merely missing one sample.** Structural JSON parsing finds complete ID/epoch coverage: all relevant c01s04 pairs have **1319** scored items and all c01s07 selection pairs have **500**. There is no ±1 denominator uncertainty. Four c01s04 and two c01s07 discordance counts change substantially below.

c01s07's **15-log** count is correct; its displayed size breakdown omitted a second n=150 run. Correct breakdown: **150×2 + 200×8 + 500×4 + 1000×1 = 15 logs**. The omitted n=150 entry is the default final-model artifact check at 01:46:15.

## Receipt, sources, and reproducibility

Bundle root `B`:

`/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1`

Immutable receipt: `B/formal-2026-09-02T210446.182614+0000.json`, SHA256 `ce6b5d4fc25286f7b48d282561fcf8134163fe3715f919ec346fd4bb8b46fda9`. Its `jobs` entries identify **c01s04/job90816** and **c01s07/job90819**, exactly matching `B/<cell>/status.json` (`complete:true`, `eligible:true`).

Exact status-resolved raw roots:

- `R4 = /home/robtang_google_com/gangda_workspace/agentic-world-model-gangda-exp-protocol-evolve/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1_c01s04_formal_r4/gsm8k_google_gemma-3-4b-pt_90816`
- `R7 = /home/robtang_google_com/gangda_workspace/agentic-world-model-gangda-exp-protocol-evolve/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1_c01s07_formal_r4/gsm8k_google_gemma-3-4b-pt_90819`

All developer JSON sources are under `R4/task/logs/` or `R7/task/logs/`. `inventory.tsv` contains **every full path and full SHA256**, plus model, status, n, scored/correct counts, accuracy, token cap, concurrency and memory. `structural-audit.json` additionally preserves per-ID/epoch scores, input/target/rendered-prompt hashes, exact discordant IDs, declared ID order, completion times, metadata, mismatch counts, receipt/status/trace/script hashes and final generation-config contents.

Reproduce using only Python's standard library:

```bash
python /tmp/window04-control-pairs.K96WXQ/audit_pairs.py
```

Output files in that same owned temporary directory:

| File | SHA256 |
|---|---|
| `audit_pairs.py` | `bbc7290424e87ef46934d88ddf2bcc3353cdcf227cf6400016b1ed0daf5c8101` |
| `structural-audit.json` | `8ec9db6bc17e33e69d49c9001c81b363ce2c6c719a2d8371e56aa4a97856fe41` |
| `inventory.tsv` | `fbcd2f1b0595ff34d7c0dccf70b059dd00fb01e19c81d7b63a29cbf76dbacd57` |
| `pairs.tsv` | `7c9b92eeeb6524a6e9387e11a55ffaff57afeeda058817295fb0b57c0b27731d` |

## Structural checks and matching recipe

Each source was parsed with `json.load`; this audit does **not** regex-stream nested JSON or zip completion-order arrays. A row key is the canonical typed pair **`[sample.id, sample.epoch]`**; all inspected epochs are 1 and every key is unique within its log. Binary outcome is **`sample.scores.match.value`**, exclusively `C` or `I`. The n=500 prefix is taken from the ordered `eval.dataset.sample_ids`, not the first 500 returned `samples` entries.

Across all **34 Inspect logs** (c01s04:19, c01s07:15):

- `status` is `success`; actual sample length, unique-key count, `results.total_samples`, `results.completed_samples`, and scorer `scored_samples` all agree.
- `match` is the sample score key; aggregate metric keys are **`accuracy` and `stderr`**. No sample error, missing/extra declared ID, duplicate ID/epoch, invalid binary score or aggregate-accuracy discrepancy was found. `correct/scored_n` reproduces every reported accuracy exactly to floating-point tolerance.
- The `eval.dataset.samples:1319` field describes the source test population, **not** the selected/scored n of each limited run.
- Across every requested paired comparison: **0 original-input mismatches, 0 target mismatches and 0 rendered-prompt mismatches** on common ID/epoch keys. Original input/target are hashed canonically; rendered prompts are the ordered `event=='model'` input-message lists, retaining role/content/tool-call fields but excluding volatile message IDs. The script also retains all per-model-event generation configurations.
- Both cells' `evaluate.py` SHA256 is `02b97287d5cd2179dc26c4328a35386a5a2908919a360fd97418b32003b15529`; their `templates/gemma3.jinja` SHA256 is `7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4`. Same hashes alone would not prove equal requests; the per-item rendered-input comparisons supply that additional check.

An independent `jq` structural ID+epoch join reproduced c01s04 b70/b160: **935 both correct,58 A-only,51 B-only,275 both wrong**, n1319. The old regex implementation is not retained here, so its precise bug is unresolved; the wrong counts cannot be explained as a single omitted row.

## Exact pair tables

Order is **A versus B**. `Both` means both correct; `Neither` means both incorrect. P-values are **exact two-sided McNemar/binomial**, `min(1, 2*P[Binomial(A-only+B-only,0.5) <= min(A-only,B-only)])`, computed with integer binomial coefficients. They are descriptive unadjusted comparisons of these selected developer evaluations, not equivalence tests or evidence that more evaluation was unnecessary.

### c01s04 — all n=1319

| A vs B | Both | A-only / B-only | Neither | Net A−B | Exact two-sided p | Old report |
|---|---:|---:|---:|---:|---:|---|
| b70 vs b160 | 935 | **58 / 51** | 275 | **+7** | 0.565700 | 219/211, +8 of1318: incorrect |
| b70 vs b105 | 934 | **59 / 44** | 282 | **+15** | 0.167444 | 224/208, +16: incorrect |
| b70 vs soup1 | 936 | **57 / 45** | 281 | **+12** | 0.276014 | 226/213, +13: incorrect |
| b70 vs b70_rp | 909 | **84 / 69** | 257 | **+15** | 0.257634 | 225/209, +16: incorrect |
| b105 vs b70_rp | 897 | **81 / 81** | 260 | 0 | 1.000000 | Correct:162 discordant despite equal scalar accuracy |

Correct totals: b70 **993**, b160 **986**, b105 **978**, soup1 **981**, b70_rp **978**. Thus b70/runner-up is **7 items with109 discordants**, not 8 items with430 discordants. Every pair's numerator difference now agrees exactly with its aggregate-score difference.

Source files under `R4/task/logs/` (the JSON pointers above specify the fields used):

| Alias | Exact filename |
|---|---|
| soup1 | `2026-09-04T02-33-30+00-00_gsm8k_FPB6MgdXg4sjJbKuMoBVxs.json` |
| b70 | `2026-09-04T02-39-49+00-00_gsm8k_UCNV3hoFEd5zrHugdsuUE8.json` |
| b105 | `2026-09-04T02-49-29+00-00_gsm8k_BYeNFPpFuQNPrVGVKfg9Q5.json` |
| b160 | `2026-09-04T02-52-36+00-00_gsm8k_k9intyp4y52ZMmeG7VrH9F.json` |
| b70_rp | `2026-09-04T02-59-45+00-00_gsm8k_nRmE6zjVeJDj4qzsNCGk4f.json` |

### c01s07 — selection pairs n=500, plus stable shared500

| A vs B | Both | A-only / B-only | Neither | Net A−B | Exact two-sided p | Old report |
|---|---:|---:|---:|---:|---:|---|
| g2_200 vs rft_v1 | 322 | **50 / 44** | 84 | +6 | 0.606296 | Correct |
| g2_200 vs g3_150 | 315 | **57 / 43** | 85 | +14 | 0.193348 | 74/60 of499: incorrect |
| g3_150 vs g3_100 | 317 | **41 / 41** | 101 | 0 | 1.000000 | 61/61: incorrect; **82**, not122 discordant |
| g2_200@500 vs final@1000 shared500 | 356 | **16 / 6** | 122 | +10 | 0.052479 | Counts correct; noise interpretation needs correction |

Correct totals: g2_200 **372/500**, rft_v1 **366/500**, g3_150 and g3_100 each **358/500**. The n1000 log's first500 declared IDs equal the g2_200 n500 IDs **in both order and set**; its shared-prefix accuracy is **362/500=.724**, remaining500 **361/500=.722**, total **723/1000=.723**. The observed near-equality between these two halves is not proof of a generally absent difficulty gradient.

Source files under `R7/task/logs/`:

| Alias | Exact filename |
|---|---|
| g2_200@500 | `2026-09-04T01-41-59+00-00_gsm8k_MVKpsws3LtzJYJ6RkqhyrZ.json` |
| rft_v1@500 | `2026-09-04T01-43-23+00-00_gsm8k_dYFzpyK7VR8zLNVrCkkbQm.json` |
| omitted default@150 | `2026-09-04T01-46-15+00-00_gsm8k_HtAbhmTioyDzpHAZQUUoPk.json` |
| g3_150@500 | `2026-09-04T02-43-26+00-00_gsm8k_FJecToYrnfR3NCbAMy2Qfm.json` |
| g3_100@500 | `2026-09-04T02-46-05+00-00_gsm8k_FAM2qPLzeAadH2JvEzgiAA.json` |
| final@1000 | `2026-09-04T02-47-46+00-00_gsm8k_b82nQzscgpzgBe3foew2XE.json` |

## Serving and inference boundaries

| Compared evaluation group | Logged max_tokens / concurrency / GPU-memory fraction | Other changes/limitations |
|---|---|---|
| c01s04 selected n1319 models | **4000 /32 /.85** throughout | Task/scorer/fewshot/package/plan metadata and per-item prompts match. Different checkpoint weights remain the intended candidate contrast. |
| c01s04 b70 versus b70_rp | **4000 /32 /.85** in both JSON headers | **Repetition penalty changes to1.05 in the model generation-config file**, not the JSON header. Trace `B/c01s04/solve_parsed.txt.gz:7224–7235`, 02:58:15, copies the b70 serving directory and writes this field. Equal logged headers do not imply identical resolved serving policy. |
| c01s07 ordinary n200 reads | **1024 /32 /.35** | 8 logs; same-path SFT sample/greedy A/B is two separate generations, not duplicate JSON. |
| c01s07 four n500 selection logs | **1024 /48 /.40** | Requested protocol and per-item prompts match; checkpoint identity changes between the specified candidates. |
| c01s07 final@1000 | **2048 /48 /.40** | Compared with g2_200@500, **token cap doubles and total batch population changes**. Same shared IDs/prompts do not isolate either effect or random repeat variability. |
| c01s07 omitted default artifact check | **4000 /2 /.30**, n150 | 110/150=.733333; separate from the n150 base evaluation (3/150). |

c01s04 complete inventory is **19 logs = 150×2 +200×5 +400×6 +1319×6**. No failed/partial Inspect JSON was found in either inventoried developer log directory; scalar `{accuracy,stderr}` summary JSONs are listed separately in the output and are not counted as Inspect logs.

Artifact lineage: c01s07 installs `final_model` with `cp -r ckpt/g2_200 final_model` at **01:44:25**, trace `B/c01s07/solve_parsed.txt.gz:5854–5856`; final n1000 invocation at **02:46:45** explicitly changes `--max-tokens 2048`, lines6269–6275. Final generation config at `R7/final_model/generation_config.json` is retained and hashed, with `do_sample:true,temperature:0,top_k:64,top_p:.95`. This audit verifies trace-based lineage and config, **not an independent bytewise comparison against omitted old checkpoint shards**.

Accordingly, replace the report's “same-artifact re-read noise,” “measurement's repeatability,” and “selection inside the noise floor” claims with: **a trace-linked same-artifact, changed-serving/evaluation-size contrast flipped22 shared items (16 losses,6 gains)**. These observations do not estimate isolated repeat noise or a noise floor. Likewise p>.05 on the selected checkpoint pairs means this exact two-sided test did not reject equal marginal success probabilities; it does **not** establish equivalence, equal expected quality, or that a selection read was wasted. No new candidate or queue decision follows from this audit.
