# G delivery, useful blockers and cost — bounded operator audit, 2026-09-04

Read-only audit of completed G cells w13r01..04, jobs 91441..91444. No other in-flight traces, model calls, scorer changes or repository/shared-state edits. Four scope/access flags are owned by the parallel scope auditor and are not reassessed here. Canonical receipt/cell/result mapping is `g-result.json` alongside this report; the manifest is `experiments/posttrainbench/wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4.yaml`, preregistration `doc/spec/2026-09-03-wma-round04-probe-selection.md` (G guards lines 98–100, common cost/delivery guards 127–132, opportunity limitation 151–155).

## Delivery and retained-state counts

| Cell / job | Completed requests/responses | Final verdicts + delivered locks | Superseded reviews | Final recorded wait min | All-request lifecycle min | Retained-final USD | Retained-final wall min |
|---|---:|---:|---:|---:|---:|---:|---:|
| w13r01 / 91441 | 13 | 7 | 6 | 37.205 | 72.421 | $12.5274 | 38.885 |
| w13r02 / 91442 | 10 | 7 | 3 | 41.125 | 61.966 | $13.6254 | 42.828 |
| w13r03 / 91443 | 8 | 6 | 2 | 34.117 | 46.717 | $11.5358 | 35.629 |
| w13r04 / 91444 | 11 | 5 | 6 | 29.948 | 70.201 | $9.9815 | 31.309 |
| Total | 42 | 25 | 17 | 142.395 | 251.305 | $47.6701 | 148.651 |

Reproduction: count `task/.wma/processed/*.json`, join by request_id to `responses/*.json`, and count final `task/memory/cards/*.verdict.json` / `*.lock.json`. All 42 processed requests have one matching response with state `completed` and empty `errors`; no requests/processing records remain. All 25 final locks have `wma.state=delivered`. Every card has a retained final verdict; all final L0/L1/L3 answers are yes. Final relock ancestry totals 17, matching 42 minus 25. No recorded WMA timeout or `--no-wma-wait` use found. Scientist Bash `Command timed out after 2m` messages are polling-shell/tool timeouts and are not evidence of a timed-out WMA review. Final locks' request IDs resolve to completed queue responses.

The 25 retained private transcript files each have one init/session: these preserve the last review, not all 42 sessions. Seventeen superseded verdict files/transcripts/cost records were overwritten. Some early answers survive in scientist-returned verdict summaries and relock reasons; do not pretend those are complete archived verdicts. Parent reproduced ledger: 25 final / 21 scored / 4 access-flagged. Clean terminal L0/L1 hit 1.0 cannot measure early failed-proposal recall, because these final cards include repaired proposals.

Final recorded waits are 8,543.7 s total, 341.748 s/final lock and 35.599 min/cell. All-request lifecycle is response.completed_at minus request.created_at: 15,078.307 s, 359.007 s/request and 62.826 min/cell. These are separate recorded measures. Lifecycle includes request handling and does not equal API time; last-lock waited_s is not the cumulative wait on overwritten locks. Neither is measured GPU idle: preparation, other work and GPU activity can overlap. No retrospective GPU-idle estimate is made.

Retained-final cost is $47.6701 ($1.906804/verdict; $11.917525/cell), with 148.650587 wall min (5.946023 min/verdict). This is the accepted-final cost denominator specified by preregistration, **not complete WMA spend**. Known whole-review spend is bounded below by $47.6701; the additional cost of 17 overwritten reviews is unknown here, and no finite upper bound is established. Do not fill it with zero, scale final cost by 42/25, or assert the 1.5x matched-baseline cost guard from this bounded audit alone.

## Three concrete card timelines

### 1. w13r04 / exp-02: an explicit negative L0 changed the training implementation

- First request `20260904T082824.312222Z-f449bd4b`, created 08:28:24, completed 08:35:27. Scientist receives `L0_runs=no@0.58; L1_valid=yes@0.7; L3=yes@0.85` at 08:36:36 (solve_parsed lines 4860–4864). Evidence summary identifies the 262144-vocabulary logits-memory term and estimates roughly 95 GB against the H100's usable memory (4892–4896). This is a static risk estimate, not a measured original-code OOM.
- Scientist reads the loss implementation and at 08:38:12 changes the trainer to compute logits only at supervised positions (5090–5119). At 08:38:41 it relocks explicitly crediting the negative WMA argument (5164–5166; also final exp-02.lock.json.relocked_from[0]). It later repairs a missed save-steps argv edit and the new implementation's autocast/throughput issue; there are four requests for this card. A later smoke OOM involved a leftover process, so it does **not** experimentally confirm the original 95 GB estimate.
- Final request `20260904T092100.019729Z-3b634924` completes 09:27:52; scientist reads final yes at 09:28:50 (6954–6957), performs the requested small load/stop smoke (6980–7045), then launches the real 24k-row run at 09:31:03 (7055–7062). Final lock waited_s=390.4.
- Interpretation: explicit negative L0 and actual repair are retained, with a visible returned-verdict-before-launch sequence. It is a useful feasibility-intervention opportunity, but prevented-OOM GPU hours and score benefit are unobserved counterfactuals. This also involved substantial scientist debugging, so “saved 95 GB” is not a measured benefit.

### 2. w13r04 / exp-04: reproduced save-path blocker repaired before launch

- Request `20260904T125655.409078Z-ca07ba6b`, 12:56:55 → 13:03:55. Scientist sees `L0_runs=no@0.8; L1_valid=no@0.75; L3=yes@0.6` at 13:04:57 (12169–12185). The precondition names inherited do_sample=false with incompatible sampling fields causing `GenerationConfig.save_pretrained` to fail.
- Scientist applies a clean in-memory GenerationConfig at 13:05:13 and tests the exact inherited configuration. Tool output at 13:05:15 says `REPRODUCED: ValueError GenerationConfig is invalid` and `clean config saves fine` (12195–12237). This is stronger than only repeating the WMA's claim.
- Relock request `20260904T130554.159221Z-32bbc639` returns at 13:12:18; final yes is read at 13:13:56 (12419–12422). The corrected train command starts at 13:14:01 (12426–12439). Final lock waited_s=370.4; relock reason records the blocker and repair.
- Interpretation: a genuine encountered negative L0/L1 blocker led to a verified local repair before the expensive launch. No GPU-hour savings estimate is assigned: the unchanged full run was not executed. One such opportunity in one cell is not population-level recall preservation.

### 3. w13r03 / exp-03: a genuine missed blocker hidden by the final repaired card

- First request `20260904T115156.571176Z-08ceb6af`, 11:51:56 → 11:57:57. Scientist reads `L0_runs=yes@0.9; L1_valid=yes@0.85` at 11:59:04 (7327–7332); returned suggestions flag dtype, selection and few-shot issues but not the inherited GenerationConfig save failure (7341–7352). Full training starts at 11:59:20 after that verdict (7363–7377).
- At 13:12:39 training logs show 1260/1260 and `[1:09:03]`; the subsequent traceback is `GenerationConfig.validate(strict=True)` rejecting do_sample=false/temperature=0.0 (7564–7596). The scientist records that the checkpoint folder has config.json and no weights, patches the script and relocks (7670–7716). Thus approximately 69:03 of observed training wall was spent before a failed save. It is not an estimate derived from planned-minus-actual, and it is not claimed as measured utilization-integrated GPU time.
- Repair request `20260904T131353.314226Z-95a18e88` completes 13:19:52; final yes is read at 13:20:48 and now asks for an actual save smoke (7785–7803). The final card is later successful, so its final L0 hit cannot score the first positive against the failed original proposal. No retrospective ledger edit was made.
- Interpretation: useful blocker detection was **not universal**. This is a real missed opportunity on the same class as case 2, independently of whatever the scope auditor finds. It does not establish that G caused the miss relative to baseline, because there is no matched replay of this exact original proposal.

## Guard disposition

No evidence of lost delivery in the retained queue and final-lock records: 42/42 request completions and 25/25 final delivered verdicts. The three manually inspected card timelines place each examined expensive launch after its returned review; this bounded audit does not assert a full per-launch timing rate for all cards. Two explicit negative reviews on two cards in one cell drove repairs, including one independently reproduced deterministic failure. At least one comparable failure was missed in another cell. Therefore useful blockers remain possible under G, while matched recall preservation and net benefit remain unconfirmed. Scope/leak guard belongs to the separate audit. No G promotion, score-gain or whole-spend cost-pass claim follows from this report.

## Exact canonical source roots

- w13r01 / 91441: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4_w13r01_formal_r13/gsm8k_google_gemma-3-4b-pt_91441`.
- w13r02 / 91442: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4_w13r02_formal_r13/gsm8k_google_gemma-3-4b-pt_91442`.
- w13r03 / 91443: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4_w13r03_formal_r13/gsm8k_google_gemma-3-4b-pt_91443`.
- w13r04 / 91444: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4_w13r04_formal_r13/gsm8k_google_gemma-3-4b-pt_91444`.

Cited line numbers refer to each canonical root’s uncompressed `solve_parsed.txt`; request evidence is in `task/.wma/processed` and `responses`, final locks/verdicts in `task/memory/cards`.
