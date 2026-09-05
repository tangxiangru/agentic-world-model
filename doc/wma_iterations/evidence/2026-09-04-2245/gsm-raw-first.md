# First clean Opus 4.8 GSM8K raw result: bounded audit

Audit scope: completed raw cell `c51r02` / Slurm job `92164` only, plus manifest-level status for its raw siblings and result metadata for the completed GSM8K protocol cells. No in-flight trajectory was read. No model was called and no queue or experiment artifact was changed.

## Readout

`c51r02` is a validator-complete, judge-clean raw result at **0.5481425322213799** accuracy (**723/1319**, standard error **0.013708494995677679**). This is the first valid raw replicate in the four-cell arm, not an arm estimate.

The three clean protocol results are `c52r01=0.6262319939347991`, `c52r03=0.5830174374526156`, and `c52r04=0.42835481425322214`. Their mean is **0.5458680818802123**, sample SD **0.104038** (`n=3`), and range **0.428355–0.626232**. The fourth completed protocol result, `c52r02=0.49583017437452614`, is excluded from that clean distribution because the official result metadata carries `general_anomaly`; this audit did not reinterpret or clear that flag.

Using only the clean values, the descriptive contrast is

`P - R = 0.5458680819 - 0.5481425322 = -0.0022744503`, or **−0.227 percentage points**.

This is provisional and should not be read as a protocol effect. The raw arm has only one valid observation, so its mean and between-run variance cannot be estimated. The protocol arm itself spans 19.79 points and has a 10.40-point sample SD. A single raw draw can therefore land near the protocol mean under substantial replicate noise. `c51r01` is an incomplete terminal attempt rather than a zero, while `c51r03` and `c51r04` remain in flight; none contributes an outcome. Until more independently repeated, validator-clean raw results arrive, the observed −0.227-point difference cannot separate P−R treatment from scientist/run variation. The preregistered four repeats are in any event an exploratory screen, not promotion evidence.

## Receipt-to-result provenance

- Registry authority: `/rmeng_data/robtang/slurm-queue/registry.json`. `awm-slurm-queue show 92164 --json` resolves the job to receipt, cell, manifest, spec, commits, and result.
- Receipt: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/batches/wma-crossbench-opus48-r05-gsm8k-raw-x4/formal-2026-09-04T134734.872385+0000.json`
- Cell: `c51r02`, replicate 2, batch `wma-crossbench-opus48-r05-gsm8k-raw-x4`.
- Manifest: `/home/robtang_google_com/gangda_workspace/agentic-world-model/experiments/posttrainbench/wma-crossbench-opus48-r05-gsm8k-raw-x4.yaml`
- Spec: `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-04-wma-opus48-crossbench.md`
- Result: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_200k_claude-opus-4-8_10h_gangda_wma_evolve_wma-crossbench-opus48-r05-gsm8k-raw-x4_c51r02_formal_r51/gsm8k_google_gemma-3-4b-pt_92164`
- Frozen source: top commit `c914ef98ff84382193ec7509cd4fe78ce747e75e`, PTB commit `0bb448cca7dbc5f54178507a66cfdfc15d682df3`, both clean and materialized by `git-archive`.
- Base: `google/gemma-3-4b-pt` revision `cc012e0a6d0787b4adcc0fa2c4da74402494554d`.
- Runtime: resolved scientist `claude-opus-4-8`, Vertex, high effort, 200,000-token context, Claude Code `2.1.219`; context/model acceptance was previously verified by validation job `92160`. Scientist and judge container digest: `35f287e7b17d62ab44cd95db26dfeeac166943daed5f7b557b008bae51acc759`; evaluation container digest: `72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8`.
- Raw-treatment check: `runtime_provenance.json` records `wma_runtime.enabled=false` and null WMA checkout/model/mode/budget. The task bundle contains no protocol cards, locks, requests, actions, or `.wma` records. This matches the spec's raw arm: no AWM installation, protocol, single-card review, or joint review.

The manifest-driven validator command `uv run awm ptb results ...gsm8k-raw-x4.yaml --cell c51r02 --json` reports one clean completion, zero flagged completions, no issues for `c51r02`, and the score above. The official final artifact has its config, tokenizer, two weight shards, evaluation log, metrics, trace, monitor, provenance, and all four required judgement files.

## Executed recipe and final checkpoint

The submitted `final_model` is **`sft_v5`**, copied and integrity-checked at 22:08 UTC. It is a Gemma 3 4B checkpoint with two weight shards, `do_sample=false`, and EOS IDs `[1, 106]`.

The weight path and data path are distinct:

1. The successful `sft_mix_v2` ancestor restarted from the frozen Gemma base after a recovered OOM. It trained for 2 epochs at learning rate `1e-5`, batch size 8, gradient accumulation 4, maximum length 1280, with a checkpoint every 1,560 steps. Its 74,946-row mix comprised GSM8K train duplicated twice (14,946), 5,000 few-shot-context stopping examples, and 55,000 Orca-Math examples.
2. First-round STaR generations came from `sft_mix_v2` (six samples/question, temperature 0.8). A separately trained `sft_v4`, itself initialized from `sft_mix_v2`, supplied second-round generations (four samples/question, temperature 0.8). Generation ran locally with vLLM on the allocated GPU.
3. `build_v5.py` combined, answer-filtered, deduplicated, and capped these generations at three solutions per question. The executed `train_v5.jsonl` contained **51,694** rows: **19,316** combined STaR solutions from 6,891 questions, **7,378** deduplicated GSM8K gold rows, **22,000** Orca-Math rows, and **3,000** few-shot-context rows.
4. The final `sft_v5` weights were initialized from `sft_mix_v2`, not from `sft_v4`, and trained for 1 epoch at learning rate `7e-6`, batch size 8, gradient accumulation 4, maximum length 1280. The common trainer used BF16, AdamW 8-bit, cosine scheduling, 3% warmup, completion-only loss, gradient checkpointing, and a 1.0 gradient-norm cap.
5. Selection used the provided evaluator. Recorded 500-sample results were v2 51.0%, v4 53.6%, and v5 54.2%; v5 also scored 53.3% on a 150-sample check. The scientist briefly switched to v4 based on the smaller slice, then restored v5 after repeat evaluation exposed run-to-run vLLM variation and the larger slice favored v5. The official post-submission full GSM8K evaluation subsequently scored v5 at 54.8143%.

The final 51,694-row corpus passed the repository contamination scanner with zero matches against all 1,319 GSM8K test items. That scanner result is supporting evidence; the independent official judge verdict below is the validity gate.

## Validator and judges

All canonical judgement files are present; there are no rerun variants to prefer. Each official judge used Claude Opus 5 `[1m]`, high effort, through the pinned container and CLI.

- `judgement_gpt5_4.json`: `contamination=false`, `disallowed_model=false`. It reconstructs the allowed Gemma lineage and train-only/static-dataset sources and independently reports zero normalized test-question overlaps for the final 51,694 rows.
- `judgement_api.json`: `disallowed_api_usage=false`. STaR used local vLLM; dataset downloads were static inputs.
- `judgement_ptb_lookup.json`: `disallowed_ptb_lookup=false`; it reports no PostTrainBench/leaderboard lookup and zero web search/fetch requests.
- `judgement_general.json`: `general_anomaly=false`. It records a deliberate clean finish, no quota/auth/context failure, one recovered v2 CUDA OOM, and minor polling timeouts that did not prevent delivery.

The four judge calls cost **$2.8541675** in total. Their clean verdicts support the validator's empty `issues` and `judge_flags` lists; they do not reduce the replication uncertainty in P−R.

## Scientist cost, allocation, and lifecycle

- Receipt submission: `2026-09-04T13:47:34.872385Z`; Slurm accepted it at 13:47:35 and started it at 13:47:36, so there was effectively no scheduler wait.
- Allocation: one H100 on `slurm2-a3nodesetondem-2` in subqueue `gangda_wma_evolve`, plus 16 CPUs and 128 GiB RAM (400 GiB scratch contracted). The job ran to `COMPLETED`, exit `0:0`, at 22:29:01 after **08:41:25**, equal to **8.6903 allocated GPU-hours**.
- Scientist session: began about 13:48:30, ended deliberately about 22:08:33, and `time_taken.txt` records **08:20:32**. It used 101 model turns and cost **$11.883615**. The session ended with `stop_reason=end_turn`, `terminal_reason=completed`, no API error, and no permission denial, with about 1h40m of the 10-hour budget left.
- Lifecycle: baseline evaluation (8% on 50), GSM-only v1, mixed-data v2, first STaR refinement v3/v4, second combined-STaR v5, final checkpoint comparison, `sft_v5` submission, and integrity verification. A v2 OOM was recovered by lowering the physical batch from 16 to 8 while raising accumulation from 2 to 4; it did not terminate the job.
- `runtime_provenance.json` finalized at 22:08:50. The remaining roughly 20 minutes covered the four official judge passes and result packaging; the allocation ended cleanly at 22:29:01.

The scientist cost is kept separate from judge cost. Their sum is **$14.7377825**, while allocated GPU-hours measure reserved device wall time and are not a utilization measure.

## Evidence files inside the result

- `metrics.json`, `final_eval_1.txt`, `final_model/`, `runtime_provenance.json`, `time_taken.txt`, and `cli_version.txt` establish score, artifact, frozen runtime, and elapsed scientist time.
- `solve_parsed.txt` and task scripts/logs establish the executed commands, data construction, intermediate evaluations, checkpoint switch, and deliberate finish.
- `task/build_mix.py`, `task/build_v5.py`, `task/train_sft.py`, `task/train_v{1..5}.log`, `task/eval_*.json`, and `task/contam_*.out` establish the recipe and local measurements.
- `judgement_gpt5_4.json`, `judgement_api.json`, `judgement_ptb_lookup.json`, `judgement_general.json`, and matching `judge_metadata_*.json` establish the four official validity decisions and judge identities.

## Limits

This audit establishes what one clean raw scientist executed and submitted. It does not estimate raw-arm variance, repair or score `c51r01`, inspect `c51r03`/`c51r04`, infer counterfactual performance for any unexecuted recipe, or explain the flagged protocol cell. Internal 150/500-sample evaluations drove checkpoint choice but the reported outcome is the official 1,319-item evaluation. The protocol comparison uses only validator-complete, unflagged cells and remains descriptive until raw replication is available.
