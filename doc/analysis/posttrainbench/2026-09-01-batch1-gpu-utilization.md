# PTB Batch 1 GPU utilization and phase analysis

**Status:** interim live-run snapshot

**Snapshot window:** approximately 2026-09-01 04:55–07:32 UTC

**Batch:** `gsm8k-aime2025-opus5-4x4x2-batch1-v3`

**Scope:** 32 effective formal cells, about 2.4–2.5 elapsed hours per cell

This document records empirical observations from our self-run PostTrainBench batch. It is not an
experiment specification or a stable upstream harness fact. Update this same report after the ten-hour
agent phase and final evaluation complete.

## 1. Run identity and effective job mapping

The original formal receipt is:

```text
data/ptb/batches/gsm8k-aime2025-opus5-4x4x2-batch1-v3/
  formal-2026-09-01T045457.788033+0000.json
```

It froze top-level commit `9a545ed8f8b502c0489faae198b4772da26bc8b3` and PTB commit
`cec6d1190518c7801fc4fb29cf145565908f538d`.

Eight Qwen3-1.7B cells in that receipt failed preflight because the compute-node validator accepted
only sharded safetensors indexes, while this pinned model legitimately uses a monolithic
`model.safetensors`. The replacement receipt is:

```text
data/ptb/batches/gsm8k-aime2025-opus5-4x4x2-batch1-v3/
  formal-retry1-2026-09-01T045857.507936+0000.json
```

It froze top-level commit `ff4766684d1ddf720f67d39fe5351bf1dd2cb518` and PTB commit
`0c11fa19e87c74666c1b6d1604884dcf31828718`.

For analysis, retry jobs replace the same cell IDs in the original receipt:

```text
g01 g05 g09 g13 a01 a05 a09 a13
```

Never count the eight failed original job IDs together with their replacements. At the snapshot,
all 32 effective jobs were `RUNNING`, eight per node on `slurm2-a3nodesetondem-[0-3]`.

The formal run was launched after the user explicitly waived waiting for the v3 pilots to finish.
Pilot jobs `87022` and `87023` were then cancelled to release their two GPUs. This report therefore
does not claim that the runbook's pilot-audit gate passed; the targeted GPU-mapping and final-eval
cache smokes passed before formal launch, but the full v3 pilot flow did not complete.

## 2. Measurement method

Each active workspace contains `system_monitor.log`, sampled every 60–61 seconds. During the run
the file is job-local:

```text
/mnt/localssd/posttrainbench/$USER/$JOB_ID/
  posttrain_container_*/job_dir/task/system_monitor.log
```

The analysis joins that telemetry to timestamped `solve_out.txt` events and to the live process
cgroup (`/slurm/uid_*/job_$JOB_ID/...`). Definitions:

- **compute:** sampled `utilization.gpu > 0`;
- **resident but not computing:** GPU memory is nonzero, but sampled utilization is zero;
- **full idle:** GPU memory is zero and there is no compute process;
- **idle interval:** consecutive full-idle samples, weighted by actual timestamp gaps.

The final open sample is not extrapolated. Boundaries have about ±1 minute uncertainty. vLLM is
bursty, so a zero-utilization sample with model memory still resident is not treated as full idle.

## 3. Aggregate findings

Across 32 effective cells:

| State | Share of observed GPU time |
|---|---:|
| Actual GPU compute | 80.6% |
| Model resident, sampled utilization zero | about 4.4% |
| Fully idle | 15.0% |

There were 183 full-idle intervals:

| Statistic | Duration |
|---|---:|
| Median | 3.0 min |
| Mean | 3.9 min |
| P90 | 9.0 min |
| Maximum | 26.1 min |

Of those intervals, 101 lasted at least three minutes, 45 at least five minutes, 16 at least ten
minutes, and four at least fifteen minutes.

### 3.1 Time evolution

Idle is front-loaded. About 60% of all observed idle minutes occurred in the first 30 minutes.

| Time since agent start | Full-idle share |
|---|---:|
| 0–30 min | 44.7% |
| 30–60 min | 2.6% |
| 60–90 min | 10.4% |
| 90–120 min | 7.0% |
| 120–150 min | 9.8% |

Every cell first made a model resident on its GPU within about 2–6 minutes.

### 3.2 Task, setup, and base-model effects

The trace-aligned task snapshots showed about 16.1% full idle for GSM8K and 14.4% for AIME2025.
GSM8K spent much more time in vLLM, about 23% versus 9% for AIME, because its agents more often
generated rejection-sampling data.

No material effort/context gradient was visible:

| Setup | Compute | Full idle |
|---|---:|---:|
| max / 1M | 81.0% | 15.0% |
| xhigh / 1M | 80.6% | 15.0% |
| high / 1M | 79.5% | 16.2% |
| max / 200K | 81.5% | 14.1% |

Base-model differences were also small. SmolLM3 had the highest sampled compute share, about 83%;
the Qwen and Gemma groups were around 80%. Agent strategy and failure recovery explain more
variance than effort, context window, or base-model size.

## 4. Typical phase structure

The dominant pattern is:

```text
read harness and inspect environment
→ baseline vLLM evaluation
→ download/filter/tokenize/decontaminate data
→ SFT
→ save and package checkpoint
→ vLLM checkpoint evaluation or RFT generation
→ CPU-side result analysis and next-stage data selection
→ another SFT or GRPO stage
```

The first 5–15 minutes are commonly full idle because downloading, parquet/JSONL processing,
tokenization, contamination checks, and training-script authoring are CPU/network work. Once a
model is loaded, SFT is the longest continuous high-utilization phase. Checkpoint packaging,
process cleanup, data-pool construction, and model reloads normally create 2–6 minute gaps.

Long `sleep` trace events do not by themselves mean idle: agents often sleep while a background
trainer continues at 95–100% GPU utilization. A long sleep becomes waste only when the child has
already exited or crashed.

## 5. GSM8K cells

| Cell | Full idle | Longest idle | Snapshot interpretation |
|---|---:|---:|---|
| g01 | 20.4% | 15 min | downloads/data build, SFT OOM repair; later SFT2 |
| g02 | 16.7% | 9 min | SFT → checkpoint eval → RFT generation → GRPO |
| g03 | 16.0% | 6 min | full-eval diagnosis; later vLLM RFT generation |
| g04 | 9.6% | 4 min | SFT → eval/RFT → SFT2; one of the steadiest cells |
| g05 | 11.2% | 7 min | data prep and CE repair; later vLLM RFT generation |
| g06 | 13.5% | 9 min | rejection generation and SFT/script restart |
| g07 | 24.3% | 18 min | self-matching `pgrep -f` watcher; self-recovered at 07:29 |
| g08 | 15.8% | 5 min | repeated vLLM sampling/RFT; many resident batch gaps |
| g09 | 12.5% | 6 min | GRPO checkpoint failure; detected, repaired, and resumed |
| g10 | 19.7% | open at snapshot | GRPO save failure followed by blind `sleep 1800` |
| g11 | 12.5% | 10 min | data prep; long SFT then long vLLM RFT; transition at snapshot |
| g12 | 28.3% | 27 min | continuation SFT OOM hidden by fixed sleep; later recovered |
| g13 | 13.8% | 4 min | long vLLM RFT → SFT → GRPO; efficient pipeline |
| g14 | 21.1% | 12 min | GRPO diagnosis, context compaction, EOS/masking fixes |
| g15 | 11.2% | 11 min | startup repair followed by nearly continuous SFT |
| g16 | 11.2% | 9 min | decontamination/near-dup checks → SFT → vLLM RFT |

At about 07:31 UTC, GSM8K had five SFT cells, four GRPO cells, five vLLM eval/RFT cells, one
transitioning cell, and one anomalously idle cell (`g10`).

### 5.1 Confirmed GSM8K failure-recovery cases

- **g07:** SFT2 completed, but two watchers used `pgrep -f` with a pattern that matched their own
  command lines. The agent stopped them after about 18 idle minutes and resumed checkpoint eval.
- **g10:** GRPO failed at checkpoint step 50/260 because `do_sample=False` was paired with
  `temperature=0.0` in an invalid generation config. The agent then entered a fixed 30-minute
  sleep and had not inspected the failure by the snapshot.
- **g12:** continuation SFT OOMed, but a 1700-second fixed sleep delayed detection by about 27
  minutes. The agent reduced batch size and recovered.
- **g09:** a related GRPO checkpoint failure was detected and repaired after about six minutes.
- **g14:** twelve idle minutes were used for genuine diagnosis and script changes, not a deadlock.

## 6. AIME2025 cells

| Cell | Full idle | Longest idle | Snapshot interpretation |
|---|---:|---:|---|
| a01 | 10.0% | 11 min | startup data preparation, then SFT |
| a02 | 13.3% | 13 min | math-data download/filtering, then SFT |
| a03 | 16.0% | 10 min | checkpoint eval followed by SFT2 |
| a04 | 18.0% | 10 min | data pool/decontamination and trainer restarts |
| a05 | 21.3% | 11 min | checkpoint evaluation and next-stage data prep |
| a06 | 10.0% | 9 min | sustained main SFT |
| a07 | 13.3% | 10 min | checkpoint/grid eval and SFT2 preparation |
| a08 | 14.0% | 9 min | SFT1 checkpoint pass@1/dev evaluation |
| a09 | 21.3% | 17 min | longer initial data preparation |
| a10 | 21.3% | 13 min | A1 eval analysis, packaging, and v1 preparation |
| a11 | 12.7% | 12 min | block-dataset preparation, then SFT |
| a12 | 5.3% | 4 min | one of the highest-utilization cells |
| a13 | 18.0% | 7 min | stopped chain, removed Liger, rewrote trainer, resumed |
| a14 | 10.7% | 6 min | AIME24/HMMT25 temperature sweep |
| a15 | 13.6% | 9 min | data preparation followed by sustained SFT |
| a16 | 14.9% | 8 min | dev sweep, v2 data selection, then SFT2 |

At about 07:30 UTC, 14 AIME cells were updating weights with SFT and two were running vLLM
checkpoint/dev sweeps. No AIME agent had launched GRPO, PPO, or another RL method. No AIME cell
showed a persistent hang or GPU-isolation failure.

## 7. What vLLM was doing

Observed vLLM uses were local inference, never an external model API:

- baseline measurement before training;
- checkpoint evaluation and decoding-parameter sweeps;
- GSM8K rejection-sampling generation from the `openai/gsm8k` train split;
- pass@k sampling on AIME checkpoints;
- local development evaluation on AIME2024, HMMT, and in some agents AIME2025.

For example, `g02` loaded `ckpt/sft1`, sampled eight solutions per GSM8K training problem at
temperature 1.0/top-p 0.97, kept at most four exact-answer-correct generations, and wrote RFT data.
`a07` used vLLM to compare temperature/repetition-penalty settings for a SmolLM checkpoint.
`a16` compared greedy and sampled decoding for a Gemma checkpoint on AIME24/AIME25.

## 8. Operational lessons

Most full idle is legitimate R&D overhead. Confirmed avoidable waste in this snapshot is only a
small fraction of total GPU time, but it is concentrated in failure detection. Recommended rules:

1. Use `wait "$pid"` and inspect the exit code instead of fixed long sleeps.
2. Do not use raw `pgrep -f` patterns that can match the watcher itself; retain the child PID.
3. After the first 30 minutes, treat a fully idle GPU for five minutes as a warning and ten minutes
   as an investigation trigger. Do not auto-kill solely on utilization.
4. Distinguish full idle from a vLLM-resident zero-utilization sample.
5. Record the last log modification time and process existence in any watchdog notification.
6. Run multiple decoding configurations inside one vLLM server where possible to avoid reloads.
7. Pipeline CPU data preparation/decontamination with the previous GPU stage when allowed.
8. Keep monitoring actions receipt- and job-ID-scoped; never cancel by username or partition.

## 9. Limitations and follow-up

- These are partial-run observations, not final ten-hour utilization numbers.
- One-minute sampling can miss short inference bursts and shifts boundaries by about one minute.
- `python` process names require trace correlation to distinguish SFT from GRPO.
- AIME2025 evaluation access is permitted by the benchmark prompt, but repeated checkpoint or
  decoding selection against AIME2025 can introduce selection bias and should be reported.
- Recompute this report after agent solve, judges, and final evaluation finish. Separate agent-solve
  GPU utilization from post-agent judge/evaluation overhead in the final version.

Related documents:

- [Batch specification](../../spec/2026-09-01-ptb-gsm8k-aime2025-batch1.md)
- [Slurm runbook](../../reference/posttrainbench_dual_task_slurm_runbook.md)
