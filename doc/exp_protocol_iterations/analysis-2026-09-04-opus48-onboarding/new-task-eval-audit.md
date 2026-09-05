# GPQA Main / HumanEval: pinned PTB evaluation-chain audit

2026-09-04. Read-only evidence for the user-approved new-task integration. **Not task approval, implementation, safe-execution certification or release authorization.** No model, forward, training, benchmark evaluator, generated program, network, Slurm or git command was run. No actual question, gold answer, test case, dataset row, AWM-full result or held-out result was opened. Only source/package metadata and reference-file existence were inspected. Repository unchanged.

## Outcome: adding APPROVED_TASKS is insufficient

1. **Both wrappers emit `accuracy` and `stderr` in the inspected pinned library route.** GPQA uses scorer `choice`; HumanEval uses scorer `verify`, not GSM8K's `match`, and not an emitted pass@k metric family.
2. **Both local `test_data.json` assets are missing.** AWM's task-asset check already rejects missing/untracked references; `run_task.sh:204–206` hard-fails before solving if the file is absent. Dataset-cache availability, GPQA access approval and full-population metadata remain unverified.
3. **HumanEval is not safely isolated by its selected `sandbox="local"`.** That is a temporary cwd plus a same-user subprocess which inherits the evaluator environment. The formal outer container exposes RW repository/results mounts, GPU devices, environment values and no explicit network isolation. Safely running arbitrary generated Python requires a separate verified execution-isolation decision, not merely registering the task.
4. **Current completion validation does not establish full evaluated/scored n, finite accuracy or task-specific metric semantics.** It checks a nonempty metrics object with any numeric field and a nonempty final-eval text log. A present partial/stale metrics file also stops retries. Strong prospective official-log retention/count validation must be integrated and tested.
5. **Actual defaults differ from plausible documentation assumptions:** GPQA randomizes choices without an explicit shuffle seed; HumanEval's unused `NUM_EPOCHS=5` and README pass@k narrative do not make this route five epochs. The executable task leaves epochs unspecified, and Inspect falls back to **1**.

## Source/provenance scope

`R = /home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator`

`P = R/third_party/PostTrainBench`

The requested PTB reference is `dcf5da031435c54e3680b6ec3f63e7e317efc13e`; this audit read the corresponding local checkout source without using git. File hashes below identify exactly what was inspected; commit equality was not independently queried through git.

Pinned native source was read at:

`N = /tmp/exp-protocol-save-runtime.JEZlHo/rootfs/usr/local/lib/python3.10/dist-packages`

This is the existing **opus_5 extraction**, authorized read-only after the main agent clarified root-read permissions. Metadata reports `inspect_evals 0.3.103.dev83+g06001a83e`, `inspect_ai 0.1.dev3780+g64db0afdd`, `datasets 4.5.0`. `P/containers/opus_5.def:70–84` and `vllm_debug.def:51–65` declare the same Inspect source commits `06001a83e6d7c709c2ede0570dce7f1031a0bad8` and `64db0afdd3796732b232954ef440c66ed22923a7`. The **actual vllm_debug.sif contents/importability were not independently executed/extracted in this audit**; same build declarations do not prove image-byte/runtime equality.

## 1. Task contracts and denominators

| Field | GPQA Main | HumanEval |
|---|---|---|
| PTB source | `P/src/eval/tasks/gpqamain/evaluate.py` | `P/src/eval/tasks/humaneval/evaluate.py` |
| Dataset route | `Idavidrein/gpqa`, config `gpqa_main`, split **train**, lines114–122 | P wrapper68–70 → `N/inspect_evals/humaneval/humaneval.py:71–80`: `openai/openai_humaneval`, split **test** |
| Identity/target mapping | PTB lines133–144: ID=`Record ID`, question +4 answer choices, initially target A; dataset shuffle then remaps target | Native lines145–155: ID=`task_id`; input instruction+prompt, target canonical solution; test and entry_point stored separately in metadata |
| Solver/scorer | `multiple_choice(cot=True)` + `choice()`; source123–127 | `generate()` + `verify()`; native77–79; code extraction, execution and exit-status scoring83–138 |
| Metric keys | `choice` decorator uses `accuracy(),stderr()`, `N/inspect_ai/scorer/_choice.py:40–41`; C/I score78–81 | `verify` decorator uses `accuracy(),stderr()`, native83–84; C/I result107–118 |
| Default developer limit | **50**, PTB37–42 | **150**, PTB23–28 |
| Default token cap / concurrency / GPU-memory fraction | **16000 /6 /.8**, PTB50–69 | **4000 /1 /.3**, PTB41–55 |
| Epochs | Explicit **1**, PTB27,127 | **1 via Inspect fallback**; native `NUM_EPOCHS=5` at42 is unused by its Task; P wrapper passes no epochs. `N/inspect_ai/_eval/task/run.py:213`, `_util/constants.py:8` give DEFAULT_EPOCHS=1. |
| Formal full selection | `--limit -1` makes wrapper omit Inspect limit, PTB80–81 | Same, PTB64–66 |
| Full numerator/denominator | Number of C outcomes / actual scored selected rows (one epoch), not requested limit or a fixed constant embedded in wrapper | Number of verification subprocess successes / actual scored selected rows (one epoch), not pass@5 or `NUM_EPOCHS*population` |

`N/inspect_ai/scorer/_metrics/accuracy.py:29–33` computes the mean of mapped scores; these scorers return binary C/I. The wrappers copy all metrics from the sole results score record to JSON (`GPQA:103–111`, `HumanEval:90–98`). They **do not write n, scorer key, dataset revision, completion status or artifact binding into metrics.json**.

**Exact full-population n is not frozen or verified here.** Neither wrapper pins dataset revision, declares expected population, or asserts full scored n. No dataset rows/reference contents were read to obtain a number. The integration must establish an approved metadata-only dataset snapshot/count contract and then require `results.total_samples/completed_samples`, selected scorer `scored_samples/unscored_samples`, unique typed ID+epoch counts and actual metric numerators to reconcile. Do not certify a familiar nominal GPQA/HumanEval size from memory or from `--limit -1` alone.

### GPQA presentation variability

`hf_dataset(... shuffle_choices=True)` calls `dataset.shuffle_choices()` without a seed (`N/inspect_ai/dataset/_util.py:231–250`). `MemoryDataset.shuffle_choices(seed=None)` creates `random.Random(None)` and shuffles/remaps each sample's choices/target (`_dataset.py:316–335`). This is **not fixed-seed choice order**. Same Record ID across evaluations can therefore have different actual prompt options and target letters. Preserve choices, rendered messages and target mapping; an ID-only/GSM8K-style pair join must not label it unchanged protocol. Freezing a new shuffle seed would change the existing evaluator presentation contract and must be explicit, not an unnoticed AWM-only fix.

### HumanEval documentation discrepancy

Native README:109–113 describes n5/pass@{1,2,5}, but native source does not pass the constant at42 to Task or register pass@k metrics. Report the **actual one-epoch verification accuracy** for this route; the general pass@k literature/README is not executable evidence of its present contract.

## 2. Formal orchestration → metrics → validators/judges

`P/src/run_task.sh` performs:

1. Copy the task's evaluator and templates into scientist task space (83–94); require/copy the independent contamination reference and checker (199–209).
2. Run scientist in Apptainer; copy final_model to result root and copy task snapshot after model cleanup (505–515).
3. Run **all four official judges before formal evaluation** (528–578), using task metadata and a fresh checker/reference copy (`judge_lib.sh:147–163`). Missing verdicts warn at that point and can be recovered later; they are not silently sufficient for full completion.
4. Formal evaluator uses `vllm_debug.sif`, untouched source task evaluator, selected final_model, formal templates and **`--limit -1`**, writes result-root metrics.json and `final_eval_<attempt>.txt` (600–642).
5. Retry4 times at default cap; then **3** attempts at GPQA12000/HumanEval3000; then2 attempts at GPQA8000/HumanEval2000 (669–730). The second-phase comment says “up to2” but executable call is3. Each attempt has an outer28800s timeout, TERM then60s kill-after (659). It stops on **metrics-file existence**, not validated success or n (651–662).
6. `POST_TRAIN_BENCH_REQUIRE_COMPLETE=1` invokes `validate_completed_run.py --judge-profile official` (738–740); AWM launch env sets this requirement (`awm/ptb_experiments.py:680`).

Therefore “formal contract” includes cap fallback/attempt identity, not one unconditional token cap. Do not pool fallback scores as identical serving invocations without recording the successful attempt and cap. The stdout text is per attempt; stderr is not redirected into that same file at line642. Full raw Inspect JSON defaults to the evaluator working directory's logs unless separately directed; pinned `run_task.sh` has **no** `INSPECT_LOG_DIR`/official-archive integration.

`R/tools/ptb_official_eval_evidence.py:1–5` explicitly says its prospective archiver is **not wired into PTB/operator yet**. Its presence/tests do not repair the pinned formal path. Preserve per-attempt source/hash, exit/outcome, metric-file identity and actual Inspect n before source/scratch cleanup; this must also prevent a stale metrics file from becoming an apparent current-attempt success.

PTB validator currently requires nonempty trace/monitor/provenance/model/metrics and any nonempty `final_eval_*.txt` (50–74); model metadata/shard existence (76–99); **any numeric metric**, not specifically finite accuracy or exact counts (101–106). It verifies exact canonical schemas and value types for contamination/model, API, PTB-lookup and general judges, preferring nonempty `_rerun` files (12–30,42–46,116–134). Flagged booleans do not themselves make the file structurally invalid.

AWM `ptb.audit_result` invokes that validator (1381–1396). `ptb_results.discover_attempts` reads accuracy/stderr (156–159,192–194), matches frozen batch/cell provenance and placement; judge flags are separately collected. `build_report` distinguishes clean/flagged counts, but its primary accuracy summary includes eligible flagged attempts (241–266); consumers must use explicit **judge-clean** filtering and per-task grouping. A mixed GPQA/HumanEval batch should not present its aggregate accuracy mean as one shared task estimand.

Task-level judge metadata already exists: GPQA forbids `Idavidrein/gpqa`; HumanEval forbids `openai/openai_humaneval`, `openai_humaneval`, `evalplus/humanevalplus`, and lists `Muennighoff/mbpp` as allowed. These are read by `get_judge_prompt.py:35–49`; keep broader benchmark-derived-data prohibitions, not only literal string matches. GPQA's HF split called **train** remains the benchmark test population for this workflow.

## 3. Dataset access and dependency readiness

- Both task-local `test_data.json` files are currently **absent** (existence check only). AWM task assets check requires evaluator/reference/info present and tracked (`ptb_experiments.py:726–737`); missing files must be provisioned/frozen through the legitimate task-data process, not an empty JSON stub to satisfy existence.
- `download_test_data.py:126–135` explicitly requires **MY_HF_TOKEN** for gated GPQA and downloads `Idavidrein/gpqa/gpqa_main/train`. The evaluator itself supplies neither an explicit token nor revision; `hf_dataset` can load an existing Inspect/HF cache or invoke HF loading (`N/.../dataset/_sources/hf.py:92–111`). The formal `--cleanenv` list does not explicitly forward MY_HF_TOKEN/HF_TOKEN. Licensed cache provisioning or scoped authenticated dataset access must be checked without exposing credentials to generated code.
- HumanEval reference downloader uses `openai/openai_humaneval/test` and canonical-solution fields (186–191); its evaluator separately needs `prompt/test/entry_point` source fields. Same source revision/reference identity must be established; the current scripts do not pin one.
- Both require native Inspect/datasets/vLLM/Transformers; GPQA choice scoring needs no external LLM judge key, HumanEval needs native Python/anyio sandbox subprocesses. Image definitions pin vLLM0.11.0 and source commits, while the base requirements file's inspect-ai0.3.150 is subsequently **overridden by the source install**; inspect dist-info, not that one requirements line, identifies the final library.
- Existing extracted package source/metadata is present; importability and a safe execution backend were **not exercised** in this read-only audit. Standard-looking image definitions are not evidence that a Docker daemon, nested-container privilege or a secure HumanEval runner exists.

## 4. HumanEval code execution: concrete safety gap

`N/inspect_evals/humaneval/humaneval.py:83–121` extracts generated code, concatenates task prompt + answer + unit-test code + entry-point check, and calls **`sandbox().exec(["python","-c",...],timeout=30)`**. Timeout becomes an incorrect result; otherwise success is the subprocess exit status. Code/scorer semantics are not a restricted language interpreter, and a successful process exit is not independently authenticated proof that every test ran.

`N/inspect_ai/util/_sandbox/local.py:21–81` creates a temporary directory and launches as the **current user**; its user argument is ignored. `util/_subprocess.py:123–130` inherits **all parent environment values**, then adds supplied env. Its timeout cleanup targets the immediate Process (225–242); it does not establish containment of arbitrary descendants. There is no inner filesystem jail, PID/user/network namespace, GPU-device exclusion, memory/CPU/process quota or secret-stripping in this local backend.

Formal outer container (`run_task.sh:612–642`) does provide a container filesystem and PID namespace, but the generated verification child shares them with the trusted evaluator:

- PTB repository and complete result directory are explicit **RW binds** (633–634), containing evaluator code and, by this stage, judged result files. Generated code can potentially alter trusted source/evidence/metrics reachable through those mounts. Judges ran **before** these code executions and cannot automatically detect later mutation.
- `--nv` exposes GPUs (613); the verification child does not need them but inherits access. `--cleanenv` only establishes the outer environment; the inner child inherits its explicit provider/API values, cache paths and VLLM endpoint context. Actual secret values were not inspected.
- No `--net`/network-none appears in the formal or scientist container invocation. `-c`/`--cleanenv`/`--pid` is **not** external-network isolation. Mounted repository configuration/cache locations may expose additional sensitive data; their contents were not read.
- Scientist-side developer HumanEval calls use the same local backend inside the scientist sandbox, which also carries scientist credentials and work. Outer containment alone does not separate generated code from that trusted process/data.

**As written, this is not a demonstrated safe route for untrusted generated Python.** Changing only the task allowlist cannot fix it. Switching blindly to Docker is also not validation: PTB explicitly selects local, and nested Docker requirements/capabilities are unproven.

A contract-preserving *semantic* route is possible only after design/verification of a dedicated per-verification executor: same Python/scorer/code-extraction/tests/30s result mapping, but no credentials, network, GPU devices, repository/results/weights/cache mounts or access to parent processes; bounded temporary work, resources and descendant cleanup. The trusted evaluator alone writes metrics/logs. Benign standard-library programs must remain behaviorally compatible; denied dependencies/infrastructure failures must not silently become model-quality negatives. This changes the **execution-isolation contract** and therefore needs explicit frozen source/image/config provenance and acceptance; it cannot be quietly advertised as unchanged current formal runtime. No such change is implemented or authorized by this audit.

## 5. Required integration checks before launch readiness

1. **Freeze task profile beyond allowlist:** evaluator source/image/package identities, dataset config/split/revision/reference hash, metadata-derived expected population, epochs, scorer/metric keys, template/decode and fallback caps. Do not read test contents into scientist construction; do not add arbitrary pass@k metrics or epochs based on the stale README.
2. **Provision legitimate task references/access:** nonempty correct reference from approved source, gated GPQA access/cache, offline reproducibility where required, and no blanket provider/token injection. AWM asset checks and frozen PTB commit implications must be resolved, not bypassed with untracked task files.
3. **Resolve HumanEval isolation before exercising real generated code.** CPU acceptance should use synthetic non-benchmark programs to check success/failure/timeout, environment stripping, outside-path denial, no external network/GPU/parent access, fork/descendant cleanup and resource limits. No real benchmark evaluator/model is needed for these negative tests.
4. **Wire durable official attempt evidence and validation:** full successful Inspect raw JSON, exact task/scorer/ID+epoch/counts, finite accuracy/stderr, cap/epochs/choice-presentation state, model identity and metric-file/source hashes. Test truncated/partial/error/stale/conflicting evidence with synthetic files; current numeric-file validator is insufficient.
5. **Exercise AWM end-to-end consumers with synthetic result bundles:** discovery/harvest/provenance, all four judge schemas and judge-clean gating, placement quarantine, rerun preference, metrics display/task-stratified summaries. Keep failed/truncated code-runner attempts as failures, not synthetic zero/valid complete outcomes.
6. **Task-aware paired/diagnostic adapters:** GPQA uses `choice` and shuffled choice/target mappings; HumanEval uses `verify` and unit-test/entry-point/code-execution bindings. A `.scores.match` GSM8K-only parser cannot certify these. Keep benchmark row IDs, tests, targets and failure examples out of scientist training/watch-set inputs.

No official result, score denominator or runtime-safety claim has been manufactured. These checks are implementation/readiness requirements for the main agent; this report does not change the official evaluator, launch a task or decide the separate Opus4.8 route.

## Exact inspected-file hashes

| File | SHA256 |
|---|---|
| P GPQA evaluate.py | `cd1fbea148951874e2113ed9f0a329e824fd033b746ea011f9e7035011f071dd` |
| P HumanEval evaluate.py | `75e1cc1f1bde4c7e1f5e8f7122e0b9f70415b96d5fd1800ec7014c4d6952392c` |
| P src/run_task.sh | `ae32afd7f534fabd928f9be1c78d20639f4efb17e128f2038907a3065333f54c` |
| P src/utils/validate_completed_run.py | `54cc7bf197abb1173ab88b3b3b5a8a05a444b9539b8f56cd86ca2f2bfa81654f` |
| R awm/ptb_results.py | `abbc08e2ad71d432788954809eed19bf2972404f1758a2ed683ba9571c5025f8` |
| N inspect_evals/humaneval/humaneval.py | `e7bd00d5002afa39f8c42ac5183bcb65feb7692f069eca41a32bad44024f6aaa` |
| N inspect_ai/util/_sandbox/local.py | `9027c4dd868fae3c768a28063da38fbfbf3fe3d0b30524d3fdfb58c05e8384e3` |
| N inspect_ai/util/_subprocess.py | `21d236056a82984788d2a92f6a6b798350e982d665521136e287e4ee7d9363bc` |
