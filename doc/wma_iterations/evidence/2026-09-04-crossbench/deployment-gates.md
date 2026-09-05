# Deployment gates: Opus 4.8 / 200k and isolated WMA broker

Read-only review at 2026-09-04 12:49 UTC. No sbatch/srun/container/model execution, no repository edits, and no validation record fabricated. Paths below are relative to `/home/robtang_google_com/gangda_workspace/agentic-world-model` unless absolute.

## Observed environment and ownership

`/rmeng_data/robtang/bin/awm-slurm-queue current --subqueue gangda_wma_evolve --summary` returned OWNERSHIP OK,9/16 GPUs allocated (node2:4,node3:5), no pending at12:49 UTC. This is allocation, not utilization. PTB local configuration is:

- scheduler slurm; partition `ptb-a3`; reservation `robtang-ptb-a3`;
- subqueue `gangda_wma_evolve`; nodelist `slurm2-a3nodesetondem-[2-3]`;
- ownership registry `/rmeng_data/robtang/slurm-queue/registry.json`;
- apptainer `/rmeng_data/robtang/tools/apt-root/usr/bin/apptainer`, library path `/rmeng_data/robtang/tools/apt-root/usr/lib/x86_64-linux-gnu`;
- containers `/rmeng_data/robtang/ptb-containers/opus_5.sif` and `vllm_debug.sif` exist;
- Vertex `CLAUDE_CODE_USE_VERTEX=1`, project `sercan-v1`, region `global`, `VERTEX_REGION_CLAUDE_4_8_OPUS=global`.

I have not verified node2/node3 Landlock ABI or actual provider permission/model resolution. Presence of the routing variable proves neither.

## Gate 1: genuine provider/profile validation

The genuine existing probe is `third_party/PostTrainBench/src/commit_utils/slurm/context_probe.sh AGENT MODEL RECORD ISOLATED_HOME`, invoked by `single_task.sbatch:223` under `--runtime-smoke` after the normal node preflight and one-H100 torch smoke.

Within the owned Slurm runtime-smoke job, the exact existing payload is:

```bash
bash src/commit_utils/slurm/context_probe.sh \
  "$AGENT" 'claude-opus-4-8' \
  "$POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD" \
  "$JOB_SCRATCH/context-probe-home"
```

AGENT must refer to a tracked profile declaring the requested effort and `PTB_AGENT_REQUESTED_CONTEXT_TOKENS=200000`. The current existing profile `claude_vertex_max_200k` is max/200000 and non-AWM. If the approved experiment is high/200k or uses AWM, a matching tracked scaffold/profile/allowlist must first be created and approved as a setup; selecting max merely because it exists changes the treatment. `claude_vertex_high_awm` currently declares the old1M setup and cannot represent200k by renaming a record.

The probe's real Claude command is `claude --print --verbose --output-format stream-json --model MODEL --effort PROFILE_EFFORT --setting-sources '' --safe-mode --no-session-persistence --dangerously-skip-permissions 'Reply with exactly OK.'`, inside the pinned SIF, cleanenv, isolated home. It needs the profile's env_passthrough routing, functioning GCE metadata/Vertex auth from the allocated node, the SIF CLI, and egress. No scientist credential file is required by this probe. Do not print credentials when checking auth.

The probe writes RECORD and sibling `*.stream.json`. Schema1 record includes:verified,verified_at,provider,project,region,slurm_job_id,node,gpu_ids,cli_version,container/container_sha256,requested_model,resolved_model,canonical_model,requested/resolved_context_tokens,effort,terminal_reason,api_error_status,raw_trace/raw_trace_sha256. It computes the actual image digest. A process success with no API error and exact context200000 is required by the script.

**Additional evidence inspection is necessary:** the script's `verified` predicate checks context and API success but does not require the resolved/canonical model to equal Opus4.8; `local_issues` also checks requested_model, not resolved_model. Inspect the raw `modelUsage` and canonicalModel for model identity and fallback before accepting the record. A valid Opus5/200k record cannot serve Opus4.8. Preserve the original raw trace and hashes. This is not PTB score evidence.

After real completion, `uv run awm ptb check MANIFEST` verifies model request,route,CLI2.1.219,container hash,effort,verified=true,and exact requested/resolved context against the manifest. Formal submit additionally freezes the validation file digest into each cell receipt. Do not overwrite a prior profile's record in place; use a new named path such as `data/ptb/context-validation/claude-opus-4-8-200k-high.json` when high is the frozen effort.

## Receipt gap in the current context-smoke entry point

`awm/ptb_experiments.py:1036 submit_context_smokes` runs `local_issues(require_context=False)+site_issues`, calls `build_launches(...purpose='context-smoke')`, adds `POST_TRAIN_BENCH_REQUIRE_CONTEXT_VALIDATION=0`, then submits `--runtime-smoke --walltime 00:15:00` and returns job IDs. It **does not** hold jobs,write a receipt,register ownership,freeze source,check held routing,or release after registration. `submit.sh` does not register either. Therefore the current command below is a real provider probe entry but **not currently a receipt-backed safe submission command for this line**:

```bash
uv run awm ptb context-smoke MANIFEST --cell REPRESENTATIVE_CELL
```

Minimum correct adaptation before executing it: reuse the existing formal submission lifecycle (`submit` around lines1080–1265) for a validation-purpose receipt. Build launches with `hold=True,purpose='context-smoke'`; preserve clean/source snapshot, frozen top/PTB SHA and validation target; allow the context output to be absent only for this validation kind. Submit each held runtime-smoke, record returned actual IDs immediately, call `slurm_queue.register_receipt`, run `held_job_routing_issues` against exact node[2-3], then release only those actual registered IDs. Failed registration/routing leaves the actual jobs held. A small operator wrapper using these existing helpers is possible; the current CLI does not implement it. Do not create a dummy provider validation to get ordinary submit through its dependency gate.

Use a distinct validation kind/receipt and keep its jobs out of scientific harvest,judge-clean result counts,promotion comparisons and queue experiment-reserve claims. One profile request is enough per distinct model/effort/image/route; if only kernel differs, repeat the broker OS gate on each node, not redundant provider calls solely to fill pending.

## Gate 2: real isolated broker under the production SIF

The normal sidecar in `src/run_task.sh:304–342` uses the same scientist SIF but **no --nv**, cleanenv, `/opt/awm:ro` frozen private checkout, `/history:ro`, `/session:ro` with separately writable cards and `.wma`, private output and dedicated home/tmp. Broker subprocesses then use Landlock ABI>=3 and seccomp. Tests on the host do not prove the same policy establishes inside the allocated-node SIF.

Current `--runtime-smoke` only tests torch GPU count plus context_probe, then exits. It never starts `run_task.sh` or `awm.wma.isolation`. There is no existing broker/container-smoke CLI or runtime-smoke hook in the inspected code. A named committed validation payload must be added or an operator-owned validation batch entrypoint must invoke it; it must use the receipt lifecycle above, not an ad hoc unregistered sbatch.

Minimum container command design for that payload (variables must resolve to the receipt-frozen image/checkouts and the validation job's own scratch):

```bash
apptainer exec --containall --cleanenv \
  --env PYTHONPATH=/opt/awm --env PYTHONNOUSERSITE=1 \
  --bind "$FROZEN_AWM:/opt/awm:ro" \
  --bind "$VALIDATION_INPUTS:/smoke:ro" \
  --bind "$JOB_SCRATCH/broker-tmp:/tmp" \
  --home "$JOB_SCRATCH/broker-home:/home/ben" \
  --pwd /home/ben --writable-tmpfs "$PINNED_IMAGE" \
  python3 /smoke/broker_smoke.py
```

`broker_smoke.py` above is a proposed new payload, **not an existing file or executed command**. Keep it small and explicitly fail,never pytest-skip,on unsupported Landlock/seccomp. Reuse real production `isolated_tools`, `run_probe`, `broker_call` and MCP stdio transport, creating synthetic canaries in private scratch. Required checks:

1. CPU/static exported-card read succeeds; unexported synthetic file/cache read and source write fail; a library indirect read and child inherit the denial.
2. AF_INET,AF_INET6,AF_UNIX sockets and `/proc/1/environ` reads fail; credentials/cache env are absent; normal child works but setsid/setpgid escape fails.
3. Actual MCP initialize/tools-list/tools-call works; allowed read succeeds,unexported read returns tool error,run returns output,write_result is published only to intended output. A fixture's fake model is insufficient for the next gate.
4. Preserve image digest,top/PTB/code SHA,node/job/UID,kernel release,Landlock ABI,Python/CLI version,checks,stdout/stderr,exit code and transport transcript in a separate broker-smoke record. It is not the provider context validation JSON and not a PTB result. No fixed schema currently exists for it.

`tests/test_wma_isolation.py` contains direct implementations of these canaries, but its `sandbox` fixture converts unavailable Landlock/seccomp into pytest.skip. Merely running pytest with exit0 is not deployment proof. The production `isolated_tools` self-check fails closed and must remain so.

## Gate 3: genuine model → broker round trip

After Gates1/2, use a bounded actual Claude run in the same production-shaped sidecar container, using its frozen `awm.wma.backends.CLIBackend`/review entry and a synthetic prelaunch card with explicit harmless code export. Require the real model to call list_inputs/read_file/run/write_result and produce a schema-valid verdict through broker publication. Preserve stream-json plus measured isolation metadata (`landlock-seccomp-mcp-v1`, exported hashes,limits),request/reply,actual model/effort/context,cost and timestamps. The broker CLI adds `--bare --disable-slash-commands --tools '' --strict-mcp-config --mcp-config PATH`; check installed CLI help recognizes them before paying for the request. `context_probe` tests --safe-mode etc but does not test these broker flags or the transport.

This is an actual API request and is intentionally **not executed by this read-only audit**. The existing high_awm scientist scaffold and `run_task.sh` differ from the broker's private mount/auth surface. In particular the sidecar currently passes VERTEX_REGION_CLAUDE_5_OPUS but not VERTEX_REGION_CLAUDE_4_8_OPUS explicitly; verify the sidecar's chosen model under global routing, or add the explicit4.8 routing variable if that model is intended for WMA. The scientist's4.8 profile does not silently change the WMA model; freeze the two independently.

## Concrete remaining blockers

- Current approved tasks are only gsm8k/aime2025; approved agent setups contain Opus5 only. Opus4.8,GPQA (`gpqa`),and BFCL (`bfcl`) need the authorized contract/setup additions and tracked complete benchmark assets before launcher validation. Do not loosen arbitrary unrelated guard constants.
- No genuine Opus4.8 context record exists in `data/ptb/context-validation` at inspection.
- Context-smoke currently lacks receipt lifecycle and frozen source; no existing broker-SIF smoke entry.
- Node2/node3 kernel/container broker capability and real model→MCP transport remain unverified; source/unit tests do not clear them.
- Exact scientist effort/profile and WMA model need to match the parent's approved manifest; this audit does not infer them from an available scaffold.

Once these gates pass, baseline/control cells on the same runtime contract can be queued immediately without waiting for unrelated old tails. This audit submitted zero jobs and produced no success claim for an unexecuted validation.
