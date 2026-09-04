# Opus 4.8 cross-benchmark launch record

Status: preparing production acceptance; no new scientific jobs submitted yet.

Study contract: `doc/spec/2026-09-04-wma-opus48-crossbench.md`. The user first
requested the broader comparison and then explicitly waited for implementation
completion. The new implementation record states completion; this task then
reproduced the shared-UID isolation failure, fixed the resource policy without
removing the I/O boundary, and added explicit single/multi-self/multi-joint
study modes. The existing code and analysis work was preserved.

The scientist and WMA model route is Opus4.8/high/200k; the diagnostic partner
remains Opus5/max. The host route probe returned MODEL_ROUTE_OK. Shared tests
exercise real kernel file/network/inheritance canaries and the protocol flows;
the production SIF and real model/broker gate are still required.

The planned scientific matrix is GSM8K/BFCL/HumanEval x R/P/S/M/J x four repeats.
GPQA-main is blocked by the dataset's access permission, after a legitimate
request using the existing Hugging Face login. BFCL and HumanEval contamination
assets were downloaded through the repository helper and their hashes recorded.

At the initial check the queue was ownership-clean but underfilled: nine GPUs
allocated and zero pending, 80 validator/judge-clean old results under the old
contracts. The current c10r08 tail and G/H wave remain running. They are not
dependencies for this new study's independently specified baselines.

Evidence and deployment audits: `evidence/2026-09-04-crossbench/`.

## Deployment compatibility and validation

At 2026-09-04T13:34:11.028776+00:00, the selected exp_protocol/WMA/sandbox/launcher/operator regression suite passed, including real Landlock/seccomp canaries in the non-root shared UID environment. The new high/200k scientist scaffold shell test and changed-source Ruff checks also passed.

The common deployment changes add explicit study modes, keep current review/action/version checks, supervise each probe group without imposing a UID-wide NPROC=128, and register/hold/route validation jobs through the formal receipt lifecycle. Validation jobs are excluded from scientific harvesting. Opus4.8 formal WMA cells additionally require a matching real SIF/model/broker acceptance artifact; the receipt freezes its hash. The completion hook discovers all receipt-backed wma-prefixed study manifests within the same subqueue and stratifies reports by task/model/mode.
