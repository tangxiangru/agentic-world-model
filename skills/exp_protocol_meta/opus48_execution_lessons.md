# Opus4.8 execution lessons, 2026-09-05

Use these process lessons when reviewing or constructing the next packages.
The first eight-attempt comparison has six clean results, one flagged complete
result and one startup/truncated scientist result. It is not an eight-clean window.

- Record the effective serving route. `do_sample:false` alone does not establish
  greedy behavior under the pinned vLLM route; preserve selected file bytes,
  request fields and native resolution separately. A verified file hash is not
  evidence that its intended decoder was used. Do not automatically choose a
  decoder or impute the gain observed in another cell.
- Use current inputs for sampling, persist generated data, then lock a new
  training card against its actual inputs. Keep strict hashes. Detect inevitable
  card/source/API failures before allocating an engine. Do not certify future
  data or treat a lock override as authority to bypass runtime invariants.
- Wrapped execution time includes training and evaluation. Classifier hits for
  `protocol_hours`, first model publication, RL and waiting require raw-trace
  confirmation. A tool name or TODO is not an executed model/artifact event.
- Deployment identity includes worker-readable paths. Never submit from a
  node-local `/tmp` development checkout. Use a verified shared WorkDir,
  entrypoint and context-record location; local image/source checks do not prove
  visibility on a worker. Preserve early failed attempts separately from actual
  scientist/model outcomes and count spend across replacement batch names.
- Environment acceptance is its own receipt-backed operation. Synthetic native
  checks, source/image/data identities, actual process admission and raw records
  do not become benchmark scores or clean scientist cells. A node-specific pass
  admits only that frozen node subset; every placement consumer must agree.

The completed review and E-repair decision are in
`doc/exp_protocol_iterations/analysis-2026-09-05-opus48-wave1/` and
`doc/spec/2026-09-05-exp-protocol-e-repair-discovery.md`. Recipe observations,
including BF16, RFT and few-shot effects, remain conditional precedents rather
than mandatory scientist rules.
