# Packaging audit

Status: portable **static experiment-design bundle**. The package preserves the
frozen 1,000 checkpoint-by-generation-policy cells; it is not by itself a
self-contained evaluator or authorization to launch them.

## Preserved design

- 1,000 unique experiment rows, 400 selected checkpoint IDs, and 10 requested
  full benchmark passes per row.
- Exact checkpoint IDs/URIs, benchmark assignments, development/locked-session
  splits, four phase files, seven generation-policy payloads, and seven complete
  request templates.
- The selected-ID semantic digest and all recorded external provenance hashes.
- No labels, raw trajectories, model tensors, tokenizer files, credentials, or
  prior completion text were copied into this directory.

## Portable-reference changes

Only references changed during packaging:

1. Workstation-local absolute repository references became repo-relative or
   immutable remote references.
2. Local Hugging Face cache paths became authenticated, immutable URLs under
   dataset revision `446127629d7b271d537390e69bfb2d960a3aa515`.
3. Raw trajectory/evaluator references became authenticated, immutable URLs
   under dataset revision `cc2ac9d884a7d962a6024ab0d5cd8ed3370070de`.
4. Markdown links to unbundled analysis files became non-clickable provenance
   paths so a fresh clone does not present them as bundled links.

The historical `/home/ben/task/...` script strings in checkpoint records were
not rewritten: they describe what the original scientist declared, not files
that this package promises to execute.

The original `matrix_summary.json.source_hashes` map is intentionally retained.
It hashes generation-time inputs, including the ignored source artifact; it is
not the checksum manifest for this normalized package. Use
[`bundle_files.sha256.json`](bundle_files.sha256.json) for packaged bytes and
[`source_pins.json`](source_pins.json) for source classification and revisions.

## External launch prerequisites

The evaluator code, benchmark prompt data, scorers, templates, full runtime
image, model weights, and tokenizer artifacts are not bundled. Their pinned
references and recorded hashes are in `protocol.json` and `source_pins.json`.
Private Hugging Face objects may require independent credentials; none are in
this package. Before spending cells, the runbook must still enforce all gates in
`protocol.json`, `checkpoint_audit.md`, and `runtime_preflight.md`.

## Integrity and design-equivalence checks

Packaging verified that:

- the source and packaged matrices contain the same 1,000 JSON objects in the
  same order;
- all four source and packaged phase files are byte-identical;
- `selected_ids_400.json`, `selected_checkpoints.jsonl`, all generation-policy
  payloads, and all request templates are byte-identical;
- no workstation-specific absolute paths or dead relative Markdown links remain;
- all packaged JSON and JSONL parse successfully.

The SHA-256 inventory covers every packaging-owned file except itself. Root-owned
`README.md`, `AGENT_HANDOFF.md`, and `RUNBOOK.md` are explicitly excluded so
their authors can finalize them without invalidating this manifest.
