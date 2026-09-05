# Window04 focused audit: card provenance and rendered stop-token semantics

Read-only focused reviewer correction, 2026-09-04. No repository edits, GPU work,
training/evaluation command, Slurm action, or git mutation. This report is evidence,
not a candidate/queue/promotion decision.

## Source map

Workspace: `/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator`.

`B03` = workspace + `/results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r03`.
`B05` = workspace + `/results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r05`.
`T03` / `T05` = uncompressed line numbers of the corresponding
`solve_parsed.txt.gz`. All trace timestamps below are UTC.
`R03` / `R05` = workspace +
`/doc/exp_protocol_iterations/trace-reviews/window04-local/reports/g01r03.md`
and `g01r05.md`.

The raw result path was resolved through each bundle's `status.json`:

- g01r03, job90649:
  `/home/robtang_google_com/gangda_workspace/agentic-world-model-gangda-exp-protocol-evolve/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1_g01r03_formal_r1/gsm8k_google_gemma-3-4b-pt_90649`.
- g01r05, job90651:
  `/home/robtang_google_com/gangda_workspace/agentic-world-model-gangda-exp-protocol-evolve/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1_g01r05_formal_r1/gsm8k_google_gemma-3-4b-pt_90651`.

Both bundles name shipped AWM `4ae3d87c446bbda9732537a72b2f0fb3f96ac35a`.

## 1. Rule7: content redaction is not full rule compliance, nor contamination proof

### The exact rule and harness boundary

Both shipped `task/skills/exp_protocol/SKILL.md` files, L107–108, say:

> The benchmark's test copy is input to the contamination checker only.
> No item from it in `failure_examples`, `watch_set`, or training data.

There is no express ID-only exception. Do not silently reinterpret this as only
"no verbatim text" while calling the result full compliance. Read literally,
test-selected watch sets still contain benchmark items even if represented by IDs.
If the intended policy permits evaluator-log IDs, that is an unresolved policy
boundary to clarify, not an exemption already present in the shipped wording.

The raw g01r05 `prompt.txt`, L8 and L21, expressly allows `evaluate.py` queries
and `--limit` for development. L25–30 restricts the supplied test-data copy to
the contamination checker; L35–36 prohibits test questions/answers and specific
test-item-derived examples in training, but allows general style/domain matching.
Thus permitted evaluator feedback, explicit stricter card-rule breaches, and
training contamination must be reported separately. The source of the card data
here is evaluator output, not demonstrated direct extraction from the separate
`../test_data.json` reference file.

### g01r03: explicit card-rule breach extends beyond "once"

- `B03/task/memory/cards/exp-02.yaml`, L34–36, stores the Cody/Amir cookie test
  question with gold `20` in `failure_examples[0]`; the source is the retained
  developer Inspect log `2026-09-03T18-38-43+00-00_gsm8k_9RNbQSCRKwfMvXTEW3Hotb.json`.
  The accepted card write is T03 L4867 onward, 19:01:21 (an earlier write at
  19:00:36 was rejected for not reading the file). This is an explicit breach of
  the card clause. "Verbatim" should mean content-level reproduction; the trace's
  source item contains a non-breaking space that the card normalizes.
- `exp-03.yaml` L30–33 has three test IDs with question descriptions and golds
  `400`, `150`, `3`. `exp-04.yaml` L30–34 and `exp-05.yaml` L32–36 each retain four
  test-item descriptions with golds `150`, `3`, `50`, `9500`. They do not become
  allowed failure examples by paraphrasing the question. R03 L132's "broken once"
  understates the strict rule's scope even though exp02 is the clearest full-text case.
- The diagnostic lineage is explicit: `B03/task/scripts/diagnose_eval.py` L24–48
  reads Inspect samples and writes `wrong_ids`, target, model answer and output tail;
  later cards cite those diagnostics.

### g01r05: final watch files are ID-only; historical and literal qualifications matter

- T05 L4674–4693, 18:55:50: the scientist initially writes all189 incorrect base
  dev200 samples as `{id, question, gold}` to `analysis/exp-01_failures.jsonl`.
  The output prints the same Cody item and gold20. This is an observed transient
  test-content watch-set artifact, not a hypothetical inference from a card field.
- T05 L4702–4716, 18:56:00–18:56:01: it rewrites that file to `{id}` only, explicitly
  citing rule7. This is a genuine corrective content-redaction action. Do not report
  the cell as never having stored the question/gold watch set.
- T05 L6340–6354, 22:03:10: it creates `analysis/exp-02_failures.jsonl` directly from
  the58 incorrect exp02 developer Inspect samples, selecting only `id`.
- Both bundle and raw retained files were read: exp01 has189 rows; exp02 has58;
  every retained row has exactly the single key `id`.
- `B05/task/memory/cards/exp-02.yaml` L37–43 deliberately withholds question/gold,
  but retains item IDs in `failure_examples` and names the189-row file as `watch_set`.
  `exp-03.yaml` L37–44 and `exp-04.yaml` L34–41 similarly use test-derived failure
  examples and the58-item watch set. Their `model_output` fields also retain
  item-specific numeric/semantic detail, so placeholder `question`/`gold` fields
  are not proof of complete absence of item information.
- `B05/task/scripts/eval_report.py` L62–71 reads these IDs to report fixed,
  still-failing and regression counts by matching another Inspect log. The observed
  consumer is diagnostic evaluation, not training.

Correct R05 L137/L155's "fully compliant counterpart" framing: this is a
content-redacted evaluator-derived watch-set practice, with a transient full-content
write and an unresolved/violated literal no-test-items card/watch-set clause. It is
not a clean binary compliant/noncompliant source pair for a text-only guard.

### What this does and does not establish about training and score

Observed training lineage is separated from these diagnostics:

- g01r03 `scripts/build_data.py` L22–23,59–88,94–106 loads GSM8K train demos and
  OpenMathInstruct2 GSM8K-family source rows; `scripts/train_sft.py` L33–58 reads
  only the command-named JSONL's `prompt`/`completion` for model inputs and labels.
  The exp02 card L52–70 names `data/sft_v1.jsonl`; later training cards name the
  corresponding larger built files. The read code does not consume card YAML or
  analysis watch sets.
- g01r05 `scripts/build_sft_data.py` L38–39,104–150 uses OMI2 and GSM8K train;
  `gen_rft.py` L21–57 gathers GSM8K train plus OMI2 augmented questions;
  `mix_rft.py` L19–42 mixes those generated rows with an unseen-source replay slice.
  `train_sft.py` L133–160 reads command-named JSONL into rendered supervised tokens.
  Exp06's `--exclude-questions` input is the previous RFT source file, not a test
  watch set (`exp-06.yaml` L59–61).

This is evidence against direct card/watch-set ingestion in the inspected paths,
not a universal proof that cards can never affect training or scores. Cards and
evaluator diagnostics demonstrably inform scientist decisions. A clean lexical
contamination checker also cannot exclude every semantic/test-derived training
path (the harness says so). No such specific forbidden training derivation is
established by this focused audit. Replace R03 L135's "No score consequence
(cards never enter training)" with the bounded statement: "Explicit card-rule
breach; no demonstrated training-data contamination or quantified score effect in
the inspected training lineage."

Do not invalidate g01r03's official0.8278999241849886 (82.79%) on these assumptions.
Its status remains complete/eligible with no judge flags; g01r05 likewise remains
official0.7611827141774071. This audit changes neither validation nor score.

### Hazards in the bare question/gold scanner proposal (no implementation/design decision)

R03 L149 onward proposes scanning free text for any benchmark question or gold
answer. Bare numeric gold matches such as `3`, `20`, `150` are non-specific: they
occur in legitimate train examples, model outputs, metrics, counts and reasoning.
Even whole-token matching would reject unrelated valid content; substring matching
is worse. A number hit cannot establish benchmark-item provenance. Such a scanner
would also miss paraphrases, aliases and ID-only test selection, while exact-text
matching can miss whitespace/Unicode normalization differences (already present
in the Cody example). It risks both frequent nuisance blocks and false reassurance.
Reading the checker-only test reference for a new unrelated scanner also requires
an explicit access-contract reconciliation. g01r05 is not an unqualified compliant
negative-control source for that proposal. None of this selects or rejects a candidate.

## 2. g01r05 stop_token_consistent: raw-schema false alarm, not rescued termination

### Original trainer before the failure already terminated the supervised sequence

The original `scripts/train_sft.py` is preserved in T05's Write at18:39:05
(L3392 onward), well before the18:56:52 lock failure:

- T05 L3428–3435: `build_strings()` creates the model prefix and then
  `full = f"{prefix}{completion.strip()}<end_of_turn>\n"`.
- T05 L3518–3543: it reads raw `completion`, renders `full`, tokenizes it with
  `add_special_tokens=False`, drops overlength/retokenization-drift rows, and assigns
  `labels = [-100] * len(pre_ids) + full_ids[len(pre_ids):]`. The appended terminator
  therefore belongs to the supervised target, not just an ignored/prompt segment.
- T05 L3547–3549 asserts EOT at the final or penultimate token for the first2000
  retained rows and prints that sampled targets end with EOT.
- The pre-failure smoke run actually exercised this renderer: launch18:44:16
  T05 L3888–3893; result18:44:54 L3921–3928 reports exact grader-template equality,
  EOT id106,400 retained rows, and "all sampled targets end with <end_of_turn>".
  (This is retained evidence, not a new training run; launch-scope compliance is a
  separate audit.)

The displayed edits before18:56:52 add speed/memory controls and few-shot data
prefixes, not removal of the renderer's EOT. In particular the old-string patch
at18:57:14 still names the preexisting `build_strings(..., r["completion"])` call.

### What failed and what changed

- Shipped `preflight.py` at commit4ae3d87, L32–35,95–105,190–218, reads the first500
  raw JSONL rows, picks `completion` before other fields, and tests raw suffixes.
  It never invokes the training renderer.
- T05 L4888–4909,18:56:52: correct raw-field observation `0/500` completion strings
  end with EOT; lock refuses. It is an actual preflight FAIL, but a false alarm for
  the alleged model-level missing-termination defect.
- T05 L4918–4919,18:57:05: scientist says "The stop-token check is right to fail".
  That is the scientist's interpretation; it is contradicted as a runtime-defect
  diagnosis by their own earlier renderer and smoke output.
- T05 L4921–4969,18:57:14: builder changes raw `completion: target` to
  `completion: body + STOP_TOKEN`, adding `completion_body: body`. Dedup and length
  filtering switch to the same bare body. Trainer gains `body_of`, checks the raw
  consistency, and feeds the body (without EOT) into the SAME `build_strings` which
  appends EOT. The old and new renderer paths produce the same supervised string
  for the same prompt/body. The patch avoids double-appending by changing both
  producer and consumer; it does not newly supply a previously absent training EOT.
- Pre-rebuild output at18:54:14 (T05 L4458–4465) and post-rebuild at18:57:19
  (L4996–5004) each report95642 rows and identical source counts. Printed completion
  character statistics increase by13, exactly the literal EOT length. This supports
  raw-representation change, not a new termination-learning intervention.
- At18:57:41, T05 L5045–5060 reports contamination0/95642, raw stop check500/500 and
  successful lock. Launch18:57:46 is T05 L5065–5069. The failure-to-lock interval is
 49seconds; do not count an alleged2.3h rescued run as an observed benefit.
- Final artifacts corroborate this path: `B05/task/scripts/train_sft.py` L33–47,
 133–166 and `build_sft_data.py` L92–100. Raw `sft_v1.jsonl` still has the old
 completion-only/no-EOT schema; raw `sft_v2.jsonl` has the body+EOT/body schema.
 The exact pre-rebuild sft_v2 file was overwritten, so do not claim a retained
 byte-for-byte pre/post dataset comparison.

### Score and ledger inference corrections

R05 L14,L126–130,L143 should not attribute the SFT gain to this preflight repair,
call it a true missing-termination positive, or claim it reverses ledger#9.
This is another instance of the raw-field/script-owned-termination mismatch
described by ledger#9/#17, with compliance cost appearing as a data rewrite rather
than an override. Zero recorded overrides does not mean zero nuisance failures.
This is a mechanism classification, not a whole-ledger/candidate decision.

The directly matched n200 observation is0.055 ->0.710 (+65.5pp), accompanied by
non-cap outputs0.59 ->0.99 (`exp-02.yaml` L119–132). It compares the base against
a two-epoch95.6k-row SFT model plus target formatting, teacher reasoning, few-shot
augmentation and greedy output config. It does not isolate termination from the
other interventions, and especially does not isolate a checker whose old training
code already supervised EOT. The card itself L48–53 distinguishes the two deficits
and predicts stopping alone would recover the4pp gap0.055 ->0.095, with teacher
data improving reasoning. Do not mix base n200 with post-SFT n1319 as a matched
comparison in the causal description.

Even the diagnostic is more narrowly "not reported as max_tokens" than literal
EOT compliance: `scripts/eval_report.py` L35–37 increments `stopped` whenever
`stop_reason != "max_tokens"`; L57 names it `share_stopped_before_cap`. It does
not distinguish EOT from EOS or other termination reasons. `exp-01_diagnostic.json`
L2–6 and `exp-02_report.json` L3–7 provide the actual measures.

Recommended factual replacement: "Preflight rejected un-terminated raw completion
fields, prompting a49-second schema/consumer adjustment and recheck. The existing
training renderer already appended and supervised EOT; this was not evidence of a
rescued mis-terminated run. The SFT package improved matched dev200 accuracy by65.5pp
and reduced cap hits, with no isolated attribution to the checker or stop token."
