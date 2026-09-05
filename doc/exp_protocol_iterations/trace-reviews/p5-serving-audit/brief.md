# P5 supplemental serving-contract review

This supplements the already-closed Window03/strict-guard review; it is not a new eight-clean window. No additional cells are counted or launched.

Supplemental read-only P5 review of already completed exp_protocol trace windows, not a new eight-clean analysis window. You are a trace reviewer. Follow skills/exp_protocol_meta/trace_review.md and .claude/agents/trace-reviewer.md, overridden to read-only tools and final-output reports: NO file edits, git mutations, network, model/evaluator execution, or Slurm operations. Do not read live/running experiment outputs. Read only the three harvested bundles and relevant frozen evaluator source/docs.

Question: Does existing evidence isolate serving concurrency as the cause of developer-vs-official GSM8K score differences, or only show confounded/non-repeatable reads? Audit three cells:
1 results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r01 (official .7103866565579985).
2 results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/g01s02 (official .7247915087187263).
3 same strict batch/g01s07 (official .7354056103108415).
Full cohort review is already done; scope is evaluator-contract ambiguity. Existing per-cell reports are in doc/exp_protocol_iterations/trace-reviews/window03-local/cells/g01r01.md and round01-strict-guard-addendum/cells/g01s02.md,g01s07.md; use them as locators, verify raw trace and retained actual Inspect JSON/logs, not summaries alone. Read relevant metrics.md rules. Use tools/exp_protocol_cell_read.py and trace_timeline as needed; never load whole traces. Each claim needs trace L and timestamp or exact retained source path/JSON pointer.

For each cell build a compact matrix for final artifact developer read(s), official read, and any matched repeat: exact checkpoint/artifact evidence, actual n/items evidence, accuracy, concurrency, GPU memory utilization, max_tokens, seed, dtype/quantization, decoding settings, templates/evaluator code identity, process reuse/restart and whether each is known/unknown/different. Distinguish requested command settings from resolved evaluator metadata; latest similarly named logs are not necessarily final artifact. Separate within-config repeat variability from cross-config gaps, and identical-correctness from byte-identical responses. Do NOT assume greedy determinism or concurrency causality. No new statistical significance from unpaired aggregate scores.

Return one Markdown report with three per-cell sections, a cross-cell conclusion, exact missing evidence that would distinguish causes, and at most one justified protocol-surface proposal if two cells establish a distinct mechanism not merely C-style uncertainty. No experiment should be proposed simply to inflate held work. A no-new-candidate conclusion is valid. Do not change the frozen evaluator contract. Limit the report to relevant evidence (~2000 words), mark unknowns honestly.
