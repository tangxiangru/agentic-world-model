# E repair construction record

Reference commit: dcfa742. Isolated branch: codex/exp-protocol-e-repair-20260905.
This is a new reviewable candidate, not a mutation of frozen E or a promotion.
The component spec and directions ledger contain accepted/rejected alternatives.

## Validation

Final changed-surface native CPU suite: **91 passed in17.05s** (sampling, readiness,
decode evidence, execution and Stop hook). It uses the extracted fixed image's
Python3.10/vLLM0.11/Transformers4.57.3 and actual local Gemma tokenizer. Native
ModelConfig resolver and SamplingParams are real; engines/outputs are inert.
No model forward, training, evaluation, GPU device or queue call was performed.
Native resolver tests confirm do_sample-only has no temperature override and
explicit temperature0 survives. Existing native-source hash/signature tests pass.

Initial ordinary host run:46passed/6skipped (missing native packages). First native
run51passed/1skip (tokenizer cache not yet linked), then84passed after readonly
cache linkage and execution coverage; final91 includes the Stop hook change.
These are successive verification stages, not independent scientific samples.

Ruff checks and git diff whitespace checks pass. The skill-creator quick validator
reports only the existing exp_protocol underscore naming restriction and exits
early. Name is deliberately preserved per user instruction; this is not claimed
a full validator pass. Frontmatter, reference routing and Python syntax were
inspected independently. No text-matching tests were added.

Reproduction uses bwrap readonly root, fresh /dev (no GPUs), only the CPU test
output directory writable, CUDA_VISIBLE_DEVICES empty and HF/Transformers offline
flags. Runtime path: /tmp/exp-protocol-save-runtime.JEZlHo/rootfs. Candidate tests
are read from this checkout; pytest's pure-Python package is borrowed from the
operator .venv. A read-only data/ptb/hf symlink points at the existing cache; it is
not part of the candidate. The additional --unshare-net option was rejected by
the outer sandbox's NETLINK_ROUTE restriction, so execution retained the managed
network restriction and offline flags; no network requests were used.

An optional full23-module protocol/sandbox regression was started separately
(session82826) and remained live without a completed output at this record.
It is not included in the91passed claim. The targeted changed-surface verification
is complete; any later full-suite result or failure must be recorded separately.

## Remaining validation

Independent forward review remains owned by the planner and must use the frozen
candidate without an answer key. CPU evidence cannot establish actual GPU engine
startup/shutdown, time saved, model loadability, official scores or package effect.
Receipt-backed GPU discovery is required for those claims. No future training
input hash is waived; no temperature is rewritten; no global cleanup is added.
