#!/usr/bin/env python3
"""Give the evaluation container the Hugging Face token.

PostTrainBench's evaluator loads the benchmark dataset inside
``vllm_debug.sif``, but the exec forwards only the OpenAI/OpenRouter/vLLM
keys. That is fine for an ungated benchmark such as gsm8k. GPQA Main is
gated: ``inspect_ai.hf_dataset`` calls ``datasets.load_dataset`` and fails
with ``DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated dataset on
the Hub. You must be authenticated to access it.`` Every evaluation attempt
then fails and the run produces no metrics.json and no per-card score.

``datasets`` reads ``HF_TOKEN`` from the environment, and PTB's .env already
carries the token as ``MY_HF_TOKEN``. Forward it into the evaluation exec
only. ``run_card_evaluation`` is derived from ``run_evaluation`` at runtime
(``declare -f``), so patching this one function covers the per-checkpoint
evaluation too.
"""

from __future__ import annotations

import sys
from pathlib import Path


MARK = "# awm: gated benchmark datasets need HF auth inside the evaluator"
# The agent exec carries the same VLLM/PYTHONNOUSERSITE pair, so anchor on the
# OPENROUTER line, which only the evaluation exec has.
ANCHOR = '        --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \\\n'
BLOCK = (
    '        --env HF_TOKEN="${MY_HF_TOKEN}" \\\n'
    '        --env HUGGING_FACE_HUB_TOKEN="${MY_HF_TOKEN}" \\\n'
)


def _marked(text: str) -> bool:
    return 'HF_TOKEN="${MY_HF_TOKEN}"' in text


def apply(text: str) -> str:
    if _marked(text):
        return text
    if text.count(ANCHOR) != 1:
        raise SystemExit("run_task.sh: expected exactly one evaluation env block")
    return text.replace(ANCHOR, ANCHOR + BLOCK, 1)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "src/run_task.sh")
    old = path.read_text()
    new = apply(old)
    if new != old:
        path.write_text(new)
        print(f"{path}: patched (HF token for the evaluation container)")
    else:
        print(f"{path}: already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
