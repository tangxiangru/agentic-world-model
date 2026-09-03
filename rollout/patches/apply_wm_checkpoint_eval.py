#!/usr/bin/env python3
"""Officially evaluate every archived card checkpoint, not only final_model.

Recorder-mode runs preserve one checkpoint per experiment card under
``task/wm/checkpoints/<card>/`` (``awm wm archive``). PostTrainBench only
copies ``final_model`` out before ``delete_hf_models.py`` strips weights from
the task dir, and only evaluates ``final_model`` — so every card except the
adopted one would lose its checkpoint and never earn a test-set label. This
adds two mechanical steps:

1. next to the ``final_model`` copy, copy ``task/wm/checkpoints`` to
   ``$EVAL_DIR/wm_checkpoints`` (before the weight strip);
2. after PTB's own evaluation finishes, run the byte-identical evaluation —
   ``run_evaluation`` renamed and re-pointed via ``declare -f | sed`` at
   runtime, so the protocol stays whatever PTB's is — once per archived
   checkpoint, writing ``$EVAL_DIR/wm_metrics/<card>.json`` and a log per card.

The official ``metrics.json`` path is untouched and runs first; the sweep only
consumes GPU time after PTB is done with it. Idempotent: running it twice
changes nothing. Applied by rollout/setup.sh to the private checkout.

    python rollout/patches/apply_wm_checkpoint_eval.py <ptb>/src/run_task.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

MARK_COPY = "# --- awm: preserve card checkpoints (rollout/patches/apply_wm_checkpoint_eval.py) ---"
MARK_SWEEP = "# --- awm: evaluate card checkpoints (rollout/patches/apply_wm_checkpoint_eval.py) ---"

COPY_ANCHOR = (
    'if [ -d "${JOB_DIR}/task/final_model" ]; then\n'
    '    cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"\n'
    "fi\n"
)

COPY_BLOCK = f"""
{MARK_COPY}
if [ -d "${{JOB_DIR}}/task/wm/checkpoints" ]; then
    cp -r "${{JOB_DIR}}/task/wm/checkpoints" "$EVAL_DIR/wm_checkpoints"
fi
"""

SWEEP_ANCHOR = (
    'echo "================================"\n'
    'echo "======= EVALUATION DONE ========"\n'
    'echo "================================"\n'
)

SWEEP_BLOCK = f"""
{MARK_SWEEP}
# One official evaluation per archived card checkpoint. run_card_evaluation is
# run_evaluation with only the model path, the output file, and the log
# re-pointed, derived at runtime so the evaluation protocol cannot drift from
# PTB's own.
if [ -d "$EVAL_DIR/wm_checkpoints" ]; then
    mkdir -p "$EVAL_DIR/wm_metrics"
    eval "$(declare -f run_evaluation | sed \\
        -e 's/^run_evaluation/run_card_evaluation/' \\
        -e 's|$EVAL_DIR/final_model|$AWM_WM_CKPT|' \\
        -e 's|${{EVAL_DIR}}/metrics.json|$AWM_WM_OUT|' \\
        -e 's|$EVAL_DIR/final_eval_${{eval_num}}.txt|$AWM_WM_LOG|')"
    for _awm_ckpt in "$EVAL_DIR"/wm_checkpoints/*/; do
        _awm_card="$(basename "$_awm_ckpt")"
        [ -f "$_awm_ckpt/config.json" ] || continue
        export AWM_WM_CKPT="${{_awm_ckpt%/}}"
        export AWM_WM_OUT="$EVAL_DIR/wm_metrics/${{_awm_card}}.json"
        export AWM_WM_LOG="$EVAL_DIR/wm_metrics/${{_awm_card}}_eval.txt"
        for _awm_args in "" "$MAX_TOKENS_ARG"; do
            [ -f "$AWM_WM_OUT" ] && break
            echo "card checkpoint evaluation: $_awm_card (max_tokens='$_awm_args')"
            timeout --signal=TERM --kill-after=60s 7200s bash -c "$(declare -f run_card_evaluation with_huggingface_overlay); run_card_evaluation \\"$_awm_args\\" card" || true
        done
        [ -f "$AWM_WM_OUT" ] || echo "card checkpoint evaluation FAILED: $_awm_card" >&2
    done
fi
"""


def apply(text: str) -> str:
    if MARK_COPY in text and MARK_SWEEP in text:
        return text
    if text.count(COPY_ANCHOR) != 1:
        raise SystemExit("run_task.sh: expected exactly one final_model copy block; "
                         "the runner changed shape — update apply_wm_checkpoint_eval.py")
    if text.count(SWEEP_ANCHOR) != 1:
        raise SystemExit("run_task.sh: expected exactly one EVALUATION DONE banner; "
                         "the runner changed shape — update apply_wm_checkpoint_eval.py")
    text = text.replace(COPY_ANCHOR, COPY_ANCHOR + COPY_BLOCK, 1)
    return text.replace(SWEEP_ANCHOR, SWEEP_ANCHOR + SWEEP_BLOCK, 1)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "src/run_task.sh")
    text = path.read_text()
    patched = apply(text)
    if patched != text:
        path.write_text(patched)
        print(f"patched {path}")
    else:
        print(f"{path} already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
