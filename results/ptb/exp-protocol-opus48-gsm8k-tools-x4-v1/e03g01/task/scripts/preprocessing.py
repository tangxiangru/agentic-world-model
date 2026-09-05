"""Pure formatter for RenderedTrainingBundle. No model-executing imports.

Rows are {prompt, completion} where prompt is the already-rendered gemma3 chat
prompt (ends with '<start_of_turn>model\n') and completion is the supervised
target ending with '<end_of_turn>'. Prompt is pre-rendered, so prompt_mode
must be 'pre_rendered'.
"""
from awm.exp_protocol.rendered_training import RenderedParts


def render(row, *, template, settings, rng):
    return RenderedParts(prefix=row["prompt"], target=row["completion"])
