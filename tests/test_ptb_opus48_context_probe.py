import json
import runpy
from pathlib import Path

import pytest

probe = runpy.run_path(str(Path(__file__).parents[1] / "tools/ptb_opus48_context_probe.py"))


def raw(*, model="claude-opus-4-8[1m]", context=1_000_000, cli="2.1.219", result="OK"):
    return "\n".join(json.dumps(event) for event in [
        {"type": "system", "subtype": "init", "claude_code_version": cli},
        {"type": "result", "is_error": False, "result": result,
         "modelUsage": {model: {"contextWindow": context}}},
    ])


def test_matching_native_cli_provider_result():
    assert probe["summarize"](raw(), 0)["verified"]


@pytest.mark.parametrize("changes", [
    {"model": "claude-opus-5[1m]"}, {"context": 200_000},
    {"context": "1000000"}, {"cli": "2.1.260"}, {"result": "model unavailable"},
])
def test_requested_model_or_context_is_not_inferred_from_a_success_code(changes):
    assert not probe["summarize"](raw(**changes), 0)["verified"]


def test_missing_result_and_nonzero_exit_are_unverified():
    assert not probe["summarize"]("apptainer: command not found", 127)["verified"]
    assert not probe["summarize"](raw(), 1)["verified"]
