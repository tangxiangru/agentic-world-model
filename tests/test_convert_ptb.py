"""PostTrainBench conversion: committed samples always, full runs when present.

The committed samples under ``tests/data/posttrainbench/`` are verbatim line
subsets of the four runs the converters were written against — one per CLI
format (see ``make_samples.py``); the ``full_runs`` fixture points at those four
complete runs and skips when they are not on this machine.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import pytest

from awm.paths import raw_dir
from awm.traj import convert_claude_code, convert_codex, convert_cursor, convert_opencode
from awm.traj.posttrainbench import (
    NoAgentOutput,
    RunDir,
    build_run,
    compact,
    convert_run_dir,
    detect_harness,
    event_kind,
    iso_from_ms,
    iter_run_dirs,
    make_run_dir,
    parse_agent_config,
    parse_run_dir_name,
    read_line_stream,
    sniff_harness,
)
from awm.traj.schema import (
    MAIN_AGENT,
    events_path,
    meta_path,
    read_events,
    read_meta,
    validate_stream,
    write_run,
)

CLAUDE_CFG = "claude_non_api_max_claude-opus-4-8_10h_run1"
CLAUDE_RUN = "gsm8k_Qwen_Qwen3-1.7B-Base_17315721"
CODEX_CFG = "codex_non_api_high_gpt-5.4_10h_run1"
CODEX_RUN = "gsm8k_Qwen_Qwen3-1.7B-Base_16934887"
OPENCODE_CFG = "opencode_opencode_kimi-k2.5_10h_run2"
OPENCODE_RUN = "gpqamain_Qwen_Qwen3-1.7B-Base_16853750"
CURSOR_CFG = "cursor_cli_cursor-grok-4.5-high_10h_run2"
CURSOR_RUN = "healthbench_google_gemma-3-4b-pt_17417310"

FULL_RUN_DIRS = (
    ("claude", CLAUDE_CFG, CLAUDE_RUN),
    ("codex", CODEX_CFG, CODEX_RUN),
    ("opencode", OPENCODE_CFG, OPENCODE_RUN),
    ("cursor", CURSOR_CFG, CURSOR_RUN),
)

#: The seven published scaffolds produce four wire formats. ``glmx``, ``kimi``
#: (kimi_claude) and ``qwen3max`` are Claude Code pointed at another provider,
#: so they must sniff as claude-code; ``cursor`` must NOT, which is the whole
#: point of sniffing over a window instead of on the first object.
SCAFFOLD_HARNESS = {
    "claude": "claude-code",
    "glmx": "claude-code",
    "kimi": "claude-code",
    "qwen3max": "claude-code",
    "codex": "codex",
    "cursor": "cursor-cli",
    "opencode": "opencode",
}

#: The complete runs the acceptance numbers below were measured on. They are
#: too large to commit, so these tests read them from the fetched release and
#: skip when it is absent. AWM_PTB_SAMPLES points at a directory holding
#: ``run_claude/``, ``run_codex/``, ``run_opencode/`` and ``run_cursor/`` instead.
FULL_SAMPLES = Path(os.environ["AWM_PTB_SAMPLES"]) if "AWM_PTB_SAMPLES" in os.environ else None


@pytest.fixture
def ptb_samples(sample_dir: Path) -> Path:
    return sample_dir / "posttrainbench"


@pytest.fixture
def full_runs() -> dict[str, RunDir]:
    out = {}
    for name, cfg, run in FULL_RUN_DIRS:
        path = raw_dir("posttrainbench") / cfg / run
        if FULL_SAMPLES is not None and (FULL_SAMPLES / f"run_{name}" / "solve_out.txt").exists():
            path = FULL_SAMPLES / f"run_{name}"
        if not (path / "solve_out.txt").exists():
            pytest.skip(f"full {name} sample not available ({path})")
        out[name] = RunDir(path=path, agent_config=cfg, **parse_agent_config(cfg),
                           **parse_run_dir_name(run))
    return out


def _by_type(events, kind):
    return [e for e in events if e.type == kind]


def _sample_run(ptb_samples: Path, cfg: str, run: str) -> RunDir:
    return make_run_dir(cfg, ptb_samples / cfg / run)


# --- directory conventions -------------------------------------------------


def test_parse_agent_config():
    assert parse_agent_config(CLAUDE_CFG) == {
        "agent": "claude",
        "config": "non_api_max",
        "model": "claude-opus-4-8",
        "hours": 10.0,
        "gpus": None,
        "experiment": "run1",
        "run_index": 1,
        "context_1m": False,
    }
    assert parse_agent_config(CODEX_CFG)["model"] == "gpt-5.4"
    # "_1m_" marks the 1M-context variant and doubles the separator before the hours.
    m1 = parse_agent_config("claude_non_api_max_claude-fable-5_1m__10h_run1")
    assert (m1["model"], m1["context_1m"], m1["config"]) == ("claude-fable-5", True, "non_api_max")
    bare = parse_agent_config("codex_non_api_max_gpt-5.6-sol_10h")
    assert (bare["run_index"], bare["experiment"]) == (None, "")


def test_the_experiment_name_is_free_form_not_just_runN():
    # Upstream's template ends in an arbitrary ${EXPERIMENT_NAME}; one of the 62
    # published configurations uses it for something other than a repetition
    # index, and before this was parsed it aborted the whole conversion.
    odd = parse_agent_config("claude_claude-opus-4-6_10h_run1_old_container")
    assert odd["experiment"] == "run1_old_container"
    assert odd["run_index"] == 1
    # and it stays distinguishable from the sibling it shares 27 trajectories
    # with, which differs on nothing else a groupby would see.
    plain = parse_agent_config("claude_claude-opus-4-6_10h_run1")
    assert plain["experiment"] == "run1"
    assert {k: v for k, v in odd.items() if k != "experiment"} == {
        k: v for k, v in plain.items() if k != "experiment"
    }


def test_a_gpu_count_is_read_when_published_and_absent_otherwise():
    # `_${NUM_GPUS}gpu` is emitted only above one GPU, so its absence is unknown,
    # not one. No configuration in the current release carries it; parsing it is
    # what keeps the next one that does from being unparsable.
    multi = parse_agent_config("claude_non_api_max_claude-fable-5_1m__10h_8gpu_run1")
    assert (multi["gpus"], multi["run_index"], multi["model"]) == (8, 1, "claude-fable-5")
    assert parse_agent_config("claude_x_10h_8gpu")["experiment"] == ""
    assert parse_agent_config(CLAUDE_CFG)["gpus"] is None


def test_an_experiment_name_can_never_swallow_the_hours():
    # `rest` is lazy, so widening the tail's character class is what would let
    # the hours bind to an earlier token and silently change `hours` and `model`
    # — both of which reach the index. Every word of the tail must be
    # letter-initial; an hours token is not, and a word cannot contain "_".
    # Widen the class and these three go red.
    early = parse_agent_config("claude_x_2h_bar_10h_run1")
    assert (early["hours"], early["model"], early["config"]) == (10.0, "bar", "x_2h")
    dup = parse_agent_config("codex_gpt_10h_10h_run1")
    assert (dup["hours"], dup["model"], dup["config"]) == (10.0, "10h", "gpt")
    # A digit-initial tail is upstream emitting nothing of the sort, so it
    # raises rather than being read as an experiment name.
    with pytest.raises(ValueError):
        parse_agent_config("claude_x_10h_run1_1m")


def test_a_hyphen_in_the_experiment_name_is_not_a_second_outage():
    # The suffix that broke the conversion was `_old_container`; `_old-container`
    # would have been a second one, so words may hold "-" and "." internally.
    assert parse_agent_config("claude_x_10h_run1_old-container")["experiment"] == (
        "run1_old-container"
    )


def test_a_name_with_no_model_left_raises_rather_than_indexing_off_the_end():
    with pytest.raises(ValueError):
        parse_agent_config("a_1m_10h")


def test_parse_run_dir_name():
    assert parse_run_dir_name(CLAUDE_RUN) == {
        "benchmark": "gsm8k",
        "hf_org": "Qwen",
        "base_model": "Qwen3-1.7B-Base",
        "cluster_id": "17315721",
    }


def test_iter_run_dirs(ptb_samples: Path):
    runs = {r.agent: r for r in iter_run_dirs(ptb_samples)}
    assert set(runs) == {"claude", "codex", "cursor", "opencode"}
    assert runs["claude"].run_id == f"{CLAUDE_CFG}__{CLAUDE_RUN}"
    assert runs["codex"].benchmark == "gsm8k"
    assert runs["cursor"].benchmark == "healthbench"
    assert runs["opencode"].model == "kimi-k2.5"


def test_iter_run_dirs_skips_the_site_catalogue(tmp_path: Path):
    # fetch puts viewer_data/index.json in raw/ on every batch, so that directory
    # sits beside the configurations on any fetched mirror. It is flat today, so
    # nothing under it is reached; if upstream ever nests it per run, the name
    # guard is what stops "viewer_data" being parsed as an agent configuration.
    for cfg, run in ((CLAUDE_CFG, CLAUDE_RUN), ("viewer_data", CLAUDE_RUN)):
        d = tmp_path / cfg / run
        d.mkdir(parents=True)
        (d / "solve_out.txt").write_text("{}\n")
    (tmp_path / "viewer_data" / "index.json").write_text("{}")
    assert [r.agent_config for r in iter_run_dirs(tmp_path)] == [CLAUDE_CFG]


def test_make_run_dir_rejects_junk(tmp_path: Path):
    with pytest.raises(ValueError):
        make_run_dir("not-a-config", tmp_path / "nor-a-run")


# --- the line reader -------------------------------------------------------


def test_read_line_stream(ptb_samples: Path):
    claude = list(read_line_stream(ptb_samples / CLAUDE_CFG / CLAUDE_RUN / "solve_out.txt"))
    codex = list(read_line_stream(ptb_samples / CODEX_CFG / CODEX_RUN / "solve_out.txt"))
    # Measured: every claude line carries the "[ISO] " prefix, no codex line does.
    assert all(ts for ts, _o, _n, _r in claude[1:])
    assert not any(ts for ts, _o, _n, _r in codex)
    # Non-JSON launcher output is passed through, not dropped.
    assert [n for _ts, o, n, _r in claude if o is None] == list(range(1, 11))
    assert [n for _ts, o, n, _r in codex if o is None] == list(range(1, 11))
    assert [n for _ts, _o, n, _r in claude] == list(range(1, len(claude) + 1))
    assert claude[10][1]["type"] == "system"


def test_detect_harness(ptb_samples: Path):
    got = {r.agent: detect_harness(r.solve_out) for r in iter_run_dirs(ptb_samples)}
    assert got == {agent: SCAFFOLD_HARNESS[agent] for agent in got}
    assert set(got.values()) == {"claude-code", "codex", "cursor-cli", "opencode"}


def test_detect_harness_unknown(tmp_path: Path):
    p = tmp_path / "solve_out.txt"
    p.write_text('not json\n{"type": "something-else"}\n', encoding="utf-8")
    assert detect_harness(p) == "unknown"


def test_a_cursor_stream_is_never_read_as_claude_code():
    """Cursor's first object is an exact Claude Code marker.

    ``{"type": "system", "subtype": "init"}`` is byte-compatible between the two
    CLIs, and so are ``assistant``, ``user`` and ``result``. Deciding on the
    first object sent all 56 cursor runs to the Claude Code converter, which
    dropped their ``thinking`` and ``tool_call`` lines — 128k events — while
    reporting a clean conversion. The sniffer therefore votes over a window and
    tests cursor's own markers first.
    """
    claude_shaped = [
        (None, {"type": "system", "subtype": "init", "session_id": "s"}, 1, ""),
        (None, {"type": "assistant", "message": {"id": "m", "content": []}}, 2, ""),
        (None, {"type": "user", "message": {"content": "go"}}, 3, ""),
    ]
    assert sniff_harness(claude_shaped) == "claude-code"
    # One cursor-only line anywhere in the window is decisive, however many
    # shared names precede it.
    for tell in ("thinking", "tool_call", "connection", "retry", "interaction_query"):
        rows = claude_shaped + [(None, {"type": tell}, 4, "")]
        assert sniff_harness(rows) == "cursor-cli", tell


def test_sniff_harness_says_none_when_there_is_no_agent_output(tmp_path: Path):
    """41 published runs are a few hundred bytes of shell error and no JSON.

    That is a run that did not happen, not a format the converter failed to
    read, and ``build_run`` has to distinguish the two: one is skipped, the
    other is a failure that must red the exit code.
    """
    run_dir = tmp_path / CLAUDE_CFG / CLAUDE_RUN
    run_dir.mkdir(parents=True)
    (run_dir / "solve_out.txt").write_text(
        "/usr/local/bin/solve.sh: line 6: opencode: command not found\n", encoding="utf-8"
    )
    run = make_run_dir(CLAUDE_CFG, run_dir)
    assert detect_harness(run.solve_out) == "none"
    with pytest.raises(NoAgentOutput):
        build_run(run)


def test_sniff_harness_ignores_json_that_is_not_an_event():
    # A training subprocess's generation_config.json, interleaved into the
    # stream: it parses, it has no `type`, and it votes for nothing.
    blob = {"bos_token_id": 151643, "eos_token_id": 151645}
    assert sniff_harness([(None, blob, 1, "")]) == "none"
    assert sniff_harness([(None, blob, 1, ""), (None, {"type": "turn.started"}, 2, "")]) == "codex"


def test_the_sniff_window_is_bounded():
    # A solve_out reaches 1.5 GB; the sniffer may not read all of it. Past the
    # window the answer is fixed, which is the cost of the bound and is why the
    # window is 4x the largest measured distance to a decisive marker.
    from awm.traj.posttrainbench import SNIFF_WINDOW

    rows = [(None, {"type": "system"}, i, "") for i in range(SNIFF_WINDOW)]
    assert sniff_harness(rows + [(None, {"type": "tool_call"}, 999, "")]) == "claude-code"
    assert sniff_harness(rows[:-1] + [(None, {"type": "tool_call"}, 999, "")]) == "cursor-cli"


# --- shared line helpers ---------------------------------------------------


def test_event_kind_rejects_everything_that_is_not_a_type_string():
    assert event_kind({"type": "assistant"}) == "assistant"
    assert event_kind({"bos_token_id": 151643}) is None
    assert event_kind({"type": 7}) is None
    assert event_kind(None) is None
    assert event_kind(["type"]) is None


@pytest.mark.parametrize("converter", [convert_codex, convert_claude_code, convert_opencode,
                                       convert_cursor])
def test_a_json_line_with_no_type_never_reaches_a_converter_branch(converter):
    """Two runs raised ``KeyError: 'type'`` mid-conversion on a blob the
    launcher interleaved into the stream. Every converter skips it."""
    rows = [(None, {"bos_token_id": 151643, "transformers_version": "4.57.1"}, 1, "")]
    events, _extra = converter.convert(rows, "r")
    assert events == []


def test_a_foreign_json_line_is_counted_apart_from_unparsable_text(
    ptb_samples: Path, tmp_path: Path
):
    """It parsed, so it is not "non-JSON"; it is not an event, so it is not a
    line the converter read. Neither counter may absorb the other."""
    _events, before = build_run(_sample_run(ptb_samples, CODEX_CFG, CODEX_RUN))

    dest = tmp_path / CODEX_CFG / CODEX_RUN
    dest.mkdir(parents=True)
    lines = (ptb_samples / CODEX_CFG / CODEX_RUN / "solve_out.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    lines.insert(12, '{"bos_token_id": 151643}')
    (dest / "solve_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    events, meta = build_run(make_run_dir(CODEX_CFG, dest))
    assert meta.extra["n_foreign_json_lines"] == 1
    assert meta.extra["foreign_json_lines"] == [{"line": 13, "text": '{"bos_token_id": 151643}'}]
    assert meta.extra["n_non_json_lines"] == before.extra["n_non_json_lines"]
    assert len(events) == before.n_events


def test_iso_from_ms():
    assert iso_from_ms(1755262528196) == "2025-08-15T12:55:28.196Z"
    # A timestamp is never invented: absent, non-numeric and bool all say so.
    assert iso_from_ms(None) is None
    assert iso_from_ms("1755262528196") is None
    assert iso_from_ms(True) is None


def test_compact_elides_by_size_and_says_it_did():
    out = compact({"n": 3, "ok": True, "big": "x" * 3000, "small": [1, 2], "dup": "v"},
                  limit=100, drop=("dup",))
    assert out == {"n": 3, "ok": True, "big": "<elided 3002 chars>", "small": [1, 2]}
    assert compact("not a dict") is None
    assert compact({}) is None


# --- committed samples -----------------------------------------------------


def test_claude_sample_converts(ptb_samples: Path):
    run = make_run_dir(CLAUDE_CFG, ptb_samples / CLAUDE_CFG / CLAUDE_RUN)
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.harness == "claude-code"
    assert meta.source == "posttrainbench"
    assert meta.model == "claude-opus-4-8"
    assert meta.benchmark == "gsm8k"
    assert meta.budget == {"hours": 10.0}
    assert meta.duration_s == 10 * 3600 + 5 * 60 + 1
    assert meta.final_score["value"] == pytest.approx(0.6118271417740713)
    assert meta.flags["contamination"] is False
    assert meta.extra["n_non_json_lines"] == 10
    assert meta.extra["n_lines"] == len(list(read_line_stream(run.solve_out)))
    # The experiment name has to reach the persisted record: it is the only
    # field separating the two configurations that share 27 trajectories.
    assert meta.extra["experiment"] == "run1"
    assert set(meta.source_paths) >= {"solve_out", "metrics", "time_taken", "judgement"}

    # One tool_use per "Tool call" line in upstream's own rendering.
    parsed = (run.path / "solve_parsed.txt").read_text(encoding="utf-8")
    assert len(_by_type(events, "tool_use")) == parsed.count("Tool call") == 8

    # Both sessions of the trimmed sample are recorded; the second is still open.
    sessions = meta.extra["sessions"]
    assert [s["index"] for s in sessions] == [0, 1]
    assert sessions[0]["num_turns"] == 69
    assert "result_line" not in sessions[1]
    assert meta.cost_usd == pytest.approx(4.789912)


def test_a_published_gpu_count_reaches_the_budget(ptb_samples: Path):
    # `budget_gpus` is an index column and is NA for every run in the release,
    # because no configuration name carries the suffix. When one does, the count
    # has to get there — and when one does not, the key must stay absent rather
    # than be filled in with a 1 nobody published (see test_claude_sample_converts).
    run = make_run_dir(
        "claude_non_api_max_claude-opus-4-8_10h_8gpu_run1", ptb_samples / CLAUDE_CFG / CLAUDE_RUN
    )
    _events, meta = build_run(run)
    assert meta.budget == {"hours": 10.0, "gpus": 8}


def test_claude_harness_origin(ptb_samples: Path):
    run = make_run_dir(CLAUDE_CFG, ptb_samples / CLAUDE_CFG / CLAUDE_RUN)
    events, _meta = build_run(run)
    kinds = {e.extra["kind"] for e in events if e.origin == "harness" and e.extra}
    assert kinds == {"session_start", "task_started", "task_notification", "task_updated",
                     "rate_limit_event"}
    # Background tasks are notifications about an existing Bash call, not tool calls.
    assert not any(e.type == "tool_use" for e in events if e.origin == "harness")
    started = next(e for e in events if e.extra and e.extra.get("kind") == "task_started")
    assert started.parent_tool_use.startswith("toolu_")
    assert started.text == "Locate inspect_evals gsm8k task files"


def test_claude_turns_and_usage(ptb_samples: Path):
    run = make_run_dir(CLAUDE_CFG, ptb_samples / CLAUDE_CFG / CLAUDE_RUN)
    events, _meta = build_run(run)
    # Turns are API responses, they never go backwards, and they do not reset
    # when the launcher restarts the CLI mid-run.
    turns = [e.turn for e in events if e.turn is not None]
    assert turns == sorted(turns)
    assert max(turns) == 7
    with_usage = [e for e in events if e.usage]
    assert len(with_usage) == len({e.turn for e in with_usage})
    assert set(with_usage[0].usage) <= {"in", "out", "cache_read", "cache_write"}


def test_claude_tool_results(ptb_samples: Path):
    run = make_run_dir(CLAUDE_CFG, ptb_samples / CLAUDE_CFG / CLAUDE_RUN)
    events, _meta = build_run(run)
    results = _by_type(events, "tool_result")
    assert all(e.role == "user" and e.parent_tool_use for e in results)
    calls = {e.tool_use_id: e.tool for e in _by_type(events, "tool_use")}
    linked = [e for e in results if e.parent_tool_use in calls]
    assert linked and all(e.tool == calls[e.parent_tool_use] for e in linked)
    # ToolSearch answers with tool_reference blocks rather than text.
    blocky = [e for e in results if e.extra and "content_blocks" in e.extra]
    assert blocky and blocky[0].text is None


def test_codex_sample_converts(ptb_samples: Path):
    run = make_run_dir(CODEX_CFG, ptb_samples / CODEX_CFG / CODEX_RUN)
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.harness == "codex"
    assert meta.model == "gpt-5.4"
    assert meta.final_score["value"] == pytest.approx(0.4268385140257771)
    assert meta.duration_s is None  # no time_taken.txt in this run directory
    assert meta.flags["judgement_unavailable"] == "Entry not found"
    assert meta.extra["n_non_json_lines"] == 10
    assert meta.extra["thread_id"] == "019cc3e4-2307-7cf3-a8e0-489866e4a1cd"

    # No timestamps exist in this format; none are invented.
    assert all(e.ts is None for e in events)
    assert meta.t_start is None and meta.t_end is None

    # One event per item, however many item.* messages it produced.
    ids = [e.tool_use_id for e in _by_type(events, "tool_use")]
    assert len(ids) == len(set(ids))
    assert meta.extra["unfinished_items"] == ["item_140", "item_4"]

    usage = [e for e in events if e.usage]
    assert len(usage) == 1 and usage[0].i == 0
    assert usage[0].usage == {"in": 20976581, "out": 46993, "cache_read": 20707840}


# --- OpenCode ---------------------------------------------------------------


def test_opencode_sample_converts(ptb_samples: Path):
    run = _sample_run(ptb_samples, OPENCODE_CFG, OPENCODE_RUN)
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.harness == "opencode"
    assert meta.model == "kimi-k2.5"
    assert meta.benchmark == "gpqamain"
    assert meta.final_score["value"] == pytest.approx(0.19196428571428573)
    assert meta.extra["session_ids"] == ["ses_39e724c78ffepielaTUCNqN3Og"]

    # One line is one call AND its result: OpenCode publishes the completed
    # state, so both events come from the same line.
    assert meta.n_by_type == {"tool_use": 15, "tool_result": 15, "text": 9}
    assert meta.tools == {"bash": 6, "read": 3, "invalid": 2, "edit": 1, "write": 1,
                          "todowrite": 1, "websearch": 1}
    calls = _by_type(events, "tool_use")
    results = _by_type(events, "tool_result")
    assert [e.tool_use_id for e in calls] == [e.parent_tool_use for e in results]
    assert len({e.tool_use_id for e in calls}) == len(calls)

    # Every line timestamps itself in-band, so nothing is missing and nothing
    # was invented from the launcher's prefix.
    assert all(e.ts for e in events)
    assert meta.t_start == "2026-02-15T13:46:44.509Z"


def test_opencode_cost_is_the_sum_of_its_steps_not_the_last_one():
    """The opposite of Claude Code's cumulative ``total_cost_usd``. 263 of the
    319 costed runs are non-monotone, so taking the last would under-report."""
    def finish(mid, cost):
        return (None, {"type": "step_finish", "part": {"messageID": mid, "cost": cost,
                                                       "tokens": {"input": 1, "output": 2}}}, 1, "")

    def text(mid, s):
        return (None, {"type": "text", "part": {"messageID": mid, "text": s}}, 1, "")

    rows = [text("m1", "a"), finish("m1", 0.5), text("m2", "b"), finish("m2", 0.25)]
    _events, extra = convert_opencode.convert(rows, "r")
    assert extra["step_costs"] == [0.5, 0.25]
    assert extra["cost_usd"] == pytest.approx(0.75)
    assert extra["tokens"] == {"in": 2, "out": 4}


def test_opencode_drops_the_disagreeing_total_token_key():
    """``tokens.total`` matches four different formulas across the corpus; the
    four components are self-consistent and are what the events carry."""
    tokens = {"input": 10, "output": 3, "reasoning": 2, "total": 99,
              "cache": {"read": 7, "write": 1}}
    assert convert_opencode.map_usage(tokens) == {
        "in": 10, "out": 3, "reasoning_out": 2, "cache_read": 7, "cache_write": 1
    }
    assert convert_opencode.map_usage({}) is None
    assert convert_opencode.map_usage(None) is None


def test_opencode_a_synthetic_prompt_is_the_scaffolding_not_the_model():
    rows = [
        (None, {"type": "text", "part": {"messageID": "m", "text": "thinking out loud"}}, 1, ""),
        (None, {"type": "text", "part": {"messageID": "m", "synthetic": True,
                                         "text": "Continue if you have next steps"}}, 2, ""),
    ]
    events, extra = convert_opencode.convert(rows, "r")
    assert [(e.origin, e.role) for e in events] == [("agent", "assistant"), ("harness", "user")]
    assert extra["n_synthetic_prompts"] == 1


def test_opencode_a_step_that_emitted_nothing_is_counted_not_silently_dropped():
    """463 steps finish having emitted neither text nor a call, so their usage
    has no event to sit on. The run total must still include them."""
    rows = [
        (None, {"type": "step_start", "part": {"messageID": "m1"}}, 1, ""),
        (None, {"type": "step_finish", "part": {"messageID": "m1", "cost": 0.1,
                                                "tokens": {"input": 5, "output": 1}}}, 2, ""),
    ]
    events, extra = convert_opencode.convert(rows, "r")
    assert events == []
    assert extra["n_steps"] == 1 and extra["n_steps_without_events"] == 1
    assert extra["tokens"] == {"in": 5, "out": 1}
    # ...and the committed sample has exactly one such step: the one whose only
    # tool call make_samples.py leaves out for size.
    assert extra["cost_usd"] == pytest.approx(0.1)


def test_opencode_tool_use_id_is_part_id_because_call_id_collides():
    """The kimi runs re-emit ``functions.bash:0`` call after call — 1,950
    colliding re-emissions — so ``callID`` cannot link a result to its call."""
    def call(pid):
        return (None, {"type": "tool_use", "part": {
            "messageID": "m", "id": pid, "callID": "functions.bash:0", "tool": "bash",
            "state": {"status": "completed", "input": {"command": "ls"}, "output": "a",
                      "metadata": {"exit": 0}}}}, 1, "")

    events, _extra = convert_opencode.convert([call("prt_1"), call("prt_2")], "r")
    calls = _by_type(events, "tool_use")
    assert [e.tool_use_id for e in calls] == ["prt_1", "prt_2"]
    assert {e.extra["call_id"] for e in calls} == {"functions.bash:0"}


def test_opencode_is_error_stays_unknown_when_the_stream_does_not_say():
    """``metadata.exit`` is None for 1,176 calls (killed or timed out). Unknown
    is not success."""
    def result(tool, metadata, status="completed"):
        rows = [(None, {"type": "tool_use", "part": {
            "messageID": "m", "id": "p", "tool": tool,
            "state": {"status": status, "output": "", "metadata": metadata}}}, 1, "")]
        events, _extra = convert_opencode.convert(rows, "r")
        return _by_type(events, "tool_result")[0].is_error

    assert result("bash", {"exit": 0}) is False
    assert result("bash", {"exit": 137}) is True
    assert result("bash", {}) is None
    assert result("read", {}) is None
    # OpenCode refusing a malformed tool call is a call that really failed.
    assert result("invalid", {}) is True
    assert result("bash", {}, status="error") is True


def test_opencode_error_lines_become_harness_events():
    rows = [(None, {"type": "error", "error": {"name": "ProviderAuthError", "data": {
        "message": "AI_APICallError: Rate limit"}}}, 3, "")]
    events, extra = convert_opencode.convert(rows, "r")
    assert [(e.origin, e.type, e.text) for e in events] == [
        ("harness", "text", "AI_APICallError: Rate limit")
    ]
    assert extra["errors"] == [{"line": 3, "name": "ProviderAuthError",
                                "message": "AI_APICallError: Rate limit"}]


def test_opencode_an_unknown_line_type_still_produces_an_event():
    events, extra = convert_opencode.convert([(None, {"type": "tool_pending"}, 1, "")], "r")
    assert extra["unknown_line_kinds"] == {"tool_pending": 1}
    assert [(e.origin, e.extra["kind"]) for e in events] == [("harness", "tool_pending")]


# --- Cursor -----------------------------------------------------------------


def test_cursor_sample_converts(ptb_samples: Path):
    run = _sample_run(ptb_samples, CURSOR_CFG, CURSOR_RUN)
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.harness == "cursor-cli"
    assert meta.model == "cursor-grok-4.5-high"
    assert meta.final_score["value"] == pytest.approx(0.2288255673906463)
    assert meta.extra["api_model"] == "Cursor Grok 4.5 High"
    assert meta.n_by_type == {"text": 26, "thinking": 8, "tool_use": 22, "tool_result": 15}
    assert set(meta.tools) == {"shell", "await", "edit", "read", "grep", "glob",
                               "updateTodos", "webSearch", "webFetch"}
    # Cursor publishes no cost at all; a zero would be a fabrication.
    assert meta.cost_usd is None
    assert meta.tokens == {"in": 207388, "out": 3114, "cache_read": 1801856}
    # The one `result` line carries the run's only usage, on its first turn.
    with_usage = [e for e in events if e.usage]
    assert len(with_usage) == 1 and with_usage[0].turn == 0


def test_cursor_is_the_one_harness_that_publishes_the_prompt(ptb_samples: Path):
    """``claude --print`` never echoes its prompt, so a user text event there is
    the launcher. Cursor does echo it, and the benchmark's own instruction is
    the human's, not scaffolding."""
    run = _sample_run(ptb_samples, CURSOR_CFG, CURSOR_RUN)
    events, meta = build_run(run)
    human = [e for e in events if e.origin == "human"]
    assert len(human) == 1
    assert human[0].extra == {"kind": "prompt"}
    assert human[0].text.startswith("We want to train the small LLM")
    assert "Store your best trained model in the folder `final_model`" in human[0].text
    assert meta.n_by_origin["human"] == 1


def test_cursor_carries_the_prompt_whichever_shape_it_arrives_in():
    """All 56 runs use a one-element block list. Reading only the bare string
    form — which the identically-named Claude Code line uses — left the whole
    instruction in an extra bag with the event's text set to None."""
    def prompt(content):
        rows = [(None, {"type": "user", "message": {"role": "user", "content": content}}, 1, "")]
        events, _extra = convert_cursor.convert(rows, "r")
        return events[0]

    listed = prompt([{"type": "text", "text": "train the model"}])
    assert (listed.text, listed.extra) == ("train the model", {"kind": "prompt"})
    assert prompt("train the model").text == "train the model"
    # A block that is not text is kept beside the prompt rather than dropped.
    mixed = prompt([{"type": "text", "text": "train it"}, {"type": "image", "source": {}}])
    assert mixed.text == "train it"
    assert mixed.extra["content_blocks"] == {"blocks": [{"type": "image", "source": {}}]}
    assert prompt(None).text is None


def test_cursor_thinking_deltas_are_joined_at_the_first_delta():
    """98,719 of the 133k lines are thinking deltas and the text is on them, not
    on the ``completed`` marker."""
    rows = [
        (None, {"type": "thinking", "subtype": "delta", "text": "Let me ",
                "timestamp_ms": 1755262528196}, 4, ""),
        (None, {"type": "thinking", "subtype": "delta", "text": "check.",
                "timestamp_ms": 1755262529000}, 5, ""),
        (None, {"type": "thinking", "subtype": "completed"}, 6, ""),
    ]
    events, _extra = convert_cursor.convert(rows, "r")
    assert [(e.type, e.text) for e in events] == [("thinking", "Let me check.")]
    assert events[0].ts == "2025-08-15T12:55:28.196Z"
    assert events[0].source_ref == {"file": "solve_out.txt", "line": 4}


def test_cursor_thinking_is_flushed_even_when_its_completed_never_came():
    """A block is closed by the next line of any kind, so a thinking event still
    lands before the tool call it reasoned about."""
    rows = [
        (None, {"type": "thinking", "subtype": "delta", "text": "run it"}, 1, ""),
        (None, {"type": "tool_call", "subtype": "started", "model_call_id": "mc1",
                "tool_call": {"toolCallId": "t1", "shellToolCall": {
                    "args": {"command": "ls"}}}}, 2, ""),
    ]
    events, _extra = convert_cursor.convert(rows, "r")
    assert [e.type for e in events] == ["thinking", "tool_use"]
    assert [e.turn for e in events] == [0, 0]


def test_cursor_replayed_tool_lines_are_counted_not_converted_twice():
    """After a reconnect the CLI replays the transcript: 331 excess ``started``
    lines against 356 reconnects. The command did not run again."""
    started = {"type": "tool_call", "subtype": "started", "model_call_id": "mc1",
               "tool_call": {"toolCallId": "t1", "shellToolCall": {"args": {"command": "ls"}}}}
    completed = {"type": "tool_call", "subtype": "completed", "model_call_id": "mc1",
                 "tool_call": {"toolCallId": "t1", "shellToolCall": {
                     "result": {"success": {"stdout": "a\n", "exitCode": 0}}}}}
    rows = [(None, started, 1, ""), (None, {"type": "connection"}, 2, ""),
            (None, started, 3, ""), (None, completed, 4, "")]
    events, extra = convert_cursor.convert(rows, "r")
    assert [e.type for e in events] == ["tool_use", "text", "tool_result"]
    assert extra["n_replayed_tool_lines"] == 1
    assert extra["n_reconnects"] == 1


def test_cursor_replay_shows_up_on_the_committed_sample(ptb_samples: Path):
    # The sample keeps lines 785-796 of the run for exactly this: a
    # connection/retry pair followed by two started lines and one completed
    # replayed verbatim.
    _events, meta = build_run(_sample_run(ptb_samples, CURSOR_CFG, CURSOR_RUN))
    assert meta.extra["n_replayed_tool_lines"] == 3
    assert (meta.extra["n_reconnects"], meta.extra["n_retry_lines"]) == (2, 2)


def test_cursor_a_completion_whose_start_never_arrived_still_has_a_call():
    rows = [(None, {"type": "tool_call", "subtype": "completed", "model_call_id": "mc1",
                    "tool_call": {"toolCallId": "t9", "readToolCall": {
                        "result": {"success": {"content": "hello"}}}}}, 1, "")]
    events, _extra = convert_cursor.convert(rows, "r")
    assert [e.type for e in events] == ["tool_use", "tool_result"]
    assert events[0].extra == {"start_line_missing": True}
    assert events[0].tool == events[1].tool == "read"
    assert events[1].text == "hello"


def test_cursor_turns_are_numbered_in_a_second_pass():
    """``model_call_id`` is on the assistant and tool_call lines but not on
    thinking, and a response streams its thinking FIRST. Deciding a turn inline
    would put that thinking in the previous response — and would mis-split
    parallel tool calls, whose completions arrive out of order."""
    def tool(mcid, cid, sub):
        body = {"args": {"command": "ls"}} if sub == "started" else {
            "result": {"success": {"stdout": "", "exitCode": 0}}}
        return (None, {"type": "tool_call", "subtype": sub, "model_call_id": mcid,
                       "tool_call": {"toolCallId": cid, "shellToolCall": body}}, 1, "")

    rows = [
        (None, {"type": "system", "subtype": "init", "session_id": "s"}, 1, ""),
        (None, {"type": "thinking", "subtype": "delta", "text": "plan"}, 2, ""),
        tool("mc1", "a", "started"),
        tool("mc1", "b", "started"),
        tool("mc1", "b", "completed"),   # parallel calls complete out of order
        tool("mc1", "a", "completed"),
        (None, {"type": "thinking", "subtype": "delta", "text": "next"}, 7, ""),
        (None, {"type": "assistant", "model_call_id": "mc2",
                "message": {"content": "done"}}, 8, ""),
    ]
    events, extra = convert_cursor.convert(rows, "r")
    assert extra["n_turns"] == 2
    # init is harness and precedes turn 0, so it takes no turn; the thinking
    # that opens each response belongs to that response, not the previous one.
    assert [(e.type, e.turn) for e in events] == [
        ("text", None), ("thinking", 0), ("tool_use", 0), ("tool_use", 0),
        ("tool_result", 0), ("tool_result", 0), ("thinking", 1), ("text", 1),
    ]


def test_cursor_reads_the_outcome_from_the_key_not_a_status_field():
    def result(tool_key, payload):
        rows = [(None, {"type": "tool_call", "subtype": "completed",
                        "tool_call": {"toolCallId": "t", tool_key: {"result": payload}}}, 1, "")]
        events, _extra = convert_cursor.convert(rows, "r")
        return _by_type(events, "tool_result")[0]

    assert result("shellToolCall", {"success": {"exitCode": 0}}).is_error is False
    # A shell "success" that exited non-zero is still a failed command.
    assert result("shellToolCall", {"success": {"exitCode": 1}}).is_error is True
    for key in ("failure", "error", "spawnError"):
        e = result("webFetchToolCall", {key: {"error": "no route to host"}})
        assert (e.is_error, e.extra["outcome"], e.text) == (True, key, "no route to host")
    # No result object at all says nothing, and nothing is what is recorded.
    assert result("shellToolCall", None).is_error is None


def test_cursor_shell_results_prefer_the_interleaved_output():
    """``interleavedOutput`` is stdout and stderr in the order the model saw
    them, and it is set on 30 results where ``stdout`` is empty."""
    def text_of(payload):
        rows = [(None, {"type": "tool_call", "subtype": "completed", "tool_call": {
            "toolCallId": "t", "shellToolCall": {"result": {"success": payload}}}}, 1, "")]
        events, _extra = convert_cursor.convert(rows, "r")
        return _by_type(events, "tool_result")[0]

    both = text_of({"stdout": "out", "interleavedOutput": "out+err", "exitCode": 0})
    assert both.text == "out+err"
    # Whichever field was promoted onto the event is not repeated in the extra.
    assert "interleavedOutput" not in both.extra["result"]
    assert "stdout" not in both.extra["result"]
    assert text_of({"stdout": "only stdout", "exitCode": 0}).text == "only stdout"
    assert text_of({"interleavedOutput": "", "stdout": "", "exitCode": 0}).text is None


def test_cursor_drops_the_re_derived_shell_argument_but_nothing_else():
    """``args.parsingResult`` re-derives ``command`` as an AST: 13.6 MB of the
    37 MB of tool arguments in this release, restating a string that is kept.
    Everything else is the agent's own bytes and is not summarised."""
    args = {"command": "python train.py", "parsingResult": {"nodes": ["..."] * 500},
            "isBackground": True, "blob": "x" * 5000}
    rows = [(None, {"type": "tool_call", "subtype": "started", "tool_call": {
        "toolCallId": "t", "shellToolCall": {"args": args}}}, 1, "")]
    events, _extra = convert_cursor.convert(rows, "r")
    assert events[0].args == {"command": "python train.py", "isBackground": True,
                              "blob": "x" * 5000}


def test_cursor_an_edit_result_keeps_the_diff_and_not_the_two_snapshots():
    payload = {"diffString": "@@ -1 +1 @@", "beforeFullFileContent": "a" * 4000,
               "afterFullFileContent": "b" * 4000, "linesAdded": 1}
    rows = [(None, {"type": "tool_call", "subtype": "completed", "tool_call": {
        "toolCallId": "t", "editToolCall": {"result": {"success": payload}}}}, 1, "")]
    events, _extra = convert_cursor.convert(rows, "r")
    result = _by_type(events, "tool_result")[0]
    assert result.text == "@@ -1 +1 @@"
    assert result.extra["result"] == {"linesAdded": 1}


def test_cursor_the_clis_own_chatter_is_harness_and_never_dropped():
    rows = [
        (None, {"type": "connection", "status": "reconnecting"}, 1, ""),
        (None, {"type": "retry", "attempt": 2}, 2, ""),
        (None, {"type": "interaction_query", "query_type": "webSearch"}, 3, ""),
        (None, {"type": "system", "subtype": "task_notification", "title": "done"}, 4, ""),
    ]
    events, extra = convert_cursor.convert(rows, "r")
    assert [e.origin for e in events] == ["harness"] * 4
    assert [e.extra["kind"] for e in events] == [
        "connection", "retry", "interaction_query", "task_notification"
    ]
    # None of these is an unrecognised type; the counter stays empty.
    assert extra["unknown_line_kinds"] == {}
    assert (extra["n_reconnects"], extra["n_retry_lines"]) == (1, 1)


def test_write_and_read_back(ptb_samples: Path, tmp_path: Path):
    run = make_run_dir(CODEX_CFG, ptb_samples / CODEX_CFG / CODEX_RUN)
    ep = convert_run_dir(run, tmp_path)
    events = list(read_events(ep))
    meta = read_meta(tmp_path / f"{run.run_id}.meta.json")
    assert len(events) == meta.n_events
    validate_stream(events, meta.run_id)


def test_claude_subagent_rows():
    """The Task tool nests rows under parent_tool_use_id; the sample has none."""
    spawn = {
        "type": "assistant",
        "message": {
            "id": "msg_a",
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "content": [
                {"type": "tool_use", "id": "toolu_task", "name": "Task",
                 "input": {"description": "measure the baseline"}}
            ],
        },
    }
    child = {
        "type": "assistant",
        "parent_tool_use_id": "toolu_task",
        "message": {"id": "msg_b", "content": [{"type": "text", "text": "on it"}]},
    }
    rows = [(None, spawn, 1, ""), (None, child, 2, "")]
    events, _extra = convert_claude_code.convert(rows, "r")
    validate_stream(events, "r")
    assert [e.agent_id for e in events] == [MAIN_AGENT, "toolu_task"]
    assert [e.i for e in events] == [0, 0]
    assert events[1].parent_tool_use == "toolu_task"


def test_claude_usage_counted_once_per_message_id():
    """A sub-agent's lines can split a parent message; the repeated usage on its
    second half must not be counted twice."""
    def assistant(mid, text, parent=None):
        row = {
            "type": "assistant",
            "message": {"id": mid, "usage": {"input_tokens": 10, "output_tokens": 4},
                        "content": [{"type": "text", "text": text}]},
        }
        if parent:
            row["parent_tool_use_id"] = parent
        return row

    rows = [(None, assistant("msg_a", "first half"), 1, ""),
            (None, assistant("msg_b", "child", parent="toolu_task"), 2, ""),
            (None, assistant("msg_a", "second half"), 3, "")]
    events, _extra = convert_claude_code.convert(rows, "r")
    validate_stream(events, "r")
    # `out` is absent by design: an assistant line's output_tokens is a snapshot
    # of a response still streaming — see the next two tests.
    assert [e.usage for e in events] == [{"in": 10}, {"in": 10}, None]


def test_claude_output_tokens_come_from_the_result_not_the_assistant_lines():
    """An assistant message's ``output_tokens`` is the count at the moment the
    block was emitted, not the response's: measured over the corpus it sums to
    3.56 M against the result lines' 94.5 M, a 26.5x undercount that read as
    "Claude Code writes 2% of what codex writes". Only the result is believed."""
    def assistant(mid, out):
        return {"type": "assistant",
                "message": {"id": mid,
                            "usage": {"input_tokens": 100, "output_tokens": out,
                                      "cache_read_input_tokens": 5000},
                            "content": [{"type": "text", "text": "x" * 4000}]}}

    result = {"type": "result", "total_cost_usd": 1.5,
              "usage": {"input_tokens": 100, "output_tokens": 3000,
                        "cache_read_input_tokens": 5000}}
    rows = [(None, assistant("msg_a", 7), 1, ""), (None, result, 2, "")]
    events, extra = convert_claude_code.convert(rows, "r")
    assert events[0].usage == {"in": 100, "cache_read": 5000}  # no `out` anywhere per event
    assert extra["tokens"] == {"in": 100, "out": 3000, "cache_read": 5000}
    assert extra["tokens_source"] == "result"


def test_claude_tokens_sum_over_sessions_while_the_cost_does_not():
    """The two live on the same line and disagree: ``total_cost_usd`` is
    cumulative over the run (monotone on all 735 runs with a result) and
    ``usage`` is per session, so one is the last and the other is the sum. A run
    killed before any result has no output count at all, and says so."""
    def result(cost, out):
        return {"type": "result", "total_cost_usd": cost,
                "usage": {"input_tokens": 10, "output_tokens": out}}

    rows = [(None, result(4.0, 700), 1, ""), (None, result(9.5, 300), 2, "")]
    _events, extra = convert_claude_code.convert(rows, "r")
    assert extra["cost_usd"] == 9.5
    assert extra["tokens"] == {"in": 20, "out": 1000}

    killed = [(None, {"type": "assistant",
                      "message": {"id": "m", "usage": {"input_tokens": 8, "output_tokens": 3},
                                  "content": [{"type": "text", "text": "hi"}]}}, 1, "")]
    _events, extra = convert_claude_code.convert(killed, "r")
    assert "tokens" not in extra and extra["tokens_source"] == "assistant_messages"


def test_claude_flags_truncated_tool_results():
    """Claude Code caps a background task's output in-band; the result is not
    the whole thing and must say so."""
    capped = ("<output>\nOutput truncated (2KB total). Full output saved to: "
              "/tmp/tasks/bfdrh8iig.output\n</output>")
    rows = [
        (None, {"type": "assistant", "message": {"id": "m", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "TaskOutput", "input": {}}]}}, 1, ""),
        (None, {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": capped}]}}, 2, ""),
        (None, {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "all of it"}]}}, 3, ""),
    ]
    events, _extra = convert_claude_code.convert(rows, "r")
    assert [e.truncated for e in events] == [False, True, False]


def test_claude_an_unknown_line_type_still_produces_an_event():
    """The if/elif chain that had no ``else`` is how the 56 cursor runs lost
    128k lines while reporting a clean conversion. Detection sends those
    elsewhere now, but a stream may still carry a type this converter has never
    seen, and it must say so rather than drop it."""
    rows = [(None, {"type": "compact_boundary", "trigger": "auto"}, 1, "")]
    events, extra = convert_claude_code.convert(rows, "r")
    assert extra["unknown_line_kinds"] == {"compact_boundary": 1}
    assert [(e.origin, e.type) for e in events] == [("harness", "text")]
    assert events[0].extra["kind"] == "compact_boundary"


def test_codex_error_lines_become_harness_events():
    """A dropped response stream is a bare {"type": "error"} line; it is the CLI
    talking, and no input line may vanish."""
    rows = [
        (None, {"type": "turn.started"}, 1, ""),
        (None, {"type": "item.completed",
                "item": {"id": "i1", "type": "agent_message", "text": "hi"}}, 2, ""),
        (None, {"type": "error", "message": "Reconnecting... 1/5"}, 3, ""),
    ]
    events, _extra = convert_codex.convert(rows, "r")
    validate_stream(events, "r")
    assert [(e.origin, e.type) for e in events] == [("agent", "text"), ("harness", "text")]
    assert events[1].text == "Reconnecting... 1/5"
    assert events[1].extra == {"kind": "error"}


def test_codex_usage_is_cumulative_over_the_thread():
    """One thread per run and a running total on every ``turn.completed``:
    monotone on all 76 multi-turn runs. A turn's own usage is the delta and the
    run's total is the last line — summing them counts a 6-turn reprompt run
    about four times over."""
    def turn(n, item, total_in, total_out):
        return [(None, {"type": "turn.started"}, n, ""),
                (None, {"type": "item.completed",
                        "item": {"id": item, "type": "agent_message", "text": item}}, n + 1, ""),
                (None, {"type": "turn.completed",
                        "usage": {"input_tokens": total_in, "output_tokens": total_out}},
                 n + 2, "")]

    rows = turn(1, "i1", 30_000, 60_000) + turn(4, "i2", 63_000, 110_000)
    events, extra = convert_codex.convert(rows, "r")
    validate_stream(events, "r")
    assert [e.usage for e in events] == [{"in": 30_000, "out": 60_000},
                                         {"in": 33_000, "out": 50_000}]
    assert extra["tokens"] == {"in": 63_000, "out": 110_000}
    assert extra["tokens_source"] == "turn.completed"
    # The line's own number is kept beside the delta rather than thrown away.
    assert [t["usage_cumulative"]["out"] for t in extra["turns"]] == [60_000, 110_000]


def test_codex_a_thread_that_compacts_never_reports_a_negative_delta():
    """One published thread's input count falls between turns. A turn that
    consumed a negative number of tokens is not a thing to record, and the run's
    total stays whatever the last line said."""
    rows = [
        (None, {"type": "turn.started"}, 1, ""),
        (None, {"type": "item.completed",
                "item": {"id": "i1", "type": "agent_message", "text": "a"}}, 2, ""),
        (None, {"type": "turn.completed", "usage": {"input_tokens": 900, "output_tokens": 10}},
         3, ""),
        (None, {"type": "turn.started"}, 4, ""),
        (None, {"type": "item.completed",
                "item": {"id": "i2", "type": "agent_message", "text": "b"}}, 5, ""),
        (None, {"type": "turn.completed", "usage": {"input_tokens": 400, "output_tokens": 25}},
         6, ""),
    ]
    events, extra = convert_codex.convert(rows, "r")
    assert [e.usage for e in events] == [{"in": 900, "out": 10}, {"out": 15}]
    assert extra["tokens"] == {"in": 400, "out": 25}


def test_codex_a_run_killed_before_turn_completed_reports_no_tokens():
    """16 runs never reach one. No usage is better than a turn's worth called
    the run's."""
    rows = [(None, {"type": "turn.started"}, 1, ""),
            (None, {"type": "item.completed",
                    "item": {"id": "i1", "type": "agent_message", "text": "a"}}, 2, "")]
    _events, extra = convert_codex.convert(rows, "r")
    assert "tokens" not in extra and extra["tokens_source"] == "none"


# --- the two complete runs -------------------------------------------------


def test_full_claude_acceptance(full_runs: dict[str, RunDir]):
    run = full_runs["claude"]
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.n_events == 639
    assert meta.n_by_type == {"text": 208, "thinking": 111, "tool_use": 160, "tool_result": 160}
    assert meta.n_by_origin == {"agent": 552, "harness": 87}
    assert meta.tools == {"Bash": 93, "Read": 32, "Write": 11, "TaskUpdate": 10,
                          "TaskCreate": 6, "Edit": 6, "ToolSearch": 2}
    # 208 text events = 121 assistant text blocks + 87 harness injections.
    assert len([e for e in events if e.type == "text" and e.origin == "agent"]) == 121

    parsed = (run.path / "solve_parsed.txt").read_text(encoding="utf-8")
    assert len(_by_type(events, "tool_use")) == parsed.count("Tool call") == 160

    assert meta.extra["n_lines"] == 663
    assert meta.extra["n_non_json_lines"] == 11
    assert meta.extra["n_sessions"] == 13
    # All thirteen sessions share one session id: the launcher restarts the CLI
    # with --continue, so sessions are told apart by init order, not by id.
    assert len(meta.extra["session_ids"]) == 1
    # total_cost_usd is cumulative over the run, so the run's cost is the last.
    assert meta.cost_usd == pytest.approx(17.26357075)
    # Its usage is not: 13 per-session payloads, summed. The 155 assistant
    # messages claim 1,652 output tokens between them, which is the streaming
    # snapshot and not this run's 150,503.
    assert meta.extra["tokens_source"] == "result"
    assert meta.tokens == {"in": 10910, "out": 150503,
                           "cache_read": 21901354, "cache_write": 399127}
    assert sum(e.usage.get("out", 0) for e in events if e.usage) == 0
    assert meta.final_score["value"] == pytest.approx(0.6118271417740713)
    assert meta.duration_s == 36301.0
    assert meta.t_start == "2026-06-07T21:31:06Z"


def test_full_codex_acceptance(full_runs: dict[str, RunDir]):
    run = full_runs["codex"]
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.n_events == 283
    assert meta.n_by_type == {"thinking": 106, "text": 41, "tool_use": 73, "tool_result": 63}
    # 126 command_execution *messages* (63 started + 63 completed) are 63 items.
    assert meta.tools == {"command_execution": 63, "file_change": 8, "web_search": 1,
                          "todo_list": 1}
    assert len(_by_type(events, "tool_result")) == 63
    assert all(e.ts is None for e in events)
    assert meta.extra["n_lines"] == 300
    assert meta.extra["n_non_json_lines"] == 10
    assert meta.extra["unfinished_items"] == []
    # One turn, so its cumulative usage IS the run's. Codex has no cost field.
    assert meta.extra["tokens_source"] == "turn.completed"
    assert meta.tokens == {"in": 20976581, "out": 46993, "cache_read": 20707840}
    assert meta.extra["turns"][0]["usage"] == meta.tokens
    assert meta.cost_usd is None
    assert meta.final_score["value"] == pytest.approx(0.4268385140257771)


def test_full_opencode_acceptance(full_runs: dict[str, RunDir]):
    run = full_runs["opencode"]
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.harness == "opencode"
    assert meta.n_events == 200
    assert meta.n_by_type == {"tool_use": 76, "tool_result": 76, "text": 48}
    assert meta.n_by_origin == {"agent": 200}
    assert meta.tools == {"bash": 49, "todowrite": 10, "read": 7, "websearch": 5,
                          "invalid": 2, "codesearch": 1, "write": 1, "edit": 1}
    assert meta.extra["n_lines"] == 235
    assert meta.extra["n_non_json_lines"] == 13
    assert meta.extra["n_steps"] == 49
    assert meta.extra["step_reasons"] == {"tool-calls": 48, "stop": 1}
    # One usage payload per step, each on that step's first event.
    assert len(meta.extra["step_costs"]) == 49
    assert len([e for e in events if e.usage]) == 49
    assert meta.extra["n_steps_without_events"] == 0
    # The sum of 49 per-step costs, not the largest or the last.
    assert meta.cost_usd == pytest.approx(0.65321648)
    assert meta.cost_usd > max(meta.extra["step_costs"])
    assert meta.tokens == {"in": 379723, "out": 20539, "cache_read": 4547071}
    assert meta.t_start == "2026-02-15T13:46:44.509Z"
    assert meta.duration_s == 2370.0


def test_full_cursor_acceptance(full_runs: dict[str, RunDir]):
    run = full_runs["cursor"]
    events, meta = build_run(run)
    validate_stream(events, meta.run_id)

    assert meta.harness == "cursor-cli"
    assert meta.n_events == 485
    assert meta.n_by_type == {"text": 91, "thinking": 108, "tool_use": 143, "tool_result": 143}
    assert meta.n_by_origin == {"agent": 419, "harness": 65, "human": 1}
    assert meta.tools == {"shell": 80, "await": 21, "edit": 18, "read": 12, "webSearch": 5,
                          "updateTodos": 3, "grep": 2, "glob": 1, "webFetch": 1}
    # 1,342 thinking lines collapse to 108 blocks — one per `completed` marker.
    assert meta.extra["n_lines"] == 1738
    assert meta.extra["n_turns"] == 112
    assert (meta.extra["n_reconnects"], meta.extra["n_retry_lines"]) == (16, 16)
    assert meta.extra["n_replayed_tool_lines"] == 10
    assert meta.extra["unknown_line_kinds"] == {}
    # Turns are contiguous from 0 and never go backwards.
    turns = [e.turn for e in events if e.turn is not None]
    assert turns == sorted(turns) and max(turns) == 111
    assert meta.cost_usd is None
    assert meta.tokens == {"in": 207388, "out": 3114, "cache_read": 1801856}
    assert meta.final_score["value"] == pytest.approx(0.2288255673906463)


#: Upstream's renderer names each call ``Tool call — <tool> (<id>)`` on a line of
#: its own. Anchoring matters: the bare phrase "Tool call" also appears inside
#: tool OUTPUT (overwhelmingly on bfcl, whose task text is about tool calling),
#: and counting that made the cross-check disagree with a correct conversion.
_RENDERED_CALL = re.compile(r"^\s*Tool call — (\S+) \(([^)]+)\)\s*$", re.MULTILINE)


@pytest.mark.needs_data
def test_whole_batch_converts(ptb_raw: Path, tmp_path: Path):
    runs = list(iter_run_dirs(ptb_raw))
    if not runs:
        pytest.skip(f"no PostTrainBench run directories under {ptb_raw}")
    seen = Counter()
    for run in runs:
        try:
            events, meta = build_run(run)
        except NoAgentOutput:
            # The CLI died before emitting anything. 41 published runs, none
            # with a metrics.json — a run that did not happen, not a format
            # this converter failed to read.
            assert not (run.path / "metrics.json").exists(), run.run_id
            seen["no agent output"] += 1
            continue
        validate_stream(events, meta.run_id)
        # Converting cleanly is no guarantee the run can be KEPT: two opencode
        # runs carry the high half of an emoji their web result was truncated
        # inside, which every in-memory check passes and UTF-8 cannot encode.
        # The corpus is what reaches disk, so the batch test writes.
        write_run(events, meta, tmp_path, validate=False)
        events_path(tmp_path, meta.run_id).unlink()
        meta_path(tmp_path, meta.run_id).unlink()
        seen[meta.harness] += 1
        # The scaffold directory name says which CLI ran; sniffing the stream
        # has to agree with it. This is the assertion the 56 cursor runs failed
        # while converting "successfully" as Claude Code.
        assert meta.harness == SCAFFOLD_HARNESS[run.agent], run.run_id
        assert meta.final_score is not None or "metrics_unavailable" in meta.extra
        if meta.n_events == 0:
            # 11 runs opened a turn and died inside it — codex emitting
            # thread.started/turn.started, opencode a step_start/step_finish
            # pair — so the format is known and there is genuinely nothing in
            # it. A run that produced a SCORE cannot be one of them.
            seen["framing only"] += 1
            assert not (run.path / "metrics.json").exists(), run.run_id
            continue

        parsed = run.path / "solve_parsed.txt"
        if meta.harness != "claude-code" or not parsed.exists():
            continue
        rendered = _RENDERED_CALL.findall(parsed.read_text(encoding="utf-8", errors="replace"))
        if not rendered:
            # Either the parsed file is a verbatim copy of the raw stream (the
            # glmx runs) or the run made no tool call the renderer named.
            seen["not rendered"] += 1
            continue
        # Upstream's own rendering is truncated mid-file on a few runs, so what
        # it holds is a prefix of the conversion, tool name and id in order —
        # a far stronger claim than the two counts matching.
        calls = [(e.tool, e.tool_use_id) for e in _by_type(events, "tool_use")]
        assert rendered == calls[:len(rendered)], run.run_id
        seen["cross-checked"] += 1

    # No silent cap: say how much of the batch the cross-check actually covered.
    print(dict(seen))
    assert seen["cross-checked"] > seen["not rendered"]
    assert set(seen) >= {"claude-code", "codex", "cursor-cli", "opencode"}
