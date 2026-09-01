"""What counts as an evaluation, and what counts as having learned from it."""

from __future__ import annotations

import gzip
import json

import pytest

from awm.paths import events_dir
from awm.traj import eval_events

CHAMPION = "claude_non_api_claude-opus-5_10h_run1__gsm8k_Qwen_Qwen3-4B-Base_17415829"


def use(i: int, ts: str, command: str) -> dict:
    return {
        "run_id": "r", "agent_id": "main", "i": i, "type": "tool_use", "role": "assistant",
        "ts": ts, "tool": "Bash", "args": {"command": command}, "tool_use_id": f"t{i}",
    }


def result(i: int, ts: str, parent: str, text: str = "") -> dict:
    return {
        "run_id": "r", "agent_id": "main", "i": i, "type": "tool_result", "role": "user",
        "ts": ts, "parent_tool_use": parent, "text": text,
    }


def rows(events: list[dict], benchmark: str | None = None) -> list[dict]:
    return eval_events.events_for_run("r", events, benchmark)


class TestWhatCountsAsAnEvaluation:
    def test_python_invoking_it_launches(self) -> None:
        r = rows([use(0, "2026-01-01T00:00:00Z", "python evaluate.py --model-path m --limit 150")])
        assert len(r) == 1
        assert r[0]["model_path"] == "m"
        assert r[0]["limit"] == 150

    @pytest.mark.parametrize(
        "command",
        [
            "sed -n '1,260p' evaluate.py",
            "ps aux | grep -iE 'evaluate.py|train_sft' | grep -v grep | wc -l",
            "grep -rn 'ChatCompletionToolParam' /usr/local/lib/python3.10/"
            "dist-packages/inspect_ai/model/_openai.py",
            "cat evaluate.py | head -40",
        ],
        ids=["sed-reads-it", "ps-greps-it", "greps-inspect-source", "cats-it"],
    )
    def test_naming_it_is_not_running_it(self, command: str) -> None:
        assert rows([use(0, "2026-01-01T00:00:00Z", command)]) == []

    def test_codex_bash_lc_wrapper_is_unwrapped(self) -> None:
        cmd = "/bin/bash -lc 'python evaluate.py --model-path final_model --limit -1'"
        r = rows([use(0, "2026-01-01T00:00:00Z", cmd)], "gsm8k")
        assert len(r) == 1
        assert r[0]["limit"] == -1

    def test_a_read_inside_the_codex_wrapper_is_still_a_read(self) -> None:
        r = rows([use(0, "2026-01-01T00:00:00Z", "/bin/bash -lc \"sed -n '1,240p' evaluate.py\"")])
        assert r == []


class TestTier:
    @pytest.mark.parametrize(
        "limit,benchmark,expected",
        [
            (-1, "gsm8k", 4),
            (1319, "gsm8k", 4),
            (150, "gsm8k", 3),
            (164, "humaneval", 4),
            (16, "humaneval", 3),
            (150, None, 3),
            (None, "gsm8k", None),
        ],
        ids=["explicit-full", "full-by-size", "subsample", "humaneval-full",
             "humaneval-sub", "unknown-benchmark", "no-limit"],
    )
    def test_tier(self, limit, benchmark, expected) -> None:
        assert eval_events.tier_for(limit, benchmark) == expected


class TestGotSignal:
    def test_foreground_result_carrying_the_score(self) -> None:
        r = rows([
            use(0, "2026-01-01T00:00:00Z", "python evaluate.py --model-path m --limit 150"),
            result(1, "2026-01-01T00:02:00Z", "t0", '{"accuracy": 0.93, "stderr": 0.02}'),
        ])
        assert r[0]["got_signal"] is True
        assert r[0]["accuracy"] == 0.93
        assert r[0]["signal_via"] == "returned"
        assert r[0]["wait_s"] == 120.0

    def test_backgrounded_score_arrives_by_catting_the_output_file(self) -> None:
        r = rows([
            use(0, "2026-01-01T00:00:00Z",
                "nohup python evaluate.py --model-path m --limit 150 "
                "--json-output-file res/v1.json &"),
            result(1, "2026-01-01T00:00:01Z", "t0", "PID: 1031874"),
            use(2, "2026-01-01T00:40:00Z", "cat res/v1.json"),
            result(3, "2026-01-01T00:40:01Z", "t2", '{"accuracy": 0.7266}'),
        ])
        assert r[0]["got_signal"] is True
        assert r[0]["accuracy"] == pytest.approx(0.7266)
        assert r[0]["signal_via"] == "artifact"

    def test_a_score_from_an_unrelated_evaluation_does_not_count(self) -> None:
        """A result merely containing a number is not evidence for this launch."""
        r = rows([
            use(0, "2026-01-01T00:00:00Z",
                "nohup python evaluate.py --model-path m --limit 150 "
                "--json-output-file res/mine.json &"),
            result(1, "2026-01-01T00:00:01Z", "t0", "PID: 1"),
            use(2, "2026-01-01T00:40:00Z", "cat res/somebody_else.json"),
            result(3, "2026-01-01T00:40:01Z", "t2", '{"accuracy": 0.99}'),
        ])
        assert r[0]["got_signal"] is False
        assert r[0]["accuracy"] is None

    def test_truncated_output_means_no_signal(self) -> None:
        """The run that piped both evaluations through `| head -100` and never
        saw a score. It must not read as a completed verification."""
        r = rows([
            use(0, "2026-01-01T00:00:00Z",
                "python evaluate.py --model-path m --limit 150 2>&1 | head -100"),
            result(1, "2026-01-01T00:05:00Z", "t0",
                   "INFO Automatically detected platform cuda.\nLoading model shards..."),
        ])
        assert r[0]["got_signal"] is False
        assert r[0]["accuracy"] is None

    def test_log_line_spelling_is_recognised(self) -> None:
        r = rows([
            use(0, "2026-01-01T00:00:00Z",
                "nohup python evaluate.py --model-path m --limit 64 > eval_v1.log 2>&1 &"),
            result(1, "2026-01-01T00:00:01Z", "t0", "PID: 9"),
            use(2, "2026-01-01T00:30:00Z", 'grep -E "Accuracy:|Examples:" eval_v1.log | tail -2'),
            result(3, "2026-01-01T00:30:01Z", "t2", "  Examples: 64\n  Accuracy: 0.2339 (±0.0402)"),
        ])
        assert r[0]["got_signal"] is True
        assert r[0]["accuracy"] == pytest.approx(0.2339)

    def test_a_score_outside_zero_to_one_is_not_an_accuracy(self) -> None:
        r = rows([
            use(0, "2026-01-01T00:00:00Z", "python evaluate.py --model-path m --limit 150"),
            result(1, "2026-01-01T00:02:00Z", "t0", "accuracy: 93.0"),
        ])
        assert r[0]["got_signal"] is False


class TestFrame:
    def test_empty_frame_keeps_the_contract(self) -> None:
        df = eval_events.empty()
        assert list(df.columns) == list(eval_events.COLUMNS)
        assert df.empty


@pytest.mark.needs_data
class TestChampionRun:
    """The C1 comparison the reference document reports as 93.0% -> 85.5%."""

    @pytest.fixture
    def rows(self) -> list[dict]:
        path = events_dir("posttrainbench") / f"{CHAMPION}.jsonl.gz"
        if not path.exists():
            pytest.skip(f"champion run not converted: {path}")
        with gzip.open(path, "rt") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        return eval_events.events_for_run(CHAMPION, events, "gsm8k")

    def test_the_greedy_and_sampled_scores_are_recovered(self, rows: list[dict]) -> None:
        got = [r["accuracy"] for r in rows if r["got_signal"]]
        assert any(abs(a - 0.930) < 0.001 for a in got), "greedy 93.0% not found"
        assert any(abs(a - 0.855) < 0.001 for a in got), "sampled 85.5% not found"

    def test_some_evaluations_returned_nothing(self, rows: list[dict]) -> None:
        """Not every launch yields a number, and the table must show that."""
        assert any(r["got_signal"] is False for r in rows)
