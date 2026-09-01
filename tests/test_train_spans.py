"""What a training span must and must not be paired to.

The synthetic cases below are the shapes that actually broke the previous
extractor, written small enough to read: a smoke test that returns in the
foreground, a real training backgrounded behind ``nohup``, an OOM relaunch
chain, and the polling commands agents interleave while a training runs.
"""

from __future__ import annotations

import gzip
import json

import pytest

from awm.traj import train_spans
from awm.paths import events_dir

CHAMPION = "claude_non_api_claude-opus-5_10h_run1__gsm8k_Qwen_Qwen3-4B-Base_17415829"


def use(i: int, ts: str, command: str, tid: str | None = None) -> dict:
    return {
        "run_id": "r",
        "agent_id": "main",
        "i": i,
        "type": "tool_use",
        "role": "assistant",
        "ts": ts,
        "tool": "Bash",
        "args": {"command": command},
        "tool_use_id": tid or f"t{i}",
    }


def result(i: int, ts: str, parent: str, text: str = "") -> dict:
    return {
        "run_id": "r",
        "agent_id": "main",
        "i": i,
        "type": "tool_result",
        "role": "user",
        "ts": ts,
        "parent_tool_use": parent,
        "text": text,
    }


def spans(events: list[dict]) -> list[dict]:
    return train_spans.spans_for_run("r", events)


class TestWhatCountsAsALaunch:
    def test_python_invoking_a_trainer_launches(self) -> None:
        s = spans([use(0, "2026-01-01T00:00:00Z", "python train_sft.py --out ckpt/a")])
        assert len(s) == 1
        assert s[0]["out_dir"] == "ckpt/a"

    @pytest.mark.parametrize(
        "command",
        [
            "sed -i 's/save_steps=50/save_steps=40/' train_grpo.py",
            "ps aux | grep -c train_sft",
            "wc -c logs/sft1.log; head -c 3000 logs/sft1.log",
            "grep -n 'Traceback' -A 12 logs/grpo2.log",
        ],
        ids=["sed-edits-it", "ps-greps-it", "reads-its-log", "greps-its-log"],
    )
    def test_naming_a_trainer_is_not_launching_one(self, command: str) -> None:
        assert spans([use(0, "2026-01-01T00:00:00Z", command)]) == []


class TestForegroundIsExact:
    def test_span_is_the_wait_the_agent_actually_served(self) -> None:
        """Not ``train_runtime`` — the agent also waited through model loading."""
        s = spans(
            [
                use(0, "2026-01-01T00:00:00Z", "timeout 900 python train_sft.py "
                    "--max-samples 600 --out ckpt/smoke"),
                result(1, "2026-01-01T00:00:59Z", "t0", "{'train_runtime': 35.07}"),
            ]
        )
        assert s[0]["sec"] == 59.0
        assert s[0]["train_runtime_s"] == 35.07
        assert s[0]["mode"] == "foreground"
        assert s[0]["end_reason"] == "returned"


class TestBackgroundPairsByArtifact:
    def test_ends_when_the_artifact_is_consumed(self) -> None:
        s = spans(
            [
                use(0, "2026-01-01T00:00:00Z",
                    "nohup python -u train_sft.py --out ckpt/sft1 > logs/sft1.log 2>&1 &"),
                use(1, "2026-01-01T01:00:00Z", "python finalize.py ckpt/sft1 eval_sft1"),
            ]
        )
        assert s[0]["sec"] == 3600.0
        assert s[0]["end_reason"] == "consumed"

    @pytest.mark.parametrize(
        "poll",
        [
            "ls ckpt/sft1",
            "du -sh ckpt/sft1",
            "tail -c 250 logs/sft1.log; ls ckpt/sft1 2>/dev/null",
            'until grep -qE "^saved ckpt" logs/sft1.log; do sleep 90; done; ls ckpt/sft1',
        ],
        ids=["ls", "du", "tail-then-ls", "wait-loop"],
    )
    def test_polling_does_not_end_it(self, poll: str) -> None:
        """The failure that cut a 1.8h GRPO run to 1.0h: agents poll constantly."""
        s = spans(
            [
                use(0, "2026-01-01T00:00:00Z",
                    "nohup python -u train_sft.py --out ckpt/sft1 > logs/sft1.log 2>&1 &"),
                use(1, "2026-01-01T00:30:00Z", poll),
                use(2, "2026-01-01T02:00:00Z", "python finalize.py ckpt/sft1 eval_sft1"),
            ]
        )
        assert s[0]["sec"] == 7200.0
        assert s[0]["end_reason"] == "consumed"

    def test_relaunching_the_same_artifact_supersedes(self) -> None:
        """An OOM relaunch means the first attempt died; its wall clock still cost."""
        s = spans(
            [
                use(0, "2026-01-01T00:00:00Z",
                    "nohup python -u train_grpo.py --out ckpt/g --bs 8 > logs/g.log 2>&1 &"),
                use(1, "2026-01-01T00:03:00Z",
                    "nohup python -u train_grpo.py --out ckpt/g --bs 4 > logs/g.log 2>&1 &"),
                use(2, "2026-01-01T02:00:00Z", "python finalize.py ckpt/g eval_g"),
            ]
        )
        assert [r["end_reason"] for r in s] == ["superseded", "consumed"]
        assert [r["sec"] for r in s] == [180.0, 7020.0]

    def test_next_stage_training_from_it_consumes_it(self) -> None:
        s = spans(
            [
                use(0, "2026-01-01T00:00:00Z",
                    "nohup python -u train_sft.py --out ckpt/sft1 > logs/sft1.log 2>&1 &"),
                use(1, "2026-01-01T01:00:00Z",
                    "nohup python -u train_grpo.py --model ckpt/sft1 --out ckpt/g > l 2>&1 &"),
            ]
        )
        assert s[0]["end_reason"] == "consumed"
        assert s[0]["sec"] == 3600.0

    def test_unfinished_training_runs_to_the_end_of_the_run(self) -> None:
        s = spans(
            [
                use(0, "2026-01-01T00:00:00Z",
                    "nohup python -u train_sft.py --out ckpt/sft1 > logs/sft1.log 2>&1 &"),
                use(1, "2026-01-01T00:10:00Z", "nvidia-smi"),
            ]
        )
        assert s[0]["end_reason"] == "run_end"


class TestDiscarding:
    def test_removing_the_whole_artifact_abandons_it(self) -> None:
        s = spans(
            [
                use(0, "2026-01-01T00:00:00Z",
                    "nohup python -u train_grpo.py --out ckpt/g2 > logs/g2.log 2>&1 &"),
                use(1, "2026-01-01T00:20:00Z",
                    "rm -rf ckpt/g2 && cp -r eval_g150 ckpt/g150_bf16"),
            ]
        )
        assert s[0]["end_reason"] == "discarded"

    def test_removing_one_checkpoint_inside_it_is_housekeeping(self) -> None:
        """Agents delete spent checkpoints for disk while still submitting the run."""
        s = spans(
            [
                use(0, "2026-01-01T00:00:00Z",
                    "nohup python -u train_grpo.py --out ckpt/g2 > logs/g2.log 2>&1 &"),
                use(1, "2026-01-01T01:00:00Z", "rm -rf ckpt/g2/checkpoint-160"),
                use(2, "2026-01-01T01:00:04Z", "python finalize.py ckpt/g2/checkpoint-120 out"),
            ]
        )
        assert s[0]["end_reason"] == "consumed"
        assert s[0]["sec"] == 3604.0

    def test_a_cp_elsewhere_in_the_command_does_not_consume_it(self) -> None:
        s = spans(
            [
                use(0, "2026-01-01T00:00:00Z",
                    "nohup python -u train_grpo.py --out ckpt/g2 > logs/g2.log 2>&1 &"),
                use(1, "2026-01-01T00:20:00Z", "cp -r eval_g150 ckpt/g150_bf16 && ls ckpt/g2"),
                use(2, "2026-01-01T01:00:00Z", "python finalize.py ckpt/g2 out"),
            ]
        )
        assert s[0]["sec"] == 3600.0


class TestKindIsIntentNotDuration:
    @pytest.mark.parametrize(
        "command,expected",
        [
            ("python train_sft.py --max-samples 600 --out ckpt/x", "smoke"),
            ("python train_sft.py --max-samples 5000 --out ckpt/x", "smoke"),
            ("python train_grpo.py --max-steps 400 --out ckpt/x", "real"),
            ("python train_sft.py --max-samples 51000 --out ckpt/x", "real"),
            ("python train_sft.py --out ckpt/smoke", "smoke"),
            ("python train_sft.py --out ckpt/sanity1", "smoke"),
            ("python train_sft.py --data d.jsonl --out ckpt/sft1", "real"),
        ],
        ids=["tiny-slice", "at-the-bound", "max-steps-is-a-real-knob",
             "big-slice", "smoke-dir", "sanity-dir", "real"],
    )
    def test_kind(self, command: str, expected: str) -> None:
        assert spans([use(0, "2026-01-01T00:00:00Z", command)])[0]["kind"] == expected

    def test_a_long_smoke_test_is_still_a_smoke_test(self) -> None:
        """Duration must never decide kind: that read a 138s smoke test as a
        run's whole training history."""
        s = spans(
            [
                use(0, "2026-01-01T00:00:00Z",
                    "python train_sft.py --max-samples 600 --out ckpt/x"),
                result(1, "2026-01-01T03:00:00Z", "t0"),
            ]
        )
        assert s[0]["kind"] == "smoke"
        assert s[0]["sec"] == 10800.0


class TestFrame:
    def test_empty_frame_still_has_the_contract(self) -> None:
        df = train_spans.empty()
        assert list(df.columns) == list(train_spans.COLUMNS)
        assert df.empty


@pytest.mark.needs_data
class TestChampionRun:
    """The one run whose training history was reconstructed by hand.

    SFT ran 07:46:36 -> 09:02:17 and GRPO2 12:11:36 -> 13:24:12. The extractor
    this replaces reported 138.4 seconds of training for the whole run, that
    being its smoke test's real ``train_runtime``.
    """

    @pytest.fixture
    def rows(self) -> list[dict]:
        path = events_dir("posttrainbench") / f"{CHAMPION}.jsonl.gz"
        if not path.exists():
            pytest.skip(f"champion run not converted: {path}")
        with gzip.open(path, "rt") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        return train_spans.spans_for_run(CHAMPION, events)

    def test_sft_span_matches_the_hand_check(self, rows: list[dict]) -> None:
        sft = max((r for r in rows if r["out_dir"] == "ckpt/sft1"), key=lambda r: r["sec"])
        assert sft["sec"] == pytest.approx(4541.0, abs=1.0)

    def test_grpo2_span_matches_the_hand_check(self, rows: list[dict]) -> None:
        g2 = max((r for r in rows if r["out_dir"] == "ckpt/grpo2"), key=lambda r: r["sec"])
        assert g2["sec"] == pytest.approx(4356.0, abs=1.0)

    def test_total_real_training_is_hours_not_seconds(self, rows: list[dict]) -> None:
        total = sum(r["sec"] for r in rows if r["kind"] == "real")
        assert total > 4 * 3600, f"the bug this replaces reported 138.4s, got {total}s"

    def test_the_oom_relaunch_chain_is_recognised(self, rows: list[dict]) -> None:
        grpo1 = [r for r in rows if r["out_dir"] == "ckpt/grpo1"]
        assert sum(r["end_reason"] == "superseded" for r in grpo1) >= 4

    def test_smoke_tests_are_classified_by_intent(self, rows: list[dict]) -> None:
        smoke = [r for r in rows if r["kind"] == "smoke"]
        assert {r["out_dir"] for r in smoke} == {"ckpt/smoke"}
        assert all(r["mode"] == "foreground" for r in smoke)


class TestLaunchThatNeverRan:
    """``pkill …; nohup python train.py &`` where the shell dies at the pkill.

    The agent's own pattern matches the shell's command line, so ``pkill`` kills
    the shell before it reaches the launch. The command text says a training
    started; the exit code says the shell was signalled. Two of one run's eight
    recorded launches were these, and the spans they invented were then read as
    crashes because they ended within seconds.
    """

    def _events(self, result_text: str):
        return [
            {"run_id": "r", "i": 1, "type": "tool_use", "tool": "Bash",
             "tool_use_id": "t1", "ts": "2026-01-01T00:00:00Z",
             "args": {"command": 'pkill -f "train.py --data v2" 2>/dev/null; sleep 5\n'
                                 "nohup python train.py --data v2 --out ckpt/v2 &"}},
            {"run_id": "r", "i": 2, "type": "tool_result", "parent_tool_use": "t1",
             "ts": "2026-01-01T00:00:05Z", "is_error": True, "text": result_text},
            {"run_id": "r", "i": 3, "type": "tool_use", "tool": "Bash",
             "tool_use_id": "t2", "ts": "2026-01-01T02:00:00Z",
             "args": {"command": "nvidia-smi"}},
        ]

    def test_a_signalled_shell_launched_nothing(self) -> None:
        rows = train_spans.spans_for_run("r", self._events("Exit code 144"))
        assert rows == [], "the launch never ran; recording it invents a span"

    def test_a_background_launch_that_did_start_survives(self) -> None:
        events = self._events("Exit code 144")
        events[1]["is_error"] = False
        events[1]["text"] = ""
        assert len(train_spans.spans_for_run("r", events)) == 1

    def test_a_foreground_crash_is_still_a_launch(self) -> None:
        """A trainer that ran and raised is a training. Only the shell dying
        before the launch is not, so the rule must not reach the foreground."""
        events = [
            {"run_id": "r", "i": 1, "type": "tool_use", "tool": "Bash",
             "tool_use_id": "t1", "ts": "2026-01-01T00:00:00Z",
             "args": {"command": "python train.py --out ckpt/v1"}},
            {"run_id": "r", "i": 2, "type": "tool_result", "parent_tool_use": "t1",
             "ts": "2026-01-01T00:03:00Z", "is_error": True,
             "text": "torch.OutOfMemoryError: CUDA out of memory"},
        ]
        rows = train_spans.spans_for_run("r", events)
        assert len(rows) == 1 and rows[0]["end_reason"] == "returned"


class TestEffectiveBatch:
    """The number the optimiser sees, which is what tells a tested
    hyperparameter apart from a memory compensation."""

    def test_the_transformers_spelling(self) -> None:
        assert train_spans.effective_batch(
            "python train.py --per-device-train-batch-size 4"
            " --gradient-accumulation-steps 8"
        ) == 32

    def test_the_shell_variable_spelling(self) -> None:
        assert train_spans.effective_batch("BS=2 GA=16 python train.py") == 32

    def test_an_oom_compensation_leaves_it_unchanged(self) -> None:
        """``bs 4->2`` with ``accum 4->8``: the same effective batch, refactored.

        Four annotators reported this shape and each had to record it as a
        hyperparameter change, because the schema offered nothing else. Half
        the corpus's readable bs/accum changes are this.
        """
        before = train_spans.effective_batch("--bs 4 --accum 4")
        after = train_spans.effective_batch("--bs 2 --accum 8")
        assert before == after == 16

    def test_unknown_when_the_command_states_only_one(self) -> None:
        """Agents fix these inside the trainer too. Unknown is not unchanged."""
        assert train_spans.effective_batch("python train.py --bs 4") is None


class TestTimeoutCeiling:
    """``timeout 900 python train.py`` cannot have run for four hours."""

    def test_the_command_states_its_own_ceiling(self) -> None:
        assert train_spans.timeout_cap("timeout 900 python train.py") == 900
        assert train_spans.timeout_cap("timeout -k 10 600 python train.py") == 600
        assert train_spans.timeout_cap("python train.py") is None

    def test_a_span_is_clamped_to_it(self) -> None:
        """Artifact pairing put this span's end hours after the process died.
        Ten background spans, 6% of those launched under a timeout, exceeded
        the limit they ran under; the worst claimed 4.37h against six minutes.
        """
        events = [
            {"run_id": "r", "i": 1, "type": "tool_use", "tool": "Bash",
             "tool_use_id": "t1", "ts": "2026-01-01T00:00:00Z",
             "args": {"command": "nohup timeout 900 python train.py --out ckpt/v1 &"}},
            {"run_id": "r", "i": 2, "type": "tool_result", "parent_tool_use": "t1",
             "ts": "2026-01-01T00:00:01Z", "text": ""},
            {"run_id": "r", "i": 3, "type": "tool_use", "tool": "Bash",
             "tool_use_id": "t2", "ts": "2026-01-01T04:00:00Z",
             "args": {"command": "python evaluate.py --model-path ckpt/v1"}},
        ]
        rows = train_spans.spans_for_run("r", events)
        assert len(rows) == 1
        assert rows[0]["sec"] == 900
        assert rows[0]["end_reason"] == "timeout_cap"

    def test_a_span_within_its_ceiling_is_untouched(self) -> None:
        events = [
            {"run_id": "r", "i": 1, "type": "tool_use", "tool": "Bash",
             "tool_use_id": "t1", "ts": "2026-01-01T00:00:00Z",
             "args": {"command": "nohup timeout 7200 python train.py --out ckpt/v1 &"}},
            {"run_id": "r", "i": 2, "type": "tool_result", "parent_tool_use": "t1",
             "ts": "2026-01-01T00:00:01Z", "text": ""},
            {"run_id": "r", "i": 3, "type": "tool_use", "tool": "Bash",
             "tool_use_id": "t2", "ts": "2026-01-01T01:00:00Z",
             "args": {"command": "python evaluate.py --model-path ckpt/v1"}},
        ]
        rows = train_spans.spans_for_run("r", events)
        assert rows[0]["sec"] == 3600
        assert rows[0]["end_reason"] == "consumed"

