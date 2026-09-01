"""The conversion overwrote commands; the raw file still has them."""

from __future__ import annotations

from awm.traj import repair


def _raw(tmp_path):
    run = tmp_path / "fam" / "tail"
    run.mkdir(parents=True)
    (run / "solve_out.txt").write_text(
        '[2026-04-25T18:24:57Z] {"type":"item.started","item":'
        '{"id":"item_2","type":"command_execution","command":"bash timer.sh"}}\n'
        '[2026-04-25T18:25:14Z] {"type":"item.started","item":'
        '{"id":"item_9","type":"command_execution","command":"find templates -type f"}}\n'
        "not json at all\n"
        '[2026-04-25T18:26:00Z] {"type":"item.completed","item":{"id":"item_2"}}\n',
        encoding="utf-8",
    )
    return tmp_path


class TestRepair:
    def test_an_overwritten_command_is_restored(self, tmp_path) -> None:
        """A later episode's args landed on an earlier episode's event; the
        line that event points at still holds the command it really ran."""
        events = [
            {"type": "tool_use", "i": 3, "source_ref": {"line": 1},
             "args": {"command": "python scripts/train_math_lora.py --train-file x"}},
        ]
        out, n = repair.repair("fam__tail", events, root=_raw(tmp_path))
        assert n == 1
        assert out[0]["args"]["command"] == "bash timer.sh"
        assert "train_math_lora" in out[0]["args"]["command_before_repair"]

    def test_an_intact_command_is_left_alone(self, tmp_path) -> None:
        events = [
            {"type": "tool_use", "i": 5, "source_ref": {"line": 2},
             "args": {"command": "find templates -type f"}},
        ]
        out, n = repair.repair("fam__tail", events, root=_raw(tmp_path))
        assert n == 0
        assert "command_before_repair" not in out[0]["args"]

    def test_a_line_that_is_not_a_command_is_skipped(self, tmp_path) -> None:
        """Only lines the raw file proves are command_execution may overwrite."""
        events = [
            {"type": "tool_use", "i": 7, "source_ref": {"line": 3},
             "args": {"command": "whatever the conversion wrote"}},
            {"type": "tool_use", "i": 8, "source_ref": {"line": 4},
             "args": {"command": "also untouched"}},
        ]
        out, n = repair.repair("fam__tail", events, root=_raw(tmp_path))
        assert n == 0
        assert out[0]["args"]["command"] == "whatever the conversion wrote"

    def test_a_missing_raw_file_reports_zero(self, tmp_path) -> None:
        """The caller must be able to tell "nothing to fix" from "cannot look"."""
        events = [{"type": "tool_use", "i": 1, "source_ref": {"line": 1},
                   "args": {"command": "x"}}]
        out, n = repair.repair("fam__absent", events, root=tmp_path)
        assert (out, n) == (events, 0)

    def test_lost_launches_are_enumerated(self, tmp_path) -> None:
        """Raw item.started lines minus the lines any event already claims.

        Locating them is mechanical; only reading them needs judgement. And
        repairing without reinstating is worse than not repairing: a displaced
        args.command is sometimes the stream's only record of a real
        evaluation, so putting the true command back deletes it.
        """
        root = _raw(tmp_path)
        events = [{"type": "tool_use", "i": 3, "source_ref": {"line": 1},
                   "args": {"command": "bash timer.sh"}}]
        out = repair.lost("fam__tail", events, root=root)
        assert [e["args"]["command"] for e in out] == ["find templates -type f"]
        assert out[0]["origin"] == "reinstated"
        assert out[0]["ts"] == "2026-04-25T18:25:14Z"
        assert out[0]["i"] > 3, "must sort after the event it follows"

    def test_nothing_lost_when_every_line_is_claimed(self, tmp_path) -> None:
        events = [{"type": "tool_use", "i": 1, "source_ref": {"line": 1}, "args": {}},
                  {"type": "tool_use", "i": 2, "source_ref": {"line": 2}, "args": {}}]
        assert repair.lost("fam__tail", events, root=_raw(tmp_path)) == []

