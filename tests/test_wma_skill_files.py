"""The skill's own files obey the skill."""

from __future__ import annotations

import re

from awm import paths
from awm.wma import review, schema


def test_example_verdict_validates() -> None:
    v = schema.load_verdict(review.default_skill_dir() / "verdict.example.json")
    r = schema.validate_verdict(v)
    assert r.ok, r.render()


def test_skill_md_names_the_levels_the_file_and_the_offline_rule() -> None:
    text = (review.default_skill_dir() / "SKILL.md").read_text()
    assert text.startswith("---\nname: wma\n")
    assert re.search(r"^description: .+", text, re.MULTILINE)
    for needle in schema.LEVELS + ("verdict.json", "static_check", "Offline mode", "Not a new direction",
                                   "change_types", "noise floor", "change_types.md"):
        assert needle in text, needle


def test_the_manual_is_part_of_the_skill_and_names_every_change_type() -> None:
    manual = review.default_skill_dir() / "change_types.md"
    text = manual.read_text()
    for t in [f"C{i}" for i in range(2, 19)] + ["C1a", "C1b"]:
        assert f"**{t}**" in text, t
    for needle in ("tier", "noise floor", "controlled", "max_model_len", "modules_to_save", "--limit"):
        assert needle in text, needle
    example = schema.load_verdict(review.default_skill_dir() / "verdict.example.json")
    assert example["change_types"] and schema.validate_verdict(example).ok


def test_repo_symlink_and_agents_md() -> None:
    link = paths.REPO_ROOT / ".claude" / "skills" / "wma"
    assert link.is_symlink() and (link / "SKILL.md").is_file()
    assert "skills/wma/SKILL.md" in (paths.REPO_ROOT / "AGENTS.md").read_text()


def test_meta_skill_exists_and_names_the_loop() -> None:
    text = (paths.REPO_ROOT / "skills" / "wma_meta" / "SKILL.md").read_text()
    assert text.startswith("---\nname: wma_meta\n")
    for needle in ("awm wma replay", "awm wma ledger", "--limit 20", "test side", "one change", "width"):
        assert needle in text, needle
    link = paths.REPO_ROOT / ".claude" / "skills" / "wma_meta"
    assert link.is_symlink() and (link / "SKILL.md").is_file()
    assert "skills/wma_meta/SKILL.md" in (paths.REPO_ROOT / "AGENTS.md").read_text()


def test_the_protocol_skill_on_this_line_puts_the_verdict_inside_the_lock() -> None:
    """The scientist's only knowledge of the WMA is the protocol skill. Since 2026-09-03 the verdict is
    part of the lock: `lock` asks and waits, and the run may not start before it returns (Round 01
    showed 21/22 verdicts arriving after the launch under the old background step). WMA-aware line only."""
    text = (paths.REPO_ROOT / "skills" / "exp_protocol" / "SKILL.md").read_text()
    assert "The verdict is part of the lock" in text
    assert "Do not start the run before `lock` has returned" in text
    assert "only after lock has returned" in text
    assert "awm wma review --dir {dir} exp-NN --background" not in text
    assert "Do not wait for it to launch" not in text
    assert "no world-model agent is attached to this cell" in text      # the control arm's whole answer
    assert "--background" in text                                        # still there for batch second opinions
