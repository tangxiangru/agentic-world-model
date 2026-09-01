"""The scientist gets the protocol; never the meta skill."""

from __future__ import annotations

import os

import pytest

from awm.exp_protocol import install


def test_both_tools_get_the_skill_the_symlink_and_the_agents_block(tmp_path) -> None:
    written = install.install(tmp_path, "both")
    assert (tmp_path / "skills" / "exp_protocol" / "SKILL.md").is_file()
    assert (tmp_path / "skills" / "exp_protocol" / "pitfalls.yaml").is_file()
    link = tmp_path / ".claude" / "skills" / "exp_protocol"
    assert link.is_symlink() and os.readlink(link) == "../../skills/exp_protocol"
    assert (link / "SKILL.md").is_file()
    agents = (tmp_path / "AGENTS.md").read_text()
    assert install.BEGIN in agents and "skills/exp_protocol/SKILL.md" in agents
    assert all(p.exists() or p.is_symlink() for p in written)


def test_codex_only_writes_no_claude_symlink(tmp_path) -> None:
    install.install(tmp_path, "codex")
    assert not (tmp_path / ".claude").exists()
    assert (tmp_path / "AGENTS.md").is_file()


def test_claude_only_writes_no_agents_md(tmp_path) -> None:
    install.install(tmp_path, "claude")
    assert not (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / ".claude" / "skills" / "exp_protocol").is_symlink()


def test_install_is_idempotent_and_keeps_existing_agents_text(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Task\n\nkeep me\n")
    install.install(tmp_path, "both")
    install.install(tmp_path, "both")
    text = (tmp_path / "AGENTS.md").read_text()
    assert text.count(install.BEGIN) == 1 and "keep me" in text


def test_refuses_to_install_the_meta_skill(tmp_path, monkeypatch) -> None:
    meta = tmp_path / "src" / "exp_protocol_meta"
    meta.mkdir(parents=True)
    (meta / "SKILL.md").write_text("---\nname: exp_protocol_meta\n---\n")
    monkeypatch.setenv("AWM_EXP_PROTOCOL_DIR", str(meta))
    with pytest.raises(install.InstallError):
        install.install(tmp_path / "task", "both")
