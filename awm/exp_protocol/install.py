"""Put the scientist skill where the scientist's tool will find it.

Claude Code discovers ``.claude/skills/<name>/SKILL.md`` and reads
``CLAUDE.md``; Codex reads ``AGENTS.md``. The skill itself lives once, at
``skills/exp_protocol/``; the rest is a symlink and one marked block in each
tool's instruction file, so the protocol is pointed at explicitly and not left
to whether the tool decides to open the skill. The meta skill is never copied:
a scientist must not read how it is being iterated on.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from .preflight import skill_dir

BEGIN = "<!-- exp_protocol:begin -->"
END = "<!-- exp_protocol:end -->"
AGENTS_BLOCK = f"""{BEGIN}
## Experiment protocol

Before any model-training or evaluation experiment, read `skills/exp_protocol/SKILL.md`
and follow it: write the card (sections 0-4), `awm exp_protocol check`, `awm exp_protocol lock`,
then launch; fill sections 5-6 and `awm exp_protocol close` afterwards. Read
`skills/exp_protocol/pitfalls.yaml` before your first launch.
{END}
"""


class InstallError(ValueError):
    pass


def _make_writable(root: Path) -> None:
    if not root.is_dir():
        return
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            continue
        try:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass


def _pointer_block(path: Path) -> Path:
    existing = path.read_text() if path.is_file() else ""
    if BEGIN in existing and END in existing:
        head, rest = existing.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        text = head + AGENTS_BLOCK.rstrip("\n") + tail
    elif BEGIN in existing:
        # A truncated block: everything from the dangling marker on was ours; replace it.
        head, _ = existing.split(BEGIN, 1)
        text = head + AGENTS_BLOCK
    else:
        text = (existing.rstrip("\n") + "\n\n" if existing.strip() else "") + AGENTS_BLOCK
    path.write_text(text if text.endswith("\n") else text + "\n")
    return path


def install(target: Path, tool: str = "both") -> list[Path]:
    if tool not in ("claude", "codex", "both"):
        raise InstallError(f"tool must be claude, codex, or both, not {tool!r}")
    src = skill_dir()
    if src.name != "exp_protocol" or not (src / "SKILL.md").is_file():
        raise InstallError(f"{src} is not the scientist skill (skills/exp_protocol); refusing")
    target = Path(target)
    written: list[Path] = []

    dst = target / "skills" / "exp_protocol"
    if dst.is_symlink():
        dst.unlink()  # a link to somewhere else is not the skill; the real files go here
    # Merge-copy: the skill's files are refreshed, anything the scientist added alongside them stays.
    _make_writable(dst)  # a previous copy from a read-only source would refuse the refresh
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The copy belongs to the scientist, whatever mode the source had (a read-only mount, say).
    _make_writable(dst)
    written.append(dst)

    if tool in ("claude", "both"):
        link = target / ".claude" / "skills" / "exp_protocol"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        os.symlink("../../skills/exp_protocol", link)
        written.append(link)
        written.append(_pointer_block(target / "CLAUDE.md"))

    if tool in ("codex", "both"):
        written.append(_pointer_block(target / "AGENTS.md"))
    return written
