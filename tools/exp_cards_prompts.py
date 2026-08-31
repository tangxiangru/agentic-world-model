"""Write one filled extractor prompt per run, so a subagent can be pointed at a file.

Reads ``<data>/exp-cards/<split>/manifest.json`` (from render_card_digests.py)
and the prompt template ``tools/exp_cards_prompt.md``; writes
``<data>/exp-cards/<split>/prompts/<run_ref>.md`` and creates the empty output
directory ``results/exp-cards/<split>/<side>/<run_ref>/``.

Usage:
    python3 tools/exp_cards_prompts.py [split-name]
    python3 tools/exp_cards_prompts.py --todo         # list run_refs with no cards yet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from awm import paths  # noqa: E402

TEMPLATE = ROOT / "tools/exp_cards_prompt.md"


def main(name: str = "gsm8k-gemma-holdout-v1", todo_only: bool = False) -> int:
    data = paths.data_root() / "exp-cards" / name
    out_root = ROOT / "results/exp-cards" / name
    manifest = json.loads((data / "manifest.json").read_text())
    template = TEMPLATE.read_text()
    (data / "prompts").mkdir(exist_ok=True)
    todo = []
    for ref, m in sorted(manifest.items()):
        out_dir = out_root / m["side"] / ref
        out_dir.mkdir(parents=True, exist_ok=True)
        if not any(out_dir.glob("exp-*.yaml")):
            todo.append((ref, m["side"], m["digest_chars"]))
        if todo_only:
            continue
        task_dir = data / "task" / ref
        fills = {
            "{run_ref}": ref,
            "{trained_model}": str(m["trained_model"]),
            "{digest}": str(data / "digests" / f"{ref}.md"),
            "{task_dir}": str(task_dir),
            "{task_files}": ", ".join(m["task_files"]) or "(none)",
            "{out_dir}": str(out_dir),
        }
        prompt = template
        for key, value in fills.items():
            prompt = prompt.replace(key, value)
        (data / "prompts" / f"{ref}.md").write_text(prompt)
    if todo_only:
        for ref, side, chars in todo:
            print(f"{ref}\t{side}\t{chars}")
        print(f"# {len(todo)} runs without cards", file=sys.stderr)
    else:
        print(f"{len(manifest)} prompts -> {data / 'prompts'}; {len(todo)} runs without cards")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raise SystemExit(main(args[0] if args else "gsm8k-gemma-holdout-v1", "--todo" in sys.argv))
