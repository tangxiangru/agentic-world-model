"""Post-process reconstructed experiment cards: validate, lint, join, index.

Steps, all deterministic (see doc/exp-card-specs/extraction-protocol.md):

1. Parse every ``results/exp-cards/<split>/<side>/<run_ref>/exp-*.yaml``;
   report files that do not parse or lack the top-level sections.
2. Check ``setup.command.argv`` against the digest block at
   ``provenance.launch_i`` (whitespace-collapsed substring).
3. Lint the serialised YAML for agent identity: agent model names and harness
   names from the catalogue, plus the run's own experiment/run strings.
4. Join ``outcome.official_accuracy`` from the pinned catalogue onto the
   adopted card whose output is the submission (last ``adopt`` in launch order).
5. Write ``results/exp-cards/<split>/index.md`` and ``coverage.json``.

Usage:
    python3 tools/exp_cards_post.py [split-name] [--no-join]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from awm import paths  # noqa: E402

SECTIONS = ("problem", "hypothesis", "setup", "evaluation", "result", "conclusion", "provenance")
# "openai" is deliberately absent: `openai/gsm8k` is the dataset id.
HARNESS_WORDS = ("claude code", "claude_code", "codex", "cursor", "opencode", "kimi", "glmx",
                 "qwen3max", "anthropic")


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def digest_blocks(text: str) -> dict[int, str]:
    blocks: dict[int, str] = {}
    cur: int | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^--- \[(\d+)\]", line)
        if m:
            if cur is not None:
                blocks[cur] = "\n".join(buf)
            cur, buf = int(m.group(1)), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        blocks[cur] = "\n".join(buf)
    return blocks


def identity_terms(catalog_rows: list[dict], run: str) -> list[str]:
    terms = set()
    for row in catalog_rows:
        for key in ("agent_model", "agent"):
            v = row.get(key)
            if isinstance(v, str) and len(v) >= 4:
                terms.add(v.lower())
    config, run_name = run.split("/", 1)
    terms.add(config.lower())
    terms.add(run_name.lower())
    terms.add(run_name.rsplit("_", 1)[-1])  # cluster id
    return sorted(terms)


def main(name: str = "gsm8k-gemma-holdout-v1", join: bool = True) -> int:
    out_root = ROOT / "results/exp-cards" / name
    data = paths.data_root() / "exp-cards" / name
    sources = json.loads((data / "sources.json").read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    catalog_path = paths.raw_dir("posttrainbench") / "viewer_data" / "index.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.is_file() else []
    if isinstance(catalog, dict):
        catalog = catalog.get("runs") or catalog.get("rows") or list(catalog.values())
    by_run = {}
    for row in catalog:
        exp, rn = row.get("experiment"), row.get("run_name") or row.get("run")
        if exp and rn:
            by_run[f"{exp}/{rn}"] = row

    problems: list[str] = []
    rows: list[dict] = []
    coverage: Counter = Counter()
    n_cards = 0
    for side in ("train", "test"):
        for run_dir in sorted((out_root / side).glob("r-*")):
            ref = run_dir.name
            run = sources.get(ref)
            cards = sorted(run_dir.glob("exp-*.yaml"))
            if not cards or not (run_dir / "index.md").is_file():
                continue  # nothing written, or an extractor is still writing
            digest = digest_blocks((data / "digests" / f"{ref}.md").read_text())
            terms = identity_terms(catalog, run) if run else []
            parsed: list[tuple[Path, dict]] = []
            for path in cards:
                text = path.read_text()
                try:
                    card = yaml.safe_load(text)
                except yaml.YAMLError as exc:
                    problems.append(f"{path}: YAML error {str(exc)[:80]}")
                    continue
                if not isinstance(card, dict):
                    problems.append(f"{path}: not a mapping")
                    continue
                # Cards written while the live template carried world-model-agent
                # fields: strip them so every card follows the frozen schema.
                stripped = False
                setup = card.get("setup") or {}
                for key in ("base_model", "resume_argv", "output_dir", "progress"):
                    if key in setup:
                        setup.pop(key); stripped = True
                res = card.get("result") or {}
                if "pings_acted_on" in res:
                    res.pop("pings_acted_on"); stripped = True
                diag = ((card.get("evaluation") or {}).get("diagnostic")) or {}
                if isinstance(diag, dict) and ("items" in diag or "metric" in diag):
                    card["evaluation"]["diagnostic"] = {
                        "what": diag.get("what"), "command": [], "path": diag.get("items")}
                    stripped = True
                if stripped:
                    path.write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True, width=100))
                    text = path.read_text()
                missing = [s for s in SECTIONS if s not in card]
                if missing:
                    problems.append(f"{path}: missing {missing}")
                prov = card.get("provenance") or {}
                li = prov.get("launch_i")
                argv = ((card.get("setup") or {}).get("command") or {}).get("argv") or []
                if li is not None and argv:
                    block = _ws(digest.get(int(li), ""))
                    def _norm(t: str) -> str:
                        return _ws(t.replace('"', "").replace("'", "").replace("\\", ""))
                    head = _norm(" ".join(str(a) for a in argv[:3]))[:60]
                    block = _norm(block)
                    if head and head not in block:
                        problems.append(f"{path}: argv head {head!r} not in digest block [{li}]")
                # The extractor names itself in provenance.extractor; that is not the
                # run's agent, so drop the line before linting.
                low = "\n".join(l for l in text.lower().splitlines()
                                 if not l.strip().startswith("extractor:"))
                hits = [t for t in terms if t in low] + [w for w in HARNESS_WORDS if w in low]
                if hits:
                    problems.append(f"{path}: identity leak {hits[:3]}")
                parsed.append((path, card))
                n_cards += 1
                for sec in SECTIONS[:-1]:
                    v = card.get(sec) or {}
                    for k, val in (v.items() if isinstance(v, dict) else []):
                        if val not in (None, [], {}, ""):
                            coverage[f"{sec}.{k}"] += 1
            # official score onto the submitted card
            adopted = [(p, c) for p, c in parsed
                       if ((c.get("conclusion") or {}).get("decision") == "adopt")]
            acc = manifest.get(ref, {}).get("accuracy")
            if acc is None and run in by_run:
                acc = by_run[run].get("accuracy")
            if join and adopted and acc is not None:
                path, card = adopted[-1]
                card.setdefault("outcome", {})["official_accuracy"] = acc
                path.write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True, width=100))
            for path, card in parsed:
                s = card.get("setup") or {}
                r = card.get("result") or {}
                c = card.get("conclusion") or {}
                meas = r.get("measurements") or []
                best = max((m.get("value") for m in meas if isinstance(m.get("value"), (int, float))),
                           default=None)
                rows.append({
                    "side": side, "run_ref": ref, "card": path.stem,
                    "model": manifest.get(ref, {}).get("trained_model"),
                    "launch_i": (card.get("provenance") or {}).get("launch_i"),
                    "family": (s.get("method") or {}).get("family"),
                    "parent": (s.get("parent_checkpoint") or {}).get("origin"),
                    "datasets": ",".join(str(d.get("source")) for d in (s.get("data") or [])),
                    "execution": r.get("execution"),
                    "best": best,
                    "verdict": c.get("verdict"), "decision": c.get("decision"),
                    "stated_hyp": ((card.get("provenance") or {}).get("stated_by_agent") or {}).get("hypothesis"),
                    "official": (card.get("outcome") or {}).get("official_accuracy"),
                })

    lines = ["# Index of reconstructed cards", "",
             "| side | run_ref | card | base model | launch_i | family | parent | data sources | exec | best own eval | verdict | decision | hyp stated | official |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) if r[k] is not None else "" for k in
                     ("side", "run_ref", "card", "model", "launch_i", "family", "parent", "datasets",
                      "execution", "best", "verdict", "decision", "stated_hyp", "official")) + " |")
    (out_root / "index.md").write_text("\n".join(lines) + "\n")
    (out_root / "coverage.json").write_text(json.dumps(
        {"cards": n_cards, "runs": len({r['run_ref'] for r in rows}),
         "field_coverage": dict(sorted(coverage.items())), "problems": problems}, indent=1))
    print(f"{n_cards} cards over {len({r['run_ref'] for r in rows})} runs; {len(problems)} problems")
    for p in problems[:30]:
        print("  ", p)
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raise SystemExit(main(args[0] if args else "gsm8k-gemma-holdout-v1", "--no-join" not in sys.argv))
