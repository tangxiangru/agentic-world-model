"""Write the study's prompt files into a PostTrainBench checkout.

Every prompt is PostTrainBench's own ``prompt.txt`` plus study sections
inserted before ``## Rules`` — nothing else changes, so a cell differs from the
corpus runs only by what was added:

    prompt_fulltraj.txt      PTB prompt + "Prior runs"                          (C1: raw files, no WMA)
    prompt_wm.txt            PTB prompt + "The world-model agent"               (C3: WMA over memory)
    prompt_wm_fulltraj.txt   PTB prompt + "Prior runs" + "The world-model agent" (C2: raw files + WMA)

The WMA section is ``input/wma_section.md`` (the source of truth); the rendered
``prompt_wm.txt`` is also written to ``input/instruction.md`` for reference.
``get_prompt.py`` fills ``{model} {benchmark} {num_hours} ...`` by plain
replacement, so the sections use those placeholders too. Run by
rollout/setup.sh:

    python rollout/build_prompts.py <ptb checkout>
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PRIOR_RUNS_SECTION = """## Prior runs
- `/home/ben/prior_runs/` holds read-only published trajectory records of previous attempts at this same task (post-training a base model for {benchmark} under these rules) by other autonomous agents. Start with `/home/ben/prior_runs/INDEX.md`: one row per run with its base model, agent, official accuracy, split side, and path.
- Every run directory has exactly `solve_out.txt` (the complete session trajectory), `metrics.json` (the official score), and `time_taken.txt`. Optional upstream artifacts and `task/` workspace snapshots are deliberately excluded.
- The corpus does not include prior scripts, checkpoints, private caches, or referenced local data files. Reimplement any idea or recipe you infer from a trace in your own workspace.
- You may read anything that is present there and reuse any idea or recipe. Every rule below still applies: a prior run's training data is not exempt from contamination rules, and a prior run's base model is not yours unless it is `{model}`.
- Reading files under `/home/ben/prior_runs` is an allowed exception to the working-directory rule.

"""

WMA_SECTION = (ROOT / "input" / "wma_section.md").read_text()

PINNED_BASE_SECTION = """## Pinned base checkpoint
- This study permits only the official `google/gemma-3-4b-pt` base at revision `cc012e0a6d0787b4adcc0fa2c4da74402494554d`. It is already complete at the immutable read-only path `/home/ben/pinned-base/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d`.
- Load that exact snapshot locally. Do not inspect cache credential files, authenticate to Hugging Face, download a substitute, or use any mirror, fork, repack, instruction-tuned model, or other revision (including an `unsloth/*` model).

"""

SMOKE_SECTION = """## One-hour peer-session smoke protocol
This is a labelled integration smoke, not a production research cell. Exercise the complete two-session path: message the world-model agent with one concrete plan before training, run one optimizer step from the pinned base, evaluate and leave a complete loadable `final_model/`, then tell the world-model agent what you shipped. The smoke tests mechanics, not model quality.

"""


def _insert_before_rules(prompt: str, *sections: str) -> str:
    anchor = "## Rules"
    if anchor not in prompt:
        raise SystemExit("PTB prompt.txt has no '## Rules' heading; update build_prompts.py")
    block = "".join(sec if sec.endswith("\n\n") else sec.rstrip("\n") + "\n\n" for sec in sections)
    return prompt.replace(anchor, block + anchor, 1)


def ptb_fulltraj(ptb_prompt: str) -> str:
    """C1: the PTB prompt with raw priors and the pinned-base rule."""
    return _insert_before_rules(ptb_prompt, PRIOR_RUNS_SECTION, PINNED_BASE_SECTION)


def wm_prompt(ptb_prompt: str, *, fulltraj: bool) -> str:
    """C2/C3: the PTB prompt with the world-model section (and prior runs for C2)."""
    if fulltraj:
        return _insert_before_rules(
            ptb_prompt, PRIOR_RUNS_SECTION, PINNED_BASE_SECTION, WMA_SECTION
        )
    return _insert_before_rules(ptb_prompt, PINNED_BASE_SECTION, WMA_SECTION)


def smoke_prompt(prompt: str) -> str:
    """Add a short integration-smoke directive without changing production prompts."""
    return _insert_before_rules(prompt, SMOKE_SECTION)


def find_ptb_prompt(ptb: Path | None) -> Path:
    for cand in ([ptb] if ptb else []) + [ROOT / "third_party" / "PostTrainBench"]:
        f = Path(cand) / "src" / "eval" / "general" / "prompt.txt"
        if f.is_file():
            return f
    raise SystemExit("no PostTrainBench prompt.txt found; pass the checkout path or init the submodule")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptb", nargs="?", type=Path)
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="write only into the supplied private PTB checkout",
    )
    args = parser.parse_args()
    ptb = args.ptb.resolve() if args.ptb else None
    if args.no_review and ptb is None:
        parser.error("--no-review requires a PTB checkout")
    ptb_prompt = find_ptb_prompt(ptb).read_text()
    out_review = None if args.no_review else HERE / "prompts"
    if out_review:
        out_review.mkdir(exist_ok=True)
    wm = wm_prompt(ptb_prompt, fulltraj=False)
    wm_fulltraj = wm_prompt(ptb_prompt, fulltraj=True)
    files = {
        "prompt_fulltraj.txt": ptb_fulltraj(ptb_prompt),
        "prompt_wm.txt": wm,
        "prompt_wm_fulltraj.txt": wm_fulltraj,
        "prompt_wm_smoke.txt": smoke_prompt(wm),
        "prompt_wm_fulltraj_smoke.txt": smoke_prompt(wm_fulltraj),
    }
    for name, text in files.items():
        if out_review:
            (out_review / name).write_text(text)
        if ptb:
            (ptb / "src" / "eval" / "general" / name).write_text(text)
    (ROOT / "input" / "instruction.md").write_text(files["prompt_wm.txt"])
    destinations = []
    if out_review:
        destinations.append(str(out_review))
    if ptb:
        destinations.append(str(ptb / "src/eval/general"))
    print(
        f"wrote {', '.join(files)} to {' and '.join(destinations)}; "
        "input/instruction.md = prompt_wm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
