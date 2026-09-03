"""Write the study's prompt files into a PostTrainBench checkout.

Every prompt is PostTrainBench's own ``prompt.txt`` plus study sections
inserted before ``## Rules`` — nothing else changes, so a cell differs from the
corpus runs only by what was added:

    prompt_record.txt        PTB prompt + "Experiment log"                      (recorder: no priors, cards by command)
    prompt_noprior.txt       PTB prompt + nothing historical                    (C0: no prior runs, no registration)
    prompt_fulltraj.txt      PTB prompt + "Prior runs"                          (C1, retired: raw files, no WMA)
    prompt_wm.txt            PTB prompt + "Experiment log"                      (C3, retired)
    prompt_wm_fulltraj.txt   PTB prompt + "Prior runs" + "Experiment log"       (C2, retired)

The experiment-log section is ``input/wma_section.md`` (the source of truth). A
normal review build also writes ``prompt_record.txt`` to ``input/instruction.md``;
``--no-review`` leaves the source tree untouched.
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

SESSION_COMPLETION_SECTION = """## Session completion
A final assistant response ends the current Claude invocation. Background tasks and waiters do not themselves re-invoke you; the launcher may resume this conversation while budget remains. Do not finish, say you are waiting, or rely on a future notification while training, saving, or evaluation is still active. Keep polling with tool calls within their timeout until the work completes. Before claiming completion, verify that `final_model/config.json` and model weight files exist.

"""

RECORD_SMOKE_SECTION = """## One-hour recorder smoke protocol
This is a labelled integration smoke, not a production research cell. Exercise the complete registration path: implement one minimal experiment, fill sections 1-4 of its card and run `awm wm submit` before launching it, run one optimizer step from the assigned base model, evaluate, fill sections 5-6 and submit the same card again, leave a `final_model/`, and record what you shipped with `awm wm outcome`. The smoke tests mechanics, not model quality.

"""

SMOKE_SECTION = """## One-hour peer-session smoke protocol
This is a labelled integration smoke, not a production research cell. Exercise the complete two-session path: message the world-model agent with one concrete plan before training, run one optimizer step from the assigned base model, evaluate and leave a `final_model/`, then tell the world-model agent what you shipped. The smoke tests mechanics, not model quality.

"""


def _insert_before_rules(prompt: str, *sections: str) -> str:
    anchor = "## Rules"
    if anchor not in prompt:
        raise SystemExit("PTB prompt.txt has no '## Rules' heading; update build_prompts.py")
    block = "".join(sec if sec.endswith("\n\n") else sec.rstrip("\n") + "\n\n" for sec in sections)
    return prompt.replace(anchor, block + anchor, 1)


def ptb_noprior(ptb_prompt: str) -> str:
    """C0: the PTB prompt with no prior information; only the shared completion note."""
    return _insert_before_rules(ptb_prompt, SESSION_COMPLETION_SECTION)


def ptb_record(ptb_prompt: str) -> str:
    """Recorder: the PTB prompt plus the experiment-log section. No prior information."""
    return _insert_before_rules(ptb_prompt, WMA_SECTION, SESSION_COMPLETION_SECTION)


def ptb_fulltraj(ptb_prompt: str) -> str:
    """C1: the PTB prompt with raw priors."""
    return _insert_before_rules(
        ptb_prompt, PRIOR_RUNS_SECTION, SESSION_COMPLETION_SECTION
    )


def wm_prompt(ptb_prompt: str, *, fulltraj: bool) -> str:
    """C2/C3: the PTB prompt with the world-model section (and prior runs for C2)."""
    if fulltraj:
        return _insert_before_rules(
            ptb_prompt,
            PRIOR_RUNS_SECTION,
            WMA_SECTION,
            SESSION_COMPLETION_SECTION,
        )
    return _insert_before_rules(ptb_prompt, WMA_SECTION, SESSION_COMPLETION_SECTION)


def smoke_prompt(prompt: str, section: str = SMOKE_SECTION) -> str:
    """Add a short integration-smoke directive without changing production prompts."""
    return _insert_before_rules(prompt, section)


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
    record = ptb_record(ptb_prompt)
    wm = wm_prompt(ptb_prompt, fulltraj=False)
    wm_fulltraj = wm_prompt(ptb_prompt, fulltraj=True)
    files = {
        "prompt_record.txt": record,
        "prompt_record_smoke.txt": smoke_prompt(record, RECORD_SMOKE_SECTION),
        "prompt_noprior.txt": ptb_noprior(ptb_prompt),
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
    if out_review:
        (ROOT / "input" / "instruction.md").write_text(files["prompt_record.txt"])
    destinations = []
    if out_review:
        destinations.append(str(out_review))
    if ptb:
        destinations.append(str(ptb / "src/eval/general"))
    print(f"wrote {', '.join(files)} to {' and '.join(destinations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
