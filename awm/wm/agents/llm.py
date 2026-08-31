"""The autonomous arms: an LLM that reads the evidence itself and advises with citations.

Two registrations of one class:

* ``llm``  — sources: WMA memory (precedents) + the raw prior runs, if mounted.
* ``traj`` — sources: the raw prior runs only. This is study condition C2:
  the scientist has the raw files, and so does the world-model agent.

The agent is Claude Code in non-interactive mode (``claude -p``) with
read-only tools, pointed at the card, the grounding report, the proposed
contract, the observation bundle, and ``prior_runs_root``. It must answer
with one JSON object; every claim carries ``{path, locator}`` evidence, and
the runtime's grounding lint drops anything that does not resolve under the
session dir, the memory root, or the prior-runs root. If the call fails or
the answer does not parse, the arm degrades to its deterministic parent —
the scientist still gets the bundle, never silence.

The credential is whatever ``claude`` already has in the sandbox (the same
OAuth token the scientist runs on); no new key reaches the scientist. A
``fake`` backend returns canned JSON for tests.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .base import Advice, Brief, WorldModelAgent, observation_evidence, observation_summary
from .retrieval import RetrievalAgent

READ_ONLY_TOOLS = "Read,Grep,Glob,LS,Bash(ls:*),Bash(head:*),Bash(tail:*),Bash(wc:*),Bash(grep:*),Bash(rg:*),Bash(cat:*),Bash(find:*)"

BRIEF_SCHEMA = """{
  "summary": "<two lines: is the problem real, what similar runs did, what you expect>",
  "objections": [{"field": "<evaluation.* or setup.*>", "severity": "blocking|advisory", "fix": "<one line>"}],
  "prediction": {"metric": "<the card's metric>", "horizon": "final", "delta_mean": <float vs parent>, "delta_sd": <float>, "basis": "<n runs, which>"} | null,
  "evidence": [{"path": "<absolute path under an allowed root>", "locator": "<line range | item ids | run name>", "observation": "<one line>"}],
  "recommendation": "run|amend|withdraw"
}"""

OBS_SCHEMA = """{
  "kind": "notice|yield_request|decision",
  "summary": "<two lines: what this observation shows against parent, previous checkpoint, and similar runs>",
  "evidence": [{"path": "<absolute path under an allowed root>", "locator": "<...>", "observation": "<one line>"}],
  "prediction": {"metric": "<the selection metric>", "horizon": "final", "delta_mean": <float vs parent>, "delta_sd": <float>, "basis": "<...>"} | null,
  "request_evaluators": ["<names from the contract's on_request list, only if kind is yield_request>"],
  "recommendation": "continue|abort|select:<obs-id>|null"
}"""


class LLMAgent(RetrievalAgent):
    arm = "llm"
    sources: tuple[str, ...] = ("memory", "prior_runs")

    # ------------------------------------------------------------ hooks

    def on_proposal(self, card, grounding, memory, config) -> Brief:
        if "memory" in self.sources:
            brief = RetrievalAgent.on_proposal(self, card, grounding, memory, config)
        else:
            brief = WorldModelAgent.on_proposal(self, card, grounding, memory, config)
        roots = self._roots(config)
        if "prior_runs" in self.sources and not roots.get("prior_runs"):
            return _degrade(brief, f"prior_runs_root {config.get('prior_runs_root')!r} is not a directory", config)
        out = self._ask(config, self._brief_prompt(card, grounding, brief, roots, config), "brief")
        if not out:
            return _degrade(brief, self._last_error or "agent call returned nothing parsable", config)
        brief.produced_by = "llm"
        if isinstance(out.get("summary"), str) and out["summary"].strip():
            brief.summary = out["summary"].strip()
        brief.objections = [o for o in out.get("objections") or [] if _valid_objection(o)]
        brief.prediction = _valid_prediction(out.get("prediction"))
        brief.evidence = [e for e in out.get("evidence") or [] if _valid_evidence(e)]
        rec = out.get("recommendation")
        if rec in ("run", "amend", "withdraw"):
            brief.summary += f" Recommendation: {rec}."
        return brief

    def on_observation(self, observation, history, contract, card, memory, config) -> Advice:
        if "memory" in self.sources:
            advice = RetrievalAgent.on_observation(self, observation, history, contract, card, memory, config)
        else:
            advice = Advice(kind="notice", summary=observation_summary(observation, contract),
                            evidence=observation_evidence(observation))
        roots = self._roots(config)
        if "prior_runs" in self.sources and not roots.get("prior_runs"):
            return _degrade(advice, f"prior_runs_root {config.get('prior_runs_root')!r} is not a directory", config)
        out = self._ask(config, self._obs_prompt(card, contract, observation, history, advice, roots, config), "observation")
        if not out:
            return _degrade(advice, self._last_error or "agent call returned nothing parsable", config)
        advice.produced_by = "llm"
        kind = out.get("kind")
        if kind in ("notice", "yield_request", "decision"):
            advice.kind = kind
        if isinstance(out.get("summary"), str) and out["summary"].strip():
            advice.summary = out["summary"].strip()
        advice.evidence = [e for e in out.get("evidence") or [] if _valid_evidence(e)] + advice.evidence
        advice.prediction = _valid_prediction(out.get("prediction"))
        allowed = set(contract.get("on_request") or [])
        advice.request_evaluators = [n for n in out.get("request_evaluators") or [] if n in allowed]
        if advice.kind == "yield_request" and not advice.request_evaluators:
            advice.kind = "notice"
        rec = out.get("recommendation")
        if rec in ("continue", "abort") or (isinstance(rec, str) and rec.startswith("select:")):
            advice.recommendation = rec
        return advice

    def on_close(self, card, state, result, memory) -> None:
        if not result:
            return
        con = result.get("conclusion") or {}
        memory.note(card["card_id"], f"# {card['card_id']} ({self.arm})\n\n"
                                     f"claim: {card['hypothesis'].get('claim')}\n\n"
                                     f"verdict: {con.get('verdict')} · decision: {con.get('decision')}\n\n"
                                     f"{con.get('summary', '')}\n")

    # ------------------------------------------------------------ prompts

    def _roots(self, config) -> dict[str, str | None]:
        pr = config.get("prior_runs_root") if "prior_runs" in self.sources else None
        if pr and not Path(pr).is_dir():
            pr = None
        return {"session": config.get("_session_dir"), "memory": config.get("memory_root") if "memory" in self.sources else None,
                "prior_runs": pr}

    def _preamble(self, roots, config) -> str:
        allowed = ", ".join(f"`{v}`" for v in roots.values() if v)
        lines = [
            "You are the world-model agent beside an autonomous scientist that is post-training a base model.",
            "You never train anything and you never decide for the scientist; you ground, compare, and predict, citing files.",
            f"You may read only under these roots: {allowed}. Do not write anything. Do not run training or evaluation.",
        ]
        if roots.get("prior_runs"):
            lines.append(
                f"Prior runs: `{roots['prior_runs']}` holds previous attempts at this task by other agents. Start with "
                f"`{roots['prior_runs']}/INDEX.md` (base model, agent, official accuracy, path per run). Prefer runs on the "
                "same base model. In a run, `solve_parsed.txt` is the condensed trace, `task/` has the scripts it wrote and "
                "its own eval outputs (`eval_*.json`), `metrics.json` its official score.")
        if roots.get("memory"):
            lines.append(f"WMA memory: `{roots['memory']}/structured/*.jsonl` (cards, observations, interactions) and "
                         f"`{roots['memory']}/raw/`.")
        lines.append(
            "Every number and every claim in your answer must cite evidence as {path, locator, observation} under the "
            "allowed roots; uncited claims are discarded. Do not guess values. If the evidence is thin, say so and return "
            "prediction: null.")
        lines.append("Answer with ONE JSON object in a ```json fence as the last thing you write, and nothing after it.")
        return "\n".join(lines)

    def _brief_prompt(self, card, grounding, brief, roots, config) -> str:
        sd = Path(config["_session_dir"])
        cdir = sd / "wm" / "cards" / card["card_id"]
        if "memory" in self.sources:
            pre = ("- memory precedents (deterministic top-k retrieval, k=" + str(config.get("retrieval_k", 5)) + "):\n"
                   + ("\n".join(f"  - {p['card_id']} (similarity {p['similarity']}): best-vs-parent {p['delta_best_vs_parent']}, "
                                 f"{p.get('decision')}, {p.get('raw_dir')}" for p in brief.precedents) or "  - none"))
        else:
            pre = "- memory: not a source for this arm"
        return f"""{self._preamble(roots, config)}

## Task: the brief
The scientist proposed experiment card `{card['card_id']}`:
- card: `{cdir / 'card.yaml'}`
- grounding checks (mechanical, already run): `{cdir / 'grounding' / 'report.json'}` — {sum(g['passed'] for g in grounding)}/{len(grounding)} passed
- proposed evaluation contract: `{cdir / 'contract.proposed.yaml'}`
- base model: {card['setup'].get('base_model')} · method: {card['setup']['method'].get('family')} · data: {[d.get('source') for d in card['setup'].get('data', [])]}
- claim: {card['hypothesis'].get('claim')}
{pre}

Read the card. Then look for prior runs that tried something like it on this base model: what recipe, what their own
dev evals showed over time, what they scored. Judge (a) whether the stated problem matches the cited failures,
(b) whether the evaluation section can resolve the expected effect (n vs stderr, comparator under the same protocol,
a diagnostic for the named mechanism), and (c) what outcome to expect, with a spread, from the closest prior runs.

Return:
```json
{BRIEF_SCHEMA}
```"""

    def _obs_prompt(self, card, contract, observation, history, advice, roots, config) -> str:
        sd = Path(config["_session_dir"])
        cdir = sd / "wm" / "cards" / card["card_id"]
        hist = "\n".join(f"- {h['obs_id']} step {h['checkpoint'].get('step')} ({(h.get('fraction') or 0):.0%}): "
                         + "; ".join(f"{n}={m['value']:.4f}" + (f" ({m['delta_vs_parent']:+.4f} vs parent)" if m.get('delta_vs_parent') is not None else "")
                                     for n, m in h['evaluators'].items()) for h in history) or "- none yet"
        return f"""{self._preamble(roots, config)}

## Task: read an observation
Card `{card['card_id']}`: `{cdir / 'card.yaml'}`; frozen contract: `{cdir / 'contract.yaml'}`
(selection metric `{contract['selection']['evaluator']}`, on_request evaluators {contract.get('on_request') or []},
rules {[r['id'] for r in contract.get('rules', [])]}).
New observation `{observation['obs_id']}` at step {observation['checkpoint'].get('step')} ({(observation.get('fraction') or 0):.0%} of training):
`{cdir / 'observations' / observation['obs_id'] / 'observation.json'}`
Deterministic reading: {advice.summary}
Earlier observations:
{hist}
Parent values: `{cdir / 'parent'}/<evaluator>/normalized.json`.

Compare this curve with how the closest prior runs on this base model evolved (their own `eval_*.json` over time in
`task/`, and their final `metrics.json`). Decide whether the scientist should hear anything beyond the bundle:
- `notice` if nothing needs a reply;
- `yield_request` only if one of the on_request evaluators would resolve a real ambiguity now — name it;
- `decision` only if the evidence already says stop (predicted final not above parent + noise) or select — say which.

Return:
```json
{OBS_SCHEMA}
```"""

    # ------------------------------------------------------------ backends

    _last_error: str | None = None

    def _ask(self, config, prompt: str, kind: str) -> dict[str, Any] | None:
        self._last_error = None
        backend = config.get("wma_backend", "claude-cli")
        log_dir = Path(config["_session_dir"]) / "wm" / "agent-calls"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{int(time.time())}-{kind}"
        (log_dir / f"{stamp}.prompt.md").write_text(prompt)
        if backend == "fake":
            canned = (config.get("wma_fake") or {}).get(kind)
            (log_dir / f"{stamp}.response.json").write_text(json.dumps(canned, indent=2))
            if not isinstance(canned, dict):
                self._last_error = "fake backend: no canned answer"
            return canned if isinstance(canned, dict) else None
        if backend != "claude-cli":
            self._last_error = f"unknown wma_backend {backend!r}"
            return None
        text = self._claude_cli(config, prompt, log_dir / f"{stamp}.raw.json")
        out = parse_json_answer(text) if text else None
        if text and not out:
            self._last_error = "answer had no parsable JSON object"
        (log_dir / f"{stamp}.response.json").write_text(json.dumps(out, indent=2) if out else "null\n")
        return out

    def _claude_cli(self, config, prompt: str, raw_path: Path) -> str | None:
        argv = ["claude", "-p", prompt, "--output-format", "json", "--model", str(config.get("wma_model", "claude-opus-4-8")),
                "--max-turns", str(int(config.get("wma_max_turns", 40))), "--allowedTools", READ_ONLY_TOOLS]
        for root in (config.get("prior_runs_root"), config.get("memory_root")):
            if root and Path(str(root)).is_dir():
                argv += ["--add-dir", str(root)]
        env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
        try:
            proc = subprocess.run(argv, cwd=config["_session_dir"], env=env, capture_output=True, text=True,
                                  timeout=float(config.get("wma_call_timeout_s", 900)), check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            raw_path.write_text(json.dumps({"error": repr(exc)}))
            self._last_error = f"claude call failed: {type(exc).__name__}"
            return None
        raw_path.write_text(proc.stdout or proc.stderr or "")
        if proc.returncode != 0:
            self._last_error = f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout or '')[-200:].strip()}"
            return None
        try:
            payload = json.loads(proc.stdout)
            return payload.get("result") if isinstance(payload, dict) else proc.stdout
        except json.JSONDecodeError:
            return proc.stdout


class TrajAgent(LLMAgent):
    """C2: the autonomous agent over the raw prior runs only; memory is not consulted."""

    arm = "traj"
    sources = ("prior_runs",)


# ---------------------------------------------------------------- degradation

class AgentDegraded(RuntimeError):
    """An autonomous arm could not produce its answer and wma_strict forbids falling back."""


def _degrade(result, reason: str, config):
    """Mark a deterministic fallback as such; in strict mode refuse instead.

    The label of a study cell must equal what ran: a C2 cell whose agent
    silently answered as the null arm is not a C2 cell. So every fallback is
    stamped ``produced_by: deterministic`` + ``degraded: <reason>`` on the ping
    and in the ledger, and ``wma_strict: true`` (the harness default for the
    autonomous arms) turns a fallback into a hard error at the brief.
    """
    result.produced_by = "deterministic"
    result.degraded = reason
    result.summary = f"[agent degraded: {reason}] " + result.summary
    if config.get("wma_strict") and isinstance(result, Brief):
        raise AgentDegraded(reason)
    return result


# ---------------------------------------------------------------- parsing / validation

def parse_json_answer(text: str) -> dict[str, Any] | None:
    fences = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = fences[::-1] if fences else []
    if not candidates:
        start = text.rfind("{")
        while start >= 0:
            try:
                obj = json.loads(text[start:text.rfind("}") + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                start = text.rfind("{", 0, start)
        return None
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _valid_objection(o: Any) -> bool:
    return isinstance(o, dict) and isinstance(o.get("field"), str) and o.get("severity") in ("blocking", "advisory") \
        and isinstance(o.get("fix"), str)


def _valid_evidence(e: Any) -> bool:
    return isinstance(e, dict) and isinstance(e.get("path"), str) and e.get("path").startswith("/") \
        and isinstance(e.get("locator"), (str, int))


def _valid_prediction(p: Any) -> dict[str, Any] | None:
    if not isinstance(p, dict):
        return None
    try:
        mean, sd = float(p["delta_mean"]), float(p["delta_sd"])
    except (KeyError, TypeError, ValueError):
        return None
    if sd < 0:
        return None
    return {"metric": str(p.get("metric", "accuracy")), "horizon": str(p.get("horizon", "final")),
            "delta_mean": mean, "delta_sd": sd, "basis": str(p.get("basis", ""))}
