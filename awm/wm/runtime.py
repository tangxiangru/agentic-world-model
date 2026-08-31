"""The deterministic runtime: propose, freeze, checkpoint, worker, seal, finalize.

One ``Session`` per scientist directory. State per card lives in
``wm/cards/<card_id>/state.json``; every transition is appended to
``wm/events.jsonl``. The agent is consulted at two points only —
``on_proposal`` and ``on_observation`` — and everything it says passes
through the grounding lint before it becomes a ping.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .agents import make_agent
from .agents.base import Advice, Brief
from .evaluators import freeze_evaluators, run_evaluator, watch_transitions
from .memory import Memory
from .protocol import Ledger, Mailbox, render_ping
from .schema import (
    CONFIG_SCHEMA,
    HOOK_ABORT,
    HOOK_CONTINUE,
    HOOK_YIELD,
    OBSERVATION_SCHEMA,
    SEAL_SCHEMA,
    STATE_SCHEMA,
    WMError,
    dump_json,
    dump_yaml,
    inside,
    load_json,
    load_yaml,
    now,
    plan_hash,
    sha256_file,
    validate_card,
    validate_contract,
    validate_result,
)


_SENSITIVE_RESUME_ENV_NAME = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|AUTHORIZATION|CREDENTIAL)", re.IGNORECASE
)
_SENSITIVE_RESUME_ENV_PREFIXES = (
    "ANTHROPIC_",
    "APPTAINERENV_",
    "CLAUDE_CODE_",
    "GOOGLE_",
    "SINGULARITYENV_",
    "VERTEX_",
)
_SENSITIVE_RESUME_ENV_VALUE = re.compile(
    r"(?:\bhf_[A-Za-z0-9]{16,}\b|\bsk-[A-Za-z0-9_-]{16,}\b|"
    r"\bya29\.[A-Za-z0-9._~-]{16,}\b|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----)"
)


def _safe_resume_environment(environment: dict[str, str]) -> dict[str, str]:
    """Keep trainer state while excluding credentials inherited from Claude/Vertex."""
    return {
        name: value
        for name, value in environment.items()
        if not name.startswith(_SENSITIVE_RESUME_ENV_PREFIXES)
        and _SENSITIVE_RESUME_ENV_NAME.search(name) is None
        and _SENSITIVE_RESUME_ENV_VALUE.search(value) is None
    }

HOOK_SRC = Path(__file__).with_name("hook_example.py")


def default_config(session_dir: Path, arm: str = "null") -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "session_id": f"{session_dir.name}-{int(time.time())}",
        "arm": arm,
        "split_side": "train",
        "disclosure": "arm hidden from the scientist",
        "memory_root": os.environ.get("AWM_WM_MEMORY") or str(_data_root() / "wm-memory"),
        "memory_readonly": False,
        "submission": str(session_dir / "submission"),
        "submission_mode": "symlink",   # or "copy": adopt copies the sealed checkpoint into `submission`
        "memory_sides": ["train"],      # which split sides the agent may retrieve from
        "wma_provider": "vertex",
        "wma_model": os.environ.get("AWM_WMA_MODEL"),
        "wma_corpus_kind": "cards",
        "wma_corpus_root": os.environ.get("AWM_WMA_CORPUS_ROOT"),
        "wma_command": "claude",
        "wma_effort": "high",
        "wma_max_budget_usd": 1.0,
        "wma_timeout_s": 900,
        # Opt in above one only in a controlled harness. Every attempt must
        # still pass the same fail-closed tool/citation/provider validators.
        "wma_validation_attempts": 1,
        "official_argv": None,      # default: python evaluate.py --model-path {checkpoint} --limit {n} --json-output-file {out}/metrics.json
        "official_cwd": None,       # default: the session dir
        "custom_argv": None,        # default: python -m awm.wm.score_items ...
        "budgets_min": {"total_runtime": 150, "requested_subset": 30},
        "timeouts_s": {"brief": 1800, "yield_request": 600, "decision": 900},
        "amend_limit": 3,
        "standing_fractions": [0.25, 0.5, 0.75, 1.0],
        "regress_threshold": 0.03,
        "auto_relaunch": True,
        "spawn_worker": True,
        "trainer_exit_timeout_s": 900,
        "hash_weights": False,
        "poll_s": 2.0,
    }


def _data_root() -> Path:
    try:
        from awm import paths

        return paths.data_root()
    except Exception:  # noqa: BLE001
        return Path.cwd() / "data"


class Session:
    def __init__(self, session_dir: str | os.PathLike):
        self.dir = Path(session_dir).expanduser().resolve()
        self.wm = self.dir / "wm"
        self.config_path = self.wm / "config.yaml"
        if not self.config_path.is_file():
            raise WMError(f"{self.wm} is not initialised; run: awm wm --dir {self.dir} init")
        self.config = load_yaml(self.config_path)
        self.ledger = Ledger(self.wm / "events.jsonl")
        self.inbox = self.wm / "inbox.md"
        self.cards_dir = self.wm / "cards"
        self.memory = Memory(Path(self.config["memory_root"]), session=self.config["session_id"],
                             arm=self.config["arm"], split_side=self.config.get("split_side", "train"),
                             readonly=bool(self.config.get("memory_readonly")),
                             visible_sides=tuple(self.config.get("memory_sides") or ["train"]))
        self.agent = make_agent(self.config["arm"], session_dir=self.dir)

    # ------------------------------------------------------------ init

    @classmethod
    def init(cls, session_dir: str | os.PathLike, *, arm: str = "null", **overrides: Any) -> Session:
        d = Path(session_dir).expanduser().resolve()
        wm = d / "wm"
        wm.mkdir(parents=True, exist_ok=True)
        cfg_path = wm / "config.yaml"
        cfg = load_yaml(cfg_path) if cfg_path.is_file() else default_config(d, arm)
        cfg["arm"] = arm
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
        dump_yaml(cfg_path, cfg)
        inbox = wm / "inbox.md"
        if not inbox.exists():
            inbox.write_text("# WMA inbox\n\nOne line per ping, newest last. Lines marked REPLY NEEDED "
                             "are answered with `awm wm reply <card>/<ping> --choose <option>`.\n\n")
        shutil.copy2(HOOK_SRC, wm / "hook_example.py")
        (wm / "cards").mkdir(exist_ok=True)
        s = cls(d)
        if hasattr(s.agent, "validate"):
            s.agent.validate(s.memory, s.config)
        s.ledger.append("session_init", arm=arm, config=str(cfg_path))
        return s

    # ------------------------------------------------------------ card state

    def card_dir(self, card_id: str) -> Path:
        return self.cards_dir / card_id

    def state(self, card_id: str) -> dict[str, Any]:
        path = self.card_dir(card_id) / "state.json"
        if not path.is_file():
            raise WMError(f"no card {card_id} under {self.cards_dir}")
        return load_json(path)

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now()
        dump_json(self.card_dir(state["card_id"]) / "state.json", state)

    def mailbox(self, card_id: str) -> Mailbox:
        return Mailbox(self.card_dir(card_id), self.inbox, self.ledger)

    def card(self, card_id: str) -> dict[str, Any]:
        return load_yaml(self.card_dir(card_id) / "card.yaml")

    def contract(self, card_id: str) -> dict[str, Any]:
        return load_yaml(self.card_dir(card_id) / "contract.yaml")

    def cards(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.cards_dir.glob("exp-*/state.json")):
            out.append(load_json(p))
        return out

    def pending_replies(self) -> list[dict[str, Any]]:
        out = []
        for st in self.cards():
            if st["status"] == "closed":
                continue
            out.extend(self.mailbox(st["card_id"]).pending())
        return out

    # ------------------------------------------------------------ propose / brief

    def propose(self, card_path: str | os.PathLike) -> dict[str, Any]:
        card = load_yaml(Path(card_path))
        grounding = validate_card(card, self.dir)
        card_id = card["card_id"]
        lock_dir = self.wm / "locks"
        if lock_dir.is_symlink():
            raise WMError(f"proposal lock directory must not be a symlink: {lock_dir}")
        lock_dir.mkdir(exist_ok=True)
        lock_path = lock_dir / f"{card_id}.propose.lock"
        try:
            lock_fd = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            raise WMError(f"cannot open safe proposal lock {lock_path}: {exc}") from exc
        lock = os.fdopen(lock_fd, "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock.close()
            raise WMError(
                f"{card_id} already has a propose call in progress; wait for that command "
                "instead of starting another"
            ) from exc
        try:
            return self._propose_locked(card_path, card, grounding)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    def _propose_locked(
        self,
        card_path: str | os.PathLike,
        card: dict[str, Any],
        grounding: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Complete one proposal while its card-scoped inter-process lock is held."""
        card_id = card["card_id"]
        cdir = self.card_dir(card_id)
        if (cdir / "state.json").is_file():
            st = self.state(card_id)
            if st["status"] != "draft":
                raise WMError(f"{card_id} is {st['status']}; a material change is a new card")
            pending = self.mailbox(card_id).pending()
            if pending:
                raise WMError(f"{card_id} has an unanswered {pending[0]['kind']} ({pending[0]['ping_id']}); "
                              f"reply to it (amend) instead of re-proposing")
        else:
            cdir.mkdir(parents=True)
            st = {"schema_version": STATE_SCHEMA, "card_id": card_id, "status": "draft",
                  "created_at": now(), "brief_rounds": 0, "observations": [], "standing_taken": [],
                  "accepted_requests": [], "budget_used_min": {"total_runtime": 0.0, "requested_subset": 0.0},
                  "parent": {}, "trainer": None, "worker": None, "pending_yield": None, "seal": None,
                  "aborted": False, "final_seen": False, "override": None, "obs_counter": 0}
            self._save_state(st)
            self.ledger.append("card_proposed", card_id=card_id, source=str(card_path))
        dump_yaml(cdir / "card.yaml", card)
        self._materialize_grounding(card, grounding, cdir / "grounding")
        return self._brief(card, grounding, st)

    def _materialize_grounding(self, card: dict[str, Any], grounding: list[dict[str, Any]], gdir: Path) -> None:
        gdir.mkdir(parents=True, exist_ok=True)
        dump_json(gdir / "report.json", {"checks": grounding, "at": now()})
        cited = [e["path"] for e in card["problem"]["evidence"]] + [card["problem"]["watch_set"]["path"]]
        cited += [ex["source"] for ex in card["problem"]["failure_examples"]]
        for i, p in enumerate(dict.fromkeys(map(str, cited))):
            src = Path(p)
            if src.is_file() and src.stat().st_size <= 5_000_000:
                shutil.copy2(src, gdir / f"{i:02d}-{src.name}")

    def _brief(self, card: dict[str, Any], grounding: list[dict[str, Any]], st: dict[str, Any]) -> dict[str, Any]:
        card_id = card["card_id"]
        cdir = self.card_dir(card_id)
        try:
            brief: Brief = self.agent.on_proposal(card, grounding, self.memory, self.config)
        except Exception as exc:
            self.ledger.append(
                "agent_failed",
                card_id=card_id,
                where="brief",
                arm=self.config["arm"],
                corpus_kind=self.config.get("wma_corpus_kind"),
                model=self.config.get("wma_model"),
                produced_by="none",
                degraded=f"{type(exc).__name__}: {exc}"[:500],
            )
            raise WMError(
                "the autonomous world-model could not produce a validated brief; "
                "the cell failed closed and must not be labelled as an llm-arm result"
            ) from exc
        agent_meta = self._agent_meta(brief)
        if self.agent.arm == "llm" and (
            brief.produced_by != "llm" or brief.degraded is not None
        ):
            self.ledger.append("agent_degraded", card_id=card_id, where="brief", **agent_meta)
            raise WMError("the llm arm degraded; refusing a mislabeled study cell")
        validate_contract(brief.contract)
        dump_yaml(cdir / "contract.proposed.yaml", brief.contract)
        dump_json(cdir / "grounding" / f"brief-{st['brief_rounds'] + 1}.json", {
            "precedents": brief.precedents, "objections": brief.objections, "prediction": brief.prediction,
            "evidence": brief.evidence, "audit": brief.audit, "agent": agent_meta,
        })
        blocking = [o for o in brief.objections if o.get("severity") == "blocking"]
        evidence = [{"path": str(cdir / "grounding" / "report.json"), "locator": "checks",
                     "observation": f"{sum(g['passed'] for g in grounding)}/{len(grounding)} checks passed"}]
        for p in brief.precedents[:5]:
            evidence.append({"path": p.get("raw_dir"), "locator": p["card_id"],
                             "observation": f"similarity {p['similarity']}; best-vs-parent "
                                            f"{p['delta_best_vs_parent']}; {p.get('decision')}"})
        for o in brief.objections:
            evidence.append({"path": str(cdir / "card.yaml"), "locator": o.get("field"),
                             "observation": f"[{o.get('severity')}] {o.get('fix')}"})
        agent_evidence = _lint_evidence(
            brief.evidence,
            self.dir,
            Path(self.config["memory_root"]),
            extra_roots=_wma_evidence_roots(self.config),
        )
        if len(agent_evidence) != len(brief.evidence):
            self.ledger.append("lint", card_id=card_id,
                               dropped=len(brief.evidence) - len(agent_evidence), phase="brief")
            if self.agent.arm == "llm":
                raise WMError("llm brief contained evidence outside the session or visible memory")
        evidence.extend(agent_evidence)
        if brief.audit:
            self.ledger.append("wma_call", card_id=card_id, **brief.audit)
        options = [
            {"id": "accept", "label": "freeze the card and contract as proposed", "consequence": "parent is scored; you may launch"},
            {"id": "amend", "label": "submit a revised evaluation section (--amend FILE)", "consequence": f"new brief; {self.config['amend_limit'] - st['brief_rounds'] - 1} rounds left"},
            {"id": "override", "label": "freeze over objections (--why required)", "consequence": f"{len(blocking)} blocking objection(s) recorded as overridden"},
            {"id": "withdraw", "label": "close the card without running", "consequence": "execution not_run, decision abandon_line"},
        ]
        st["brief_rounds"] += 1
        self._save_state(st)
        ping = self.mailbox(card_id).send(
            "brief", brief.summary, evidence=evidence, prediction=brief.prediction, options=options,
            timeout_action={"action": "remain_draft", "after_s": self.config["timeouts_s"]["brief"]},
            raised_by="runtime", extra={"contract_path": str(cdir / "contract.proposed.yaml"),
                                        "blocking_objections": len(blocking), "arm_disclosed": False,
                                        "wma_audit": brief.audit, "agent": agent_meta})
        return ping

    def _agent_meta(self, result: Brief | Advice) -> dict[str, Any]:
        return {
            "arm": self.config["arm"],
            "corpus_kind": (
                self.config.get("wma_corpus_kind") if self.config["arm"] == "llm" else None
            ),
            "provider": self.config.get("wma_provider") if self.config["arm"] == "llm" else None,
            "model": self.config.get("wma_model") if self.config["arm"] == "llm" else None,
            "produced_by": result.produced_by,
            "degraded": result.degraded,
            "strict": self.config["arm"] == "llm",
        }

    # ------------------------------------------------------------ replies

    def reply(self, ping_ref: str, choice: str, *, why: str | None = None,
              amend: str | None = None) -> dict[str, Any]:
        card_id, ping_id = self._split_ping_ref(ping_ref)
        mb = self.mailbox(card_id)
        ping = mb.ping(ping_id)
        st = self.state(card_id)
        amend_path = str(Path(amend).resolve()) if amend else None
        mb.record_reply(ping_id, choice, why=why, amend=amend_path)
        self.agent.on_reply(ping, {"choice": choice, "why": why}, self.memory)
        out: dict[str, Any] = {"card_id": card_id, "ping_id": ping_id, "choice": choice}
        if ping["kind"] == "brief":
            out.update(self._after_brief_reply(st, ping, choice, why, amend_path))
        elif ping["kind"] == "yield_request":
            out.update(self._after_yield_reply(st, ping, choice))
        elif ping["kind"] == "decision":
            worker_alive = st.get("worker") and _alive(st["worker"].get("pid"))
            if worker_alive:
                out["note"] = "recorded; the worker holding this decision will apply it"
            else:
                out.update(self._apply_decision(st, ping, choice))
        return out

    def _split_ping_ref(self, ref: str) -> tuple[str, str]:
        if "/" in ref:
            card_id, ping_id = ref.split("/", 1)
            return card_id, ping_id
        matches = [st["card_id"] for st in self.cards() if (self.card_dir(st["card_id"]) / "pings" / f"{ref}.yaml").is_file()]
        if len(matches) != 1:
            raise WMError(f"ambiguous ping {ref!r}; use <card_id>/<ping_id>")
        return matches[0], ref

    def _after_brief_reply(self, st, ping, choice, why, amend_path) -> dict[str, Any]:
        card_id = st["card_id"]
        if choice == "withdraw":
            return self._withdraw(st, why)
        if choice == "amend":
            if st["brief_rounds"] >= int(self.config["amend_limit"]):
                raise WMError(f"amend limit ({self.config['amend_limit']}) reached; accept, override, or withdraw")
            amended = load_yaml(Path(amend_path))
            if amended.get("card_id") != card_id:
                raise WMError("the amended card must keep the same card_id")
            grounding = validate_card(amended, self.dir)
            dump_yaml(self.card_dir(card_id) / "card.yaml", amended)
            self._materialize_grounding(amended, grounding, self.card_dir(card_id) / "grounding")
            new = self._brief(amended, grounding, self.state(card_id))
            return {"next_ping": new["ping_id"], "printed": render_ping(new, self.card_dir(card_id) / "pings" / f"{new['ping_id']}.yaml")}
        if choice == "override":
            if ping.get("blocking_objections", 0) == 0:
                pass  # override without blocking objections is just an accept with a reason
            st["override"] = {"ping_id": ping["ping_id"], "why": why, "at": now()}
        return self._freeze(st)

    def _withdraw(self, st, why) -> dict[str, Any]:
        card_id = st["card_id"]
        result = self.card(card_id)
        result["result"] = {"execution": "not_run", "wall_h": 0.0, "output_checkpoint": None,
                            "measurements": [], "watch_set_result": None, "failure": None}
        result["conclusion"] = {"verdict": "inconclusive", "mechanism_verdict": "not_tested",
                                "decision": "abandon_line",
                                "summary": f"Withdrawn at the brief: {why or 'no reason given'}", "next_step": None}
        dump_yaml(self.card_dir(card_id) / "card.yaml", result)
        st["status"] = "closed"
        st["closed_at"] = now()
        self._save_state(st)
        self.ledger.append("card_closed", card_id=card_id, how="withdraw")
        self._record_to_memory(card_id, result)
        return {"status": "closed", "decision": "abandon_line"}

    # ------------------------------------------------------------ freeze

    def _freeze(self, st: dict[str, Any]) -> dict[str, Any]:
        card_id = st["card_id"]
        cdir = self.card_dir(card_id)
        card = self.card(card_id)
        contract = load_yaml(cdir / "contract.proposed.yaml")
        freeze_evaluators(contract, cdir / "evaluators")
        validate_contract(contract)
        dump_yaml(cdir / "contract.yaml", contract)
        parent = Path(card["setup"]["parent_checkpoint"]["path"])
        manifest = {
            "card_sha256": sha256_file(cdir / "card.yaml"),
            "plan_sha256": plan_hash(card),
            "contract_sha256": sha256_file(cdir / "contract.yaml"),
            "evaluators": {e["name"]: e["hash"] for e in contract["evaluators"]},
            "parent_manifest": _checkpoint_manifest(parent, hash_weights=False),
            "frozen_at": now(),
        }
        dump_json(cdir / "manifest.json", manifest)
        self.ledger.append("card_frozen", card_id=card_id, manifest=str(cdir / "manifest.json"))
        # score the parent through every evaluator: the comparators
        started = time.monotonic()
        for spec in contract["evaluators"]:
            metrics = run_evaluator(spec, parent, cdir / "parent" / spec["name"], self.config, self.dir)
            st["parent"][spec["name"]] = metrics
        st["budget_used_min"]["total_runtime"] += (time.monotonic() - started) / 60
        st["status"] = "frozen"
        st["frozen_at"] = now()
        self._save_state(st)
        self.ledger.append("parent_scored", card_id=card_id,
                           values={k: v["value"] for k, v in st["parent"].items()})
        summary = "; ".join(f"{k}={v['value']:.3f}" for k, v in st["parent"].items())
        self.mailbox(card_id).send(
            "notice", f"Frozen. Parent scored: {summary}. You may launch training; "
                      f"standing yields at {contract['standing_yields']['cadence']['values']}.",
            evidence=[{"path": str(cdir / "parent" / k / "normalized.json"), "locator": k,
                       "observation": f"{v['metric']}={v['value']:.4f}"} for k, v in st["parent"].items()],
            raised_by="runtime")
        return {"status": "frozen", "parent": {k: v["value"] for k, v in st["parent"].items()},
                "contract": str(cdir / "contract.yaml")}

    # ------------------------------------------------------------ checkpoint hook

    def checkpoint(self, card_id: str, path: str | os.PathLike, *, step: int, final: bool = False) -> int:
        st = self.state(card_id)
        ckpt = Path(path).resolve()
        if st.get("aborted") or st["status"] in ("awaiting_review", "closed"):
            # a straggling trainer after abort/select: tell it to stop, whatever state the card is in
            self.ledger.append("hook", card_id=card_id, step=step, code=HOOK_ABORT, why=f"card {st['status']}")
            return HOOK_ABORT
        if st["status"] not in ("frozen", "running"):
            raise WMError(f"{card_id} is {st['status']}; checkpoints are accepted only while frozen/running")
        if not inside(ckpt, self.dir):
            raise WMError(f"checkpoint {ckpt} is outside the session directory")
        if st.get("pending_yield"):
            # a previous yield has not been processed; do not stack another
            self.ledger.append("hook", card_id=card_id, step=step, code=HOOK_CONTINUE, why="yield already pending")
            return HOOK_CONTINUE
        if st["status"] == "frozen":
            st["status"] = "running"
            st["started_at"] = now()
            self.ledger.append("training_started", card_id=card_id)
        # remember how to relaunch: the hook's parent is the trainer
        trainer = {"pid": os.getppid(), "cwd": os.getcwd(), "seen_at": now()}
        env_path = self.card_dir(card_id) / "launch.env.json"
        if not env_path.exists():
            dump_json(env_path, _safe_resume_environment(dict(os.environ)))
            os.chmod(env_path, 0o600)
        st["trainer"] = trainer
        contract = self.contract(card_id)
        total = int(contract["standing_yields"]["progress"]["total"])
        fraction = min(step / total, 1.0) if total else None
        due = [f for f in contract["standing_yields"]["cadence"]["values"]
               if fraction is not None and fraction + 1e-9 >= float(f) and f not in st["standing_taken"]]
        requested = list(st.get("accepted_requests") or [])
        if final:
            st["final_seen"] = True
        if not due and not requested and not final:
            st["last_hook"] = {"step": step, "code": HOOK_CONTINUE, "at": now()}
            self._save_state(st)
            self.ledger.append("hook", card_id=card_id, step=step, code=HOOK_CONTINUE)
            return HOOK_CONTINUE
        st["standing_taken"] = sorted(set(st["standing_taken"]) | set(due))
        st["accepted_requests"] = []
        st["pending_yield"] = {"checkpoint": str(ckpt), "step": step, "fraction": fraction, "final": final,
                               "standing": due, "requested": requested, "trainer_pid": trainer["pid"], "at": now()}
        st["last_hook"] = {"step": step, "code": HOOK_YIELD, "at": now()}
        self._save_state(st)
        self.ledger.append("hook", card_id=card_id, step=step, code=HOOK_YIELD, standing=due,
                           requested=[r["ping_id"] for r in requested], final=final)
        if self.config.get("spawn_worker", True):
            self._spawn_worker(card_id)
        return HOOK_YIELD

    def _spawn_worker(self, card_id: str) -> int:
        log = self.card_dir(card_id) / "logs"
        log.mkdir(exist_ok=True)
        out = (log / f"worker-{int(time.time())}.log").open("w")
        proc = subprocess.Popen([sys.executable, "-m", "awm.cli", "wm", "--dir", str(self.dir), "worker", card_id],
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(self.dir))
        st = self.state(card_id)
        st["worker"] = {"pid": proc.pid, "started_at": now()}
        self._save_state(st)
        self.ledger.append("worker_spawned", card_id=card_id, pid=proc.pid)
        return proc.pid

    # ------------------------------------------------------------ worker

    def run_worker(self, card_id: str) -> dict[str, Any]:
        """Process the pending yield for a card: wait for the trainer, evaluate, ping, act."""
        cdir = self.card_dir(card_id)
        lock = (cdir / "worker.lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WMError(f"another worker holds {card_id}") from exc
        try:
            st = self.state(card_id)
            st["worker"] = {"pid": os.getpid(), "started_at": now()}
            self._save_state(st)
            py = st.get("pending_yield")
            if not py:
                return {"card_id": card_id, "note": "nothing pending"}
            self._wait_for_trainer(py.get("trainer_pid"))
            contract = self.contract(card_id)
            card = self.card(card_id)
            names = list(contract["standing_yields"]["evaluators"]) if (py["standing"] or py["final"]) else []
            for r in py["requested"]:
                for n in r["evaluators"]:
                    if n not in names:
                        names.append(n)
            cause = {"standing": py["standing"], "requested": [r["ping_id"] for r in py["requested"]],
                     "final": py["final"]}
            obs = self._observe(st, card, contract, Path(py["checkpoint"]), py["step"], py["fraction"], names, cause,
                                requested_names=[n for r in py["requested"] for n in r["evaluators"]])
            outcome = self._advise_and_act(st, card, contract, obs)
            st = self.state(card_id)
            st["pending_yield"] = None
            st["worker"] = None
            self._save_state(st)
            return {"card_id": card_id, "obs_id": obs["obs_id"], **outcome}
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    def _wait_for_trainer(self, pid: int | None) -> None:
        if not pid or pid == os.getpid() or pid == os.getppid():
            return
        deadline = time.monotonic() + float(self.config.get("trainer_exit_timeout_s", 900))
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(float(self.config.get("poll_s", 2.0)))
        if _alive(pid):
            self.ledger.append("warning", what="trainer still alive after yield timeout", pid=pid)

    def _observe(self, st, card, contract, ckpt: Path, step: int, fraction, names: list[str],
                 cause: dict[str, Any], *, requested_names: list[str]) -> dict[str, Any]:
        card_id = st["card_id"]
        st["obs_counter"] = int(st.get("obs_counter", 0)) + 1
        obs_id = f"obs-{st['obs_counter']}"
        odir = self.card_dir(card_id) / "observations" / obs_id
        odir.mkdir(parents=True, exist_ok=True)
        prev = st["observations"][-1] if st["observations"] else None
        started = time.monotonic()
        evaluators: dict[str, Any] = {}
        requested_min = 0.0
        for name in names:
            spec = next(e for e in contract["evaluators"] if e["name"] == name)
            t0 = time.monotonic()
            m = run_evaluator(spec, ckpt, odir / name, self.config, self.dir)
            m["delta_vs_parent"] = _delta(m, st["parent"].get(name))
            m["delta_vs_prev"] = _delta(m, (prev or {}).get("evaluators", {}).get(name))
            evaluators[name] = m
            if name in requested_names:
                requested_min += (time.monotonic() - t0) / 60
        wall = time.monotonic() - started
        watch = None
        if "watch" in evaluators:
            watch = watch_transitions(st["parent"].get("watch", {}).get("items"), evaluators["watch"].get("items"))
        obs = {
            "schema_version": OBSERVATION_SCHEMA, "obs_id": obs_id, "card_id": card_id,
            "checkpoint": {"path": str(ckpt), "step": step, "manifest": _checkpoint_manifest(ckpt, hash_weights=False)},
            "fraction": fraction, "cause": cause, "evaluators": evaluators, "watch": watch,
            "wall_s": round(wall, 1), "at": now(),
            "input_hashes": {"contract": sha256_file(self.card_dir(card_id) / "contract.yaml"),
                             "evaluators": {n: evaluators[n]["evaluator_hash"] for n in evaluators}},
        }
        history = [self._load_obs(card_id, o["obs_id"]) for o in st["observations"]]
        obs["rules_fired"] = _fire_rules(contract, history + [obs])
        dump_json(odir / "observation.json", obs)
        st["observations"].append({"obs_id": obs_id, "step": step, "fraction": fraction,
                                   "evaluators": {n: {k: m.get(k) for k in ("value", "n", "stderr", "delta_vs_parent", "delta_vs_prev")}
                                                  for n, m in evaluators.items()},
                                   "watch": watch, "path": str(odir / "observation.json")})
        st["budget_used_min"]["total_runtime"] += wall / 60
        st["budget_used_min"]["requested_subset"] += requested_min
        self._save_state(st)
        self.ledger.append("observation", card_id=card_id, obs_id=obs_id, step=step,
                           values={n: m["value"] for n, m in evaluators.items()}, rules=obs["rules_fired"],
                           wall_s=obs["wall_s"])
        self.memory.record_observation(card, obs)
        return obs

    def _load_obs(self, card_id: str, obs_id: str) -> dict[str, Any]:
        return load_json(self.card_dir(card_id) / "observations" / obs_id / "observation.json")

    def _advise_and_act(self, st, card, contract, obs) -> dict[str, Any]:
        card_id = st["card_id"]
        history = [self._load_obs(card_id, o["obs_id"]) for o in st["observations"][:-1]]
        try:
            advice: Advice = self.agent.on_observation(
                obs, history, contract, card, self.memory, self.config
            )
        except Exception as exc:
            self.ledger.append(
                "agent_failed",
                card_id=card_id,
                where=obs["obs_id"],
                arm=self.config["arm"],
                corpus_kind=self.config.get("wma_corpus_kind"),
                model=self.config.get("wma_model"),
                produced_by="none",
                degraded=f"{type(exc).__name__}: {exc}"[:500],
            )
            raise WMError(
                "the autonomous world-model could not produce validated observation advice; "
                "the cell failed closed and must not be labelled as an llm-arm result"
            ) from exc
        agent_meta = self._agent_meta(advice)
        if self.agent.arm == "llm" and (
            advice.produced_by != "llm" or advice.degraded is not None
        ):
            self.ledger.append(
                "agent_degraded", card_id=card_id, where=obs["obs_id"], **agent_meta
            )
            raise WMError("the llm arm degraded; refusing a mislabeled study cell")
        if advice.audit:
            self.ledger.append("wma_call", card_id=card_id, **advice.audit)
        evidence = _lint_evidence(
            advice.evidence,
            self.dir,
            Path(self.config["memory_root"]),
            extra_roots=_wma_evidence_roots(self.config),
        )
        if len(evidence) < len(advice.evidence):
            self.ledger.append("lint", card_id=card_id, dropped=len(advice.evidence) - len(evidence))
            if self.agent.arm == "llm":
                raise WMError("llm advice contained evidence outside the session or allowed corpus")
        evidence = [{"path": str(self.card_dir(card_id) / "observations" / obs["obs_id"] / "observation.json"),
                     "locator": "evaluators", "observation": _obs_line(obs)}] + evidence
        for p in advice.precedents[:3]:
            evidence.append({"path": p.get("raw_dir"), "locator": p["card_id"],
                             "observation": f"precedent: best-vs-parent {p['delta_best_vs_parent']}; {p.get('decision')}"})
        final = bool(obs["cause"].get("final"))
        fired = obs.get("rules_fired") or []
        best = self._best_obs(st, contract)
        mb = self.mailbox(card_id)

        if final or fired or advice.kind == "decision":
            options = []
            if not final:
                options.append({"id": "continue", "label": "resume training", "consequence": "relaunch from this checkpoint"})
            if contract.get("on_request"):
                options.append({"id": "more_eval", "label": f"run {', '.join(contract['on_request'])} on this checkpoint",
                                "consequence": "same checkpoint, decision re-issued", "cost_min": self._est_cost(st, contract["on_request"])})
            options.append({"id": f"select:{best}", "label": f"select {best} (best on {contract['selection']['evaluator']})",
                            "consequence": "runtime seals it; card awaits your result"})
            if best != obs["obs_id"]:
                options.append({"id": f"select:{obs['obs_id']}", "label": f"select {obs['obs_id']} (this checkpoint)",
                                "consequence": "runtime seals it"})
            options.append({"id": "abort", "label": "stop; no relaunch", "consequence": "card awaits your result"})
            if fired:
                rule = next(r for r in contract["rules"] if r["id"] == fired[0])
                default = {"abort": "abort", "select_best": f"select:{best}", "continue": "continue"}[rule["action"]]
                if final and default == "continue":
                    default = f"select:{best}"
                raised = f"rule:{fired[0]}"
                head = f"Rule {fired[0]} fired ({rule['evaluator']}.{rule['field']} {rule['comparator']} {rule['threshold']} × {rule['count']})."
            elif final:
                default, raised = f"select:{best}", "completion"
                head = "Training complete."
            else:
                default, raised = "continue", "agent"
                head = f"Agent recommends {advice.recommendation or 'a decision'}."
            ping = mb.send("decision", f"{head} {advice.summary}", evidence=evidence, prediction=advice.prediction,
                           options=options, timeout_action={"action": default, "after_s": self.config["timeouts_s"]["decision"]},
                           raised_by=raised, observation=obs["obs_id"],
                           extra={"recommendation": advice.recommendation, "best_observation": best,
                                  "agent": agent_meta})
            reply = self._await_reply(mb, ping)
            self.memory.record_interaction(card_id, ping, reply)
            return self._apply_decision(self.state(card_id), ping, reply["choice"])

        if advice.kind == "yield_request" and advice.request_evaluators:
            allowed = [n for n in advice.request_evaluators if n in contract.get("on_request", [])]
            cost = self._est_cost(st, allowed)
            remaining = float(self.config["budgets_min"]["requested_subset"]) - st["budget_used_min"]["requested_subset"]
            if allowed and cost <= remaining:
                ping = mb.send("yield_request", advice.summary, evidence=evidence, prediction=advice.prediction,
                               options=[{"id": "accept", "label": f"run {', '.join(allowed)} at the next save", "cost_min": round(cost, 1),
                                         "consequence": "one extra yield, charged to the WMA budget"},
                                        {"id": "reject", "label": "keep training", "consequence": "nothing happens"}],
                               timeout_action={"action": "reject", "after_s": self.config["timeouts_s"]["yield_request"]},
                               raised_by="agent", observation=obs["obs_id"],
                               extra={"evaluators": allowed, "agent": agent_meta})
            else:
                self.ledger.append("request_dropped", card_id=card_id, wanted=advice.request_evaluators,
                                   allowed=allowed, cost_min=cost, remaining_min=remaining)
                mb.send("notice", advice.summary, evidence=evidence, prediction=advice.prediction,
                        raised_by=self.agent.arm, observation=obs["obs_id"],
                        extra={"agent": agent_meta})
        else:
            mb.send("notice", advice.summary, evidence=evidence, prediction=advice.prediction,
                    raised_by=self.agent.arm, observation=obs["obs_id"],
                    extra={"agent": agent_meta})
        return self._resume(self.state(card_id), obs)

    def _est_cost(self, st, names: list[str]) -> float:
        total = 0.0
        for n in names:
            m = st["parent"].get(n)
            total += (m or {}).get("wall_s", 300) / 60
        return round(total + 2.0, 1)  # + reload allowance

    def _await_reply(self, mb: Mailbox, ping: dict[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + float(ping["timeout_action"]["after_s"])
        while time.monotonic() < deadline:
            r = mb.reply(ping["ping_id"])
            if r:
                return r
            time.sleep(float(self.config.get("poll_s", 2.0)))
        return mb.record_timeout(ping["ping_id"])

    def _best_obs(self, st, contract) -> str:
        sel = contract["selection"]["evaluator"]
        higher = contract["selection"].get("direction", "higher") == "higher"
        best, best_v = None, None
        for o in st["observations"]:
            v = (o["evaluators"].get(sel) or {}).get("value")
            if v is None:
                continue
            if best_v is None or (v > best_v if higher else v < best_v):
                best, best_v = o["obs_id"], v
        return best or (st["observations"][-1]["obs_id"] if st["observations"] else "none")

    # ------------------------------------------------------------ decisions

    def _after_yield_reply(self, st, ping, choice) -> dict[str, Any]:
        if choice != "accept":
            return {"note": "declined; nothing scheduled"}
        st["accepted_requests"].append({"ping_id": ping["ping_id"], "evaluators": ping.get("evaluators", [])})
        self._save_state(st)
        self.ledger.append("request_accepted", card_id=st["card_id"], ping_id=ping["ping_id"], evaluators=ping.get("evaluators"))
        return {"note": f"{', '.join(ping.get('evaluators', []))} will run at the next save"}

    def _apply_decision(self, st, ping, choice: str) -> dict[str, Any]:
        card_id = st["card_id"]
        obs_id = ping.get("observation")
        self.ledger.append("decision_applied", card_id=card_id, ping_id=ping["ping_id"], choice=choice)
        if choice == "continue":
            obs = self._load_obs(card_id, obs_id) if obs_id else None
            return self._resume(st, obs)
        if choice == "more_eval":
            contract = self.contract(card_id)
            obs = self._load_obs(card_id, obs_id)
            card = self.card(card_id)
            new = self._observe(st, card, contract, Path(obs["checkpoint"]["path"]), obs["checkpoint"]["step"],
                                obs["fraction"], list(contract.get("on_request", [])),
                                {"standing": [], "requested": [ping["ping_id"]], "final": obs["cause"].get("final", False),
                                 "more_eval_of": obs_id},
                                requested_names=list(contract.get("on_request", [])))
            return self._advise_and_act(self.state(card_id), card, contract, new)
        if choice.startswith("select:"):
            target = choice.split(":", 1)[1]
            seal = self._seal(st, target, decision_ping=ping["ping_id"])
            st = self.state(card_id)
            st["status"] = "awaiting_review"
            self._save_state(st)
            self.ledger.append("awaiting_review", card_id=card_id, via="select", obs_id=target)
            self.mailbox(card_id).send("notice", f"Sealed {target} → {seal['checkpoint']['path']}. Fill sections 5-6 of the "
                                                 f"card and run `awm wm finalize {card_id} <card.yaml>`.",
                                       evidence=[{"path": str(self.card_dir(card_id) / "seal.json"), "locator": "sha256",
                                                  "observation": "seal written"}], raised_by="runtime")
            return {"status": "awaiting_review", "sealed": target}
        if choice == "abort":
            st["aborted"] = True
            st["status"] = "awaiting_review"
            self._save_state(st)
            self.ledger.append("awaiting_review", card_id=card_id, via="abort")
            self.mailbox(card_id).send("notice", f"Aborted at {obs_id}. No relaunch. Fill sections 5-6 of the card "
                                                 f"(execution: killed) and run `awm wm finalize {card_id} <card.yaml>`.",
                                       raised_by="runtime")
            return {"status": "awaiting_review", "aborted": True}
        raise WMError(f"unknown decision {choice!r}")

    def _resume(self, st, obs) -> dict[str, Any]:
        card_id = st["card_id"]
        if obs and obs["cause"].get("final"):
            return {"status": st["status"], "note": "final checkpoint; nothing to resume"}
        if not self.config.get("auto_relaunch", True):
            self.ledger.append("manual_resume", card_id=card_id, checkpoint=obs["checkpoint"]["path"] if obs else None)
            return {"status": "running", "resume": "manual"}
        card = self.card(card_id)
        resume_argv = card["setup"].get("resume_argv")
        if not resume_argv:
            self.mailbox(card_id).send("notice", "No resume_argv in the card; relaunch training yourself from "
                                                 f"{obs['checkpoint']['path'] if obs else 'the last checkpoint'}.", raised_by="runtime")
            return {"status": "running", "resume": "manual (no resume_argv)"}
        argv = [a.replace("{checkpoint}", obs["checkpoint"]["path"]) for a in resume_argv]
        env_path = self.card_dir(card_id) / "launch.env.json"
        env = load_json(env_path) if env_path.is_file() else dict(os.environ)
        env["AWM_SESSION_DIR"] = str(self.dir)
        cwd = (st.get("trainer") or {}).get("cwd") or card["setup"]["command"]["cwd"]
        logs = self.card_dir(card_id) / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / f"relaunch-{obs['obs_id']}.log"
        with log.open("w") as fh:
            fh.write("$ " + " ".join(shlex.quote(a) for a in argv) + "\n")
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log.open("a"), stderr=subprocess.STDOUT,
                                start_new_session=True)
        st = self.state(card_id)
        st["trainer"] = {"pid": proc.pid, "cwd": cwd, "seen_at": now(), "relaunched_from": obs["checkpoint"]["path"]}
        self._save_state(st)
        self.ledger.append("relaunched", card_id=card_id, pid=proc.pid, checkpoint=obs["checkpoint"]["path"], log=str(log))
        return {"status": "running", "resume": "relaunched", "pid": proc.pid}

    # ------------------------------------------------------------ seal / finalize

    def _seal(self, st, obs_id: str, *, decision_ping: str | None = None) -> dict[str, Any]:
        card_id = st["card_id"]
        obs = self._load_obs(card_id, obs_id)
        ckpt = Path(obs["checkpoint"]["path"])
        if not ckpt.is_dir():
            raise WMError(f"checkpoint {ckpt} no longer exists")
        cfg = ckpt / "config.json"
        if not cfg.is_file():
            raise WMError(f"{ckpt} has no config.json; not loadable")
        parsed = json.loads(cfg.read_text())
        if "model_type" not in parsed and "architectures" not in parsed:
            raise WMError(f"{cfg} lacks model_type/architectures")
        contract = self.contract(card_id)
        seal = {
            "schema_version": SEAL_SCHEMA, "card_id": card_id, "obs_id": obs_id,
            "checkpoint": {"path": str(ckpt), "manifest": _checkpoint_manifest(ckpt, hash_weights=bool(self.config.get("hash_weights")))},
            "evaluator_hashes": {e["name"]: e["hash"] for e in contract["evaluators"]},
            "metrics": {n: {k: m.get(k) for k in ("value", "n", "stderr", "delta_vs_parent")} for n, m in obs["evaluators"].items()},
            "decision_ping": decision_ping, "sealed_at": now(),
        }
        dump_json(self.card_dir(card_id) / "seal.json", seal)
        st["seal"] = {"obs_id": obs_id, "checkpoint": str(ckpt), "at": seal["sealed_at"]}
        self._save_state(st)
        self.ledger.append("sealed", card_id=card_id, obs_id=obs_id, checkpoint=str(ckpt))
        return seal

    def finalize(self, card_id: str, result_path: str | os.PathLike) -> dict[str, Any]:
        st = self.state(card_id)
        if st["status"] not in ("awaiting_review", "running", "frozen"):
            raise WMError(f"{card_id} is {st['status']}")
        if st["status"] == "running" and st.get("trainer") and _alive(st["trainer"].get("pid")):
            raise WMError("training is still running; abort or let it finish first")
        result = load_yaml(Path(result_path))
        validate_result(result, card_id)
        manifest_path = self.card_dir(card_id) / "manifest.json"
        if manifest_path.is_file():
            frozen = load_json(manifest_path).get("plan_sha256")
            if frozen and plan_hash(result) != frozen:
                raise WMError("sections 1-4 differ from the frozen card; a material change is a new card")
        decision = result["conclusion"]["decision"]
        if decision == "adopt":
            if not st.get("seal"):
                raise WMError("adopt requires a sealed checkpoint (select one via a decision ping)")
            self._adopt(st)
        dump_yaml(self.card_dir(card_id) / "card.yaml", result)
        st = self.state(card_id)
        st["status"] = "closed"
        st["closed_at"] = now()
        self._save_state(st)
        self.ledger.append("card_closed", card_id=card_id, how="finalize", decision=decision,
                           verdict=result["conclusion"]["verdict"])
        self._record_to_memory(card_id, result)
        self.mailbox(card_id).send("notice", f"Closed: {result['conclusion']['verdict']}, {decision}. Recorded to memory.",
                                   raised_by="runtime")
        return {"status": "closed", "decision": decision, "submission": self.config["submission"] if decision == "adopt" else None}

    def _adopt(self, st) -> None:
        target = Path(st["seal"]["checkpoint"])
        sub = Path(self.config["submission"])
        if not inside(sub, self.dir):
            raise WMError(f"submission {sub} is outside the session directory")
        mode = self.config.get("submission_mode", "symlink")
        if mode == "copy":
            # the benchmark runner collects a real directory (final_model/), not a link
            tmp = sub.with_name(sub.name + ".tmp")
            if tmp.is_symlink():
                tmp.unlink()
            elif tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(target, tmp, symlinks=False)
            if sub.is_symlink():
                sub.unlink()
            elif sub.exists():
                shutil.rmtree(sub)
            os.replace(tmp, sub)
        else:
            tmp = sub.with_name(sub.name + ".tmp")
            if tmp.is_symlink() or tmp.exists():
                tmp.unlink()
            tmp.symlink_to(target, target_is_directory=True)
            os.replace(tmp, sub)
        dump_json(self.wm / "incumbent.json", {"card_id": st["card_id"], "checkpoint": str(target),
                                               "obs_id": st["seal"]["obs_id"], "at": now()})
        self.ledger.append("adopted", card_id=st["card_id"], checkpoint=str(target), submission=str(sub), mode=mode)

    def _record_to_memory(self, card_id: str, result: dict[str, Any] | None) -> None:
        st = self.state(card_id)
        card = self.card(card_id)
        cdir = self.card_dir(card_id)
        contract = load_yaml(cdir / "contract.yaml") if (cdir / "contract.yaml").is_file() else None
        mb = self.mailbox(card_id)
        for p in mb.pings():
            if p["reply_required"]:
                self.memory.record_interaction(card_id, p, mb.reply(p["ping_id"]))
        self.memory.record_card(cdir, card, contract, result, st)
        self.agent.on_close(card, st, result, self.memory)

    # ------------------------------------------------------------ status

    def status(self, card_id: str | None = None) -> dict[str, Any]:
        if card_id:
            st = self.state(card_id)
            return {**st, "pending_replies": [p["ping_id"] for p in self.mailbox(card_id).pending()]}
        return {"session": self.config["session_id"], "arm": self.config["arm"],
                "cards": [{"card_id": s["card_id"], "status": s["status"], "observations": len(s["observations"]),
                           "pending_replies": [p["ping_id"] for p in self.mailbox(s["card_id"]).pending()]}
                          for s in self.cards()],
                "agent": {
                    "arm": self.config["arm"],
                    "corpus_kind": self.config.get("wma_corpus_kind"),
                    "provider": self.config.get("wma_provider"),
                    "model": self.config.get("wma_model"),
                    "strict": self.config["arm"] == "llm",
                },
                "memory": self.memory.stats()}


# ---------------------------------------------------------------- helpers

def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _delta(a, b) -> float | None:
    if not a or not b or a.get("value") is None or b.get("value") is None:
        return None
    return round(float(a["value"]) - float(b["value"]), 6)


def _checkpoint_manifest(ckpt: Path, *, hash_weights: bool) -> dict[str, Any]:
    files = {}
    if not ckpt.is_dir():
        return {"missing": True}
    for p in sorted(ckpt.iterdir()):
        if not p.is_file():
            continue
        critical = p.suffix == ".json" or p.name.startswith("tokenizer") or p.name.endswith(".model")
        entry: dict[str, Any] = {"size": p.stat().st_size}
        if critical or (hash_weights and p.suffix == ".safetensors"):
            entry["sha256"] = sha256_file(p)
        files[p.name] = entry
    return {"files": files, "n_files": len(files)}


def _fire_rules(contract: dict[str, Any], observations: list[dict[str, Any]]) -> list[str]:
    fired = []
    for rule in contract.get("rules", []):
        rows = [o for o in observations if rule["evaluator"] in o.get("evaluators", {})]
        tail = rows[-int(rule["count"]):]
        if len(tail) < int(rule["count"]):
            continue
        ok = True
        for o in tail:
            v = o["evaluators"][rule["evaluator"]].get(rule["field"])
            if v is None:
                ok = False
                break
            t = float(rule["threshold"])
            ok = ok and {"lt": v < t, "gt": v > t, "le": v <= t, "ge": v >= t}[rule["comparator"]]
        if ok:
            fired.append(rule["id"])
    return fired


def _lint_evidence(
    evidence: list[dict[str, Any]],
    session_dir: Path,
    memory_root: Path,
    *,
    extra_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    """Grounding lint: keep only evidence whose path exists under the session or memory."""
    kept = []
    roots = (session_dir, memory_root, *extra_roots)
    for e in evidence:
        p = e.get("path")
        if not p:
            continue
        path = Path(str(p))
        if path.exists() and any(inside(path, root) for root in roots):
            kept.append(e)
    return kept


def _wma_evidence_roots(config: dict[str, Any]) -> tuple[Path, ...]:
    if config.get("arm") != "llm" or config.get("wma_corpus_kind", "cards") != "raw":
        return ()
    root = config.get("wma_corpus_root") or os.environ.get("AWM_WMA_CORPUS_ROOT")
    return (Path(str(root)).expanduser().resolve(),) if root else ()


def _obs_line(obs: dict[str, Any]) -> str:
    return "; ".join(f"{n}={m['value']:.4f}" + (f" ({m['delta_vs_parent']:+.4f} vs parent)" if m.get("delta_vs_parent") is not None else "")
                     for n, m in obs["evaluators"].items())


def print_ping(ping: dict[str, Any], session: Session) -> None:
    path = session.card_dir(ping["card_id"]) / "pings" / f"{ping['ping_id']}.yaml"
    print(render_ping(ping, path))
