"""The evidence store: every tool run, indexed, and the arbiter of every claim.

The store is an append-only ledger of :class:`~onset_agent.contract.ToolRun`
records. It answers three questions, and the whole verification story rests on
them:

* **What has been done so far?** -- :meth:`EvidenceStore.digest`, the compact
  state the planner re-reads before choosing its next action. It must stay
  small: a 4B model given a 30 kB context of raw JSON stops planning and
  starts pattern-matching.
* **Where did this number come from?** -- :meth:`EvidenceStore.find_number`
  returns the run ids whose output contains it. This is what turns "the model
  said 14.2/min" into "run ``hfo_003`` measured 14.2/min", or into a struck
  sentence.
* **What did the session cost?** -- :meth:`EvidenceStore.cost`, tool calls and
  wall-clock, which is the x-axis of the cost-accuracy curve.

Numbers are indexed with a tolerance, because a model that writes ``14.2`` for
a stored ``14.23`` is rounding, not fabricating. The tolerance is deliberately
tight (:data:`NUMBER_TOLERANCE`) and applied relative to magnitude, so ``145``
does not quietly satisfy a claim of ``150``.

The store holds failed runs too. A trace in which the third call errored and
the planner recovered is a better audit record than one in which it silently
disappeared, and the failure rate is itself a reported metric.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from onset_agent.contract import CONTRACT_VERSION, ToolRun

__all__ = ["EvidenceStore", "NUMBER_TOLERANCE", "numbers_in"]

#: A claimed number matches a stored one within this relative tolerance, or
#: within the absolute floor, whichever is larger. 1% catches honest rounding;
#: it does not let 145 pass as 150.
NUMBER_TOLERANCE = 0.01
NUMBER_FLOOR = 0.05

_NUMBER_RE = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?")


def numbers_in(value, out: set[float] | None = None) -> set[float]:
    """Every finite number anywhere inside a nested JSON-ish structure."""
    out = set() if out is None else out
    if value is None or isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            out.add(round(float(value), 4))
        return out
    if isinstance(value, str):
        for token in _NUMBER_RE.findall(value):
            out.add(round(float(token), 4))
        return out
    if isinstance(value, dict):
        for item in value.values():
            numbers_in(item, out)
        return out
    if isinstance(value, (list, tuple, set)):
        for item in value:
            numbers_in(item, out)
    return out


class EvidenceStore:
    """Append-only record of one analysis session."""

    def __init__(self, subject: str = "", source: str = ""):
        self.subject = subject
        self.source = source
        self._runs: list[ToolRun] = []
        self._by_id: dict[str, ToolRun] = {}
        self._numbers: dict[str, set[float]] = {}

    # -- writing ----------------------------------------------------------
    def append(self, run: ToolRun) -> ToolRun:
        """Record a run. Duplicate ids are a programming error, not a warning."""
        if run.run_id in self._by_id:
            raise ValueError(f"run_id {run.run_id!r} is already in the store")
        self._runs.append(run)
        self._by_id[run.run_id] = run
        self._numbers[run.run_id] = numbers_in(run.output) if run.ok else set()
        return run

    # -- reading ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._runs)

    def __iter__(self):
        return iter(self._runs)

    def get(self, run_id: str) -> ToolRun | None:
        return self._by_id.get(run_id)

    def runs(self, tool: str | None = None, ok_only: bool = False) -> list[ToolRun]:
        return [r for r in self._runs
                if (tool is None or r.tool == tool) and (not ok_only or r.ok)]

    def run_ids(self) -> list[str]:
        return [r.run_id for r in self._runs]

    def all_numbers(self) -> set[float]:
        """Every number any tool returned this session."""
        out: set[float] = set()
        for values in self._numbers.values():
            out |= values
        return out

    def find_number(self, number: float) -> list[str]:
        """Run ids whose output contains ``number``, within tolerance.

        Empty means the number is unsupported: no tool in this session
        produced it, and any sentence containing it is a fabrication as far as
        this system is concerned.
        """
        target = round(float(number), 4)
        hits = []
        for run_id, values in self._numbers.items():
            for value in values:
                if _close(value, target):
                    hits.append(run_id)
                    break
        return hits

    def evidence_ids(self) -> set[str]:
        """Event-level ids (``get_event_evidence``) that a citation may use."""
        found: set[str] = set()
        for run in self._runs:
            _collect_ids(run.output, found)
        return found

    def resolve_evidence_id(self, evidence_id: str) -> str | None:
        """The run id that produced a given evidence id, or ``None``."""
        for run in self._runs:
            found: set[str] = set()
            _collect_ids(run.output, found)
            if evidence_id in found:
                return run.run_id
        return None

    # -- state the planner reads -----------------------------------------
    def digest(self, max_runs: int = 12, per_run_chars: int = 400) -> str:
        """A compact, readable summary of the session so far.

        This is what the planner sees between steps. Keeping it short is not
        cosmetic: it is the difference between a small model that plans and
        one that drowns. Long outputs are truncated with an explicit marker,
        so the model can ask for the detail again rather than invent it.
        """
        if not self._runs:
            return "No analyses have been run yet."
        lines = []
        for run in self._runs[-max_runs:]:
            args = ", ".join(f"{k}={_short(v)}" for k, v in sorted(run.input.items()))
            if not run.ok:
                lines.append(f"[{run.run_id}] {run.tool}({args}) -> FAILED: {run.error}")
                continue
            # NOT sort_keys: tools return channels in rate order, and sorting
            # them alphabetically would hand the planner a ranking that is not
            # the ranking the tool computed.
            body = json.dumps(run.output, default=str)
            if len(body) > per_run_chars:
                body = body[:per_run_chars] + "... (truncated; call again for detail)"
            lines.append(f"[{run.run_id}] {run.tool}({args}) -> {body}")
        head = f"{len(self._runs)} tool run(s) so far"
        if len(self._runs) > max_runs:
            head += f" (showing the last {max_runs})"
        return head + ":\n" + "\n".join(lines)

    def cost(self) -> dict:
        """Tool calls and wall-clock -- the x-axis of a cost/accuracy curve."""
        return {"n_tool_calls": len(self._runs),
                "n_failed": sum(not r.ok for r in self._runs),
                "runtime_s": round(sum(r.runtime_s for r in self._runs), 3),
                "by_tool": {tool: sum(r.tool == tool for r in self._runs)
                            for tool in sorted({r.tool for r in self._runs})}}

    # -- persistence ------------------------------------------------------
    def as_records(self) -> list[dict]:
        return [run.as_record() for run in self._runs]

    def save(self, path: str | Path) -> Path:
        """Write the ledger as JSON. This file *is* the audit trail."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"contract_version": CONTRACT_VERSION, "subject": self.subject,
             "source": self.source, "cost": self.cost(), "runs": self.as_records()},
            indent=2, default=str))
        return path


def _close(value: float, target: float) -> bool:
    return abs(value - target) <= max(NUMBER_FLOOR, NUMBER_TOLERANCE * abs(target))


def _short(value, limit: int = 60) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "..."


def _collect_ids(payload, found: set[str]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "evidence_id" and isinstance(value, str):
                found.add(value)
            else:
                _collect_ids(value, found)
    elif isinstance(payload, list):
        for item in payload:
            _collect_ids(item, found)
