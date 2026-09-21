"""The frozen tool contract: the only thing that crosses the model boundary.

Every analyzer is a function with a strict JSON input and a strict JSON
output, and each call is recorded as a :class:`ToolRun` carrying a ``run_id``.
Nothing else passes between the orchestrator and the analysis code.

    {
      "tool": "detect_hfo",
      "input":  {"channels": ["LA1-LA2"], "window_s": [50, 76], "threshold_sd": 5},
      "output": {"LA1-LA2": {"rate_per_min": 14.2, "n_events": 71}},
      "run_id": "hfo_003",
      "runtime_s": 1.7
    }

Two design rules follow, and both are enforced here rather than requested in a
prompt:

1. **The model never touches raw signal.** It sees numbers produced by
   deterministic, separately tested code. This is what makes the system
   auditable, and it is the answer to "why not feed the EEG to a multimodal
   model".
2. **Every number carries a run id.** A claim in the final report is
   admissible only if it resolves back to one. :mod:`onset_agent.verifier`
   is the component that enforces it; this module is what makes it possible.

Freezing
--------
``CONTRACT_VERSION`` is stamped into every run record. Bump it when a tool's
input or output shape changes, never when an implementation detail does --
the point of the contract is that the orchestration half and the analysis half
can be developed independently against a fixed interface.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["CONTRACT_VERSION", "ToolRun", "ToolSpec", "ToolRegistry", "ContractError"]

#: Bumped only when a tool's JSON input or output shape changes.
CONTRACT_VERSION = "1.0"


class ContractError(ValueError):
    """A tool call that violates the contract. Reported to the model verbatim."""


@dataclass
class ToolRun:
    """One executed tool call, exactly as the contract describes it.

    This is the unit of evidence. A report sentence is supported when the
    numbers in it can be found in some ``ToolRun.output``; it is unsupported
    otherwise, and there is no third state.
    """

    run_id: str
    tool: str
    input: dict
    output: Any = None
    runtime_s: float = 0.0
    ok: bool = True
    error: str = ""
    started_at: float = 0.0
    contract_version: str = CONTRACT_VERSION

    def as_record(self) -> dict:
        """The JSON record in the contract's shape."""
        record = {"tool": self.tool, "input": self.input, "output": self.output,
                  "run_id": self.run_id, "runtime_s": round(self.runtime_s, 3)}
        if not self.ok:
            record["error"] = self.error
        return record

    def as_json(self) -> str:
        return json.dumps(self.as_record(), default=str, sort_keys=True)


@dataclass(frozen=True)
class ToolSpec:
    """A tool's identity, its JSON schema, and the code behind it.

    ``run_prefix`` is what makes run ids readable in a trace: ``hfo_003`` is
    the third HFO detection of the session, and a reviewer scanning a report's
    citations can see which analysis produced each number without a lookup.
    """

    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any]
    run_prefix: str
    #: True when the tool recomputes something (and so may be re-run with
    #: different parameters). False for pure lookups of what was analysed.
    recomputes: bool = False

    def schema(self) -> dict:
        """OpenAI-compatible function schema (Qwen, Llama and Mistral all read this)."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


class ToolRegistry:
    """Validates, dispatches and records every tool call.

    The registry is deliberately dumb about *what* the tools do and strict
    about *how* they are called: unknown names, unknown arguments, wrong types
    and out-of-range values are rejected before any code runs. A model that
    hallucinates a parameter gets an error message, not an exception, and the
    planner can act on it.
    """

    def __init__(self, specs: list[ToolSpec], session: Any = None):
        self.specs = {spec.name: spec for spec in specs}
        self.session = session
        self._counters: dict[str, int] = {}

    # -- introspection ----------------------------------------------------
    def schemas(self) -> list[dict]:
        return [spec.schema() for spec in self.specs.values()]

    def names(self) -> list[str]:
        return list(self.specs)

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    # -- execution --------------------------------------------------------
    def next_run_id(self, tool: str) -> str:
        prefix = self.specs[tool].run_prefix if tool in self.specs else "run"
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}_{self._counters[prefix]:03d}"

    def validate(self, name: str, arguments: dict | None) -> dict:
        """Check one call against the schema. Returns coerced arguments."""
        spec = self.specs.get(name)
        if spec is None:
            raise ContractError(f"unknown tool {name!r}; available: {', '.join(self.specs)}")
        args = dict(arguments or {})
        schema = spec.parameters
        properties = schema["properties"]

        unexpected = set(args) - set(properties)
        if unexpected:
            raise ContractError(
                f"{name} does not accept {', '.join(sorted(unexpected))}; "
                f"allowed: {', '.join(sorted(properties)) or 'no arguments'}")
        for key in schema.get("required", []):
            if key not in args:
                raise ContractError(f"{name} requires {key!r}")
        for key, value in list(args.items()):
            args[key] = _coerce(name, key, value, properties[key])
        return args

    def run(self, name: str, arguments: dict | None = None) -> ToolRun:
        """Validate, execute and record one call. Never raises for model error.

        A malformed call and a crashing tool both come back as a ``ToolRun``
        with ``ok=False``: the planner sees the failure, can re-plan around it,
        and the failed call still appears in the evidence store. A tool call
        that vanishes from the audit trail is worse than one that failed.
        """
        started = time.time()
        clock = time.perf_counter()
        try:
            args = self.validate(name, arguments)
        except ContractError as exc:
            return ToolRun(run_id=self.next_run_id(name), tool=name, input=dict(arguments or {}),
                           output={"error": str(exc)}, ok=False, error=str(exc),
                           started_at=started, runtime_s=time.perf_counter() - clock)
        run_id = self.next_run_id(name)
        spec = self.specs[name]
        try:
            output = spec.handler(self.session, **args)
        except ContractError as exc:
            # The handler rejected the call for a reason the planner can act on
            # ("unknown channel", "band unusable at this sampling rate"). That
            # message is written for the model, so it is passed through intact.
            return ToolRun(run_id=run_id, tool=name, input=args,
                           output={"error": str(exc)}, ok=False, error=str(exc),
                           started_at=started, runtime_s=time.perf_counter() - clock)
        except Exception as exc:  # a bug in a tool must not end the session
            message = f"{type(exc).__name__}: {exc}"
            return ToolRun(run_id=run_id, tool=name, input=args,
                           output={"error": "the tool failed"}, ok=False, error=message,
                           started_at=started, runtime_s=time.perf_counter() - clock)
        return ToolRun(run_id=run_id, tool=name, input=args, output=output, ok=True,
                       started_at=started, runtime_s=time.perf_counter() - clock)


def _coerce(tool: str, key: str, value: Any, spec: dict) -> Any:
    """Type-check and range-clamp one argument against its schema fragment."""
    kind = spec.get("type")
    if kind == "string":
        if not isinstance(value, str):
            raise ContractError(f"{tool}.{key} must be a string")
        if "enum" in spec and value not in spec["enum"]:
            raise ContractError(f"{tool}.{key} must be one of {', '.join(spec['enum'])}")
        return value
    if kind == "integer":
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ContractError(f"{tool}.{key} must be an integer") from None
        return _clamp(number, spec)
    if kind == "number":
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ContractError(f"{tool}.{key} must be a number") from None
        return _clamp(number, spec)
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ContractError(f"{tool}.{key} must be true or false")
        return value
    if kind == "array":
        if not isinstance(value, (list, tuple)):
            raise ContractError(f"{tool}.{key} must be a list")
        items = spec.get("items", {})
        out = [_coerce(tool, f"{key}[]", item, items) for item in value]
        if "maxItems" in spec and len(out) > spec["maxItems"]:
            raise ContractError(f"{tool}.{key} accepts at most {spec['maxItems']} items")
        if "minItems" in spec and len(out) < spec["minItems"]:
            raise ContractError(f"{tool}.{key} needs at least {spec['minItems']} items")
        return out
    return value


def _clamp(number: float | int, spec: dict) -> float | int:
    """Clamp rather than reject: a threshold of 99 is a mistake, not an attack.

    Out-of-range *types* are refused above; out-of-range *values* are pulled to
    the nearest legal one, because a planner exploring parameters should be
    corrected rather than stopped. The clamped value is what appears in the
    run record, so the trace shows what actually ran.
    """
    if "minimum" in spec:
        number = max(spec["minimum"], number)
    if "maximum" in spec:
        number = min(spec["maximum"], number)
    return number
