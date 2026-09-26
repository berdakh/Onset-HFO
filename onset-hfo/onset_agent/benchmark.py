"""Measure what a *real* model does on the S0--S3 ladder.

Every orchestration number this project has published so far comes from
:class:`~onset_agent.planner.ScriptedPlanner`, a deterministic keyword policy
that speaks the tool protocol. That is the control, not the result. This module
runs the same ladder over a matrix of served models and quantizations and
scores each cell on the things a reviewer will ask about.

Four metrics, and one non-metric
--------------------------------

* **tool-call validity** -- the share of tool calls the dispatcher accepted.
  A model that cannot address the tools produces a low number here and nothing
  useful downstream.
* **unsupported-claim rate** -- the share of numeric claims in the drafted
  report that no tool output supports, from
  :class:`~onset_agent.verifier.DeterministicVerifier`. This is computed for
  *every* rung, including the ones that do not verify in production, because
  the question is what the model did, not what the guard caught.
* **verifier delta** -- when a second model instance is available,
  :func:`~onset_agent.verifier.compare_verifiers` measures true statements the
  model verifier removed against unsupported numbers it let through. This is
  the cost of verification, and it is the number nobody reports.
* **planning quality** -- top-5 overlap and rank agreement against the S0
  fixed pipeline on the identical session. S0 is not ground truth; it is the
  deterministic reference, so this measures *how far the model's choices moved
  the answer*, which is the honest version of the question.

The non-metric is **wall-clock**. It is recorded, because throwing away a
measurement is worse than labelling it, but a shared or throttled accelerator
makes it incomparable between sessions, and 8-bit ``bitsandbytes`` kernels are
slower than fp16 on pre-Ampere cards, so a quantization comparison on such a
card measures the kernel rather than the model. :func:`timing_warnings`
inspects a finished sweep and says when the timings must not be compared.

Checkpointing
-------------

Free notebook runtimes disconnect, and this sweep is long. Every cell writes
its own JSON under ``<out>/cells/`` as soon as it finishes, and
:func:`run_matrix` skips cells whose file already exists. Re-running the same
command after a disconnect resumes; there is no separate resume flag, because
a resume path that is only exercised after a failure is a resume path that
does not work.

A cell that raises is recorded with its traceback rather than aborting the
sweep, so one model that will not load cannot cost you the other eight.

Offline
-------

``backend="scripted"`` runs the whole harness with no model and no network, on
a synthetic recording. That is what the test suite uses, and what you should
run once before spending a GPU hour.

How this relates to ``scripts/run_model_ladder.py``
---------------------------------------------------

They are deliberately different shapes, and neither should grow into the
other.

``run_model_ladder.py`` is the **depth** run: one served model on a machine
you control, the four rungs, run-to-run stability over repeats, the whole
falsification suite, and a ``RESULTS.md`` ready to paste into
``docs/ORCHESTRATION.md``. It re-runs the scripted planner on the same machine
as its control rather than quoting published numbers. It is the right tool
when you have a GPU box and one model you care about, and it asks you to
record the quantization by hand.

This module is the **breadth** sweep: the model x quantization matrix the
roadmap asks for, checkpointed per cell so a notebook runtime that disconnects
does not cost the whole run, with the quantization, the accelerator and every
library version recorded automatically because a matrix is exactly where
hand-recording breaks down. It does not repeat the falsification suite or the
stability repeats --- that is the depth run's job, and duplicating it here
would mean two implementations of the same measurement drifting apart.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from onset_agent.analysis import AnalysisSession
from onset_agent.planner import Rung, run_rung
from onset_agent.verifier import DeterministicVerifier, compare_verifiers

#: The quantizations the ladder is swept over. ``fp16`` means "whatever the
#: backend loads by default", which is float16 on a card without bfloat16.
QUANTIZATIONS = ("fp16", "8bit", "4bit")

#: All four rungs, in the order they are meant to be read.
RUNGS = ("S0", "S1", "S2", "S3")

#: Approximate weights-only VRAM, in GB, for the sizes in the roadmap's ladder.
#: These are for deciding what will fit before you start, not for citing.
APPROX_VRAM_GB = {
    ("4B", "fp16"): 8.0, ("4B", "8bit"): 4.5, ("4B", "4bit"): 2.8,
    ("8B", "fp16"): 16.0, ("8B", "8bit"): 9.0, ("8B", "4bit"): 5.5,
    ("14B", "fp16"): 28.0, ("14B", "8bit"): 15.0, ("14B", "4bit"): 9.0,
}


# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One point of the sweep: a model, a precision, a rung, a recording."""

    model: str
    quantization: str
    rung: str
    subject: str

    def __post_init__(self) -> None:
        if self.quantization not in QUANTIZATIONS:
            raise ValueError(
                f"quantization {self.quantization!r} is not one of {QUANTIZATIONS}")
        Rung(self.rung)  # raises for an unknown rung

    @property
    def cell_id(self) -> str:
        """A filesystem-safe identifier. This is the checkpoint file's name."""
        safe = self.model.replace("/", "--").replace(":", "-")
        return f"{safe}__{self.quantization}__{self.rung}__{self.subject}"

    def as_dict(self) -> dict:
        return asdict(self)


def expand_matrix(models: Sequence[str], quantizations: Sequence[str] = QUANTIZATIONS,
                  rungs: Sequence[str] = RUNGS,
                  subjects: Sequence[str] = ("sub-pt01",)) -> list[Cell]:
    """Every combination, in a deterministic order.

    S0 runs no model at all, so it appears once per (subject, model) rather
    than once per quantization -- running the fixed pipeline three times at
    three precisions would produce three identical rows and imply a comparison
    that was never made.
    """
    cells: list[Cell] = []
    for subject in subjects:
        for model in models:
            for quant in quantizations:
                for rung in rungs:
                    if rung == "S0" and quant != quantizations[0]:
                        continue
                    cells.append(Cell(model=model, quantization=quant,
                                      rung=rung, subject=subject))
    return cells


def fits(size: str, quantization: str, vram_gb: float) -> bool:
    """Will this cell fit, roughly, in ``vram_gb``? Leaves 2 GB of headroom.

    Approximate on purpose: it exists so a sweep can skip cells that will
    certainly OOM, not so a number can be quoted.
    """
    need = APPROX_VRAM_GB.get((size, quantization))
    return need is None or (need + 2.0) <= vram_gb


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return "unknown"


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version  # noqa: PLC0415
        return version(name)
    except Exception:  # noqa: BLE001 - any failure here means "not installed"
        return "absent"


def environment() -> dict:
    """Everything about this machine that could change a number.

    Recorded per cell rather than once per sweep, because a notebook runtime
    can be reassigned to a different accelerator mid-sweep and the resulting
    timings would otherwise be silently pooled.
    """
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "git_commit": _git_commit(),
        "transformers": _package_version("transformers"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "torch": _package_version("torch"),
        "accelerate": _package_version("accelerate"),
        "gpu_name": "none",
        "gpu_total_gb": 0.0,
        "cuda": "none",
        "supports_bf16": False,
    }
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return env
    if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
        return env
    props = torch.cuda.get_device_properties(0)  # pragma: no cover
    env.update({  # pragma: no cover
        "gpu_name": props.name,
        "gpu_total_gb": round(props.total_memory / 1024 ** 3, 2),
        "cuda": torch.version.cuda or "none",
        "supports_bf16": bool(torch.cuda.is_bf16_supported()),
        "capability": f"{props.major}.{props.minor}",
    })
    return env


def backend_provenance(backend) -> dict:
    """What the backend can say about itself, without assuming it can."""
    info = {"backend": getattr(backend, "name", type(backend).__name__),
            "model_id": getattr(backend, "model_id", "")}
    describe = getattr(backend, "describe", None)
    if callable(describe):
        info["describe"] = describe()
    for attr in ("revision", "quantization", "dtype", "max_new_tokens"):
        if hasattr(backend, attr):
            info[attr] = str(getattr(backend, attr))
    return info


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _overlap(a: Sequence[str], b: Sequence[str], k: int = 5) -> float:
    """Share of the top ``k`` that two rankings agree on."""
    top_a, top_b = set(a[:k]), set(b[:k])
    return round(len(top_a & top_b) / k, 4) if k else 0.0


def _rank_agreement(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Spearman correlation of positions over the channels both rankings hold.

    ``None`` when fewer than three channels are shared, because a correlation
    over two points is not a measurement.
    """
    pos_a = {c: i for i, c in enumerate(a)}
    pos_b = {c: i for i, c in enumerate(b)}
    shared = sorted(set(pos_a) & set(pos_b))
    if len(shared) < 3:
        return None
    n = len(shared)
    ra = _ranks([pos_a[c] for c in shared])
    rb = _ranks([pos_b[c] for c in shared])
    mean = (n - 1) / 2
    num = sum((x - mean) * (y - mean) for x, y in zip(ra, rb, strict=True))
    den_a = sum((x - mean) ** 2 for x in ra)
    den_b = sum((y - mean) ** 2 for y in rb)
    if den_a == 0 or den_b == 0:
        return None
    return round(num / (den_a * den_b) ** 0.5, 4)


def _ranks(values: Sequence[float]) -> list[float]:
    """Midranks, so ties do not bias the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def score_result(result, *, reference_top: Sequence[str] | None = None,
                 verifier_backend=None, labels=None) -> dict:
    """Turn one :class:`~onset_agent.planner.PlannerResult` into a row.

    ``reference_top`` is the S0 ranking on the same session. ``labels`` is an
    optional :class:`~onset_hfo.cohort.SozLabels`; when present the ranking is
    also scored against the clinician's contacts with a permutation test, and
    when absent that block is simply missing rather than filled with a
    placeholder.
    """
    cost = result.cost or {}
    n_calls = int(cost.get("n_tool_calls", 0))
    n_failed = int(cost.get("n_failed", 0))

    # Applied to every rung, including those that do not verify in production:
    # the measurement is what the model wrote, not what the guard removed.
    check = DeterministicVerifier()(result.report, result.store)

    row: dict = {
        "rung": result.rung.value if hasattr(result.rung, "value") else str(result.rung),
        "subject": result.subject,
        "backend": result.backend,
        "stop_reason": result.stop_reason,
        "n_tool_calls": n_calls,
        "n_failed_tool_calls": n_failed,
        "tool_call_validity": round(1.0 - n_failed / n_calls, 4) if n_calls else None,
        "n_steps": len(result.steps),
        "n_channels_retested": result.n_retested,
        "n_sentences": check.n_sentences,
        "n_claims": check.n_claims,
        "n_struck": check.n_struck,
        "unsupported_claim_rate": round(check.unsupported_claim_rate, 4),
        "provenance_coverage": round(check.provenance_coverage, 4),
        "top_5": result.top_k(5),
        "runtime_s_tools": cost.get("runtime_s"),
    }

    if reference_top is not None:
        row["top5_overlap_vs_s0"] = _overlap(result.top_channels, reference_top)
        row["rank_agreement_vs_s0"] = _rank_agreement(result.top_channels, reference_top)

    if labels is not None:
        from onset_agent.scoring import score_result as score_against_labels  # noqa: PLC0415
        score = score_against_labels(result, labels)
        row["label_score"] = score.as_dict() if hasattr(score, "as_dict") else str(score)

    if verifier_backend is not None:
        row["verifier_delta"] = compare_verifiers(
            result.report, result.store, verifier_backend)

    return row


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


@dataclass
class SweepPaths:
    """Where a sweep keeps its state."""

    root: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def cells(self) -> Path:
        return self.root / "cells"

    def cell_file(self, cell: Cell) -> Path:
        return self.cells / f"{cell.cell_id}.json"

    def prepare(self) -> SweepPaths:
        self.cells.mkdir(parents=True, exist_ok=True)
        return self


def run_cell(cell: Cell, *, session: AnalysisSession, backend_factory: Callable,
             reference_top: Sequence[str] | None = None, verifier_backend=None,
             labels=None, max_steps: int = 12, run_id: str = "") -> dict:
    """Run one cell and return its record. Never raises for a model failure.

    A cell that fails is a datum --- "this model at this precision could not
    complete the ladder" is exactly what a quantization sweep is for --- so the
    exception is recorded and the sweep continues.
    """
    record: dict = {
        "run_id": run_id,
        "cell": cell.as_dict(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment(),
    }
    started = time.perf_counter()
    try:
        backend = None if cell.rung == "S0" else backend_factory(cell)
        record["backend_provenance"] = backend_provenance(backend) if backend else {
            "backend": "none (fixed pipeline)"}
        result = run_rung(session, rung=cell.rung, backend=backend, max_steps=max_steps)
        record["metrics"] = score_result(
            result, reference_top=reference_top,
            verifier_backend=verifier_backend, labels=labels)
        record["report"] = result.report
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 - a failed cell is a result
        record["ok"] = False
        record["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc(limit=8)}
    record["wall_clock_s"] = round(time.perf_counter() - started, 3)
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return record


def run_matrix(cells: Iterable[Cell], *, session_factory: Callable[[str], AnalysisSession],
               backend_factory: Callable[[Cell], object], out: str | Path,
               verifier_backend=None, labels_for=None, max_steps: int = 12,
               overwrite: bool = False, verbose: bool = True) -> Path:
    """Run every cell, checkpointing each one, skipping those already done.

    Parameters
    ----------
    session_factory:
        ``subject -> AnalysisSession``. Called once per subject and cached, so
        every rung and every model see the identical recording and the
        identical cached filtering --- otherwise a difference between cells
        could be a difference between sessions.
    backend_factory:
        ``Cell -> Backend``. Called once per cell, so the previous model is
        released before the next is loaded. On a 16 GB card that is the
        difference between a sweep and an out-of-memory error.
    labels_for:
        optional ``subject -> SozLabels``.
    """
    paths = SweepPaths(Path(out)).prepare()
    sessions: dict[str, AnalysisSession] = {}
    references: dict[str, list[str]] = {}
    written: list[Path] = []

    for cell in cells:
        target = paths.cell_file(cell)
        if target.exists() and not overwrite:
            if verbose:
                print(f"skip  {cell.cell_id} (already done)")
            continue

        if cell.subject not in sessions:
            sessions[cell.subject] = session_factory(cell.subject)
            # The S0 ranking is the deterministic reference every model cell is
            # compared against, so it is computed once per subject before any
            # model is loaded.
            reference = run_rung(sessions[cell.subject], rung="S0")
            references[cell.subject] = reference.top_channels

        if verbose:
            print(f"run   {cell.cell_id}")
        record = run_cell(
            cell, session=sessions[cell.subject], backend_factory=backend_factory,
            reference_top=references[cell.subject], verifier_backend=verifier_backend,
            labels=labels_for(cell.subject) if labels_for else None,
            max_steps=max_steps, run_id=paths.run_id)
        target.write_text(json.dumps(record, indent=2, default=str))
        written.append(target)
        if verbose:
            status = "ok" if record["ok"] else f"FAILED ({record['error']['type']})"
            print(f"      {status} in {record['wall_clock_s']}s")

    if verbose:
        print(f"\n{len(written)} cell(s) written to {paths.cells}")
    return paths.root


# --------------------------------------------------------------------------
# Reading a finished sweep
# --------------------------------------------------------------------------


def load_cells(out: str | Path) -> list[dict]:
    """Every checkpoint under ``out``, in a stable order."""
    cells_dir = Path(out) / "cells"
    return [json.loads(p.read_text()) for p in sorted(cells_dir.glob("*.json"))]


def summarise(out: str | Path):
    """One row per cell, as a DataFrame. Failed cells keep their row."""
    import pandas as pd  # noqa: PLC0415

    rows = []
    for record in load_cells(out):
        row = dict(record["cell"])
        row["ok"] = record.get("ok", False)
        row["wall_clock_s"] = record.get("wall_clock_s")
        row["gpu_name"] = record.get("environment", {}).get("gpu_name", "none")
        row["run_id"] = record.get("run_id", "")
        if record.get("ok"):
            metrics = dict(record.get("metrics", {}))
            delta = metrics.pop("verifier_delta", None)
            metrics.pop("label_score", None)
            row.update({k: v for k, v in metrics.items() if not isinstance(v, dict)})
            if delta:
                row["verifier_coverage_cost"] = delta.get("coverage_cost")
                row["verifier_missed"] = delta.get("hallucinations_missed_by_llm")
                row["verifier_agreement"] = delta.get("agreement")
        else:
            row["error"] = record.get("error", {}).get("type", "unknown")
        rows.append(row)
    return pd.DataFrame(rows)


def timing_warnings(frame) -> list[str]:
    """Reasons the ``wall_clock_s`` column must not be compared across rows.

    Returns an empty list only when every successful cell ran on one named
    accelerator within one sweep. Anything else --- a mixed or unknown GPU, a
    resumed sweep, or 8-bit on a pre-Ampere card where the int8 kernel is the
    slow path --- makes the timings a property of the session rather than of
    the model, and this function says so rather than leaving the reader to
    notice.
    """
    warnings: list[str] = []
    done = frame[frame["ok"]] if "ok" in frame else frame
    if done.empty:
        return ["no successful cells"]

    gpus = sorted({str(g) for g in done.get("gpu_name", [])})
    if len(gpus) > 1:
        warnings.append(f"cells ran on {len(gpus)} different accelerators: {', '.join(gpus)}")
    if gpus == ["none"]:
        warnings.append("no accelerator was detected; timings are CPU timings")
    runs = sorted({str(r) for r in done.get("run_id", []) if str(r)})
    if len(runs) > 1:
        warnings.append(
            f"cells come from {len(runs)} separate sweeps, so the machine may have changed")
    pre_ampere = {"Tesla T4", "Tesla V100", "Tesla P100", "Quadro RTX"}
    if any(any(g.startswith(p) for p in pre_ampere) for g in gpus) and \
            "8bit" in set(done.get("quantization", [])):
        warnings.append(
            "8-bit cells ran on a pre-Ampere card, where the bitsandbytes int8 path is "
            "slower than fp16: a quantization timing comparison here measures the kernel")
    return warnings


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _default_session_factory(synthetic: bool, duration_s: float, start: float,
                             task: str, run: str) -> Callable[[str], AnalysisSession]:
    def make(subject: str) -> AnalysisSession:
        if synthetic:
            from onset_hfo.synthetic import make_synthetic_recording  # noqa: PLC0415
            recording = make_synthetic_recording(duration_s=duration_s, seed=7, verbose=False)
        else:
            from onset_hfo.datasets import fetch_slice  # noqa: PLC0415
            recording = fetch_slice(subject, task, run,
                                    t_start=start, t_stop=start + duration_s)
        return AnalysisSession(recording, verbose=False)

    return make


def _default_backend_factory(kind: str, dtype: str, revision: str | None,
                             max_new_tokens: int) -> Callable[[Cell], object]:
    def make(cell: Cell):
        from onset_agent.backends import make_backend  # noqa: PLC0415
        if kind == "scripted":
            from onset_agent.planner import ScriptedPlanner  # noqa: PLC0415
            return ScriptedPlanner()
        kwargs: dict = {"max_new_tokens": max_new_tokens}
        if kind in ("transformers", "hf"):
            kwargs.update(quantization=cell.quantization, dtype=dtype, revision=revision)
        return make_backend(kind, model=cell.model, **kwargs)

    return make


def main(argv: list[str] | None = None) -> int:
    """``python -m onset_agent.benchmark`` -- sweep the ladder and summarise it."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-7B-Instruct"])
    parser.add_argument("--quantizations", nargs="+", default=list(QUANTIZATIONS),
                        choices=list(QUANTIZATIONS))
    parser.add_argument("--rungs", nargs="+", default=list(RUNGS), choices=list(RUNGS))
    parser.add_argument("--subjects", nargs="+", default=["sub-pt01"])
    parser.add_argument("--backend", default="transformers",
                        choices=["scripted", "transformers", "hf", "ollama",
                                 "openai_compat", "vllm"])
    parser.add_argument("--dtype", default="auto",
                        help="pass float16 on a card without bfloat16, such as a T4")
    parser.add_argument("--revision", default=None,
                        help="pin the model to a commit; recorded either way")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--synthetic", action="store_true",
                        help="offline: a generated recording, no download")
    parser.add_argument("--task", default="ictal")
    parser.add_argument("--run", default="01")
    parser.add_argument("--start", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--out", default="artifacts/agent_benchmark")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cells = expand_matrix(args.models, args.quantizations, args.rungs, args.subjects)
    print(f"{len(cells)} cell(s); checkpoints in {args.out}/cells\n")
    out = run_matrix(
        cells,
        session_factory=_default_session_factory(
            args.synthetic, args.duration, args.start, args.task, args.run),
        backend_factory=_default_backend_factory(
            args.backend, args.dtype, args.revision, args.max_new_tokens),
        out=args.out, max_steps=args.max_steps, overwrite=args.overwrite)

    frame = summarise(out)
    summary_path = Path(out) / "summary.csv"
    frame.to_csv(summary_path, index=False)
    print(f"\n{summary_path}")
    for warning in timing_warnings(frame):
        print(f"  timing caveat: {warning}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
