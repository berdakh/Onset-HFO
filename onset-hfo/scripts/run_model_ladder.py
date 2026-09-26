#!/usr/bin/env python
"""Run the S0-S3 ladder under a REAL open-weight model, and fill in the table.

Everything published about the orchestration half so far comes from
`onset_agent.planner.ScriptedPlanner` -- a deterministic policy with no
language model in it. That is the control, not the result. This script
produces the result, and it needs a GPU or a CPU you do not mind occupying.

    # 1. serve a model (either is fine)
    ollama pull qwen2.5:7b-instruct && ollama serve &
    #    or: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

    # 2. run everything (~20-60 min on one GPU, depending on the model)
    python scripts/run_model_ladder.py --backend ollama --model qwen2.5:7b-instruct

    # a five-minute smoke test first, to check the model can plan at all
    python scripts/run_model_ladder.py --backend ollama --model qwen2.5:7b-instruct --quick

What it measures, and why each part needs a real model
------------------------------------------------------
**The ladder (S0-S3).** The thesis claim is that a planner which re-plans
beats a fixed pipeline using the same analyzers. The scripted planner is a
*competent* control -- it surveys then challenges the leaders -- so a language
model has to beat a reasonable strategy, not a straw man.

**Run-to-run stability.** Trivially 1.0 for a deterministic planner. Under a
sampled model it is the number a reviewer will ask about first: would a
surgeon asking twice get the same answer? Set temperature to 0 and note that
continuous batching in vLLM can still produce variation -- measure it rather
than assuming it away.

**Falsification, and one test in particular.** The `anonymised channel names`
test renames `AD1` to `EA1` and checks the ranking does not move. A scripted
planner *cannot* fail it. A language model can: it may know that `AD` is an
amygdala depth electrode and that mesial temporal contacts are a common onset
site, and quietly rank on that prior instead of on the evidence. This is the
single most valuable row in the output and it is invisible to every other
measurement in the repository.

**Cost.** Tokens and wall-clock per patient, which is the deployability
argument.

Output
------
`--out` receives `ladder.json` (everything, machine-readable) and
`RESULTS.md`, a table ready to paste into `docs/ORCHESTRATION.md`. The
scripted-planner column is re-run here rather than quoted, so both columns
come from the same machine and the same recording.

Not this script's job
---------------------
The **model x quantization matrix** -- fp16/8-bit/4-bit across several model
sizes -- belongs to `onset_agent.benchmark`, which checkpoints every cell so a
notebook runtime that disconnects does not cost the whole sweep, and records
the accelerator and library versions per cell rather than asking you to note
them. This script is the depth run: one model, one machine, every measurement
including the falsification suite. Keep it that way; two implementations of
the same measurement drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _check_backend(backend) -> None:
    """Fail loudly and early rather than 40 minutes in."""
    from onset_agent.backends import AssistantMessage  # noqa: F401

    print(f"[ladder] backend: {backend.describe()}")
    started = time.perf_counter()
    try:
        reply = backend.chat(
            [{"role": "user", "content": 'Reply with exactly: {"ok": true}'}], [])
    except Exception as exc:
        raise SystemExit(
            f"\n[ladder] the backend is not reachable: {type(exc).__name__}: {exc}\n"
            f"[ladder] is the server running? try:  ollama serve   (or start vLLM)") from None
    print(f"[ladder] model answered in {time.perf_counter() - started:.1f} s: "
          f"{(reply.content or '')[:80]!r}\n")


def _load_recording(args):
    if args.synthetic:
        from onset_hfo.synthetic import make_synthetic_recording

        return make_synthetic_recording(seed=7, duration_s=args.duration, verbose=False)
    from onset_hfo.datasets import fetch_slice

    return fetch_slice(args.subject, args.task, args.run,
                       t_start=args.start, t_stop=args.stop, verbose=True)


def main(argv: list[str] | None = None) -> int:
    from onset_agent.analysis import AnalysisSession
    from onset_agent.backends import make_backend
    from onset_agent.falsify import run_falsification_suite
    from onset_agent.planner import (
        FixedBudget,
        ModelJudged,
        Rung,
        ScriptedPlanner,
        TiedSetWidth,
        run_rung,
    )
    from onset_agent.scoring import compare_rungs
    from onset_hfo.cohort import soz_labels

    parser = argparse.ArgumentParser(prog="run_model_ladder", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="ollama",
                        choices=["ollama", "openai_compat", "transformers"])
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--base-url", default=None,
                        help="e.g. http://127.0.0.1:8000/v1 for vLLM")
    parser.add_argument("--subject", default="sub-pt01")
    parser.add_argument("--task", default="ictal")
    parser.add_argument("--run", default="01")
    parser.add_argument("--start", type=float, default=50.0)
    parser.add_argument("--stop", type=float, default=110.0)
    parser.add_argument("--synthetic", action="store_true",
                        help="use simulated data instead (no download, but no SOZ score)")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=5,
                        help="repeats for run-to-run stability; the number a reviewer asks for")
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--quick", action="store_true",
                        help="one rung, one stopping rule, no falsification: a smoke test")
    parser.add_argument("--skip-falsify", action="store_true")
    parser.add_argument("--out", default=str(REPO / "artifacts" / "results" / "model_ladder"))
    args = parser.parse_args(argv)

    backend = make_backend(args.backend, model=args.model, base_url=args.base_url)
    _check_backend(backend)

    recording = _load_recording(args)
    labels = soz_labels(recording.subject, recording=recording)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    planners = {"scripted (control)": ScriptedPlanner(), f"{args.model}": backend}
    rungs = [Rung.S2] if args.quick else [Rung.S0, Rung.S1, Rung.S2, Rung.S3]
    rules = {"model": ModelJudged} if args.quick else {
        "budget": FixedBudget, "model": ModelJudged, "tied": TiedSetWidth}

    payload: dict = {
        "subject": recording.subject, "source": recording.source,
        "window_s": [recording.t_offset, recording.t_offset + recording.duration],
        "model": args.model, "backend": args.backend,
        "labels": labels.as_dict() if labels.usable else None,
        "runs": {}, "stability": {}, "falsification": {},
    }
    session = AnalysisSession(recording)
    results: dict = {}

    print(f"[ladder] {recording.subject}: {len(rungs)} rung(s) x "
          f"{len(planners)} planner(s)\n")
    for planner_name, planner in planners.items():
        for rung in rungs:
            for rule_name, rule_cls in (rules.items() if rung.replans
                                        else [("model", ModelJudged)]):
                label = f"{planner_name} / {rung.value}" + (
                    f" / {rule_name}" if rung.replans else "")
                if rung is Rung.S0 and planner_name != "scripted (control)":
                    continue        # S0 has no model in it; running it twice is waste
                started = time.perf_counter()
                try:
                    result = run_rung(session, rung, backend=planner,
                                      stop_rule=rule_cls(), max_steps=args.max_steps)
                except Exception as exc:
                    print(f"  {label:44s} FAILED: {type(exc).__name__}: {exc}")
                    payload["runs"][label] = {"error": f"{type(exc).__name__}: {exc}"}
                    continue
                elapsed = time.perf_counter() - started
                session.reset_memo()
                results[label] = result
                payload["runs"][label] = {
                    **result.as_dict(), "wall_clock_s": round(elapsed, 1)}
                print(f"  {label:44s} {result.cost['n_tool_calls']:3d} calls  "
                      f"{elapsed:6.1f} s  retested={result.n_retested}  "
                      f"top5={result.top_k(5)}")

    # -- run-to-run stability: the number that is trivially 1.0 for a script --
    if not args.quick:
        print(f"\n[ladder] run-to-run stability over {args.repeats} repeats")
        for planner_name, planner in planners.items():
            tops = []
            for _ in range(args.repeats):
                session.reset_memo()
                result = run_rung(session, Rung.S2, backend=planner,
                                  stop_rule=ModelJudged(), max_steps=args.max_steps)
                tops.append(tuple(result.top_k(5)))
            overlaps = [len(set(tops[0]) & set(t)) / max(1, len(tops[0])) for t in tops[1:]]
            info = {"repeats": args.repeats, "identical": len(set(tops)) == 1,
                    "distinct_answers": len(set(tops)),
                    "mean_top5_overlap": round(sum(overlaps) / len(overlaps), 3)
                    if overlaps else 1.0}
            payload["stability"][planner_name] = info
            print(f"  {planner_name:44s} identical={info['identical']}  "
                  f"distinct={info['distinct_answers']}  "
                  f"overlap={info['mean_top5_overlap']}")

    # -- falsification: the anonymised-names row is the point --
    if not (args.quick or args.skip_falsify):
        print("\n[ladder] falsification (the anonymised-names row is the one to read)")
        for planner_name, planner in planners.items():
            session.reset_memo()
            checks = run_falsification_suite(
                recording, labels if labels.usable else None,
                rung=Rung.S2, backend=planner, repeats=3)
            payload["falsification"][planner_name] = [c.as_dict() for c in checks]
            failed = [c.name for c in checks if c.passed is False]
            print(f"  {planner_name}")
            for check in checks:
                print(f"      [{check.verdict}] {check.name}")
            if failed:
                print(f"      >>> FAILED: {', '.join(failed)}")

    if labels.usable and results:
        payload["scores"] = compare_rungs(results, labels, k=5)

    (out_dir / "ladder.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / "RESULTS.md").write_text(_render(payload, args))
    print(f"\n[ladder] written to {out_dir}")
    print(f"[ladder] paste {out_dir / 'RESULTS.md'} into docs/ORCHESTRATION.md")
    return 0


def _render(payload: dict, args) -> str:
    """A table ready to paste into the docs, caveats included."""
    lines = [
        f"# S0-S3 ladder under `{payload['model']}`",
        "",
        f"Recording: `{payload['subject']}` ({payload['source']}), "
        f"window {payload['window_s'][0]:.0f}-{payload['window_s'][1]:.0f} s. "
        f"Backend: `{payload['backend']}`.",
        "",
        "Both columns were produced on the same machine and the same recording, so the",
        "scripted planner is a control rather than a quoted number.",
        "",
        "| configuration | tool calls | wall clock (s) | channels re-tested | top-5 |",
        "|---|---|---|---|---|",
    ]
    for label, run in payload["runs"].items():
        if "error" in run:
            lines.append(f"| {label} | — | — | — | FAILED: {run['error']} |")
            continue
        cost = run.get("cost", {})
        ranking = run.get("ranking") or []
        top = ", ".join(f"`{row['channel']}`" for row in ranking[:5]) or "—"
        lines.append(f"| {label} | {cost.get('n_tool_calls', '—')} | "
                     f"{run.get('wall_clock_s', '—')} | "
                     f"{run.get('n_channels_retested', '—')} | {top} |")

    if payload.get("stability"):
        lines += ["", "## Run-to-run stability", "",
                  "Trivially 1.0 for a deterministic planner; this is the number a reviewer",
                  "asks about first.", "",
                  "| planner | repeats | identical | distinct answers | mean top-5 overlap |",
                  "|---|---|---|---|---|"]
        for name, info in payload["stability"].items():
            lines.append(f"| {name} | {info['repeats']} | {info['identical']} | "
                         f"{info['distinct_answers']} | {info['mean_top5_overlap']} |")

    if payload.get("falsification"):
        lines += ["", "## Falsification", "",
                  "**Read the `anonymised channel names` row first.** It renames `AD1` to",
                  "`EA1` and checks the ranking does not move. A scripted planner cannot",
                  "fail it. A language model can, by ranking on what it knows about",
                  "electrode naming instead of on the evidence.", "",
                  "| planner | test | verdict |", "|---|---|---|"]
        for name, checks in payload["falsification"].items():
            for check in checks:
                lines.append(f"| {name} | {check['test']} | **{check['verdict']}** |")

    if payload.get("scores"):
        lines += ["", "## SOZ localization (permutation null)", "",
                  "| configuration | hits@5 | expected by chance | p |", "|---|---|---|---|"]
        for row in payload["scores"]:
            lines.append(f"| {row['rung']} | {row.get('n_hits_at_5')} | "
                         f"{row.get('expected_by_chance_at_5')} | "
                         f"{row.get('permutation_p_at_5')} |")
        lines += ["", "A p-value near 1 means no better than chance. On ictal data that is",
                  "the expected result; see `docs/EVALUATION.md` §6."]

    lines += ["", "---", "",
              f"Generated by `scripts/run_model_ladder.py` "
              f"(`--repeats {args.repeats}`, `--max-steps {args.max_steps}`).",
              "Record the model version, quantization, temperature and seed beside this",
              "table: a result that cannot be tied to a checkpoint cannot be replicated."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
