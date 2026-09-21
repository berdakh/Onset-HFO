"""Run the S0-S3 ablation ladder on one recording and score it.

Every rung calls the same analyzers on the same recording; only the amount of
model control changes. That is what makes any difference between the rows
attributable to orchestration rather than to signal processing.

    # offline, no download, no model -- the whole ladder in ~20 s
    python -m onset_agent.orchestrate --synthetic

    # the public recording, scored against the archive's clinician SOZ labels
    python -m onset_agent.orchestrate --subject sub-pt01 --task ictal --run 01 \
        --start 50 --stop 110 --score

    # no labels yet? run the scoring path anyway on stand-ins, and emit the CSV
    # a clinical centre fills in to replace them
    python -m onset_agent.orchestrate --synthetic --score --labels placeholder \
        --write-label-template labels/our-centre.csv
    python -m onset_agent.orchestrate --synthetic --score \
        --labels-csv labels/our-centre.csv

    # with a real open-weight model driving the planner
    ollama pull qwen2.5:7b-instruct && ollama serve &
    python -m onset_agent.orchestrate --synthetic --backend ollama

    # compare the three stopping rules on the re-planning rung
    python -m onset_agent.orchestrate --synthetic --rungs S2 --stop-rule budget,model,tied

Outputs, per rung, into ``--out``: ``evidence.json`` (the audit trail -- every
tool call, its parameters, its result, its run id), ``result.json`` (the
ranking, the report, the verification) and a combined ``ladder.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from onset_agent.analysis import AnalysisSession
from onset_agent.backends import make_backend
from onset_agent.planner import FixedBudget, ModelJudged, Rung, TiedSetWidth, run_rung
from onset_agent.scoring import compare_rungs, score_result
from onset_hfo.cohort import (
    SozLabels,
    placeholder_labels,
    soz_labels,
    write_label_template,
)
from onset_hfo.config import RESULTS_DIR

STOP_RULES = {"budget": FixedBudget, "model": ModelJudged, "tied": TiedSetWidth}


def _load_recording(args):
    if args.synthetic:
        from onset_hfo.synthetic import make_synthetic_recording
        return make_synthetic_recording(seed=args.seed, duration_s=args.duration,
                                        verbose=args.verbose)
    from onset_hfo.datasets import fetch_slice
    return fetch_slice(args.subject, args.task, args.run,
                       t_start=args.start, t_stop=args.stop)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="onset-orchestrate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    data = p.add_argument_group("recording")
    data.add_argument("--synthetic", action="store_true",
                      help="use labelled synthetic data (offline, no download)")
    data.add_argument("--seed", type=int, default=7, help="synthetic seed")
    data.add_argument("--duration", type=float, default=60.0, help="synthetic duration (s)")
    data.add_argument("--subject", default="sub-pt01")
    data.add_argument("--task", default="ictal")
    data.add_argument("--run", default="01")
    data.add_argument("--start", type=float, default=50.0)
    data.add_argument("--stop", type=float, default=110.0)

    ladder = p.add_argument_group("ladder")
    ladder.add_argument("--rungs", default="S0,S1,S2,S3",
                        help="which rungs to run, comma separated")
    ladder.add_argument("--stop-rule", default="model",
                        help="one or more of budget,model,tied (comma separated); each is "
                             "run against every re-planning rung")
    ladder.add_argument("--budget", type=int, default=8,
                        help="tool-call budget for the fixed-budget rule")
    ladder.add_argument("--max-steps", type=int, default=12)
    ladder.add_argument("--repeats", type=int, default=1,
                        help="run each configuration N times and report top-5 stability")

    model = p.add_argument_group("model")
    model.add_argument("--backend", default="scripted",
                       choices=["scripted", "ollama", "openai_compat", "transformers"],
                       help="'scripted' is a deterministic planner, not a language model")
    model.add_argument("--model", default=None)
    model.add_argument("--base-url", default=None)

    out = p.add_argument_group("output")
    out.add_argument("--falsify", action="store_true",
                     help="try to break the system: anonymised names, a shuffled "
                          "name-to-signal mapping, the leading channel removed, a "
                          "recording with no pathology, and run-to-run stability")
    out.add_argument("--score", action="store_true",
                     help="score each ranking against the clinician SOZ labels")
    out.add_argument("--labels-csv", default=None,
                     help="your own label file, instead of the archive's (see cohort.py)")
    out.add_argument("--labels", default="auto", choices=["auto", "placeholder", "none"],
                     help="'auto' uses the best real labels available and scores nothing "
                          "if there are none; 'placeholder' falls back to generated "
                          "stand-in labels when there are none, so the scoring path runs "
                          "on un-curated data (it never overrides real labels); 'none' "
                          "ignores whatever labels exist")
    out.add_argument("--placeholder-contacts", type=int, default=6,
                     help="how many contacts a placeholder label set names")
    out.add_argument("--write-label-template", default=None, metavar="PATH",
                     help="write the CSV a clinical centre fills in, prefilled with what "
                          "is known, then continue")
    out.add_argument("--out", default=None, help="directory for the audit trail")
    out.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    recording = _load_recording(args)
    backend = None if args.backend == "scripted" else make_backend(
        args.backend, model=args.model, base_url=args.base_url)
    if args.backend == "scripted":
        from onset_agent.planner import ScriptedPlanner
        backend = ScriptedPlanner()

    rungs = [Rung(r.strip().upper()) for r in args.rungs.split(",") if r.strip()]
    rules = [r.strip() for r in args.stop_rule.split(",") if r.strip()]
    for rule in rules:
        if rule not in STOP_RULES:
            p.error(f"unknown stop rule {rule!r}; choose from {', '.join(STOP_RULES)}")

    out_dir = Path(args.out or (RESULTS_DIR / f"{recording.subject}_ladder"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[orchestrate] {recording.subject} ({recording.source}), "
          f"{recording.duration:.0f} s")
    print(f"[orchestrate] planner: {backend.describe()}")
    if not getattr(backend, "is_language_model", False):
        print("[orchestrate] NOTE: the scripted planner is deterministic and is not a "
              "language model. Use --backend ollama for the real thing.")

    session = AnalysisSession(recording, verbose=args.verbose)
    results, rows, stability = {}, [], {}

    for rung in rungs:
        for rule_name in (rules if rung.replans else rules[:1]):
            label = rung.value if not rung.replans else f"{rung.value}/{rule_name}"
            tops = []
            result = None
            for _repeat in range(max(1, args.repeats)):
                session.reset_memo()   # each rung pays for its own detections
                rule = STOP_RULES[rule_name]()
                if isinstance(rule, FixedBudget):
                    rule.k = args.budget
                result = run_rung(session, rung, backend=backend, stop_rule=rule,
                                  max_steps=args.max_steps, verbose=args.verbose)
                tops.append(tuple(result.top_k(5)))
            results[label] = result
            if args.repeats > 1:
                overlap = [len(set(tops[0]) & set(t)) / max(1, len(tops[0])) for t in tops[1:]]
                stability[label] = {
                    "repeats": args.repeats,
                    "top5_overlap_with_first": [round(o, 3) for o in overlap],
                    "identical": len(set(tops)) == 1}
            cost = result.cost
            print(f"  {label:12s} calls={cost.get('n_tool_calls', 0):3d} "
                  f"({cost.get('runtime_s', 0):5.1f} s)  "
                  f"retested={result.n_retested:2d}  top5={result.top_k(5)}")
            if result.struck:
                print(f"               struck {len(result.struck)} unsupported claim(s)")
            safe = label.replace("/", "_")
            result.store.save(out_dir / safe / "evidence.json")
            (out_dir / safe / "result.json").write_text(
                json.dumps(result.as_dict(), indent=2, default=str))

    payload: dict = {"subject": recording.subject, "source": recording.source,
                     "rungs": {k: v.as_dict() for k, v in results.items()}}
    if stability:
        payload["stability"] = stability
        print("\n[orchestrate] run-to-run stability (top-5 overlap across repeats):")
        for label, info in stability.items():
            print(f"  {label:12s} identical={info['identical']} "
                  f"overlap={info['top5_overlap_with_first']}")

    if args.score or args.write_label_template:
        labels = soz_labels(recording.subject, csv=args.labels_csv, recording=recording)
        if args.labels == "placeholder" and not labels.usable:
            labels = placeholder_labels(recording.subject, session.prepared.ch_names,
                                        n_contacts=args.placeholder_contacts)
        elif args.labels == "placeholder":
            print(f"\n[orchestrate] --labels placeholder ignored: real labels exist for "
                  f"{recording.subject} (source={labels.source}). A stand-in never "
                  "overrides a real label.")
        elif args.labels == "none":
            labels = SozLabels(subject=recording.subject, source="none")

        if args.write_label_template:
            path = write_label_template(recording.subject, session.prepared.ch_names,
                                        args.write_label_template, labels=labels)
            print(f"\n[orchestrate] label template written to {path}")
            print("[orchestrate] fill in soz_contacts and pass it back with --labels-csv; "
                  "nothing else changes.")

    if args.score:
        print(f"\n[orchestrate] labels: {len(labels.soz_contacts)} SOZ contact(s), "
              f"source={labels.source}, engel={labels.engel}, "
              f"seizure_free={labels.seizure_free}, site={labels.site or 'n/a'}")
        if labels.warning:
            print(f"[orchestrate] !! {labels.warning}")
        if not labels.usable:
            print("[orchestrate] no labelled contacts for this subject; nothing to score.")
            print("[orchestrate] re-run with --labels placeholder to exercise the scoring "
                  "path anyway, or --write-label-template PATH to produce the CSV a centre "
                  "fills in.")
        else:
            rows = compare_rungs(results, labels, k=5)
            payload["labels"] = labels.as_dict()
            payload["scores"] = rows
            payload["per_rung_score"] = {k: score_result(v, labels).as_dict()
                                         for k, v in results.items()}
            print(f"\n{'rung':12s} {'calls':>5s} {'retested':>8s} {'hits@5':>6s} "
                  f"{'chance':>7s} {'p':>7s}")
            for row in rows:
                print(f"{row['rung']:12s} {row['n_tool_calls']:5d} "
                      f"{row['n_channels_retested']:8d} "
                      f"{str(row['n_hits_at_5']):>6s} "
                      f"{str(row['expected_by_chance_at_5']):>7s} "
                      f"{str(row['permutation_p_at_5']):>7s}")
            if labels.source == "placeholder":
                print("\n[orchestrate] !! the table above was scored against generated "
                      "labels and means nothing. It shows the scoring path runs.")
            elif labels.source == "synthetic_truth":
                print("\n[orchestrate] the table above was scored against the contacts the "
                      "simulator implanted events on. Those labels are exactly right, so a "
                      "result here says the scorer works -- it says nothing about a brain.")
            else:
                print("\n[orchestrate] a p-value near 1 means no better than chance. On "
                      "ictal data that is the expected result; see docs/EVALUATION.md.")

    if args.falsify:
        from onset_agent.falsify import run_falsification_suite

        print("\n[orchestrate] falsification: trying to make the system confidently wrong")
        labels = soz_labels(recording.subject, csv=args.labels_csv, recording=recording)
        checks = run_falsification_suite(recording, labels if labels.usable else None,
                                         rung=Rung.S2, backend=backend, repeats=3)
        for check in checks:
            print(f"  [{check.verdict}] {check.name}")
            print(f"         expected: {check.expectation}")
            print(f"         measured: {check.reading}")
        payload["falsification"] = [c.as_dict() for c in checks]
        failed = [c.name for c in checks if c.passed is False]
        print(f"\n[orchestrate] {len(checks) - len(failed)}/{len(checks)} passed"
              + (f"; FAILED: {', '.join(failed)}" if failed else ""))

    (out_dir / "ladder.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[orchestrate] audit trail written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
