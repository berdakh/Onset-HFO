"""Command line interface.

    python -m onset_hfo.cli run --synthetic              # offline, labelled data
    python -m onset_hfo.cli run                          # the public example slice
    python -m onset_hfo.cli run --subject sub-pt01 --task ictal --run 01 \
                               --start 50 --stop 110 --figures
    python -m onset_hfo.cli evaluate --seeds 1 7 42      # measure the detectors
    python -m onset_hfo.cli runs --subject sub-pt01      # what else is in the archive

Every command prints where it wrote its results, because the agent
(``python -m onset_agent.cli --results <dir>``) reads exactly that directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from onset_hfo.benchmark import DEFAULT_THRESHOLDS
from onset_hfo.config import (
    DEFAULT_RUN,
    DEFAULT_SUBJECT,
    DEFAULT_TASK,
    DEFAULT_TSTART,
    DEFAULT_TSTOP,
    PIPELINE_VERSION,
    RESULTS_DIR,
    THRESHOLDS,
    PipelineConfig,
    ensure_dirs,
)
from onset_hfo.outcome import FULL_RUN_S


def _cmd_run(args: argparse.Namespace) -> int:
    from onset_hfo.pipeline import run_pipeline

    ensure_dirs()
    if args.synthetic:
        from onset_hfo.synthetic import make_synthetic_recording
        recording = make_synthetic_recording(seed=args.seed, duration_s=args.duration)
    else:
        from onset_hfo.datasets import fetch_slice
        recording = fetch_slice(subject=args.subject, task=args.task, run=args.run,
                                t_start=args.start, t_stop=args.stop)

    cfg = PipelineConfig()
    cfg.top_k = args.top_k
    if args.threshold:
        value = THRESHOLDS.get(args.threshold)
        if value is None:
            try:
                value = float(args.threshold)
            except ValueError:
                print(f"[onset-hfo] --threshold must be a number or one of: "
                      f"{', '.join(THRESHOLDS)}")
                return 2
        cfg.rms.threshold_sd = value
        cfg.line_length.threshold_sd = value
        print(f"[onset-hfo] detection threshold {value:g} SD ({args.threshold})")
    result = run_pipeline(recording, cfg, with_spikes=not args.no_spikes,
                          save_to=args.out or RESULTS_DIR)
    out_dir = Path(args.out or RESULTS_DIR) / \
        f"{recording.subject}_{recording.task}_run-{recording.run}"

    if args.figures:
        from onset_hfo.viz import save_all_figures
        written = save_all_figures(result, out_dir / "figures")
        print(f"[onset-hfo] {len(written)} figures in {out_dir / 'figures'}")

    if recording.ground_truth is not None:
        from onset_hfo.evaluate import evaluate_detections
        print("\n[onset-hfo] synthetic ground truth is available, so scores can be computed:")
        for name, events in result.events.items():
            print("   " + evaluate_detections(events, recording.ground_truth, detector=name).summary())
        if result.spikes:
            print("   " + evaluate_detections(result.spikes, recording.ground_truth,
                                              detector="spike", kind="spike").summary())

    print("\n" + result.report.to_markdown().split("## Findings")[0])
    print(f"[onset-hfo] full report: {out_dir / 'report.md'}")
    print(f"[onset-hfo] ask the agent about it:  "
          f"python -m onset_agent.cli --results {out_dir} "
          f"--question 'which channels have the highest ripple rate?'")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Measure the detectors against synthetic ground truth, over several seeds."""
    import pandas as pd

    from onset_hfo.evaluate import evaluate_detections, validation_benefit
    from onset_hfo.pipeline import run_pipeline
    from onset_hfo.synthetic import make_synthetic_recording

    rows = []
    for seed in args.seeds:
        rec = make_synthetic_recording(seed=seed, duration_s=args.duration, verbose=False)
        result = run_pipeline(rec, verbose=False)
        for name, events in result.events.items():
            res = evaluate_detections(events, rec.ground_truth, detector=name)
            rows.append({"seed": seed, **res.as_dict()})
        if result.spikes:
            res = evaluate_detections(result.spikes, rec.ground_truth, detector="spike", kind="spike")
            rows.append({"seed": seed, **res.as_dict()})
        if args.verbose:
            print(f"seed {seed}:")
            print(validation_benefit(result.events["rms"], rec.ground_truth).to_string(index=False))
    table = pd.DataFrame(rows)
    summary = table.groupby(["detector", "kind"])[["precision", "recall", "f1"]].agg(["mean", "std"])
    print("\n[onset-hfo] scores over seeds " + ", ".join(str(s) for s in args.seeds))
    print(summary.round(3).to_string())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"[onset-hfo] per-seed scores written to {args.out}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Score the detectors against expert HFO markings on a real cohort."""
    from onset_hfo.benchmark import benchmark_cohort

    ensure_dirs()
    result = benchmark_cohort(
        subjects=args.subjects, n_subjects=args.n_subjects, dataset=args.dataset,
        run=args.run, t_start=args.start, t_stop=args.stop,
        thresholds=tuple(args.thresholds), detectors=tuple(args.detectors),
        bands=tuple(args.bands))
    if not len(result.scores):
        print("[onset-hfo] nothing scored: no subject produced expert markings")
        return 1
    for band in args.bands:
        table = result.summary(band)
        if not len(table):
            continue
        print(f"\n[onset-hfo] {band} band, cohort means over "
              f"{result.subjects['subject'].nunique()} subjects "
              f"({args.stop - args.start:g} s each):")
        print(table.to_string(index=False))
    print("\n[onset-hfo] operating point the data prefers:")
    print(f"   by F1:              {result.best_threshold(args.bands[0], 'f1')}")
    print(f"   by rank agreement:  {result.best_threshold(args.bands[0], 'rank')}")
    result.save(args.out)
    return 0


def _cmd_outcome(args: argparse.Namespace) -> int:
    """Ask whether the HFO map points at the tissue whose removal cured the patient."""
    from onset_hfo.outcome import outcome_study

    ensure_dirs()
    threshold = args.threshold if args.threshold is not None else None
    result = outcome_study(
        subjects=args.subjects, n_subjects=args.n_subjects, dataset=args.dataset,
        run=args.run, t_start=args.start, t_stop=args.stop,
        detector=args.detector, threshold_sd=threshold, bands=tuple(args.bands),
        drop_eloquent=not args.keep_eloquent)
    if not len(result.subjects):
        print("[onset-hfo] nothing measured: no subject had both a resected zone and a recording")
        return 1
    for metric in ("share_in_rz", "top_channel_resected", "top3_resected"):
        print(f"\n[onset-hfo] {metric}, seizure-free vs recurrence:")
        print(result.summary(metric).to_string(index=False))
    metric, band = result.PRIMARY
    if band in args.bands:
        print(f"\n[onset-hfo] pre-specified comparison ({metric}, {band} band):")
        print(f"   {result.verdict()}")
    for other_band in args.bands:
        for other_metric in ("share_in_rz", "top_channel_resected"):
            if (other_metric, other_band) == result.PRIMARY:
                continue
            print(f"\n[onset-hfo] {other_metric}, {other_band} band: "
                  f"{result.verdict(metric=other_metric, band=other_band)}")
    result.save(args.out)
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    from onset_hfo.datasets import list_runs

    table = list_runs(args.subject)
    print(table.to_string(index=False) if len(table) else "no runs found")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from onset_hfo.store import ResultStore

    store = ResultStore(args.results)
    print(json.dumps(store.metadata(), indent=2))
    print((Path(args.results) / "report.md").read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="onset-hfo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"onset-hfo {PIPELINE_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the detection pipeline on one recording")
    run.add_argument("--synthetic", action="store_true",
                     help="use the labelled synthetic recording instead of the public archive")
    run.add_argument("--subject", default=DEFAULT_SUBJECT)
    run.add_argument("--task", default=DEFAULT_TASK)
    run.add_argument("--run", default=DEFAULT_RUN)
    run.add_argument("--start", type=float, default=DEFAULT_TSTART,
                     help="start of the slice, seconds into the recording")
    run.add_argument("--stop", type=float, default=DEFAULT_TSTOP)
    run.add_argument("--duration", type=float, default=120.0,
                     help="length of the synthetic recording, seconds")
    run.add_argument("--seed", type=int, default=7, help="synthetic recording seed")
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--threshold", default=None, metavar="SD|PRESET",
                     help="detection threshold in robust SDs, or a measured preset: "
                          + ", ".join(f"{k} ({v:g})" for k, v in THRESHOLDS.items()))
    run.add_argument("--no-spikes", action="store_true", help="skip the discharge detector")
    run.add_argument("--figures", action="store_true", help="also render the standard figures")
    run.add_argument("--out", default=None, help=f"output directory (default: {RESULTS_DIR})")
    run.set_defaults(func=_cmd_run)

    ev = sub.add_parser("evaluate", help="score the detectors against synthetic ground truth")
    ev.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42])
    ev.add_argument("--duration", type=float, default=120.0)
    ev.add_argument("--out", default=None, help="write per-seed scores to this CSV")
    ev.add_argument("--verbose", action="store_true")
    ev.set_defaults(func=_cmd_evaluate)

    bench = sub.add_parser(
        "benchmark",
        help="score the detectors against expert HFO markings (ds003498)")
    bench.add_argument("--dataset", default="ds003498",
                       help="dataset with expert markings (default: ds003498)")
    bench.add_argument("--subjects", nargs="+", default=None,
                       help="subject labels; default: every subject in the dataset")
    bench.add_argument("--n-subjects", type=int, default=None,
                       help="use only the first N subjects (a quick look)")
    bench.add_argument("--run", default="01")
    bench.add_argument("--start", type=float, default=0.0)
    bench.add_argument("--stop", type=float, default=60.0,
                       help="seconds of each recording to score (default 60)")
    bench.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
                       help="detection thresholds to sweep, in robust SDs")
    bench.add_argument("--detectors", nargs="+", default=["rms", "line_length"])
    bench.add_argument("--bands", nargs="+", default=["ripple"],
                       choices=["ripple", "fast_ripple"])
    bench.add_argument("--out", default=None, help=f"output directory (default: {RESULTS_DIR})")
    bench.set_defaults(func=_cmd_benchmark)

    out = sub.add_parser(
        "outcome",
        help="test the HFO map against post-surgical seizure outcome (ds003498)")
    out.add_argument("--dataset", default="ds003498",
                     help="dataset with a resected zone and outcomes (default: ds003498)")
    out.add_argument("--subjects", nargs="+", default=None)
    out.add_argument("--n-subjects", type=int, default=None)
    out.add_argument("--run", default="01")
    out.add_argument("--start", type=float, default=0.0)
    out.add_argument("--stop", type=float, default=FULL_RUN_S,
                     help="seconds of each recording to use. The default is the whole "
                          "run; 60 gives a different answer, which is itself a result "
                          "(see docs/OUTCOME.md)")
    out.add_argument("--detector", default="rms", choices=["rms", "line_length"])
    out.add_argument("--threshold", type=float, default=None,
                     help="one threshold for every band; default is the measured "
                          "per-band operating point (2.0 SD ripples, 5.0 SD fast ripples)")
    out.add_argument("--bands", nargs="+", default=["ripple", "fast_ripple"],
                     choices=["ripple", "fast_ripple"])
    out.add_argument("--keep-eloquent", action="store_true",
                     help="keep contacts the source study excluded for evoked "
                          "motor or language responses (default: drop them)")
    out.add_argument("--out", default=None, help=f"output directory (default: {RESULTS_DIR})")
    out.set_defaults(func=_cmd_outcome)

    runs = sub.add_parser("runs", help="list the runs available for a subject in the archive")
    runs.add_argument("--subject", default=DEFAULT_SUBJECT)
    runs.set_defaults(func=_cmd_runs)

    rep = sub.add_parser("report", help="print a saved report")
    rep.add_argument("results", help="a results directory written by 'run'")
    rep.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
