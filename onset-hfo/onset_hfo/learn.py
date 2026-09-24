"""Fit and evaluate the per-contact SOZ model on a cohort.

    # 1. build the cohort feature table (downloads ~24 MB per subject, resumable)
    python -m onset_hfo.learn cohort

    # 2. the headline table: three protocols against the untrained baselines
    python -m onset_hfo.learn evaluate

    # 3. does it matter WHICH contacts the clinician labels first?
    python -m onset_hfo.learn acquire

    # 4. calibration, conformal coverage, and the exchangeability stress test
    python -m onset_hfo.learn uncertainty

    # 5. fit a deployable model, holding out the subject you will demonstrate on
    python -m onset_hfo.learn fit --holdout sub-pt01 --out artifacts/models/soz.pkl

The evaluation always scores the untrained rate baselines beside the models.
"Better than chance" is not the bar: the pipeline already ranks channels by
ripple rate for free, and a model that beats chance while losing to the rate
it was built from has established nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from onset_hfo.config import RESULTS_DIR

DEFAULT_COHORT = RESULTS_DIR / "cohort"
#: A cohort table committed to the repository, so the modelling half runs with
#: no download at all. Used when nothing has been built locally.
SHIPPED_COHORT = Path(__file__).resolve().parent.parent / "data" / "cohort"


def _load_features(path: str | Path) -> pd.DataFrame:
    """Find a feature table: the one given, the one built locally, or the shipped one."""
    path = Path(path)
    for candidate in ([path] if path.suffix in (".csv", ".gz")
                      else [path / "features.csv", path / "features.csv.gz",
                            SHIPPED_COHORT / "features.csv.gz"]):
        if candidate.exists():
            if candidate.parent == SHIPPED_COHORT:
                print(f"[learn] using the cohort table shipped with the repository "
                      f"({candidate}). Rebuild it with: python -m onset_hfo.learn cohort")
            return pd.read_csv(candidate)
    raise SystemExit(
        f"no feature table at {path}, and none shipped. Build one with:\n"
        f"    python -m onset_hfo.learn cohort")


def _cmd_cohort(args) -> int:
    from onset_hfo.batch import CohortSpec, discover_cohort, run_cohort

    spec = CohortSpec(task=args.task, pre_s=args.pre, post_s=args.post)
    subjects = args.subjects.split(",") if args.subjects else None
    plan = discover_cohort(subjects, spec=spec, verbose=True)
    if args.dry_run:
        columns = ["subject", "acq", "run", "size_mb", "seizure_onset_s", "marker_kind",
                   "start_s", "stop_s", "n_soz_contacts", "site", "usable", "reason"]
        print(plan[columns].to_string(index=False))
        return 0
    run_cohort(plan, spec=spec, out_dir=args.out, limit=args.limit, refresh=args.refresh)
    return 0


def _cmd_evaluate(args) -> int:
    from onset_hfo.models import MODELS, baseline_scores, evaluate

    features = _load_features(args.cohort)
    print(f"[learn] {features['subject'].nunique()} subjects, {len(features)} channels, "
          f"{int(features['is_soz'].sum())} labelled SOZ "
          f"({features['is_soz'].mean():.1%} prevalence)")
    by_site = features.groupby("site").agg(subjects=("subject", "nunique"),
                                           channels=("channel", "size"),
                                           soz=("is_soz", "sum"))
    print(by_site.to_string(), "\n")

    rows = []
    for column in ["rms_rate_per_min", "ll_rate_per_min", "spike_rate_per_min"]:
        rows.append(baseline_scores(features, column, args.normalisation).summary())
    for protocol in ["within_subject", "lopo", "loso"]:
        for name in (MODELS if args.model == "all" else [args.model]):
            try:
                rows.append(evaluate(features, name, args.normalisation, protocol).summary())
            except Exception as exc:
                print(f"[learn] {protocol}/{name} failed: {type(exc).__name__}: {exc}")
    table = pd.DataFrame(rows)
    columns = [c for c in ["protocol", "model", "normalisation", "n_patients", "prevalence",
                           "auprc", "auprc_lift_over_prevalence", "auroc", "precision_at_5",
                           "auprc_per_patient_median"] if c in table.columns]
    pd.set_option("display.width", 200)
    print(table[columns].to_string(index=False))

    out = Path(args.out or (Path(args.cohort) / "evaluation.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n[learn] written to {out}")
    print("[learn] read the gap, not the best row: within_subject is a ceiling that needs "
          "the answer to compute, lopo is what a new patient would get, and the untrained "
          "rate rows are what the pipeline already gives you for nothing.")
    return 0


def _cmd_uncertainty(args) -> int:
    from onset_hfo.models import evaluate
    from onset_hfo.uncertainty import (
        conformal_coverage_report,
        exchangeability_stress_test,
        expected_calibration_error,
        reliability_table,
    )

    features = _load_features(args.cohort)
    result = evaluate(features, args.model, args.normalisation, "lopo")
    print(f"[learn] {args.model} / {args.normalisation} / lopo")
    print(f"[learn] expected calibration error, uncalibrated: "
          f"{expected_calibration_error(result.scores, result.truth):.4f}")
    print("\nreliability (predicted vs observed):")
    print(reliability_table(result.scores, result.truth).to_string(index=False))

    report = conformal_coverage_report(result, alpha=args.alpha, seed=args.seed)
    print(f"\nsplit conformal at alpha={args.alpha}:")
    print(json.dumps({k: v for k, v in report.items() if k != "candidate_sets"}, indent=2))
    if report.get("candidate_sets"):
        print("\nper-patient candidate sets (channels that cannot be ruled out):")
        print(pd.DataFrame(report["candidate_sets"]).to_string(index=False))

    sites = features[features["is_soz"].notna()]
    positives = sites.groupby("subject")["is_soz"].transform("sum")
    sites = sites[positives > 0]["site"].astype(str).to_numpy()
    stress = exchangeability_stress_test(result, alpha=args.alpha, by="site", sites=sites)
    print("\nexchangeability stress test (calibrate on other sites, test on one):")
    print(json.dumps(stress, indent=2))
    if stress.get("available"):
        print("\n[learn] coverage below nominal means the guarantee did not transfer. "
              "That is the finding, not a bug: distribution shift breaks the uncertainty "
              "estimate that was supposed to protect against distribution shift.")

    out = Path(args.out or (Path(args.cohort) / "uncertainty.json"))
    out.write_text(json.dumps({"conformal": report, "stress_test": stress},
                              indent=2, default=str))
    print(f"\n[learn] written to {out}")
    return 0


def _cmd_personalize(args) -> int:
    from onset_hfo.models import evaluate, label_budget_curve

    features = _load_features(args.cohort)
    budgets = tuple(int(b) for b in args.budgets.split(","))
    curve = label_budget_curve(features, budgets=budgets, model=args.model,
                               normalisation=args.normalisation,
                               n_repeats=args.repeats, seed=args.seed)
    ceiling = evaluate(features, args.model, args.normalisation,
                       "within_subject").summary()

    columns = [c for c in ["n_labels", "labelled_fraction", "auprc", "gap_closed",
                           "auprc_lift_over_prevalence", "auroc", "precision_at_5",
                           "auprc_per_patient_median"] if c in curve.columns]
    pd.set_option("display.width", 200)
    print(f"[learn] {args.model} / {args.normalisation}, "
          f"{args.repeats} label draws per patient\n")
    print(curve[columns].to_string(index=False))
    print(f"\nwithin-subject ceiling: auprc={ceiling['auprc']}  "
          f"precision_at_5={ceiling['precision_at_5']}  (needs every label)")
    print("\n[learn] n_labels=0 IS leave-one-patient-out, so the two ends of this curve "
          "are comparable by construction. labelled_fraction is what you are actually "
          "asking a clinician for: read it before believing a budget is modest.")
    out = Path(args.out or (Path(args.cohort) / "label_budget.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out, index=False)
    print(f"[learn] written to {out}")
    return 0


def _cmd_acquire(args) -> int:
    from onset_hfo.models import ACQUISITION, active_learning_comparison

    features = _load_features(args.cohort)
    budgets = tuple(int(b) for b in args.budgets.split(","))
    seeds = tuple(int(s) for s in str(args.seeds).split(","))
    table = active_learning_comparison(features, budgets=budgets, model=args.model,
                                       normalisation=args.normalisation,
                                       n_repeats=args.repeats, seeds=seeds)
    print("strategies:")
    for name, description in ACQUISITION.items():
        print(f"  {name:12s} {description}")
    print("\nEvery strategy is scored on the same held-out contacts: each patient's")
    print("contacts are split once into an evaluation pool and a labelling pool, from")
    print("the seed alone and never from the strategy. Without that the comparison")
    print("would be four different exams.\n")

    columns = [c for c in ["n_labels", "strategy", "auprc", "lift_over_random", "lift_sd",
                           "n_seeds_beating_random", "n_seeds", "auroc", "precision_at_5"]
               if c in table.columns]
    pd.set_option("display.width", 200)
    print(table[columns].to_string(index=False))

    out = Path(args.out or (Path(args.cohort) / "active_learning.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n[learn] written to {out}")
    print("\n[learn] lift_over_random is the column this exists for, and "
          "n_seeds_beating_random is how much to trust it: a strategy that wins on three "
          "seeds out of five has not been shown to win.")
    return 0


def _cmd_fit(args) -> int:
    from onset_hfo.models import fit_soz_model

    features = _load_features(args.cohort)
    holdout = args.holdout.split(",") if args.holdout else None
    model = fit_soz_model(features, model=args.model, normalisation=args.normalisation,
                          alpha=args.alpha, seed=args.seed, holdout=holdout)
    path = model.save(args.out)
    print(json.dumps(model.describe(), indent=2))
    print(f"\n[learn] saved to {path}")
    print("[learn] use it as an agent tool:")
    print("    from onset_agent import AnalysisSession, Rung, run_rung")
    print("    from onset_agent.planner import ConformalWidth")
    print(f"    session = AnalysisSession(recording, soz_model=SozModel.load('{path}'))")
    print("    run_rung(session, Rung.S2, stop_rule=ConformalWidth(max_width=5))")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="onset-learn", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    cohort = sub.add_parser("cohort", help="build the per-channel feature table")
    cohort.add_argument("--subjects", default=None, help="comma-separated; default is all")
    cohort.add_argument("--task", default="ictal")
    cohort.add_argument("--pre", type=float, default=25.0, help="seconds before onset")
    cohort.add_argument("--post", type=float, default=35.0, help="seconds after onset")
    cohort.add_argument("--limit", type=int, default=None)
    cohort.add_argument("--refresh", action="store_true", help="re-run cached subjects")
    cohort.add_argument("--dry-run", action="store_true", help="print the plan, download nothing")
    cohort.add_argument("--out", default=str(DEFAULT_COHORT))
    cohort.set_defaults(func=_cmd_cohort)

    for name, func, help_text in [
            ("evaluate", _cmd_evaluate, "score the models and the untrained baselines"),
            ("personalize", _cmd_personalize,
             "how much does letting a clinician label k contacts buy?"),
            ("acquire", _cmd_acquire,
             "does it matter WHICH contacts the clinician labels?"),
            ("uncertainty", _cmd_uncertainty, "calibration, conformal coverage, stress test"),
            ("fit", _cmd_fit, "fit a deployable model with a conformal threshold")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--cohort", default=str(DEFAULT_COHORT))
        p.add_argument("--model",
                       default={"evaluate": "all", "personalize": "logistic",
                                "acquire": "logistic"}.get(name, "gradient_boosting"))
        p.add_argument("--normalisation", default="raw", choices=["raw", "z", "rank"])
        p.add_argument("--out", default=None)
        if name != "evaluate":
            p.add_argument("--seed", type=int, default=0)
        if name in ("uncertainty", "fit"):
            p.add_argument("--alpha", type=float, default=0.1)
        if name == "acquire":
            p.add_argument("--budgets", default="2,5,10",
                           help="comma-separated label budgets")
            p.add_argument("--seeds", default="0,1,2,3,4",
                           help="comma-separated seeds; one seed is not a result, because "
                                "the evaluation split moves the lift by more than the "
                                "difference between strategies")
            p.add_argument("--repeats", type=int, default=3,
                           help="repeats; only the random strategy is stochastic, but the "
                                "evaluation split varies with the seed")
        if name == "personalize":
            p.add_argument("--budgets", default="0,1,2,5,10,20,40",
                           help="comma-separated label budgets; 0 is leave-one-patient-out")
            p.add_argument("--repeats", type=int, default=5,
                           help="label draws per patient (which contacts a clinician "
                                "happens to label is a lottery)")
        if name == "fit":
            p.add_argument("--holdout", default=None,
                           help="comma-separated subjects to exclude from BOTH fitting and "
                                "calibration, so the model can be demonstrated on them")
            p.set_defaults(out="artifacts/models/soz.pkl")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
