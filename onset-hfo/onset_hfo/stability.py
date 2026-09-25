"""Is the channel ranking stable enough to carry a clinical claim?

This module exists because of a mistake. :mod:`onset_hfo.outcome` first ran on
the first 60 seconds of each recording and found that the busiest fast-ripple
channel was inside the resection in 12 of 13 seizure-free patients and 2 of 7
recurrences -- AUC 0.82, p = 0.007. Re-run on the whole 300-second run, the
same code and the same pre-specified metric gave AUC 0.71, p = 0.12. Two
patients moved, and the conclusion moved with them.

A number that changes that much between minute one and minutes one-to-five is
not yet a measurement, and no amount of extra documentation around it fixes
that. So this module asks the question directly, in two arms that answer
different halves of it:

**Growing windows** (0-30 s, 0-60 s, ... 0-300 s) ask *does the estimate
settle as you add data?* If the curve flattens, the full-run number is a
measurement and the 60-second one was undersampled. If it is still moving at
300 s, the recording is not long enough and the archive's other runs have to
be brought in.

**Disjoint windows** (0-60 s, 60-120 s, ... 240-300 s) ask *does it depend on
which minute you look at?* This is the arm that makes the first one readable.
A growing-window curve that wanders could be convergence or could be drift;
five equally long, non-overlapping windows disagreeing with each other says it
is drift, and says how much.

The second arm also yields the most direct number in the whole exercise:
:meth:`StabilityResult.top_channel_stability` reports, per patient, how often
the *same* channel comes out busiest. `top_channel_resected` is an argmax over
6-65 channels whose Poisson rate intervals overlap heavily, and if that argmax
lands on a different channel every minute then the outcome metric is measuring
noise, whatever its AUC happens to be.

Nothing here re-tunes anything or introduces a new statistic. It runs the
existing study at different windows and reports the spread.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from onset_hfo.config import PIPELINE_VERSION, RESULTS_DIR
from onset_hfo.outcome import FULL_RUN_S, OutcomeResult, outcome_study

__all__ = [
    "GROWING_WINDOWS",
    "DISJOINT_LENGTH",
    "DEFAULT_RUNS_PER_SUBJECT",
    "StabilityResult",
    "stability_study",
    "across_runs",
    "pool_runs",
    "list_runs_per_subject",
    "plot_stability",
]

#: How many runs of each subject the across-runs study reads by default.
#:
#: ``ds003498`` holds **385 runs** across its 20 subjects -- 1 to 39 each, not
#: the 1-6 the ``nights`` column suggests, because a night contributes several
#: five-minute interictal segments. All of them is about 46 GB, which is not a
#: study anyone reruns, so the default samples the first few and
#: ``prune_cache`` keeps the footprint bounded.
DEFAULT_RUNS_PER_SUBJECT = 5

#: Window ends for the growing arm, in seconds from the start of the run.
#: 30 is below anything the clinical literature would accept and is included
#: precisely so the curve has a visibly bad end to come up from.
GROWING_WINDOWS: tuple[float, ...] = (30.0, 60.0, 120.0, 180.0, 240.0, FULL_RUN_S)

#: Length of each window in the disjoint arm. 60 s because that is the window
#: the original analysis used, so this arm answers "how lucky was that minute?"
DISJOINT_LENGTH: float = 60.0


@dataclass
class StabilityResult:
    """Every window's group statistics, per-patient values and top channels."""

    groups: pd.DataFrame      #: one row per window x arm x source x scope x band x metric
    subjects: pd.DataFrame    #: per-patient metric values, per window
    channels: pd.DataFrame    #: the busiest channel per patient, per window
    dataset: str = "ds003498"
    detector: str = "rms"
    pipeline_version: str = PIPELINE_VERSION
    primary: tuple[str, str] = ("top_channel_resected", "fast_ripple")
    windows: list[dict] = field(default_factory=list)
    #: Subjects analysed in every window, or the discrepancy if not. A curve
    #: computed over different cohorts at different points is not a curve, and
    #: the most likely cause is undramatic: one dropped download.
    cohort: dict = field(default_factory=dict)

    # -- views -------------------------------------------------------------

    def curve(self, metric: str | None = None, band: str | None = None,
              scope: str = "reviewed") -> pd.DataFrame:
        """The growing-window arm for one comparison, ordered by window length."""
        metric = metric or self.primary[0]
        band = band or self.primary[1]
        table = self.groups.query(
            "arm == 'growing' and metric == @metric and band == @band and scope == @scope")
        columns = ["window_s", "source", "n_seizure_free", "n_recurrence",
                   "mean_seizure_free", "mean_recurrence", "auc", "auc_lo", "auc_hi",
                   "p_permutation"]
        return table[columns].sort_values(["source", "window_s"]).reset_index(drop=True)

    def spread(self, metric: str | None = None, band: str | None = None,
               scope: str = "reviewed", arm: str = "disjoint") -> pd.DataFrame:
        """How far the disjoint windows of equal length disagree.

        ``auc_range`` is the plain statement of the problem: the difference
        between the most and least favourable minute of the same recording,
        analysed identically.
        """
        metric = metric or self.primary[0]
        band = band or self.primary[1]
        table = self.groups.query(
            "arm == @arm and metric == @metric and band == @band and scope == @scope")
        if table.empty:
            return table
        return (table.groupby("source")
                .agg(n_units=("auc", "size"), auc_min=("auc", "min"),
                     auc_max=("auc", "max"), auc_median=("auc", "median"),
                     auc_sd=("auc", "std"), p_min=("p_permutation", "min"),
                     p_max=("p_permutation", "max"))
                .round(3).reset_index()
                .assign(auc_range=lambda d: (d["auc_max"] - d["auc_min"]).round(3)))

    def top_channel_stability(self, band: str | None = None) -> pd.DataFrame:
        """Per patient: how often the *same* channel comes out busiest.

        Computed over the disjoint arm only, because equal-length windows are
        the only fair comparison -- a longer window is not more stable, it is
        a different measurement.

        ``modal_share`` is the fraction of windows won by that patient's most
        frequent winner. 1.0 means the same channel every time; 0.2 over five
        windows means a different channel every time, i.e. the argmax carries
        no information at that window length.
        """
        band = band or self.primary[1]
        table = self.channels.query("arm == 'disjoint' and band == @band")
        if table.empty:
            return table
        rows = []
        for (subject, source), group in table.groupby(["subject", "source"]):
            winners = group["top_channel"].dropna()
            if not len(winners):
                continue
            counts = winners.value_counts()
            rows.append({
                "subject": subject, "source": source, "band": band,
                "n_windows": int(len(winners)),
                "n_distinct_channels": int(counts.size),
                "modal_channel": counts.index[0],
                "modal_share": float(counts.iloc[0] / len(winners)),
            })
        return pd.DataFrame(rows).sort_values(["source", "subject"]).reset_index(drop=True)

    def decision_stability(self, band: str | None = None,
                           metric: str | None = None,
                           arm: str = "disjoint") -> pd.DataFrame:
        """Per patient: does the *metric* give the same answer in every window?

        :meth:`top_channel_stability` asks whether the same channel wins.
        This asks the question that actually reaches a clinician: whether the
        winner was inside the resection. The two come apart, and the gap
        between them is the useful part -- when the two or three busiest
        channels are all inside the resection (or all outside it), the argmax
        can move freely between them without the answer changing. Channel
        identity is the fragile thing; the decision built on it is less so.
        """
        band = band or self.primary[1]
        metric = metric or self.primary[0]
        frame = self.subjects
        if not len(frame):
            return pd.DataFrame()
        table = frame.query("arm == @arm and scope == 'reviewed' and band == @band")
        rows = []
        for (subject, source), group in table.groupby(["subject", "source"]):
            values = group[metric].dropna()
            if not len(values):
                continue
            rows.append({
                "subject": subject, "source": source, "band": band, "metric": metric,
                "n_windows": int(len(values)),
                "n_distinct_answers": int(values.nunique()),
                "share_inside": float(values.mean()),
                "spread": float(values.max() - values.min()),
                "stable": bool(values.nunique() == 1),
            })
        return pd.DataFrame(rows).sort_values(["source", "subject"]).reset_index(drop=True)

    def verdict(self) -> str:
        """One paragraph on whether the ranking is stable enough to use."""
        metric, band = self.primary
        lines = []
        curve = self.curve()
        for source, group in curve.groupby("source"):
            first, last = group.iloc[0], group.iloc[-1]
            lines.append(
                f"{source}: AUC {first['auc']:.2f} at {first['window_s']:g} s -> "
                f"{last['auc']:.2f} at {last['window_s']:g} s")
        spread = self.spread()
        for _, row in spread.iterrows():
            lines.append(
                f"{row['source']}: across {int(row['n_units'])} disjoint "
                f"{DISJOINT_LENGTH:g} s windows the AUC spans "
                f"{row['auc_min']:.2f}-{row['auc_max']:.2f} (range {row['auc_range']:.2f})")
        stability = self.top_channel_stability()
        if len(stability):
            for source, group in stability.groupby("source"):
                same = int((group["modal_share"] == 1.0).sum())
                lines.append(
                    f"{source}: the busiest channel is the same in every window for "
                    f"{same}/{len(group)} patients (median modal share "
                    f"{group['modal_share'].median():.2f})")
        decision = self.decision_stability()
        if len(decision):
            for source, group in decision.groupby("source"):
                stable = int(group["stable"].sum())
                lines.append(
                    f"{source}: the inside/outside ANSWER is the same in every window "
                    f"for {stable}/{len(group)} patients")
        return " | ".join(lines)

    def verdict_runs(self) -> str:
        """One paragraph on whether the answer holds from one night to another."""
        metric, band = self.primary
        lines = []
        spread = self.spread(arm="run")
        for _, row in spread.iterrows():
            lines.append(
                f"{row['source']}: across {int(row['n_units'])} runs the AUC spans "
                f"{row['auc_min']:.2f}-{row['auc_max']:.2f} "
                f"(range {row['auc_range']:.2f})")
        decision = self.decision_stability(arm="run")
        if len(decision):
            for source, group in decision.groupby("source"):
                stable = int(group["stable"].sum())
                multi = group[group["n_windows"] > 1]
                lines.append(
                    f"{source}: the inside/outside answer holds across every run for "
                    f"{stable}/{len(group)} patients "
                    f"({int((multi['stable']).sum())}/{len(multi)} of those with "
                    "more than one run)")
        pooled = self.groups.query(
            "arm == 'pooled' and metric == @metric and band == @band "
            "and scope == 'reviewed'")
        for _, row in pooled.iterrows():
            lines.append(
                f"{row['source']} pooled over runs: AUC {row['auc']:.2f} "
                f"(p {row['p_permutation']:.3f})")
        return " | ".join(lines)

    def save(self, directory: str | Path | None = None) -> Path:
        out = Path(directory or RESULTS_DIR) / f"stability_{self.dataset}"
        out.mkdir(parents=True, exist_ok=True)
        self.groups.to_csv(out / "groups.csv", index=False)
        self.subjects.to_csv(out / "subjects.csv", index=False)
        self.channels.to_csv(out / "top_channels.csv", index=False)
        if (self.groups.get("arm") == "growing").any():
            self.curve().to_csv(out / "curve.csv", index=False)
        stability = self.top_channel_stability()
        if len(stability):
            stability.to_csv(out / "top_channel_stability.csv", index=False)
        for arm in ("disjoint", "run"):
            decision = self.decision_stability(arm=arm)
            if len(decision):
                decision.to_csv(out / f"decision_stability_{arm}.csv", index=False)
            spread = self.spread(arm=arm)
            if len(spread):
                spread.to_csv(out / f"spread_{arm}.csv", index=False)
        pooled = self.groups.query("arm == 'pooled'") if len(self.groups) else pd.DataFrame()
        if len(pooled):
            pooled.to_csv(out / "pooled_groups.csv", index=False)
        (out / "run.json").write_text(json.dumps({
            "dataset": self.dataset, "detector": self.detector,
            "pipeline_version": self.pipeline_version,
            "primary_comparison": "/".join(self.primary),
            "windows": self.windows,
            "cohort": self.cohort,
            "verdict": self.verdict(),
            "verdict_runs": (self.verdict_runs()
                             if (self.groups.get("arm") == "run").any() else ""),
        }, indent=2))
        print(f"[onset-hfo] stability study written to {out}")
        return out


def _top_channels(result: OutcomeResult, detector: str) -> pd.DataFrame:
    """The busiest channel per subject, per source, per band, reviewed scope only.

    Ties are broken by channel name rather than by array order, so the same
    tie resolves the same way in every window -- otherwise this function would
    manufacture instability that the data does not have.
    """
    column = {"expert": "expert_events", detector: f"{detector}_events"}
    rows = []
    frame = result.channels
    if not len(frame):
        return pd.DataFrame(columns=["subject", "band", "source", "top_channel", "top_events"])
    reviewed = frame[frame["reviewed"] & ~frame["eloquent"]]
    for (subject, band), group in reviewed.groupby(["subject", "band"]):
        ordered = group.sort_values("channel", kind="mergesort")
        for source, events in column.items():
            counts = ordered[events].to_numpy(float)
            if not counts.sum():
                rows.append({"subject": subject, "band": band, "source": source,
                             "top_channel": None, "top_events": 0.0})
                continue
            best = int(np.argmax(counts))
            rows.append({"subject": subject, "band": band, "source": source,
                         "top_channel": ordered["channel"].iloc[best],
                         "top_events": float(counts[best])})
    return pd.DataFrame(rows)


def stability_study(growing: tuple[float, ...] = GROWING_WINDOWS,
                    disjoint_length: float = DISJOINT_LENGTH,
                    run_length: float = FULL_RUN_S,
                    dataset: str = "ds003498", detector: str = "rms",
                    bands: tuple[str, ...] = ("ripple", "fast_ripple"),
                    subjects: list[str] | None = None,
                    verbose: bool = True) -> StabilityResult:
    """Run the outcome study at several windows and collect the spread.

    Expensive and entirely offline after the first pass: every window is a
    separate byte-range fetch, cached thereafter, so re-running costs compute
    only.
    """
    plan: list[dict] = []
    for stop in growing:
        plan.append({"arm": "growing", "t_start": 0.0, "t_stop": float(stop)})
    start = 0.0
    while start + disjoint_length <= run_length + 1e-9:
        plan.append({"arm": "disjoint", "t_start": start,
                     "t_stop": start + disjoint_length})
        start += disjoint_length

    group_rows, subject_rows, channel_rows = [], [], []
    cohorts: dict[str, set[str]] = {}
    for i, window in enumerate(plan, 1):
        if verbose:
            print(f"\n[onset-hfo] === stability {i}/{len(plan)}: {window['arm']} "
                  f"[{window['t_start']:g}, {window['t_stop']:g}) s ===")
        result = outcome_study(subjects=subjects, dataset=dataset, detector=detector,
                               t_start=window["t_start"], t_stop=window["t_stop"],
                               bands=bands, verbose=False)
        if not len(result.groups):
            print(f"[onset-hfo]   no comparison possible for {window}")
            continue
        tag = {**window, "window_s": window["t_stop"] - window["t_start"]}
        cohorts[f"{window['t_start']:g}-{window['t_stop']:g}"] = set(
            result.meta["subject"]) if len(result.meta) else set()
        group_rows.append(result.groups.assign(**tag))
        subject_rows.append(result.subjects.assign(**tag))
        channel_rows.append(_top_channels(result, detector).assign(**tag))
        if verbose:
            primary = result.groups.query(
                "metric == 'top_channel_resected' and band == 'fast_ripple' "
                "and scope == 'reviewed'")
            for _, row in primary.iterrows():
                print(f"[onset-hfo]   {row['source']:>7}: AUC {row['auc']:.3f} "
                      f"(p {row['p_permutation']:.3f}), "
                      f"{row['mean_seizure_free']:.2f} vs {row['mean_recurrence']:.2f}")

    cohort = _check_cohort(cohorts)
    out = StabilityResult(
        groups=pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame(),
        subjects=pd.concat(subject_rows, ignore_index=True) if subject_rows else pd.DataFrame(),
        channels=pd.concat(channel_rows, ignore_index=True) if channel_rows else pd.DataFrame(),
        dataset=dataset, detector=detector, windows=plan, cohort=cohort)
    if not cohort.get("consistent", True):
        print("\n[onset-hfo] WARNING: the windows were not computed over the same "
              "patients, so differences between them are not attributable to the "
              "window alone. Missing per window: "
              + json.dumps(cohort["missing"], sort_keys=True))
    if verbose:
        print(f"\n[onset-hfo] {out.verdict()}")
    return out


def _check_cohort(cohorts: dict[str, set[str]]) -> dict:
    """Did every window analyse the same patients?

    Cohort sweeps in this package skip a subject that fails rather than
    aborting, which is right for a one-off benchmark and wrong here: a window
    that quietly lost one patient to a dropped download would show up as a
    real change in the curve. This turns that into a visible flag.
    """
    if not cohorts:
        return {"consistent": True, "n_subjects": 0, "missing": {}}
    full = set().union(*cohorts.values())
    missing = {window: sorted(full - seen) for window, seen in cohorts.items()
               if full - seen}
    return {"consistent": not missing, "n_subjects": len(full),
            "subjects": sorted(full), "missing": missing}


def plot_stability(result: StabilityResult, path: str | Path | None = None, dpi: int = 140):
    """Three panels: does it settle, does it depend on which minute, does the argmax hold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from onset_hfo.viz import PALETTE, _fig

    colors = {"expert": PALETTE["series_1"], result.detector: PALETTE["series_2"]}
    fig, axes = _fig(1, 3, figsize=(13.5, 4.2))

    # -- A: growing windows ------------------------------------------------
    ax = axes[0]
    curve = result.curve()
    for source, group in curve.groupby("source"):
        color = colors.get(source, PALETTE["series_3"])
        ax.fill_between(group["window_s"], group["auc_lo"], group["auc_hi"],
                        color=color, alpha=0.13, linewidth=0)
        ax.plot(group["window_s"], group["auc"], "-o", color=color, label=source,
                markersize=4.5, linewidth=1.8)
    ax.axhline(0.5, color=PALETTE["ink_soft"], linewidth=1, linestyle=":")
    ax.text(curve["window_s"].max(), 0.505, "chance", ha="right", va="bottom",
            fontsize=8, color=PALETTE["ink_soft"])
    ax.axhline(0.85, color=PALETTE["rejected"], linewidth=1, linestyle="--", alpha=0.7)
    ax.text(curve["window_s"].max(), 0.855, "80% power floor (13 vs 7)", ha="right",
            va="bottom", fontsize=8, color=PALETTE["rejected"])
    ax.set_xlabel("window length from start of run (s)")
    ax.set_ylabel("AUC, seizure-free vs recurrence")
    ax.set_title("A  More data: does it settle?", fontsize=10, loc="left",
                 color=PALETTE["ink"])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)

    # -- B: disjoint windows -----------------------------------------------
    ax = axes[1]
    metric, band = result.primary
    disjoint = result.groups.query(
        "arm == 'disjoint' and metric == @metric and band == @band and scope == 'reviewed'")
    for source, group in disjoint.groupby("source"):
        color = colors.get(source, PALETTE["series_3"])
        centre = group["t_start"] + group["window_s"] / 2.0
        ax.plot(centre, group["auc"], "-o", color=color, label=source,
                markersize=4.5, linewidth=1.8)
    full = result.curve()
    for source, group in full.groupby("source"):
        color = colors.get(source, PALETTE["series_3"])
        ax.axhline(float(group.iloc[-1]["auc"]), color=color, linewidth=1,
                   linestyle="--", alpha=0.55)
    ax.axhline(0.5, color=PALETTE["ink_soft"], linewidth=1, linestyle=":")
    ax.set_xlabel(f"centre of each disjoint {DISJOINT_LENGTH:g} s window (s)")
    ax.set_ylabel("AUC")
    ax.set_title("B  Same length, different minute\n     (dashed = whole-run value)",
                 fontsize=10, loc="left", color=PALETTE["ink"])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)

    # -- C: does the argmax hold, and does the answer? ---------------------
    ax = axes[2]
    identity = result.top_channel_stability()
    decision = result.decision_stability()
    if len(identity) and len(decision):
        sources = list(identity["source"].unique())
        groups = ["same top channel\nin every window", "same inside/outside\nanswer in every window"]
        width = 0.36
        positions = np.arange(len(groups))
        for i, source in enumerate(sources):
            same_channel = int((identity.loc[identity["source"] == source,
                                             "modal_share"] == 1.0).sum())
            same_answer = int(decision.loc[decision["source"] == source, "stable"].sum())
            total = int((identity["source"] == source).sum())
            offset = (i - (len(sources) - 1) / 2.0) * width
            bars = ax.bar(positions + offset, [same_channel, same_answer], width=width,
                          color=colors.get(source, PALETTE["series_3"]), label=source)
            for bar, value in zip(bars, [same_channel, same_answer], strict=True):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25,
                        f"{value}/{total}", ha="center", va="bottom", fontsize=8.5,
                        color=PALETTE["ink_soft"])
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, fontsize=8.5)
        ax.set_ylabel("patients")
        ax.set_ylim(0, max(len(identity["subject"].unique()) + 3, 5))
        ax.set_title(f"C  What survives a change of minute?\n"
                     f"     (across disjoint {DISJOINT_LENGTH:g} s windows)",
                     fontsize=10, loc="left", color=PALETTE["ink"])
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[onset-hfo] figure written to {path}")
    return fig


# --------------------------------------------------------------------------
# Across runs: does the answer settle across nights, not just within one?
# --------------------------------------------------------------------------


def list_runs_per_subject(dataset: str = "ds003498") -> dict[str, list[str]]:
    """Every run id that has a signal file, per subject. One bucket listing."""
    import re

    from onset_hfo.datasets import _http_get, _spec

    spec = _spec(dataset)
    keys: list[str] = []
    token = ""
    for _page in range(40):
        url = (f"{spec.base_url}/?list-type=2&prefix={spec.dataset_id}/sub-"
               f"&max-keys=1000")
        if token:
            from urllib.parse import quote
            url += f"&continuation-token={quote(token, safe='')}"
        xml = _http_get(url).decode("utf-8", "replace")
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        if "<IsTruncated>true</IsTruncated>" not in xml:
            break
        found = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not found:
            break
        token = found.group(1)
    runs: dict[str, set[str]] = {}
    for key in keys:
        match = re.search(r"/(sub-[A-Za-z0-9]+)_ses-\w+_run-(\d+)_ieeg\.eeg$", key)
        if match:
            runs.setdefault(match.group(1), set()).add(match.group(2))
    return {subject: sorted(found) for subject, found in sorted(runs.items())}


def _prune(dataset: str, subject: str, run: str, t_start: float, t_stop: float) -> None:
    """Delete one cached slice. 385 runs of this archive are ~46 GB."""
    import shutil

    from onset_hfo.config import DATA_CACHE

    directory = (Path(DATA_CACHE) / dataset /
                 f"{subject}_run-{run}_{t_start:g}-{t_stop:g}s")
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)


def pool_runs(channels: pd.DataFrame, resections: dict, detector: str,
              run_seconds: float, bands: tuple[str, ...]) -> pd.DataFrame:
    """Sum a subject's per-channel event counts across runs, then re-score.

    The counting question and the pooling question are different, and this is
    the pooling one: *if a patient contributes several nights, does adding
    them together give a firmer answer than any one night?* Rates add, so the
    pooled table is a longer recording of the same channels -- which also
    means the Poisson intervals behind :func:`onset_hfo.outcome.candidate_channels`
    narrow, and a tie that a single run could not resolve may resolve here.

    A channel counts as reviewed if it was reviewed in **any** run: the
    annotators marked events, and a channel with no event in one segment was
    not thereby unreviewed.
    """
    from onset_hfo.outcome import (
        _candidate_metrics,
        _share_in_resection,
        _top_channel_resected,
        _top_k_resected,
    )

    if not len(channels):
        return pd.DataFrame()
    detector_col = f"{detector}_events"
    pooled = (channels.groupby(["subject", "band", "channel"], as_index=False)
              .agg(zone=("zone", "first"), eloquent=("eloquent", "max"),
                   reviewed=("reviewed", "max"), n_runs=("run", "nunique"),
                   expert_events=("expert_events", "sum"),
                   **{detector_col: (detector_col, "sum")}))
    rows = []
    for (subject, band), group in pooled.groupby(["subject", "band"]):
        if band not in bands or subject not in resections:
            continue
        usable = group[group["reviewed"].astype(bool) & ~group["eloquent"].astype(bool)]
        if not len(usable):
            continue
        zones = usable.set_index("channel")["zone"]
        n_runs = int(group["n_runs"].max())
        minutes = n_runs * run_seconds / 60.0
        for source, column in (("expert", "expert_events"), (detector, detector_col)):
            rates = usable.set_index("channel")[column].astype(float)
            rows.append({
                "subject": subject, "band": band, "scope": "reviewed",
                "source": source, "n_runs": n_runs, "n_channels": len(usable),
                **_share_in_resection(rates, zones),
                "top_channel_resected": _top_channel_resected(rates, zones),
                "top3_resected": _top_k_resected(rates, zones, k=3),
                **_candidate_metrics(rates, zones, minutes),
            })
    return pd.DataFrame(rows)


def across_runs(runs_per_subject: int = DEFAULT_RUNS_PER_SUBJECT,
                dataset: str = "ds003498", detector: str = "rms",
                bands: tuple[str, ...] = ("ripple", "fast_ripple"),
                t_stop: float = FULL_RUN_S, subjects: list[str] | None = None,
                prune_cache: bool = True, verbose: bool = True) -> StabilityResult:
    """Does the answer settle across nights, not merely within one recording?

    :func:`stability_study` showed the estimate settles inside a single
    300-second run. That is a statement about one recording. Each patient here
    contributed several, on different nights, and whether the answer holds
    between them decides something the window study could not: whether this is
    a *per-patient* measurement or a *per-recording* one. Only the first is
    any use clinically.

    Two arms again. The ``run`` arm scores each run separately, so
    :meth:`StabilityResult.spread` and
    :meth:`StabilityResult.decision_stability` (with ``arm="run"``) report how
    far the answer moves between nights. The ``pooled`` arm adds a subject's
    runs together and scores the sum, which is what a clinician would actually
    have available.

    ``prune_cache`` deletes each slice once it has been analysed. It defaults
    to **True** because the alternative is 46 GB.
    """
    from onset_hfo.clinical import fetch_participants, resection_map
    from onset_hfo.outcome import compare_groups, outcome_subject

    resections = resection_map(dataset, verbose=verbose)
    participants = fetch_participants(dataset)
    available = list_runs_per_subject(dataset)
    if subjects is not None:
        available = {s: r for s, r in available.items() if s in subjects}

    group_rows, subject_rows, channel_rows, meta_rows = [], [], [], []
    cohorts: dict[str, set[str]] = {}
    plan = [(s, r) for s, runs in available.items() for r in runs[:runs_per_subject]]
    if verbose:
        counts = {s: len(runs[:runs_per_subject]) for s, runs in available.items()}
        print(f"[onset-hfo] {len(plan)} runs over {len(available)} subjects "
              f"(up to {runs_per_subject} each); "
              f"{sum(1 for n in counts.values() if n < 2)} subjects have only one")

    for i, (subject, run) in enumerate(plan, 1):
        resection = resections.get(subject)
        if resection is None or not len(resection):
            continue
        if verbose:
            print(f"\n[onset-hfo] === run {i}/{len(plan)}: {subject} run-{run} ===")
        try:
            rows, channels, meta = outcome_subject(
                subject, resection, run=run, t_start=0.0, t_stop=t_stop,
                dataset=dataset, detector=detector, bands=bands, verbose=False)
        except Exception as exc:
            print(f"[onset-hfo]   skipped {subject} run-{run}: "
                  f"{type(exc).__name__}: {exc}")
            continue
        finally:
            if prune_cache:
                _prune(dataset, subject, run, 0.0, t_stop)
        tag = {"arm": "run", "run": run, "t_start": 0.0, "t_stop": t_stop,
               "window_s": t_stop}
        subject_rows.append(pd.DataFrame(rows).assign(**tag))
        channel_rows.append(pd.DataFrame(channels).assign(**tag))
        meta_rows.append({**meta, **tag})
        cohorts.setdefault(run, set()).add(subject)

    subjects_df = (pd.concat(subject_rows, ignore_index=True) if subject_rows
                   else pd.DataFrame())
    channels_df = (pd.concat(channel_rows, ignore_index=True) if channel_rows
                   else pd.DataFrame())

    # Per-run group statistics, on the subjects present in EVERY run analysed.
    #
    # Two subjects of ds003498 have a single run, so an unrestricted
    # comparison would put 20 patients in run-01 and 18 in the rest, and any
    # difference between runs would partly be a difference of cohort. That is
    # the mistake the window study caught once already; it is not worth
    # making twice.
    common: set[str] = set()
    if len(subjects_df):
        by_run = subjects_df.groupby("run")["subject"].apply(set)
        common = set.intersection(*by_run) if len(by_run) else set()
        dropped = sorted(set(subjects_df["subject"]) - common)
        if dropped and verbose:
            print(f"[onset-hfo] per-run comparison restricted to the {len(common)} "
                  f"subjects present in all {len(by_run)} runs; "
                  f"excluded (too few runs): {', '.join(dropped)}")
        for run, group in subjects_df.groupby("run"):
            stats = compare_groups(group[group["subject"].isin(common)], participants)
            if len(stats):
                group_rows.append(stats.assign(arm="run", run=run, t_start=0.0,
                                               t_stop=t_stop, window_s=t_stop,
                                               n_common_subjects=len(common)))

    # Pooled: add a subject's runs together and score the sum. This arm keeps
    # every subject, including the two with a single run -- pooling one run is
    # still that patient's best available answer, and ``n_runs`` records it.
    pooled = pool_runs(channels_df, resections, detector, t_stop, bands)
    if len(pooled):
        stats = compare_groups(pooled, participants)
        group_rows.append(stats.assign(arm="pooled", run="pooled", t_start=0.0,
                                       t_stop=t_stop, window_s=t_stop))
        subject_rows.append(pooled.assign(arm="pooled", run="pooled", t_start=0.0,
                                          t_stop=t_stop, window_s=t_stop))
        subjects_df = pd.concat(subject_rows, ignore_index=True)

    out = StabilityResult(
        groups=pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame(),
        subjects=subjects_df,
        channels=channels_df,
        dataset=dataset, detector=detector,
        windows=[{"arm": "run", "subject": s, "run": r} for s, r in plan],
        cohort=_check_cohort(cohorts))
    if verbose and len(out.groups):
        print(f"\n[onset-hfo] {out.verdict_runs()}")
    return out
