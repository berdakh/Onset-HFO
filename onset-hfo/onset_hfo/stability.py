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
    "StabilityResult",
    "stability_study",
    "plot_stability",
]

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
               scope: str = "reviewed") -> pd.DataFrame:
        """How far the disjoint windows of equal length disagree.

        ``auc_range`` is the plain statement of the problem: the difference
        between the most and least favourable minute of the same recording,
        analysed identically.
        """
        metric = metric or self.primary[0]
        band = band or self.primary[1]
        table = self.groups.query(
            "arm == 'disjoint' and metric == @metric and band == @band and scope == @scope")
        if table.empty:
            return table
        return (table.groupby("source")
                .agg(n_windows=("auc", "size"), auc_min=("auc", "min"),
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
                f"{row['source']}: across {int(row['n_windows'])} disjoint "
                f"{DISJOINT_LENGTH:g} s windows the AUC spans "
                f"{row['auc_min']:.2f}-{row['auc_max']:.2f} (range {row['auc_range']:.2f})")
        stability = self.top_channel_stability()
        if len(stability):
            for source, group in stability.groupby("source"):
                same = float((group["modal_share"] == 1.0).mean())
                lines.append(
                    f"{source}: the busiest channel is the same in every window for "
                    f"{same:.0%} of patients (median modal share "
                    f"{group['modal_share'].median():.2f})")
        return " | ".join(lines)

    def save(self, directory: str | Path | None = None) -> Path:
        out = Path(directory or RESULTS_DIR) / f"stability_{self.dataset}"
        out.mkdir(parents=True, exist_ok=True)
        self.groups.to_csv(out / "groups.csv", index=False)
        self.subjects.to_csv(out / "subjects.csv", index=False)
        self.channels.to_csv(out / "top_channels.csv", index=False)
        self.curve().to_csv(out / "curve.csv", index=False)
        stability = self.top_channel_stability()
        if len(stability):
            stability.to_csv(out / "top_channel_stability.csv", index=False)
        (out / "run.json").write_text(json.dumps({
            "dataset": self.dataset, "detector": self.detector,
            "pipeline_version": self.pipeline_version,
            "primary_comparison": "/".join(self.primary),
            "windows": self.windows,
            "cohort": self.cohort,
            "verdict": self.verdict(),
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

    # -- C: does the argmax hold? ------------------------------------------
    ax = axes[2]
    stability = result.top_channel_stability()
    if len(stability):
        sources = list(stability["source"].unique())
        # Bins are the share of windows the modal channel won. The top bin is
        # closed on the right so "the same channel every time" (share 1.0)
        # lands in it rather than falling off the end.
        edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001])
        centres = (edges[:-1] + np.array([0.2, 0.2, 0.2, 0.2, 0.2]) / 2.0)
        width = 0.16 / max(len(sources), 1)
        for i, source in enumerate(sources):
            share = stability.loc[stability["source"] == source, "modal_share"]
            counts, _ = np.histogram(share, bins=edges)
            offset = (i - (len(sources) - 1) / 2.0) * width
            ax.bar(centres + offset, counts, width=width,
                   color=colors.get(source, PALETTE["series_3"]), label=source)
        ax.set_xticks(centres)
        ax.set_xticklabels(["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"],
                           fontsize=8)
        ax.set_xlabel("share of windows won by the patient's modal top channel")
        ax.set_ylabel("patients")
        ax.set_title(f"C  Is the busiest channel the same channel?\n"
                     f"     ({DISJOINT_LENGTH:g} s windows; 1.0 = never moves)",
                     fontsize=10, loc="left", color=PALETTE["ink"])
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[onset-hfo] figure written to {path}")
    return fig
