"""Running the pipeline across a cohort, to produce one labelled table.

Everything else in this repository analyses one recording. A learned model
cannot be *evaluated* on one recording: leave-one-patient-out cross-validation
needs patients, and a classifier fitted to 71 channels of a single subject has
an AUPRC that means nothing. This module is the prerequisite.

It produces a table with **one row per analysed channel per subject**, the
features the pipeline already computes, and the label from
:mod:`onset_hfo.cohort`. That table is the input to :mod:`onset_hfo.models`
and :mod:`onset_hfo.uncertainty`, and it is also what turns the agent's
ablation ladder from one patient's anecdote into a result with an interval
around it.

Design notes that matter more than they look
--------------------------------------------
**Resumable, because it will be interrupted.** 32 subjects at ~24 MB and ~8 s
each is a job that dies halfway at least once. Each subject's feature table is
cached on disk and a re-run skips it, so progress is never lost and a single
broken subject never costs the whole cohort.

**Failure is recorded, not raised.** Subjects fail for real reasons: a run
with no seizure marker, a sampling rate that cannot support the ripple band, a
montage whose contacts do not match the label spelling. Each one is written
into the failure table with its reason. A cohort run that quietly analysed 19
of 32 subjects and never said so is how a paper gets a number nobody can
reproduce.

**The window follows the seizure, not the clock.** Every centre marks its
seizures at a different time in the file, so a fixed 50-110 s slice lands
mid-seizure for one subject and in flat baseline for another. The window is
chosen as ``[onset - pre_s, onset + post_s]`` per subject, and the actual
window used is recorded in the table.

**Features are computed once, normalised three ways.** Raw, z-scored within
subject, and rank-transformed within subject. Cross-patient comparison of raw
rates is the single most common way an iEEG model appears to work and does
not: absolute rates differ by an order of magnitude between subjects for
reasons that have nothing to do with epilepsy.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from onset_hfo.cohort import SozLabels, label_channels, soz_labels
from onset_hfo.config import BANDS, DATA_CACHE, RESULTS_DIR, PipelineConfig
from onset_hfo.datasets import fetch_slice, list_runs
from onset_hfo.pipeline import run_pipeline

__all__ = ["CohortSpec", "SubjectResult", "discover_cohort", "run_subject",
           "run_cohort", "add_within_subject_normalisation", "FEATURE_COLUMNS"]

#: The per-channel features handed to a model. Deliberately short and
#: interpretable: a clinical model that cannot be explained will not be used,
#: and every one of these is a quantity the report already states.
FEATURE_COLUMNS = [
    "rms_rate_per_min", "ll_rate_per_min", "spike_rate_per_min",
    "mean_frequency_hz", "mean_prominence_db", "mean_amplitude_uv",
    "mean_duration_ms", "frac_hfo_with_spike",
    "rate_ratio_during_before", "rate_change_during_per_min",
    "power_low_1_30", "power_gamma_30_80", "power_ripple_80_250",
]


@dataclass
class CohortSpec:
    """Which recordings to analyse, and how much of each."""

    task: str = "ictal"
    #: Seconds of pre-onset baseline and of seizure to include.
    pre_s: float = 25.0
    post_s: float = 35.0
    #: Skip runs whose binary is larger than this; the byte-range loader only
    #: downloads the slice, but a very large file usually means a long
    #: recording whose marker times need checking by hand first.
    max_size_mb: float = 600.0
    #: Fall back to this window when a run has no usable seizure marker.
    fallback_window: tuple[float, float] = (0.0, 60.0)
    require_seizure: bool = True
    #: Only analyse subjects that have SOZ labels. Without one, a row cannot
    #: contribute to training or to scoring.
    require_labels: bool = True

    @property
    def duration_s(self) -> float:
        return self.pre_s + self.post_s


def _run_token(value) -> str:
    """``1`` and ``"01"`` both become ``"01"``.

    BIDS run entities are zero-padded strings, but a plan table that has been
    through a CSV comes back with ``run`` parsed as an integer, and
    ``str(1)`` builds a URL for ``run-1`` that does not exist. Every subject
    then fails with a 404 -- which the failure table reports faithfully, and
    which is still a wasted cohort run.
    """
    text = str(value).strip()
    return f"{int(text):02d}" if text.isdigit() else text


@dataclass
class SubjectResult:
    subject: str
    ok: bool
    reason: str = ""
    features: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Deciding what to run
# --------------------------------------------------------------------------


def discover_cohort(subjects: list[str] | None = None, spec: CohortSpec | None = None,
                    verbose: bool = True) -> pd.DataFrame:
    """List the runs worth analysing, with their seizure marker and labels.

    Costs a few small HTTPS requests per subject and downloads no signal, so
    it is the right thing to run before committing to a cohort. The returned
    table is the plan: inspect it, filter it, then hand it to
    :func:`run_cohort`.
    """
    from onset_hfo.datasets import (
        _dataset_url,
        _http_get,
        _seizure_times,
        read_tsv_text,
    )

    spec = spec or CohortSpec()
    subjects = subjects or default_subjects()
    rows = []
    for subject in subjects:
        labels = soz_labels(subject)
        try:
            runs = list_runs(subject)
        except Exception as exc:
            rows.append({"subject": subject, "usable": False,
                         "reason": f"could not list runs: {type(exc).__name__}"})
            continue
        runs = runs[runs["task"] == spec.task]
        if not len(runs):
            rows.append({"subject": subject, "usable": False,
                         "reason": f"no {spec.task} run with a signal file"})
            continue
        run = runs.sort_values("size_mb").iloc[0]
        run_id = _run_token(run["run"])
        stem = (f"{subject}/ses-presurgery/ieeg/{subject}_ses-presurgery_"
                f"task-{run['task']}_acq-{run['acq']}_run-{run_id}")
        try:
            events = read_tsv_text(
                _http_get(_dataset_url(f"{stem}_events.tsv")).decode("utf-8", "replace"))
            onset, offset, kind = _seizure_times(events, with_kind=True)
        except Exception as exc:
            onset = offset = None
            kind = f"unreadable: {type(exc).__name__}"

        reasons = []
        if spec.require_labels and not labels.usable:
            reasons.append("no SOZ labels")
        if spec.require_seizure and onset is None:
            reasons.append("no seizure marker")
        if float(run["size_mb"]) > spec.max_size_mb:
            reasons.append(f"run is {run['size_mb']} MB (> {spec.max_size_mb})")

        start = max(0.0, onset - spec.pre_s) if onset is not None else spec.fallback_window[0]
        stop = start + spec.duration_s
        rows.append({
            "subject": subject, "task": run["task"], "acq": run["acq"], "run": run_id,
            "size_mb": run["size_mb"], "seizure_onset_s": onset, "seizure_offset_s": offset,
            "marker_kind": kind, "start_s": round(start, 2), "stop_s": round(stop, 2),
            "n_soz_contacts": len(labels.soz_contacts), "label_source": labels.source,
            "site": labels.site, "engel": labels.engel, "seizure_free": labels.seizure_free,
            "usable": not reasons, "reason": "; ".join(reasons)})
    frame = pd.DataFrame(rows)
    if verbose and len(frame):
        usable = int(frame["usable"].sum())
        print(f"[cohort] {usable}/{len(frame)} subjects usable")
        for reason, count in frame.loc[~frame["usable"], "reason"].value_counts().items():
            print(f"[cohort]   excluded ({count}): {reason}")
    return frame


def default_subjects() -> list[str]:
    """Every subject in the archive that has both signals and a clinical row."""
    from onset_hfo.cohort import clinical_table, normalize_subject
    from onset_hfo.datasets import list_subjects

    known = set(clinical_table()["_key"])
    return [s for s in list_subjects() if normalize_subject(s) in known]


# --------------------------------------------------------------------------
# Running one subject
# --------------------------------------------------------------------------


def run_subject(plan_row, config: PipelineConfig | None = None,
                labels: SozLabels | None = None, verbose: bool = False) -> SubjectResult:
    """Analyse one subject and return its per-channel feature table."""
    subject = str(plan_row["subject"])
    try:
        recording = fetch_slice(subject, str(plan_row["task"]), _run_token(plan_row["run"]),
                                t_start=float(plan_row["start_s"]),
                                t_stop=float(plan_row["stop_s"]),
                                acq=str(plan_row["acq"]), verbose=verbose)
    except Exception as exc:
        return SubjectResult(subject, False, f"download/load failed: {type(exc).__name__}: {exc}")

    cfg = config or PipelineConfig()
    if not BANDS.usable(recording.sfreq, BANDS.ripple):
        return SubjectResult(subject, False,
                             f"sampling rate {recording.sfreq:g} Hz cannot support the "
                             f"ripple band")
    try:
        result = run_pipeline(recording, cfg, verbose=verbose)
    except Exception as exc:
        return SubjectResult(subject, False,
                             f"pipeline failed: {type(exc).__name__}: {exc}\n"
                             + traceback.format_exc(limit=2))

    labels = labels or soz_labels(subject, recording=recording)
    frame = _features_from_result(result, recording)
    if frame is None or not len(frame):
        return SubjectResult(subject, False, "the pipeline produced no channels")

    marks = label_channels(list(frame["channel"]), labels,
                           contacts_of=result.prepared.contacts_of)
    frame = frame.merge(marks[["channel", "is_soz", "matched_contacts"]], on="channel")
    frame.insert(0, "subject", subject)
    frame["site"] = labels.site
    frame["engel"] = labels.engel
    frame["seizure_free"] = labels.seizure_free
    frame["label_source"] = labels.source
    frame["trustworthy_label"] = labels.trustworthy
    frame["window_start_s"] = recording.t_offset
    frame["window_stop_s"] = recording.t_offset + recording.duration
    frame["marker_kind"] = recording.seizure_marker
    frame["sfreq_hz"] = recording.sfreq

    meta = {"n_channels": len(frame), "n_soz_channels": int(frame["is_soz"].sum()),
            "sfreq_hz": recording.sfreq, "marker_kind": recording.seizure_marker,
            "label_source": labels.source, "site": labels.site}
    if meta["n_soz_channels"] == 0:
        return SubjectResult(subject, False,
                             "no analysed channel touches a labelled contact (the labelled "
                             "electrode was probably dropped as bad, or is spelled "
                             "differently in channels.tsv)", frame, meta)
    return SubjectResult(subject, True, "", frame, meta)


def _features_from_result(result, recording) -> pd.DataFrame | None:
    """Turn one :class:`~onset_hfo.pipeline.PipelineResult` into feature rows."""
    prep = result.prepared
    rates = result.rates
    if "rms" not in rates:
        return None

    frame = rates["rms"][["channel", "n_events", "rate_per_min", "mean_frequency_hz",
                          "mean_prominence_db", "mean_amplitude_uv", "mean_duration_ms",
                          "n_with_spike"]].copy()
    frame = frame.rename(columns={"rate_per_min": "rms_rate_per_min",
                                  "n_events": "rms_n_events"})
    frame["frac_hfo_with_spike"] = np.where(
        frame["rms_n_events"] > 0, frame["n_with_spike"] / frame["rms_n_events"], 0.0)

    if "line_length" in rates:
        ll = rates["line_length"][["channel", "rate_per_min"]].rename(
            columns={"rate_per_min": "ll_rate_per_min"})
        frame = frame.merge(ll, on="channel", how="left")
    else:
        frame["ll_rate_per_min"] = np.nan

    spikes = result.spike_rates[["channel", "rate_per_min"]].rename(
        columns={"rate_per_min": "spike_rate_per_min"})
    frame = frame.merge(spikes, on="channel", how="left")

    # Rate change across the marked onset. Absent for runs with no usable
    # marker; left as NaN rather than zero, because "we could not measure it"
    # and "it did not change" are different facts.
    frame["rate_ratio_during_before"] = np.nan
    frame["rate_change_during_per_min"] = np.nan
    change = result.rate_change
    if change is not None and len(change):
        cols = {c.lower(): c for c in change.columns}
        before = cols.get("rate_before_per_min") or cols.get("before_per_min")
        during = cols.get("rate_during_per_min") or cols.get("during_per_min")
        if before and during:
            sub = change[["channel", before, during]].rename(
                columns={before: "_before", during: "_during"})
            frame = frame.merge(sub, on="channel", how="left")
            frame["rate_change_during_per_min"] = frame["_during"]
            # +1 in the denominator: a ratio built on a zero baseline is
            # infinite, and infinity is not a feature.
            frame["rate_ratio_during_before"] = frame["_during"] / (frame["_before"] + 1.0)
            frame = frame.drop(columns=["_before", "_during"])

    power = _relative_band_power(prep)
    frame = frame.merge(power, on="channel", how="left")
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _relative_band_power(prep) -> pd.DataFrame:
    """Fraction of total power in each band, per channel, on the raw signal."""
    sf = prep.sfreq
    top = 0.9 * sf / 2.0
    bands = {"power_low_1_30": (1.0, 30.0), "power_gamma_30_80": (30.0, 80.0),
             "power_ripple_80_250": tuple(BANDS.ripple)}
    window = np.hanning(prep.data.shape[1])
    freqs = np.fft.rfftfreq(prep.data.shape[1], 1.0 / sf)
    psd = np.abs(np.fft.rfft(prep.data * window, axis=1)) ** 2
    total = psd[:, (freqs >= 1.0) & (freqs <= top)].sum(axis=1)
    total = np.where(total > 0, total, np.finfo(float).eps)
    out = {"channel": list(prep.ch_names)}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < min(hi, top))
        out[name] = (psd[:, mask].sum(axis=1) / total) if mask.any() else np.nan
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Running the cohort
# --------------------------------------------------------------------------


def run_cohort(plan: pd.DataFrame | None = None, spec: CohortSpec | None = None,
               out_dir: str | Path | None = None, config: PipelineConfig | None = None,
               refresh: bool = False, limit: int | None = None,
               verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyse every usable subject in ``plan``. Returns ``(features, failures)``.

    Resumable: each subject's table is cached under ``out_dir/subjects`` and a
    re-run skips what is already there unless ``refresh`` is set.
    """
    spec = spec or CohortSpec()
    plan = discover_cohort(spec=spec, verbose=verbose) if plan is None else plan
    out_dir = Path(out_dir or (RESULTS_DIR / "cohort"))
    cache = out_dir / "subjects"
    cache.mkdir(parents=True, exist_ok=True)

    usable = plan[plan["usable"]] if "usable" in plan.columns else plan
    if limit is not None:
        usable = usable.head(int(limit))

    frames, failures = [], []
    for position, (_, row) in enumerate(usable.iterrows(), start=1):
        subject = str(row["subject"])
        path = cache / f"{subject}.csv"
        if path.exists() and not refresh:
            frames.append(pd.read_csv(path))
            if verbose:
                print(f"[cohort] ({position}/{len(usable)}) {subject}: cached")
            continue
        if verbose:
            print(f"[cohort] ({position}/{len(usable)}) {subject}: "
                  f"{row['start_s']:.0f}-{row['stop_s']:.0f} s ...", flush=True)
        outcome = run_subject(row, config=config, verbose=False)
        if outcome.ok and outcome.features is not None:
            outcome.features.to_csv(path, index=False)
            frames.append(outcome.features)
            if verbose:
                print(f"[cohort]     {outcome.meta['n_channels']} channels, "
                      f"{outcome.meta['n_soz_channels']} SOZ, "
                      f"{outcome.meta['sfreq_hz']:g} Hz, {outcome.meta['site']}")
        else:
            failures.append({"subject": subject, "reason": outcome.reason,
                             **(outcome.meta or {})})
            if verbose:
                print(f"[cohort]     SKIPPED: {outcome.reason.splitlines()[0]}")

    features = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=["subject", "channel", *FEATURE_COLUMNS]))
    failure_frame = pd.DataFrame(failures, columns=["subject", "reason"]) \
        if not failures else pd.DataFrame(failures)

    if len(features):
        features = add_within_subject_normalisation(features)
        features.to_csv(out_dir / "features.csv", index=False)
    failure_frame.to_csv(out_dir / "failures.csv", index=False)
    (out_dir / "plan.csv").write_text(plan.to_csv(index=False))
    (out_dir / "summary.json").write_text(json.dumps({
        "n_subjects_planned": int(len(usable)),
        "n_subjects_analysed": int(features["subject"].nunique()) if len(features) else 0,
        "n_subjects_failed": int(len(failure_frame)),
        "n_channels": int(len(features)),
        "n_soz_channels": int(features["is_soz"].sum()) if len(features) else 0,
        "sites": (features.groupby("site")["subject"].nunique().to_dict()
                  if len(features) else {}),
        "cache_dir": str(DATA_CACHE),
    }, indent=2, default=str))

    if verbose:
        print(f"\n[cohort] {features['subject'].nunique() if len(features) else 0} subjects, "
              f"{len(features)} channels, "
              f"{int(features['is_soz'].sum()) if len(features) else 0} labelled SOZ")
        if len(failure_frame):
            print(f"[cohort] {len(failure_frame)} subject(s) failed; see failures.csv")
        print(f"[cohort] written to {out_dir}")
    return features, failure_frame


def add_within_subject_normalisation(features: pd.DataFrame,
                                     columns: list[str] | None = None) -> pd.DataFrame:
    """Add ``*_z`` and ``*_rank`` versions of each feature, computed per subject.

    Why this is not optional: absolute event rates differ between subjects by
    an order of magnitude, for reasons that include electrode type, amplifier,
    sedation and how much of the recording is seizure. A model trained on raw
    rates across patients mostly learns to recognise the patient. Ranking
    within a subject throws that away and keeps the only thing that transfers
    -- which channels stood out *in their own recording*.

    ``_z`` is a robust z-score (median and MAD), so one very loud channel does
    not flatten the rest. ``_rank`` is the percentile within the subject, which
    is scale-free and the strongest baseline in practice.
    """
    out = features.copy()
    columns = columns or [c for c in FEATURE_COLUMNS if c in out.columns]
    for column in columns:
        grouped = out.groupby("subject")[column]
        median = grouped.transform("median")
        mad = grouped.transform(lambda s: (s - s.median()).abs().median())
        scale = (1.4826 * mad).replace(0, np.nan)
        out[f"{column}_z"] = ((out[column] - median) / scale).fillna(0.0)
        out[f"{column}_rank"] = grouped.rank(pct=True, na_option="keep")
    return out
