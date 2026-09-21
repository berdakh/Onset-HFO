"""Live analysis tools: the agent can re-run a detector, not just read a table.

What changed, and why it is the whole point
-------------------------------------------
The original agent (:mod:`onset_agent.tools`) is a set of read-only queries
over a *saved* analysis. It can report that a channel had 14.2 ripples/min at
the threshold someone chose hours ago. It cannot ask the question a clinician
asks next:

    "Does that survive a stricter threshold?"

This module makes that question answerable. Every tool here executes the real
pipeline -- the same detectors, the same validation, the same code paths as
``python -m onset_hfo.cli run`` -- against a cached recording, with parameters
the planner chooses at call time. Re-planning stops being a rephrasing of
known results and becomes a new measurement.

Making it fast enough to be interactive
---------------------------------------
Naively, "re-run detection at 6 SD" means preprocessing and filtering 71
channels again: several seconds per call, and a planner needs many calls. Two
caches fix that, and both are keyed on the values that actually change the
answer:

* **The prepared montage** (channel selection, high-pass, notch, bipolar) is
  computed once per session. It cannot depend on a detector threshold.
* **The band-passed signal** is cached per band. The band-pass is the
  expensive step; thresholding the feature trace is cheap. So the second call
  at a different ``threshold_sd`` costs a fraction of the first, which is
  exactly the call pattern robustness checks produce.

A repeated call with identical parameters is served from a memo cache and
recorded as a fresh run id with ``cached: true`` in its output, so the audit
trail still shows that the planner asked twice.

What these tools deliberately do not do
---------------------------------------
No tool writes a file, opens a URL, executes code, changes the subject, or
returns a recommendation. The patient is fixed when the session is opened --
no tool takes a subject argument, so no model output can change which
recording is being analysed. Tool results are data to be quoted, never
instructions to be followed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from onset_agent.contract import ContractError, ToolRegistry, ToolSpec
from onset_hfo.config import BANDS, PipelineConfig
from onset_hfo.datasets import Recording
from onset_hfo.detectors.base import bandpass
from onset_hfo.detectors.line_length import detect_line_length
from onset_hfo.detectors.rms import detect_rms
from onset_hfo.detectors.spike import detect_spikes as _detect_spikes
from onset_hfo.metrics import (
    channel_rates,
    compare_rankings,
    detector_agreement,
    leader_separation,
)
from onset_hfo.preprocess import Prepared, prepare
from onset_hfo.validate import flag_spike_cooccurrence, validate_events

__all__ = ["AnalysisSession", "build_registry", "ANALYSIS_TOOLS"]

_HFO_DETECTORS = {"rms": detect_rms, "line_length": detect_line_length}
#: Detection on more channels than this at once is slow enough to stall a
#: planning loop; the planner is told to narrow its request instead.
MAX_CHANNELS_PER_CALL = 96


class AnalysisSession:
    """One cached recording, re-analysable at any parameters.

    Parameters
    ----------
    recording:
        From :func:`onset_hfo.datasets.fetch_slice` (public data) or
        :func:`onset_hfo.synthetic.make_synthetic_recording` (offline).
    config:
        Default thresholds. A tool argument overrides the default for that
        call only; the session's config is never mutated, so two calls with
        the same arguments always return the same numbers.
    """

    def __init__(self, recording: Recording, config: PipelineConfig | None = None,
                 verbose: bool = False, soz_model=None):
        self.recording = recording
        self.config = config or PipelineConfig()
        self.verbose = verbose
        #: An optional :class:`~onset_hfo.models.SozModel`. When present, the
        #: ``estimate_soz_probability`` tool works and the planner can stop on
        #: the width of its candidate set. Everything else runs without it.
        self.soz_model = soz_model
        self._model_features: pd.DataFrame | None = None
        self._prepared: Prepared | None = None
        self._filtered: dict[tuple[float, float], np.ndarray] = {}
        self._memo: dict[tuple, Any] = {}

    # -- lazily built, then reused ---------------------------------------
    @property
    def prepared(self) -> Prepared:
        """Channel selection, filtering and montage -- computed once."""
        if self._prepared is None:
            self._prepared = prepare(self.recording, self.config.preprocess,
                                     verbose=self.verbose)
        return self._prepared

    def filtered(self, band: tuple[float, float]) -> np.ndarray:
        """Band-passed data for the whole slice, cached per band."""
        key = (float(band[0]), float(band[1]))
        if key not in self._filtered:
            self._filtered[key] = bandpass(self.prepared.data, self.prepared.sfreq, key)
        return self._filtered[key]

    def model_features(self):
        """The per-channel feature table the learned model expects.

        Built with the *same* code the cohort was built with
        (:mod:`onset_hfo.batch`), so a feature means the same thing at
        training time and at prediction time. Reimplementing it here to save
        a pipeline run is how a model silently starts reading a different
        quantity from the one it was fitted on.

        Cached for the session: it is one full pipeline run.
        """
        if self._model_features is None:
            from onset_hfo.batch import (
                _features_from_result,
                add_within_subject_normalisation,
            )
            from onset_hfo.pipeline import run_pipeline

            result = run_pipeline(self.recording, self.config, verbose=self.verbose)
            frame = _features_from_result(result, self.recording)
            frame.insert(0, "subject", self.recording.subject)
            self._model_features = add_within_subject_normalisation(frame)
        return self._model_features

    def reset_memo(self) -> None:
        """Forget cached detections, keeping the montage and the band-pass.

        Used between rungs of the ablation ladder. Preprocessing and filtering
        are shared infrastructure -- every rung would pay the same price for
        them, so charging it four times measures nothing. Detection results are
        not: if one rung's cache made the next rung's calls free, the
        cost column would say that re-planning is cheaper than not
        re-planning, which is the opposite of the truth.
        """
        self._memo.clear()

    # -- helpers ----------------------------------------------------------
    @property
    def window(self) -> tuple[float, float]:
        """The analysed slice, in original recording time."""
        prep = self.prepared
        return prep.t_offset, prep.t_offset + prep.duration

    def resolve_channels(self, channels: list[str] | None) -> list[str]:
        """Validate a channel list, or return all analysed channels.

        An unknown channel name is a contract error rather than a silent drop:
        a planner that invents ``LA1-LA2`` on a recording that has no such pair
        must find out, not receive an empty result it will read as "no events".
        """
        available = self.prepared.ch_names
        if not channels:
            return list(available)
        lookup = {c.upper(): c for c in available}
        resolved, unknown = [], []
        for name in channels:
            actual = lookup.get(str(name).upper())
            (resolved if actual else unknown).append(actual or name)
        if unknown:
            raise ContractError(
                f"unknown channel(s): {', '.join(unknown[:5])}. "
                f"This recording has {len(available)} analysed channels; "
                f"call channel_qc to list them.")
        if len(resolved) > MAX_CHANNELS_PER_CALL:
            raise ContractError(
                f"{len(resolved)} channels is more than the {MAX_CHANNELS_PER_CALL} "
                "allowed in one call; ask for fewer, or omit 'channels' to use the "
                "standard whole-recording analysis.")
        return resolved

    def view(self, window_s: list[float] | None) -> tuple[Prepared, tuple[float, float]]:
        """A :class:`Prepared` restricted to a time window, plus the clamped window.

        Times are in **original recording time** everywhere in this package,
        so a window quoted in a report can be found again in the archive file.
        """
        prep = self.prepared
        lo, hi = self.window
        if not window_s:
            return prep, (lo, hi)
        t0, t1 = float(window_s[0]), float(window_s[1])
        t0, t1 = max(lo, min(t0, t1)), min(hi, max(t0, t1))
        if t1 - t0 < 1.0:
            raise ContractError(
                f"window_s must span at least 1 s inside the analysed slice "
                f"{lo:.1f}-{hi:.1f} s; got {window_s}")
        i0 = int(round((t0 - lo) * prep.sfreq))
        i1 = int(round((t1 - lo) * prep.sfreq))
        return replace(prep, data=prep.data[:, i0:i1], t_offset=t0), (t0, t1)

    def detector_config(self, detector: str, threshold_sd: float | None,
                        band: str | None):
        """A per-call copy of a detector's config. The session's own is untouched."""
        base = getattr(self.config, detector)
        updates: dict = {}
        if threshold_sd is not None:
            updates["threshold_sd"] = float(threshold_sd)
        if band is not None:
            chosen = getattr(BANDS, band)
            if not BANDS.usable(self.prepared.sfreq, chosen):
                raise ContractError(
                    f"the {band} band ({chosen[0]:.0f}-{chosen[1]:.0f} Hz) is not usable at "
                    f"{self.prepared.sfreq:.0f} Hz sampling: it sits against the Nyquist edge, "
                    "where anti-alias filtering makes any measurement untrustworthy.")
            updates["band"] = chosen
        return replace(base, **updates) if updates else base

    def run_hfo(self, detector: str, channels: list[str], window_s: list[float] | None,
                threshold_sd: float | None, band: str | None, with_spikes: bool = False):
        """Detect, validate, and return (events, prepared view, window, cached?).

        The memo key is every argument that can change the answer. Two
        identical calls therefore return identical numbers, which is what
        makes run-to-run stability a property of the planner rather than of
        the signal processing.
        """
        cfg = self.detector_config(detector, threshold_sd, band)
        key = ("hfo", detector, tuple(channels), tuple(window_s or ()),
               cfg.threshold_sd, tuple(cfg.band), with_spikes)
        prep, window = self.view(window_s)
        if key in self._memo:
            return (*self._memo[key], prep, window, True)

        full = not window_s and len(channels) == len(self.prepared.ch_names)
        events = _HFO_DETECTORS[detector](
            prep, cfg, channels=channels,
            filtered=self.filtered(cfg.band) if full else None)
        validate_events(events, prep, self.config.validation)
        if with_spikes:
            spikes = _detect_spikes(prep, self.config.spikes)
            flag_spike_cooccurrence(events, spikes)
        rates = channel_rates(events, prep.duration, channels)
        self._memo[key] = (events, rates)
        return events, rates, prep, window, False


# --------------------------------------------------------------------------
# Tool handlers. Each returns plain JSON-able data and nothing else.
# --------------------------------------------------------------------------


def _recording_metadata(s: AnalysisSession) -> dict:
    prep = s.prepared
    rec = s.recording
    lo, hi = s.window
    onset, offset = rec.seizure
    return {
        "subject": rec.subject, "source": rec.source, "task": rec.task, "run": rec.run,
        "sampling_rate_hz": round(prep.sfreq, 1),
        "analysed_window_s": [round(lo, 2), round(hi, 2)],
        "duration_s": round(prep.duration, 2),
        "montage": prep.montage,
        "n_analysed_channels": prep.n_channels,
        "seizure_onset_s": onset, "seizure_offset_s": offset,
        "ripple_band_hz": list(BANDS.ripple),
        "fast_ripple_usable": BANDS.usable(prep.sfreq, BANDS.fast_ripple),
        "citation": rec.citation,
        "note": ("All times are seconds in the ORIGINAL recording, not offsets into the "
                 "analysed slice."),
    }


def _channel_qc(s: AnalysisSession, max_channels: int = 40) -> dict:
    """Which channels are analysable, which were dropped, and how noisy each is."""
    prep = s.prepared
    line = s.config.preprocess.line_freq
    nyquist = prep.sfreq / 2.0
    # Power right at the mains frequency relative to a neighbouring band: a
    # crude but honest line-noise index. The notch has already run, so a
    # channel that still scores high is genuinely contaminated.
    noise = []
    if line * 1.5 < nyquist:
        narrow = bandpass(prep.data, prep.sfreq, (line - 1.0, line + 1.0))
        wide = bandpass(prep.data, prep.sfreq, (line + 5.0, min(line + 25.0, 0.9 * nyquist)))
        num = np.sqrt(np.mean(narrow ** 2, axis=1))
        den = np.sqrt(np.mean(wide ** 2, axis=1)) + np.finfo(float).eps
        noise = (num / den).tolist()
    rows = []
    for i, ch in enumerate(prep.ch_names):
        row = {"channel": ch, "contacts": prep.contacts_of(ch)}
        if noise:
            row["line_noise_ratio"] = round(float(noise[i]), 3)
        rows.append(row)
    flagged = sorted((r for r in rows if r.get("line_noise_ratio", 0) > 3.0),
                     key=lambda r: -r["line_noise_ratio"])
    return {
        "n_analysed_channels": len(rows),
        "channels": rows[:max_channels],
        "truncated": len(rows) > max_channels,
        "dropped_by_dataset": sorted(s.recording.bads),
        "n_dropped_by_dataset": len(s.recording.bads),
        "preprocessing_steps": prep.steps,
        "line_noise_flagged": [r["channel"] for r in flagged],
        "line_noise_definition": (f"RMS at {line:.0f} Hz over RMS in the {line + 5:.0f}-"
                                  f"{line + 25:.0f} Hz band, after notching; above 3 is high"),
    }


def _detect_hfo(s: AnalysisSession, channels: list[str] | None = None,
                window_s: list[float] | None = None, detector: str = "rms",
                threshold_sd: float | None = None, band: str | None = None,
                k: int = 10) -> dict:
    """Run an HFO detector for real, at the requested parameters."""
    names = s.resolve_channels(channels)
    events, rates, _prep, window, cached = s.run_hfo(detector, names, window_s,
                                                     threshold_sd, band, with_spikes=False)
    cfg = s.detector_config(detector, threshold_sd, band)
    table = rates.head(max(1, int(k)))
    per_channel = {
        str(row["channel"]): {
            "rate_per_min": round(float(row["rate_per_min"]), 2),
            "n_events": int(row["n_events"]),
            "rate_ci": [round(float(row["rate_ci_low"]), 2), round(float(row["rate_ci_high"]), 2)],
            "mean_frequency_hz": _round_or_none(row.get("mean_frequency_hz")),
            "mean_prominence_db": _round_or_none(row.get("mean_prominence_db")),
        } for _, row in table.iterrows()}
    accepted = sum(e.accepted for e in events)
    separation = leader_separation(rates, top_k=s.config.top_k)
    return {
        "detector": detector,
        # The missing null hypothesis: every ranking function sorts noise, and
        # without this the planner cannot tell a quiet recording from a
        # localized one. See onset_hfo.metrics.leader_separation.
        "leader_stands_out": separation.get("distinguishable"),
        "leader_separation": separation,
        "threshold_sd": cfg.threshold_sd,
        "band_hz": list(cfg.band),
        "window_s": [round(window[0], 2), round(window[1], 2)],
        "n_channels_analysed": len(names),
        "n_candidates": len(events),
        "n_accepted": accepted,
        "n_rejected_as_artifact": len(events) - accepted,
        "channels": per_channel,
        "showing_top_k": len(per_channel),
        "cached": cached,
        "note": ("rate_per_min counts events that survived artifact validation. "
                 "Overlapping rate_ci intervals mean two channels are tied, not ranked. "
                 "If leader_stands_out is false, no channel is distinguishable from the "
                 "middle of the pack and the ordering should not be reported as a "
                 "ranking at all."),
    }


def _detect_spikes_tool(s: AnalysisSession, channels: list[str] | None = None,
                        window_s: list[float] | None = None,
                        threshold_sd: float | None = None, k: int = 10) -> dict:
    """Interictal epileptiform discharge rate, recomputed at the given threshold."""
    names = s.resolve_channels(channels)
    cfg = s.config.spikes
    if threshold_sd is not None:
        cfg = replace(cfg, threshold_sd=float(threshold_sd))
    prep, window = s.view(window_s)
    key = ("spikes", tuple(names), tuple(window_s or ()), cfg.threshold_sd)
    cached = key in s._memo
    if cached:
        spikes, rates = s._memo[key]
    else:
        spikes = [e for e in _detect_spikes(prep, cfg) if e.channel in set(names)]
        rates = channel_rates(spikes, prep.duration, names)
        s._memo[key] = (spikes, rates)
    return {
        "threshold_sd": cfg.threshold_sd,
        "band_hz": list(cfg.band),
        "window_s": [round(window[0], 2), round(window[1], 2)],
        "n_discharges": len(spikes),
        "channels": {str(r["channel"]): {"rate_per_min": round(float(r["rate_per_min"]), 2),
                                         "n_events": int(r["n_events"])}
                     for _, r in rates.head(max(1, int(k))).iterrows()},
        "cached": cached,
    }


def _spectral_power(s: AnalysisSession, channels: list[str],
                    window_s: list[float] | None = None) -> dict:
    """Relative power per band on the raw (unfiltered) signal.

    Reported as a fraction of total power in 1 Hz - 0.9 x Nyquist, so the
    numbers are comparable across channels with different absolute amplitudes.
    The ictal marker a clinician looks for -- low-voltage fast activity at
    onset -- shows up here as a rise in the ripple-band fraction together with
    a fall in the low-band fraction.
    """
    names = s.resolve_channels(channels)
    prep, window = s.view(window_s)
    sf = prep.sfreq
    top = 0.9 * sf / 2.0
    bands = {"low_1_30": (1.0, 30.0), "ied_5_60": tuple(BANDS.ied),
             "gamma_30_80": (30.0, 80.0), "ripple_80_250": tuple(BANDS.ripple)}
    out: dict[str, dict] = {}
    for ch in names:
        x = prep.data[prep.index(ch)]
        freqs = np.fft.rfftfreq(x.size, 1.0 / sf)
        psd = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
        total = float(psd[(freqs >= 1.0) & (freqs <= top)].sum()) or np.finfo(float).eps
        row = {}
        for name, (lo, hi) in bands.items():
            if hi > top:
                continue
            mask = (freqs >= lo) & (freqs < hi)
            row[name] = round(float(psd[mask].sum() / total), 4)
        row["rms_uv"] = round(float(np.sqrt(np.mean(x ** 2))), 2)
        out[ch] = row
    return {"window_s": [round(window[0], 2), round(window[1], 2)],
            "relative_power": out,
            "definition": ("fraction of total 1 Hz - 0.9 x Nyquist power falling in each "
                           "band. The bands overlap on purpose (ied_5_60 spans part of "
                           "low_1_30 and gamma_30_80), so these fractions do not sum to 1.")}


def _rate_change(s: AnalysisSession, channels: list[str] | None = None,
                 detector: str = "rms", threshold_sd: float | None = None,
                 k: int = 10) -> dict:
    """Event rate before versus during the clinician-marked seizure.

    Returns ``available: false`` rather than a number when the analysed slice
    does not straddle a marked onset. A ratio built from three events and one
    event is a number, not a finding.
    """
    onset, offset = s.recording.seizure
    lo, hi = s.window
    if onset is None or not (lo < onset < hi):
        return {"available": False,
                "reason": ("the analysed window does not contain a clinician-marked "
                           "electrographic onset"),
                "analysed_window_s": [round(lo, 2), round(hi, 2)],
                "seizure_onset_s": onset}
    before = [lo, onset]
    during = [onset, min(offset or hi, hi)]
    if before[1] - before[0] < 5 or during[1] - during[0] < 5:
        return {"available": False,
                "reason": "less than 5 s on one side of the marked onset",
                "before_s": before, "during_s": during}
    names = s.resolve_channels(channels)
    _, rates_before, _, _, _ = s.run_hfo(detector, names, before, threshold_sd, None)
    _, rates_during, _, _, _ = s.run_hfo(detector, names, during, threshold_sd, None)
    merged = rates_before[["channel", "n_events", "rate_per_min"]].merge(
        rates_during[["channel", "n_events", "rate_per_min"]],
        on="channel", suffixes=("_before", "_during"))
    merged = merged.sort_values("rate_per_min_during", ascending=False)
    rows = [{"channel": str(r["channel"]),
             "rate_before_per_min": round(float(r["rate_per_min_before"]), 2),
             "rate_during_per_min": round(float(r["rate_per_min_during"]), 2),
             "n_before": int(r["n_events_before"]), "n_during": int(r["n_events_during"])}
            for _, r in merged.head(max(1, int(k))).iterrows()]
    return {"available": True, "detector": detector,
            "before_s": [round(before[0], 2), round(before[1], 2)],
            "during_s": [round(during[0], 2), round(during[1], 2)],
            "channels": rows,
            "note": ("both raw counts are given on purpose: a large ratio built from very "
                     "few events is not a finding")}


def _compare_detectors(s: AnalysisSession, channels: list[str] | None = None,
                       window_s: list[float] | None = None,
                       threshold_sd_rms: float | None = None,
                       threshold_sd_line_length: float | None = None) -> dict:
    """Run both HFO detectors and report where they disagree, without averaging."""
    names = s.resolve_channels(channels)
    ev_a, rates_a, _, window, _ = s.run_hfo("rms", names, window_s, threshold_sd_rms, None)
    ev_b, rates_b, _, _, _ = s.run_hfo("line_length", names, window_s,
                                       threshold_sd_line_length, None)
    agreement = detector_agreement(ev_a, ev_b, "rms", "line_length")
    comparison = compare_rankings(rates_a, rates_b, "rms", "line_length",
                                  s.config.disagreement_ranks, s.config.top_k)
    disagreements = comparison[comparison["disagrees"]] if len(comparison) else comparison
    return {
        "window_s": [round(window[0], 2), round(window[1], 2)],
        "agreement": {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in agreement.items()},
        "disagreements": [{"channel": str(r["channel"]), "rank_gap": int(r["rank_gap"]),
                           "best_rank": int(r["best_rank"])}
                          for _, r in disagreements.head(10).iterrows()],
        "n_disagreements": int(len(disagreements)),
        "note": ("a channel is listed when the two detectors' ranks differ by at least "
                 f"{s.config.disagreement_ranks} places and at least one puts it in its "
                 f"top {s.config.top_k}. Disagreement is reported, never averaged away."),
    }


def _propagation_lead(s: AnalysisSession, channels: list[str] | None = None,
                      detector: str = "rms", threshold_sd: float | None = None,
                      k: int = 10) -> dict:
    """Which channel's activity begins earliest, and by how much.

    Not a connectivity measure and not a claim about causation: it is the time
    of each channel's first accepted event relative to the earliest channel in
    the set. A channel that leads by 340 ms leads *in this window, at this
    threshold*, which is why the threshold is echoed back in the result.
    """
    names = s.resolve_channels(channels)
    events, _, _, window, _ = s.run_hfo(detector, names, None, threshold_sd, None)
    onset, _ = s.recording.seizure
    first: dict[str, float] = {}
    for event in events:
        if not event.accepted:
            continue
        if event.channel not in first or event.start < first[event.channel]:
            first[event.channel] = float(event.start)
    if not first:
        return {"available": False, "reason": "no accepted events on the requested channels",
                "detector": detector}
    earliest = min(first.values())
    rows = [{"channel": ch, "first_event_s": round(t, 3),
             "lead_ms": round((t - earliest) * 1000.0, 1),
             **({"relative_to_marked_onset_ms": round((t - onset) * 1000.0, 1)}
                if onset is not None else {})}
            for ch, t in sorted(first.items(), key=lambda kv: kv[1])]
    return {"available": True, "detector": detector,
            "window_s": [round(window[0], 2), round(window[1], 2)],
            "earliest_channel": rows[0]["channel"], "channels": rows[:max(1, int(k))],
            "definition": ("lead_ms is the delay from the earliest accepted event in this "
                           "set; it describes order of appearance, not connectivity")}


def _event_evidence(s: AnalysisSession, channel: str, detector: str = "rms",
                    threshold_sd: float | None = None, k: int = 3) -> dict:
    """The strongest events on one channel, each with a citable evidence id.

    The evidence id encodes the channel, detector and window, so a reader can
    go to the archive file and look at the same samples the detector saw. This
    is the only tool whose output a citation may reference.
    """
    names = s.resolve_channels([channel])
    events, _, _, window, _ = s.run_hfo(detector, names, None, threshold_sd, None)
    accepted = sorted((e for e in events if e.accepted), key=lambda e: -e.score)
    rows = []
    for event in accepted[:max(1, int(k))]:
        rows.append({
            "evidence_id": f"{event.channel}|{event.detector}|{event.start:.3f}|{event.stop:.3f}",
            "channel": event.channel,
            "start_s": round(float(event.start), 3), "stop_s": round(float(event.stop), 3),
            "duration_ms": round(float(event.duration_ms), 1),
            "peak_frequency_hz": _round_or_none(event.peak_frequency_hz),
            "spectral_prominence_db": _round_or_none(event.spectral_prominence_db),
            "peak_amplitude_uv": round(float(event.peak_amplitude_uv), 1),
            "n_peaks": int(event.n_peaks),
        })
    return {"channel": names[0], "detector": detector,
            "window_s": [round(window[0], 2), round(window[1], 2)],
            "n_accepted_on_channel": len(accepted), "evidence": rows,
            "note": "start_s and stop_s are seconds in the original recording"}


def _estimate_soz_probability(s: AnalysisSession, alpha: float | None = None,
                              k: int = 15) -> dict:
    """Score every channel with the learned model, and return the candidate set.

    This is where the two halves of the system meet. The planner does not know
    how the model works and does not need to: it is a tool with a JSON
    contract like any other. What comes back is a calibrated probability per
    channel and, at the stated ``alpha``, the **candidate set** -- the
    channels that cannot be ruled out.

    The width of that set is the point. A patient whose candidate set is four
    channels has an actionable result; one whose set is forty has not been
    localized, however confident any individual number looks. That width is
    what :class:`onset_agent.planner.ConformalWidth` stops on.

    Requires a model, attached to the session as ``session.soz_model``. Without
    one the tool says so rather than guessing, and the planner can act on the
    refusal like any other tool error.
    """
    model = getattr(s, "soz_model", None)
    if model is None:
        raise ContractError(
            "no learned SOZ model is attached to this session. Fit one with "
            "onset_hfo.models.fit_soz_model on a cohort, then pass it to "
            "AnalysisSession(..., soz_model=model). The other tools work without it.")

    frame = s.model_features()
    probabilities = model.predict(frame)
    sets = model.conformal_sets(probabilities)
    in_set = [1 in candidate for candidate in sets]

    order = np.argsort(-probabilities)
    rows = []
    for i in order[:max(1, int(k))]:
        rows.append({"channel": str(frame["channel"].iloc[i]),
                     "probability": round(float(probabilities[i]), 4),
                     "in_candidate_set": bool(in_set[i])})
    width = int(sum(in_set))
    return {
        "alpha": model.alpha if alpha is None else float(alpha),
        "nominal_coverage": round(1.0 - model.alpha, 3),
        "n_channels": int(len(frame)),
        "candidate_set_size": width,
        "candidate_fraction": round(width / max(1, len(frame)), 4),
        "candidate_channels": [str(frame["channel"].iloc[i]) for i in order
                               if in_set[i]][:max(1, int(k))],
        "channels": rows,
        "model": model.describe(),
        "note": ("candidate_set_size is how many channels cannot be ruled out at this "
                 "coverage level. A wide set means the recording has not been localized, "
                 "not that every channel is involved. This is a measurement of ripple-band "
                 "and discharge features, not a seizure onset zone."),
    }


def _round_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 2) if np.isfinite(number) else None


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or [],
            "additionalProperties": False}


_CHANNELS = {"type": "array", "items": {"type": "string"}, "maxItems": MAX_CHANNELS_PER_CALL,
             "description": "channel names exactly as analysed, e.g. ['AD1-AD2']; "
                            "omit to use every analysed channel"}
_WINDOW = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
           "description": "[start, stop] in seconds of the ORIGINAL recording; "
                          "omit to use the whole analysed slice"}
_THRESHOLD = {"type": "number", "minimum": 1.0, "maximum": 12.0,
              "description": "detection threshold in robust SDs of the channel's own "
                             "baseline; raise it to test whether a finding survives"}
_DETECTOR = {"type": "string", "enum": ["rms", "line_length"],
             "description": "which HFO detector to run"}
_K = {"type": "integer", "minimum": 1, "maximum": 100,
      "description": "how many channels to return, highest rate first"}

ANALYSIS_TOOLS: list[ToolSpec] = [
    ToolSpec("get_recording_metadata",
             "What is being analysed: subject, sampling rate, analysed window, seizure "
             "markers, montage, channel count. Call this first.",
             _obj({}), _recording_metadata, "meta"),
    ToolSpec("channel_qc",
             "Which channels are analysable, which the dataset flagged bad, the "
             "preprocessing that was applied, and a line-noise index per channel. Call "
             "before trusting any channel you have not seen listed.",
             _obj({"max_channels": {"type": "integer", "minimum": 1, "maximum": 200}}),
             _channel_qc, "qc"),
    ToolSpec("detect_hfo",
             "RUN an HFO (ripple, 80-250 Hz) detector now, at parameters you choose, and "
             "return the event rate per channel. Raise threshold_sd and call again to test "
             "whether a high rate survives a stricter threshold.",
             _obj({"channels": _CHANNELS, "window_s": _WINDOW, "detector": _DETECTOR,
                   "threshold_sd": _THRESHOLD, "k": _K,
                   "band": {"type": "string", "enum": ["ripple", "fast_ripple"]}}),
             _detect_hfo, "hfo", recomputes=True),
    ToolSpec("detect_spikes",
             "RUN the interictal epileptiform discharge detector now and return the "
             "discharge rate per channel.",
             _obj({"channels": _CHANNELS, "window_s": _WINDOW, "threshold_sd": _THRESHOLD,
                   "k": _K}),
             _detect_spikes_tool, "spikes", recomputes=True),
    ToolSpec("spectral_power",
             "Relative power per frequency band on named channels, in a time window. Use "
             "it to check whether a channel shows low-voltage fast activity at seizure "
             "onset, or whether a high ripple rate is just broadband noise.",
             _obj({"channels": _CHANNELS, "window_s": _WINDOW}, ["channels"]),
             _spectral_power, "spectral", recomputes=True),
    ToolSpec("rate_change",
             "Event rate per channel before versus during the clinician-marked seizure, "
             "with both raw counts.",
             _obj({"channels": _CHANNELS, "detector": _DETECTOR, "threshold_sd": _THRESHOLD,
                   "k": _K}),
             _rate_change, "ratechange", recomputes=True),
    ToolSpec("compare_detectors",
             "Run both HFO detectors and report where they rank channels differently, plus "
             "their event-by-event agreement. Use it to qualify any claim about a leading "
             "channel.",
             _obj({"channels": _CHANNELS, "window_s": _WINDOW,
                   "threshold_sd_rms": _THRESHOLD, "threshold_sd_line_length": _THRESHOLD}),
             _compare_detectors, "compare", recomputes=True),
    ToolSpec("propagation_lead",
             "Order in which channels' detected activity begins, and each channel's lead "
             "in milliseconds over the earliest one.",
             _obj({"channels": _CHANNELS, "detector": _DETECTOR, "threshold_sd": _THRESHOLD,
                   "k": _K}),
             _propagation_lead, "lead", recomputes=True),
    ToolSpec("estimate_soz_probability",
             "Score every channel with the learned model and return the candidate set: "
             "the channels that cannot be ruled out at the stated coverage level. Call it "
             "to find out whether the evidence so far has narrowed the answer. Only "
             "available when a model is attached to the session.",
             _obj({"alpha": {"type": "number", "minimum": 0.01, "maximum": 0.5},
                   "k": {"type": "integer", "minimum": 1, "maximum": 50}}),
             _estimate_soz_probability, "soz", recomputes=True),
    ToolSpec("get_event_evidence",
             "The strongest detected events on one channel, each with an evidence_id, time "
             "window, peak frequency and spectral prominence. Every factual claim must cite "
             "an evidence_id from here.",
             _obj({"channel": {"type": "string"}, "detector": _DETECTOR,
                   "threshold_sd": _THRESHOLD,
                   "k": {"type": "integer", "minimum": 1, "maximum": 10}},
                  ["channel"]),
             _event_evidence, "evidence", recomputes=True),
]


def build_registry(session: AnalysisSession) -> ToolRegistry:
    """The tool registry the planner is given. This is the entire tool universe."""
    return ToolRegistry(ANALYSIS_TOOLS, session=session)
