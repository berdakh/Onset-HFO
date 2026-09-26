"""Turning a saved analysis into things a page can show.

Every function here is pure, takes a :class:`~onset_hfo.store.ResultStore`,
and returns plain data or a DataFrame. None of them imports Streamlit, which
is what makes them testable and what keeps the UI a *view*: the page arranges
what these return and does not decide anything.

The one rule this module exists to enforce
------------------------------------------

**The interface must not compute a result.** Every number a reader sees has to
be one the pipeline wrote to disk, because the moment a UI starts deriving its
own figures there are two sources of truth and the one on screen is the one
nobody validated.

There is exactly one exception, and it is deliberate:
:func:`leader_note` calls :func:`onset_hfo.metrics.leader_separation` on the
stored rate table. That function is the missing null hypothesis -- it answers
*does any channel actually stand out, or are they all tied?* -- and a page
that showed a ranking without it would be the single most misleading thing
this project could ship. It derives nothing new: it reads the counts and
intervals already in ``rates_*.csv`` through a library function the tests
cover. Treat any second exception as a bug.

Re-rendering a signal window (``app/signal.py``) is not an exception to
this rule either. The event's times, frequency and amplitude all come from the
store; the signal is fetched again only so the reader can *look* at the window
the numbers describe. Nothing is re-detected.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from onset_hfo.config import PROJECT_ROOT, RESULTS_DIR
from onset_hfo.store import ResultStore

#: A real 60 s analysis that ships with the source, so every page works on a
#: fresh clone with no download: OpenNeuro ds003029, the slice the quickstart
#: documents, gzipped.
EXAMPLE = PROJECT_ROOT / "data" / "example_analysis"

#: Cohort-study tables, committed so the evaluation and outcome pages need no
#: download and no 90-minute rerun.
STUDIES = PROJECT_ROOT / "data" / "stability"

__all__ = [
    "EXAMPLE",
    "STUDIES",
    "analyses",
    "find_results",
    "detector_band",
    "event_from_record",
    "metadata_rows",
    "ranking_table",
    "leader_note",
    "evidence_table",
    "disagreement_table",
    "agreement_note",
    "citations",
    "reload_spec",
]


def find_results(root: str | Path) -> list[Path]:
    """Saved analyses under ``root``, newest first.

    A directory qualifies when it holds ``events.csv``; the cohort studies
    (benchmark, outcome, stability) live in the same results folder and are
    not single analyses, so they are skipped rather than offered and then
    failing to load.
    """
    root = Path(root)
    if not root.exists():
        return []
    found = [p for p in root.iterdir() if p.is_dir() and (p / "events.csv").exists()]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def analyses() -> list[Path]:
    """Every saved analysis: the shipped example first, then anything local.

    Lives here rather than beside the page because it is a fact about the
    filesystem, not a decision about layout -- and because this module must
    import without Streamlit, which is what lets the tests run in CI where
    Streamlit is not installed.
    """
    found = find_results(RESULTS_DIR)
    if EXAMPLE.exists() and EXAMPLE not in found:
        found = [EXAMPLE, *found]
    return found


def metadata_rows(store: ResultStore) -> list[tuple[str, str]]:
    """The header: what was analysed, in the order a reader needs it."""
    meta = store.metadata()
    window = meta.get("analysed_window_s") or [None, None]
    band = meta.get("band_hz") or []
    rows = [
        ("Subject", str(meta.get("subject", "?"))),
        ("Source", str(meta.get("source", "?"))),
        ("Task / run", f"{meta.get('task') or '-'} / {meta.get('run') or '-'}"),
        ("Window analysed",
         f"{window[0]:g}-{window[1]:g} s" if None not in window else "-"),
        ("Sampling rate", f"{meta.get('sampling_rate_hz', '?')} Hz"),
        ("Channels analysed",
         f"{meta.get('channels_analysed', '?')} of "
         f"{meta.get('channels_in_recording', '?')}"),
        ("Band", f"{band[0]:g}-{band[1]:g} Hz" if len(band) == 2 else "-"),
        ("Pipeline version", str(meta.get("pipeline_version", "?"))),
    ]
    if meta.get("seizure_onset_s") is not None:
        rows.insert(4, ("Marked seizure onset", f"{meta['seizure_onset_s']:g} s"))
    return rows


def ranking_table(store: ResultStore, detector: str | None = None) -> pd.DataFrame:
    """Channels by event rate, with the interval that says how firm the order is.

    The interval columns are not decoration. Sixty seconds turns a rate into a
    small count, and two channels whose intervals overlap are tied, not
    ranked -- which is why they sit next to the rate rather than in a tooltip.
    """
    name = detector or (store.detectors()[0] if store.detectors() else None)
    table = store.rates.get(name)
    if table is None or not len(table):
        return pd.DataFrame()
    rows = store.list_channels(name)
    frame = pd.DataFrame(rows)
    lookup = table.set_index("channel")
    for column, label in (("rate_ci_low", "ci_low"), ("rate_ci_high", "ci_high"),
                          ("mean_frequency_hz", "mean_freq_hz"),
                          ("mean_duration_ms", "mean_duration_ms")):
        if column in lookup.columns:
            frame[label] = [lookup.at[c, column] if c in lookup.index else None
                            for c in frame["channel"]]
    return frame.round({"rate_per_min": 2, "ci_low": 2, "ci_high": 2,
                        "mean_freq_hz": 1, "mean_duration_ms": 1})


def leader_note(store: ResultStore, detector: str | None = None) -> dict:
    """Does any channel stand out, or is the ranking sorting noise?

    The documented exception to "the UI computes nothing" -- see the module
    docstring. Returns ``{"available": False, ...}`` when the stored rate
    table has no confidence intervals to reason about.
    """
    from onset_hfo.metrics import leader_separation

    name = detector or (store.detectors()[0] if store.detectors() else None)
    table = store.rates.get(name)
    if table is None or not len(table):
        return {"available": False, "reason": f"no rate table for {name!r}"}
    return leader_separation(table)


def detector_band(store: ResultStore, detector: str) -> tuple[float, float]:
    """The band a detector looked in, as the stored config recorded it.

    Needed to redraw an event: the figure band-passes the signal the same way
    the detector did, and guessing it would show the reader a different filter
    from the one the measurement came through. ``spike`` is stored under
    ``spikes`` in the config, which is the kind of detail that belongs in one
    place rather than in the page.
    """
    config = store.config.get("config", {})
    key = "spikes" if detector == "spike" else detector
    band = config.get(key, {}).get("band")
    if not band:
        band = config.get("bands", {}).get("ripple", [80.0, 250.0])
    return float(band[0]), float(band[1])


def event_from_record(store: ResultStore, record: dict):
    """Rebuild the :class:`~onset_hfo.detectors.base.Event` the store describes.

    Every field is copied from the stored record, and that is the point. The
    event figure prints the peak frequency, the prominence over background and
    the amplitude in its subtitle; building the event from times alone leaves
    those at their defaults, and the figure then says ``peak nan Hz, nan dB``
    directly underneath a panel showing 192 Hz and 12.5 dB. Two numbers
    disagreeing on one screen is the failure mode this whole project is
    organised against, so the mapping lives here, once, with a test.

    Nothing is recomputed: the values are the pipeline's, carried through.
    """
    from onset_hfo.detectors.base import Event

    def _get(key, default):
        value = record.get(key, default)
        return default if value is None else value

    return Event(
        channel=str(record["channel"]),
        start=float(record["start"]),
        stop=float(record["stop"]),
        detector=str(record["detector"]),
        band=detector_band(store, str(record["detector"])),
        score=float(_get("score", 0.0)),
        peak_amplitude_uv=float(_get("peak_amplitude_uv", 0.0)),
        peak_frequency_hz=float(_get("peak_frequency_hz", float("nan"))),
        spectral_prominence_db=float(_get("spectral_prominence_db", float("nan"))),
        n_peaks=int(_get("n_peaks", 0)),
        n_cycles=float(_get("n_cycles", float("nan"))),
        accepted=bool(_get("accepted", True)),
        reject_reason=record.get("reject_reason") or None,
        co_occurs_with_spike=bool(_get("co_occurs_with_spike", False)),
    )


def evidence_table(store: ResultStore, channel: str, detector: str | None = None,
                   k: int = 5) -> pd.DataFrame:
    """The citable windows on one channel, strongest first."""
    rows = store.evidence(channel, detector=detector, k=k)
    return pd.DataFrame(rows)


def disagreement_table(store: ResultStore) -> pd.DataFrame:
    """Channels the two detectors rank very differently.

    Reported rather than resolved, which is the pipeline's position: the
    report states both ranks and picks neither, and so does this page.
    """
    return pd.DataFrame(store.disagreements())


def agreement_note(store: ResultStore) -> dict:
    """Event-by-event agreement between the detectors, as stored."""
    return store.detector_agreement()


def citations(store: ResultStore, evidence_ids: list[str]) -> list[dict]:
    """Resolve what an answer cited, and flag anything that does not resolve.

    An id the store cannot resolve is the interesting case: it means the
    answer referred to a window that was never retrieved. The guard should
    have caught it before the answer was shown, so a row with
    ``resolved=False`` reaching this page is a bug worth seeing rather than
    an error worth hiding.
    """
    out = []
    for evidence_id in evidence_ids:
        record = store.resolve(evidence_id)
        out.append({"evidence_id": evidence_id, "resolved": record is not None,
                    **(record or {})})
    return out


def reload_spec(store: ResultStore) -> dict | None:
    """What :func:`onset_hfo.datasets.fetch_slice` needs to show this signal again.

    ``None`` when the analysis did not come from the public archive -- a
    synthetic run has no archive to fetch from, and the page says so instead
    of offering a button that cannot work.
    """
    provenance = store.provenance
    source = str(provenance.get("source", ""))
    if not source.startswith("openneuro:"):
        return None
    window = [provenance.get("slice_start_s"), provenance.get("slice_stop_s")]
    if None in window:
        return None
    return {
        "dataset": source.split(":", 1)[1],
        "subject": provenance.get("subject"),
        "run": provenance.get("run"),
        "session": provenance.get("session"),
        "task": provenance.get("task"),
        "acq": provenance.get("acq"),
        "t_start": float(window[0]),
        "t_stop": float(window[1]),
    }
