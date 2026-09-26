"""Re-drawing the signal window behind a stored event.

Separate from :mod:`app.panels` because this is the one place the interface
touches the recording again, and it should be obvious where that happens.

**Nothing is re-detected.** The event's times, frequency, prominence and
amplitude all came from the store; the recording is fetched a second time only
so the reader can *look* at the window those numbers describe.
:func:`app.panels.event_from_record` carries every stored field into the
figure, so its subtitle cannot disagree with the table above it.
"""

from __future__ import annotations

import streamlit as st

from app import panels


@st.cache_resource(show_spinner="fetching the recording…")
def _prepared(spec_key: str, spec: dict):
    """The preprocessed slice, cached: a Streamlit rerun happens on every click."""
    from onset_hfo.datasets import fetch_slice
    from onset_hfo.preprocess import prepare

    return prepare(fetch_slice(verbose=False, **spec), verbose=False)


def event_figure(store, record: dict):
    """``(figure, "")`` or ``(None, why not)``.

    Degrades rather than dangling a dead control: a synthetic analysis has no
    archive behind it, and an offline machine cannot reload a public
    recording. In both cases the stored measurements are still on screen and
    do not depend on the signal being available.
    """
    from onset_hfo.viz import plot_event

    spec = panels.reload_spec(store)
    if spec is None:
        return None, ("This analysis did not come from the public archive "
                      "(synthetic data has no recording to fetch), so the signal "
                      "cannot be drawn. The stored measurements are above.")
    try:
        prep = _prepared(str(sorted(spec.items())), spec)
    except Exception as exc:
        return None, (f"Could not load the signal: {type(exc).__name__}: {exc}. "
                      "The stored measurements are above and do not depend on it.")
    if record["channel"] not in prep.ch_names:
        return None, (f"{record['channel']} is not in the re-loaded montage, so its "
                      "window cannot be drawn.")
    return plot_event(prep, panels.event_from_record(store, record)), ""
