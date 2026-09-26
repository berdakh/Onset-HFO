"""One recording: two detectors, their disagreement, and the signal behind any event."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import panels  # noqa: E402
from app.common import banner, pick_analysis  # noqa: E402

st.set_page_config(page_title="Onset-HFO · Recording", layout="wide")
banner()
store = pick_analysis()
if store is None:
    st.stop()

meta = store.metadata()
st.title(f"{store.subject}")
st.caption(
    f"{meta.get('source')} · {meta.get('task') or '-'} run {meta.get('run') or '-'} · "
    f"{meta.get('channels_analysed')} of {meta.get('channels_in_recording')} channels · "
    f"{meta.get('duration_s', '?')} s at {meta.get('sampling_rate_hz')} Hz"
    + (f" · marked seizure onset {meta['seizure_onset_s']:g} s"
       if meta.get("seizure_onset_s") is not None else ""))

detectors = [d for d in store.detectors() if d != "spike"]


# -- does anything stand out at all? --------------------------------------

st.subheader("Does any channel actually stand out?")
note = panels.leader_note(store, detectors[0] if detectors else None)
if note.get("available"):
    (st.success if note["distinguishable"] else st.error)(note["statement"])
    st.caption(
        f"{note['n_tied_with_leader']} of {note['n_channels']} channels have a rate "
        f"interval overlapping the leader's. Leader {note['leader_rate_per_min']}/min "
        f"(95% CI {note['leader_ci'][0]}–{note['leader_ci'][1]}), median channel "
        f"{note['median_rate_per_min']}/min.")
else:
    st.info("No rate intervals stored, so this question cannot be answered here.")


# -- both detectors, side by side, disagreement in amber -------------------

st.subheader("Channel ranking — evidence, not a recommendation")
frames = []
for name in detectors:
    table = panels.ranking_table(store, name)
    if len(table):
        frames.append(table.set_index("channel")[["rank", "rate_per_min"]]
                      .rename(columns={"rank": f"{name} rank",
                                       "rate_per_min": f"{name} /min"}))
if not frames:
    st.info("No rate tables stored for this analysis.")
    st.stop()

side_by_side = pd.concat(frames, axis=1).reset_index()
rank_columns = [c for c in side_by_side.columns if c.endswith("rank")]
if len(rank_columns) > 1:
    gap = side_by_side[rank_columns].max(axis=1) - side_by_side[rank_columns].min(axis=1)
    side_by_side["rank gap"] = gap
    side_by_side["disagree"] = gap >= 5
else:
    side_by_side["disagree"] = False
side_by_side = side_by_side.sort_values(rank_columns[0])

top_k = st.sidebar.slider("Show top", 5, max(10, len(side_by_side)), 15)
shown = side_by_side.head(top_k)


def _amber(row):
    return ["background-color:#FAEEDA" if row["disagree"] else "" for _ in row]


rate_columns = [c for c in shown.columns if c.endswith("/min")]
st.dataframe(
    shown.style.apply(_amber, axis=1).format({c: "{:.1f}" for c in rate_columns}),
    width="stretch", hide_index=True)
st.caption(
    "Amber rows: the two detectors rank this channel five or more places apart. "
    "**Disagreement is a finding, not noise** — both ranks are shown and neither is "
    "preferred. Rates are events per minute over the analysed window; two channels "
    "whose intervals overlap are tied, not ranked.")


# -- the signal behind one event ------------------------------------------

st.subheader("Evidence window")
st.caption("Every rate above is a count of events like this one. This is the window "
           "the number came from.")

channel = st.selectbox("Channel", shown["channel"].tolist())
detector = st.selectbox("Detector", detectors, index=len(detectors) - 1)
events = panels.evidence_table(store, channel, detector=detector, k=8)

if not len(events):
    st.warning(f"No accepted {detector} events on {channel}.")
else:
    labels = [f"{r.start:.3f}–{r.stop:.3f} s · {r.peak_frequency_hz:.0f} Hz · "
              f"{r.spectral_prominence_db:.1f} dB over background"
              for r in events.itertuples()]
    picked = st.selectbox("Window", range(len(labels)), format_func=lambda i: labels[i])
    record = events.iloc[picked].to_dict()
    st.code(record["evidence_id"], language=None)
    st.dataframe(events, width="stretch", hide_index=True)

    from app.signal import event_figure

    figure, problem = event_figure(store, record)
    if figure is not None:
        st.pyplot(figure, width="stretch")
        st.caption("Wideband, band-passed, and the event's spectrum against the "
                   "recording's own 1/f background. A real oscillation leaves a bump "
                   "above the dashed line; filter ringing does not — which is the "
                   "single most important check in the pipeline.")
    else:
        st.info(problem)
