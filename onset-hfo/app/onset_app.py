"""The reading interface: one saved analysis, on a page, with its evidence.

    streamlit run app/onset_app.py

What this is for
----------------

Every number this project produces already carries the signal window behind
it. Until now that was true only in a JSON file -- a reader could not *look*
at the window without writing code, which makes "evidence-based" a claim
rather than a property. This page closes that gap: a rate leads to the events
behind it, an event leads to the signal it was measured on, and an answer
from the agent leads to both.

The design rule, and what follows from it
-----------------------------------------

**The page shows; it does not decide.** Every figure comes from
:class:`~onset_hfo.store.ResultStore` through :mod:`app.panels`, which is the
same read-only boundary the agent is held to. The page has no thresholds, no
parameters and no analysis of its own, so there is no way for it to disagree
with the report it is displaying. :mod:`app.panels` documents the single
deliberate exception (the "does anything stand out" null hypothesis) and why
leaving it out would be worse.

Three things it deliberately does not do:

* **No recommendation.** Not an empty panel, not a greyed-out button: the
  concept does not exist here, exactly as it does not exist in the report
  schema.
* **No resolving of disagreements.** Where the two detectors rank a channel
  differently, both ranks are shown and neither is preferred.
* **No hiding of refusals.** When the agent declines a question, the page
  shows the refusal and the reason, because that is the system working.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import panels  # noqa: E402
from onset_hfo.config import RESULTS_DIR  # noqa: E402
from onset_hfo.store import ResultStore  # noqa: E402

st.set_page_config(page_title="Onset-HFO — evidence viewer", layout="wide")

BANNER = (
    "**Research prototype — not a medical device.** Not validated on patients, "
    "not a diagnosis, and not a seizure-onset-zone finder. A high event rate is "
    "a measurement; physiological ripples occur in healthy tissue."
)


# --------------------------------------------------------------------------
# Loading. Cached per directory, because a page rerun must not re-read disk.
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _store(directory: str) -> ResultStore:
    return ResultStore(directory)


@st.cache_resource(show_spinner="fetching the recording…")
def _prepared(spec_key: str, spec: dict):
    """Re-load the signal so an event window can be drawn.

    Nothing is re-detected: the event's numbers all came from the store, and
    this is only the trace they were measured on. Cached because the fetch is
    a network call and a Streamlit rerun happens on every click.
    """
    from onset_hfo.datasets import fetch_slice
    from onset_hfo.preprocess import prepare

    recording = fetch_slice(verbose=False, **spec)
    return prepare(recording, verbose=False)


def _event_figure(store: ResultStore, record: dict):
    """The three-panel figure for one stored event, or a reason there is none."""
    from onset_hfo.viz import plot_event

    spec = panels.reload_spec(store)
    if spec is None:
        return None, ("This analysis did not come from the public archive "
                      "(synthetic data has no recording to fetch), so the signal "
                      "cannot be shown. The stored measurements are above.")
    try:
        prep = _prepared(str(sorted(spec.items())), spec)
    except Exception as exc:  # offline, or the archive is unreachable
        return None, (f"Could not load the signal: {type(exc).__name__}: {exc}. "
                      "The stored measurements are above and do not depend on it.")
    if record["channel"] not in prep.ch_names:
        return None, (f"{record['channel']} is not in the re-loaded montage, so its "
                      "window cannot be drawn.")
    return plot_event(prep, panels.event_from_record(store, record)), ""


# --------------------------------------------------------------------------
# Sidebar: which analysis, which detector
# --------------------------------------------------------------------------

st.sidebar.title("Onset-HFO")
st.sidebar.caption("Evidence viewer")

root = st.sidebar.text_input("Results folder", value=str(RESULTS_DIR))
found = panels.find_results(root)

if not found:
    st.title("Onset-HFO — evidence viewer")
    st.warning(BANNER)
    st.error(
        f"No saved analysis under `{root}`. A folder qualifies when it contains "
        "`events.csv`. Produce one with:\n\n"
        "```bash\npython -m onset_hfo.cli run --synthetic --figures\n```"
    )
    st.stop()

choice = st.sidebar.selectbox("Analysis", found, format_func=lambda p: p.name)
store = _store(str(choice))
detectors = [d for d in store.detectors() if d != "spike"] or store.detectors()
detector = st.sidebar.selectbox("Detector", detectors)
st.sidebar.divider()
st.sidebar.caption(store.metadata().get("citation", ""))


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title(f"{store.subject} — {detector}")
st.warning(BANNER)

columns = st.columns(4)
for i, (label, value) in enumerate(panels.metadata_rows(store)):
    columns[i % 4].metric(label, value)

summary = store.report_section("summary")
if summary:
    st.info(summary if isinstance(summary, str) else str(summary))

tabs = st.tabs(["Ranking", "Evidence", "Disagreements", "Ask the agent", "Report"])


# --------------------------------------------------------------------------
# 1. Ranking -- with the null hypothesis attached to it
# --------------------------------------------------------------------------

with tabs[0]:
    note = panels.leader_note(store, detector)
    if note.get("available"):
        text = note["statement"]
        (st.success if note["distinguishable"] else st.error)(text)
        st.caption(
            f"{note['n_tied_with_leader']} of {note['n_channels']} channels have an "
            f"interval overlapping the leader's. Leader {note['leader_rate_per_min']}/min "
            f"(CI {note['leader_ci'][0]}–{note['leader_ci'][1]}), "
            f"median channel {note['median_rate_per_min']}/min.")
    table = panels.ranking_table(store, detector)
    if len(table):
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption("Rates are events per minute over the analysed window. Two channels "
                   "whose intervals overlap are tied, not ranked.")
    else:
        st.info(f"No rate table stored for {detector!r}.")


# --------------------------------------------------------------------------
# 2. Evidence -- a channel, its events, and the signal behind one of them
# --------------------------------------------------------------------------

with tabs[1]:
    channels = [row["channel"] for row in store.list_channels(detector)]
    if not channels:
        st.info("No channels to show.")
    else:
        channel = st.selectbox("Channel", channels, key="evidence_channel")
        events = panels.evidence_table(store, channel, detector=detector, k=8)
        if not len(events):
            st.info(f"No accepted {detector} events on {channel}.")
        else:
            st.dataframe(events, width="stretch", hide_index=True)
            labels = [f"{r.start:.3f}–{r.stop:.3f} s  ({r.peak_frequency_hz:.0f} Hz)"
                      for r in events.itertuples()]
            picked = st.selectbox("Show the signal for", range(len(labels)),
                                  format_func=lambda i: labels[i], key="evidence_event")
            record = events.iloc[picked].to_dict()
            st.code(record["evidence_id"], language=None)
            figure, problem = _event_figure(store, record)
            if figure is not None:
                st.pyplot(figure, width="stretch")
                st.caption("Wideband, band-passed, and the event's spectrum against the "
                           "recording's own 1/f background. A real oscillation leaves a "
                           "bump above the dashed line; filter ringing does not.")
            else:
                st.info(problem)


# --------------------------------------------------------------------------
# 3. Disagreements -- stated, not resolved
# --------------------------------------------------------------------------

with tabs[2]:
    agreement = panels.agreement_note(store)
    if agreement:
        cols = st.columns(3)
        cols[0].metric("Jaccard", agreement.get("jaccard", "-"))
        cols[1].metric("Matched events", agreement.get("n_matched", "-"))
        cols[2].metric("Tolerance", f"{agreement.get('tolerance_s', '-')} s")
    table = panels.disagreement_table(store)
    if len(table):
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption("Both ranks are shown and neither is preferred — the report takes the "
                   "same position. A single-detector rate table is less certain than it "
                   "looks.")
    else:
        st.success("The two detectors rank no channel very differently in this analysis.")


# --------------------------------------------------------------------------
# 4. The agent -- and every citation resolved to a window on this page
# --------------------------------------------------------------------------

with tabs[3]:
    st.caption(
        "The model chooses what to look up and how to say it. The pipeline decides "
        "what is true. Every citation below is resolved against this analysis, and "
        "an answer whose numbers do not appear in a tool result is discarded before "
        "you see it.")
    backend_name = st.selectbox(
        "Backend", ["scripted (offline, no model)", "ollama", "openai-compatible"])
    question = st.text_input(
        "Question", placeholder="Which channel has the highest ripple rate?")

    if st.button("Ask", type="primary") and question.strip():
        from onset_agent.agent import OnsetAgent

        backend = None
        try:
            if backend_name.startswith("ollama"):
                from onset_agent.backends import OllamaBackend

                backend = OllamaBackend()
            elif backend_name.startswith("openai"):
                from onset_agent.backends import OpenAICompatBackend

                backend = OpenAICompatBackend(model="local")
            answer = OnsetAgent(store, backend=backend).ask(question)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            answer = None

        if answer is not None:
            if answer.refused:
                st.error(f"**Refused.** {answer.text}")
                if answer.reason:
                    st.caption(f"Reason: {answer.reason}")
            else:
                st.markdown(answer.text)
            if not answer.verified:
                st.warning("This answer failed verification and was not trusted.")
            st.caption(f"Backend: {answer.backend} · tools: "
                       f"{' → '.join(answer.tools_called) or 'none'}")

            resolved = panels.citations(store, answer.evidence_ids)
            if not resolved and not answer.refused:
                st.caption(
                    "This answer cites no individual event windows — it came from a "
                    "rate table rather than from single events. Ask about the evidence "
                    "on a channel to get citable windows.")
            if resolved:
                st.subheader("What it cited")
                for record in resolved:
                    if not record["resolved"]:
                        st.error(f"`{record['evidence_id']}` does not resolve to a "
                                 "retrieved window — this should have been caught by "
                                 "the guard.")
                        continue
                    with st.expander(
                            f"{record['channel']} · {record['detector']} · "
                            f"{record['start']:.3f}–{record['stop']:.3f} s"):
                        st.json({k: v for k, v in record.items() if k != "resolved"})
                        figure, problem = _event_figure(store, record)
                        if figure is not None:
                            st.pyplot(figure, width="stretch")
                        else:
                            st.info(problem)


# --------------------------------------------------------------------------
# 5. The report, as written
# --------------------------------------------------------------------------

with tabs[4]:
    st.caption("The authoritative artefact. The page above is a way of reading it.")
    for section in ["findings", "disagreements", "rate_change", "data_quality",
                    "methods", "limitations"]:
        content = store.report_section(section)
        if not content:
            continue
        with st.expander(section.replace("_", " ").title(),
                         expanded=section == "limitations"):
            if isinstance(content, list) and content and isinstance(content[0], str):
                for line in content:
                    st.markdown(f"- {line}")
            elif isinstance(content, list) and content:
                st.dataframe(pd.json_normalize(content), width="stretch",
                             hide_index=True)
            else:
                st.json(content)
    markdown = Path(store.dir) / "report.md"
    if markdown.exists():
        st.download_button("Download report.md", markdown.read_text(),
                           file_name=f"{store.subject}_report.md")
