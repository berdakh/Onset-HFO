"""The agent: cites or refuses, and every citation opens to its window."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app import panels  # noqa: E402
from app.common import banner, pick_analysis  # noqa: E402
from app.signal import event_figure  # noqa: E402

st.set_page_config(page_title="Onset-HFO · Assistant", layout="wide")
banner()
store = pick_analysis()
if store is None:
    st.stop()

st.title(f"Onset-HFO Assistant — {store.subject}")
st.caption(
    "Evidence only · scope: this recording · read-only tools. The model chooses what "
    "to look up and how to say it; the pipeline decides what is true; a checker "
    "proves it for every answer. Try asking what to resect.")

backend_label = st.sidebar.selectbox(
    "Backend", ["scripted (offline, no model)", "ollama", "openai-compatible"])
st.sidebar.caption(
    "The scripted backend runs the whole loop with no model at all, which is how "
    "the guards are tested. Point at Ollama for a real open-weight model.")

SEEDS = [
    "Which channels have the highest ripple rate?",
    "Where do the two detectors disagree?",
    "Which channels should we resect?",
]


def _backend():
    if backend_label.startswith("ollama"):
        from onset_agent.backends import OllamaBackend

        return OllamaBackend()
    if backend_label.startswith("openai"):
        from onset_agent.backends import OpenAICompatBackend

        return OpenAICompatBackend(model="local")
    return None


def _ask(question: str):
    from onset_agent.agent import OnsetAgent

    try:
        return OnsetAgent(store, backend=_backend()).ask(question)
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        return None


key = f"chat::{store.dir}::{backend_label}"
if key not in st.session_state:
    st.session_state[key] = [(q, _ask(q)) for q in SEEDS]


def _render(question: str, answer) -> None:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        if answer is None:
            st.error("The backend could not be reached.")
            return
        if answer.refused:
            st.warning(f"**Refused.** {answer.text}")
            if answer.reason:
                st.caption(f"Reason: {answer.reason}")
        else:
            st.write(answer.text)
        if not answer.verified:
            st.error("This answer failed verification and was not trusted.")
        st.caption(f"{answer.backend} · tools: "
                   f"{' → '.join(answer.tools_called) or 'none'}")

        resolved = panels.citations(store, answer.evidence_ids)
        if not resolved and not answer.refused:
            st.caption("No individual event windows cited — this answer came from a "
                       "rate table. Ask about a channel's evidence to get citable "
                       "windows.")
        for record in resolved:
            if not record["resolved"]:
                st.error(f"`{record['evidence_id']}` does not resolve to a retrieved "
                         "window — the guard should have caught this.")
                continue
            with st.expander(f"{record['channel']} · {record['detector']} · "
                             f"{record['start']:.3f}–{record['stop']:.3f} s"):
                st.json({k: v for k, v in record.items() if k != "resolved"})
                figure, problem = event_figure(store, record)
                if figure is not None:
                    st.pyplot(figure, width="stretch")
                else:
                    st.info(problem)
        if answer.trace:
            with st.expander("trace"):
                for step in answer.trace:
                    st.code(str(step)[:2000], language="json")


for question, answer in st.session_state[key]:
    _render(question, answer)

asked = st.chat_input("Ask about this recording's evidence (try a channel name)")
if asked:
    st.session_state[key].append((asked, _ask(asked)))
    st.rerun()
