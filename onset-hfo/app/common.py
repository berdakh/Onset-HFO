"""Shared chrome and loading for every page.

Mirrors `berdakh/onset`'s `app/common.py` on purpose: that prototype and this
one are meant to converge, and a reader moving between them should meet the
same banner, the same sidebar and the same page order. The difference is
underneath — that app builds a synthetic cohort at startup, this one reads
analyses and cohort studies computed from **public recordings of real
patients**.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app.panels import (  # noqa: E402,F401
    DATA_SENTENCE,
    DISCLAIMER_LEAD,
    DISCLAIMER_TAIL,
    EXAMPLE,
    STUDIES,
    analyses,
)
from onset_hfo.store import ResultStore  # noqa: E402


def banner() -> None:
    """The line that has to be on every page, and the shared sidebar row."""
    st.sidebar.markdown(
        # The first four entries are the shared link row: same names, same
        # order, in this app and in the teaching prototype's. See
        # docs/DUPLICATION.md. The last two necessarily differ -- they point
        # at the other instrument and at this repository's code.
        "[Clinical guide](https://berdakh.github.io/onset/clinical-guide.html) · "
        "[Implementation walkthrough](https://berdakh.github.io/onset/tutorial.html) · "
        "[Results & docs](https://berdakh.github.io/onset-hfo/) · "
        "[Onset project](https://berdakh.github.io/onset/) · "
        "[Teaching prototype](https://berdakh-onset.streamlit.app/) · "
        "[Code](https://github.com/berdakh/onset-hfo)")
    st.markdown(
        "<div style='background:#FDF6E6;color:#8A5A00;border:1px solid #EFDCAE;"
        "padding:8px 14px;border-radius:8px;font-size:13px'>"
        f"<b>{DISCLAIMER_LEAD}</b> {DATA_SENTENCE} {DISCLAIMER_TAIL}"
        "</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_store(directory: str) -> ResultStore:
    return ResultStore(directory)


def pick_analysis() -> ResultStore | None:
    """The sidebar's analysis picker, shared by the pages that need one."""
    found = analyses()
    if not found:
        st.error(
            "No saved analysis found, and the shipped example is missing. Produce "
            "one with `python -m onset_hfo.cli run --synthetic --figures`.")
        return None
    labels = {p: (f"{p.name} (shipped example)" if p == EXAMPLE else p.name)
              for p in found}
    choice = st.sidebar.selectbox("Recording", found, format_func=labels.get)
    return load_store(str(choice))


@st.cache_data(show_spinner=False)
def study(name: str):
    """One committed cohort-study table, or an empty frame if it is absent."""
    import pandas as pd

    path = STUDIES / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
