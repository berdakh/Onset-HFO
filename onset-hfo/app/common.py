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

from app import panels  # noqa: E402
from onset_hfo.config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from onset_hfo.store import ResultStore  # noqa: E402

#: Ships with the source so every page works on a fresh clone: 60 s of
#: sub-pt01 from OpenNeuro ds003029, the same slice the quickstart documents.
EXAMPLE = PROJECT_ROOT / "data" / "example_analysis"

#: Cohort-study tables, committed so the evaluation and outcome pages need no
#: download and no 90-minute rerun.
STUDIES = PROJECT_ROOT / "data" / "stability"


def banner() -> None:
    """The line that has to be on every page, and the tutorial link."""
    st.sidebar.markdown(
        "[Tutorial](https://berdakh.github.io/onset/tutorial.html) · "
        "[Onset project](https://berdakh.github.io/onset/) · "
        "[Code](https://github.com/berdakh/onset-hfo)")
    st.markdown(
        "<div style='background:#FDF6E6;color:#8A5A00;border:1px solid #EFDCAE;"
        "padding:8px 14px;border-radius:8px;font-size:13px'>"
        "<b>Research prototype — not a medical device.</b> Real public recordings, "
        "real expert markings, real surgical outcomes — and nothing here is validated "
        "for clinical use. Every number cites the window it came from. "
        "There is no recommendation anywhere in this product; the clinician decides."
        "</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_store(directory: str) -> ResultStore:
    return ResultStore(directory)


def analyses() -> list[Path]:
    """Every saved analysis: the shipped example first, then anything produced locally."""
    found = panels.find_results(RESULTS_DIR)
    if EXAMPLE.exists() and EXAMPLE not in found:
        found = [EXAMPLE, *found]
    return found


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
