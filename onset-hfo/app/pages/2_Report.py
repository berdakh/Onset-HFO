"""The structured, cited report — the authoritative artefact."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.common import banner, pick_analysis  # noqa: E402

st.set_page_config(page_title="Onset-HFO · Report", layout="wide")
banner()
store = pick_analysis()
if store is None:
    st.stop()

report = store.report
st.title(f"Onset-HFO Report — {store.subject}")
st.caption(f"pipeline {report.get('pipeline_version', '?')} · generated "
           f"{report.get('generated_at', '?')}")

summary = store.report_section("summary")
if summary:
    st.markdown(f"**Summary.** {summary}")


def _table(rows, label):
    st.subheader(label)
    if not rows:
        st.write("Nothing in this section for this recording.")
        return
    if isinstance(rows[0], str):
        for line in rows:
            st.write("• " + line)
        return
    flat = pd.json_normalize(rows)
    keep = [c for c in flat.columns if flat[c].map(lambda v: not isinstance(v, list)).all()]
    st.dataframe(flat[keep], width="stretch", hide_index=True)
    for row in rows:
        windows = row.get("evidence") or []
        if windows:
            st.caption(f"{row.get('channel', '?')} — evidence: " + ", ".join(
                str(w.get("evidence_id", w)) if isinstance(w, dict) else str(w)
                for w in windows[:3]))


_table(store.report_section("findings"), "Findings (top ranks, with evidence)")
_table(store.report_section("disagreements"), "Disagreements — stated, not resolved")

rate_change = store.report_section("rate_change")
if rate_change:
    _table(rate_change, "Rate change around the marked seizure")

left, right = st.columns(2)
with left:
    _table(store.report_section("data_quality"), "Data quality")
with right:
    _table(store.report_section("limitations"), "Limitations")

with st.expander("Methods"):
    _table(store.report_section("methods"), "")

st.success(
    "**There is no recommendation section** — not empty, absent. The schema has no "
    "such field. In the full system this report is written by a local open-weight "
    "model and validated before storage: every cited window must exist, every number "
    "must appear in a tool result, and directive language is rejected.")

markdown = Path(store.dir) / "report.md"
if markdown.exists():
    st.download_button("Download report.md", markdown.read_text(),
                       file_name=f"{store.subject}_report.md")
