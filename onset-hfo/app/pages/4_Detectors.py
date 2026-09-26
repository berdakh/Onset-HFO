"""How the detectors score against expert HFO markings on 20 real patients."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.common import banner  # noqa: E402

st.set_page_config(page_title="Onset-HFO · Detectors", layout="wide")
banner()

st.title("Detectors and how they are scored")
st.markdown("""
Two deliberately plain detectors that differ in **one thing only** — the feature
they threshold — so a disagreement between them is attributable to the feature
rather than to two implementations drifting apart.

- **RMS energy** (Staba et al. 2002) — root-mean-square of the band-passed signal.
- **Line length** (Gardner et al. 2007) — cumulative absolute difference.

Both then pass every candidate through the same **artifact rejection**, which is
where filter ringing is removed and where almost all of the precision comes from.

### The reference, and why it is agreement rather than accuracy

Scored against **expert HFO markings on 20 patients** of
[ds003498](https://openneuro.org/datasets/ds003498) — 41,187 marked events,
counted only on the channels the annotators actually reviewed. The reference is
the validated output of *another detector*, not a census of every oscillation,
so an event we find that it never proposed counts against us whether or not it
is real. **These are agreement numbers. Precision around 0.6 is a floor, not a
measurement.**
""")

RIPPLE = pd.DataFrame([
    ("ripple", "RMS", "1.5 SD", 0.43, 0.48, 0.42, 0.59),
    ("ripple", "RMS", "2.0 SD", 0.51, 0.38, 0.40, 0.66),
    ("ripple", "RMS", "3.0 SD", 0.59, 0.23, 0.30, 0.53),
    ("ripple", "RMS", "5.0 SD (shipped default)", 0.59, 0.12, 0.17, 0.37),
    ("ripple", "line length", "2.0 SD", 0.57, 0.27, 0.33, 0.51),
    ("fast ripple", "RMS", "2.0 SD", 0.09, 0.53, 0.15, 0.44),
    ("fast ripple", "RMS", "3.5 SD", 0.33, 0.29, 0.30, 0.59),
    ("fast ripple", "RMS", "5.0 SD", 0.54, 0.20, 0.26, 0.61),
    ("fast ripple", "RMS", "8.0 SD", 0.72, 0.11, 0.16, 0.54),
], columns=["band", "detector", "threshold", "precision", "recall", "F1",
            "channel-rank ρ"])

st.subheader("Cohort means, 20 patients")
band = st.radio("Band", ["ripple", "fast ripple"], horizontal=True)
st.dataframe(RIPPLE[RIPPLE["band"] == band].drop(columns="band"),
             width="stretch", hide_index=True)

st.subheader("What the sweep changed")
st.markdown("""
**The shipped default was wrong for interictal work, and now that is measured.**
At 5.0 SD — Staba's published value, which this pipeline inherited — the ripple
detector finds 12% of the expert-marked events and ranks channels at ρ = 0.37.
At 2.0 SD it finds 38% and ranks at ρ = 0.66. The default stays at 5.0 so that
published ictal results do not silently change; `--threshold
interictal-agreement` selects the measured one.

**The two bands want different operating points, by a factor of two and a half.**
Ripples rank best at 2.0 SD, fast ripples at 5.0 SD — an interior maximum of a
2–10 sweep, not a boundary. Running the ripple value in the fast-ripple band
drops precision from 0.54 to **0.09**: a mean of 1,142 detections per 60 s
against 228 expert-marked events. One threshold for both bands is a bug, not a
simplification, and it flattened the outcome result on the next page until it
was found.

**Channel ranking is the number that matters.** Nobody operates on an event;
they operate on tissue. ρ is the agreement between our per-channel rates and the
experts', and it peaks at a different threshold from F1 — which is why both are
reported instead of one headline.
""")

st.subheader("Against known truth, where accuracy *can* be measured")
st.dataframe(pd.DataFrame([
    ("RMS energy (ripples)", "0.956 ± 0.017", "0.528 ± 0.073", 0.679),
    ("Line length (ripples)", "0.957 ± 0.017", "0.550 ± 0.078", 0.697),
    ("Interictal discharges", "0.998 ± 0.004", "0.844 ± 0.037", 0.914),
], columns=["detector", "precision", "recall", "F1"]),
    width="stretch", hide_index=True)
st.caption(
    "Three synthetic recordings where every event is known by construction. "
    "Artifact rejection is what earns that precision: before it the RMS detector "
    "scores 0.63 — nearly all of its false positives are large transients ringing "
    "through the band-pass — and after it, 0.97, for about one point of recall.")

st.info(
    "Reproduce: `python -m onset_hfo.cli benchmark` (real data, expert markings) and "
    "`python -m onset_hfo.cli evaluate --seeds 1 7 42` (synthetic truth). "
    "Full tables and the reasoning: `docs/EVALUATION.md`.")
