"""The lab, the measured state of the work, and what is open."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import banner  # noqa: E402

st.set_page_config(page_title="Onset-HFO · Research", layout="wide")
banner()

st.title("Research programme")

st.markdown("""
**Lab.** Brain–Machine Interfaces Laboratory, School of Computing and Artificial
Intelligence, Nazarbayev University. PI: Berdakh Abibullaev. Focus: deep
learning for EEG/iEEG signal processing, BCI, and clinical translation for
epilepsy.

**Where this sits.** Onset-HFO is the signal-processing and evidence half of the
[Onset](https://berdakh.github.io/onset/) project, built on **public recordings
of real patients** rather than a simulator, so that every claim can be checked
against a reference someone else published.
""")

st.subheader("What is measured, honestly")
st.markdown("""
| Claim | Status |
|---|---|
| Detects HFOs in real interictal recordings | **Yes** — agreement with expert markings, ρ ≈ 0.66 on channel ranking |
| Beats a published detector | **No, and not claimed.** The reference *is* a published detector; these are agreement numbers |
| The map relates to surgical outcome | **Direction yes, significance no.** AUC 0.71, p = 0.12 on 13 vs 7 patients; nothing in the study survives correction for its 36 comparisons |
| The measurement is stable | **Across nights yes** (18/20 patients), **within a minute no** — a 60 s window changed the headline once |
| Distinguishes epileptic from physiological ripples | **No.** Nothing in the pipeline tries. The single biggest scientific gap |
| Validated for clinical use | **No.** Not a medical device |
""")

st.subheader("Two results this project reversed on itself")
st.info("""
**The 60-second headline.** An outcome result of AUC 0.82, p = 0.007 reached the
README and the landing page before the study was re-run on whole recordings,
where it became 0.71, p = 0.12. Corrected everywhere, with both tables kept side
by side rather than one quietly replacing the other.

**The detector gap.** That same 60-second run appeared to show our detector
trailing the experts, which made "add a morphology criterion" look like the top
priority. On whole runs the gap largely closed — because the *expert* number
came down, not because ours went up — so the priority changed to ranking
stability instead.
""")

st.subheader("What is open, in order")
st.markdown("""
1. **Physiological versus epileptic ripples.** A high ripple rate in healthy
   occipital cortex is not a finding, and nothing here can tell the two apart.
2. **A second cohort.** Everything rests on one centre, one annotation protocol,
   one surgical team. Needs an archive with HFO markings *and* resection *and*
   outcome — ds003498 is the only public one known to have all three.
3. **A hand-annotated benchmark.** The only way to turn *agreement* into
   *accuracy*: a few hundred expert-marked events on real data, reviewed here.
4. **More than two detectors.** Two matching about half their events is a
   finding; three or four would show whether that disagreement is structural.
5. **A real model driving the analysis ladder.** Every orchestration number so
   far comes from a scripted planner, so it measures the script.
""")

st.subheader("Principles inherited from the Onset project")
st.markdown("""
1. No patient on both sides of a split; normalisation fit on training patients only.
2. Every score comes with the window it looked at, or an empty evidence list and
   a note saying so.
3. Disagreement between methods is reported, not averaged away.
4. Nothing in the system recommends treatment; the clinician decides.
5. Public data first; partner-centre data only through de-identification.
6. **When a number changes, the old one stays visible next to the new one.**
""")

st.caption("Contact the PI for collaboration or to see the full system. "
           "Code: github.com/berdakh/onset-hfo · MIT licence.")
