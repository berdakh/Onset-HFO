"""Did the HFO map point at the tissue whose removal cured the patient?

The only test in this project whose reference standard is not another
algorithm — and the page that shows the two results which did *not* hold up.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.common import banner, study  # noqa: E402
from onset_hfo.config import PROJECT_ROOT  # noqa: E402

st.set_page_config(page_title="Onset-HFO · Outcome", layout="wide")
banner()

st.title("Surgical outcome, and whether any of it is stable")
st.caption("20 patients of ds003498 · whole 300 s runs · expert markings and our "
           "detector on the same channels · outcome from participants.tsv")

st.markdown("""
Every other number in this project compares an algorithm to another algorithm.
This one compares it to **what happened to the patient after surgery** — the
only reference standard in epilepsy surgery that is not another opinion.
""")

groups = study("outcome_groups_300s.csv")
tab_result, tab_window, tab_runs, tab_honesty = st.tabs(
    ["The result", "Does the window matter?", "Does the night matter?",
     "What it cannot support"])


with tab_result:
    st.subheader("Was the busiest fast-ripple channel inside the resection?")
    if len(groups):
        view = groups.query(
            "scope == 'reviewed' and band == 'fast_ripple' and "
            "metric in ['top_channel_resected', 'candidates_resected']")
        st.dataframe(
            view[["source", "metric", "mean_seizure_free", "mean_recurrence",
                  "auc", "auc_lo", "auc_hi", "p_permutation"]].round(3),
            width="stretch", hide_index=True)
    st.markdown("""
The direction is the one [Fedele et al. 2017](https://www.nature.com/articles/s41598-017-13064-1)
predicts — the study this dataset comes from — and with **13 seizure-free
patients against 7 recurrences it does not reach significance**. Nothing in the
whole study survives correction for its 36 comparisons, and
`min_detectable_auc(13, 7)` is **0.85**, so this cohort could not have
established either number.

**Concentration localises; proportion does not.** The share of a patient's HFOs
inside the resection shows nothing even for the experts (AUC 0.48 — a coin
flip), while *which channel is busiest* on the same events gives 0.71. "Most of
this patient's HFOs were inside the resection" is largely a statement about how
big the resection was.

**And the pipeline stopped picking a winner.** The data picks a single channel
in fewer than half the patients — worst case, 24 tied channels out of 37 — so
`candidates_resected` reports the share of the *tied set* that was removed.
Being honest about ties costs 0.017 AUC.
""")


with tab_window:
    st.subheader("The result that did not hold up")
    figure = PROJECT_ROOT / "docs" / "img" / "window_stability.png"
    if figure.exists():
        st.image(str(figure), width="stretch")
    st.markdown("""
The first version of this study used the **first 60 seconds** of each recording
and reported AUC **0.82, p = 0.007**. That number reached this project's README
and its landing page. Re-run on the whole 300-second run, the same code and the
same pre-specified metric give **0.71, p = 0.12**.

It was not cherry-picked: 60 s was chosen for download size before any outcome
data was touched, and the metric and band were pre-specified. It is something
more ordinary and easier to miss — **an analysis window short enough to change
the conclusion, never checked.**

Across five *disjoint* minutes of the same recordings the expert AUC spans
0.566–0.819 and its p-value **0.007 to 0.613**. The published 0.007 was the best
of five minutes.

One thing the study was not designed to find: **our detector is more stable
across windows than the expert markings** — AUC spread 0.09 against 0.25, same
busiest channel in 12/20 patients against 7/20. Read it as reproducibility, not
accuracy: the detector agrees with itself more than the experts agree with
themselves, and the experts are still closer to the outcome.
""")


with tab_runs:
    st.subheader("A whole run is a stable unit; a minute is not")
    figure = PROJECT_ROOT / "docs" / "img" / "run_stability.png"
    if figure.exists():
        st.image(str(figure), width="stretch")
    st.dataframe(pd.DataFrame([
        ("expert", "9/20", "18/20"),
        ("our RMS detector", "16/20", "16/20"),
    ], columns=["source", "same answer across 5 minutes of one run",
                "same answer across 5 runs (different nights)"]),
        width="stretch", hide_index=True)
    st.markdown("""
The archive holds **385 runs** across the 20 subjects — 1 to 39 each, not the
1–6 the `nights` column suggests, because a night contributes several
five-minute segments.

The expert markings **doubled their agreement with themselves** once the unit of
analysis became a whole run instead of a minute of one. Pooling a patient's runs
also resolves the ties: median candidate set 2 → 1, worst case 24 channels → 5.

**So the quantity this pipeline measures is a property of the patient, not of
the recording session.** That is the precondition for anything clinical, and it
now holds. Whether it *predicts outcome* is what twenty patients cannot settle.
""")


with tab_honesty:
    st.error("""
**Nothing here survives correction.** One of the study's 36 comparisons reaches
p < 0.05 uncorrected — our detector, *ripple* band (not the pre-specified
fast-ripple one), `candidates_resected`, AUC 0.786, p = 0.034, **Bonferroni
1.00**. In 36 uncorrected tests chance alone produces about two such rows, so
one is *fewer* than expected. It is on this page because leaving it out would
be the same sin as quoting it.
""")
    st.markdown("""
- **Thirteen against seven is a very small study.** Only a very large separation
  (AUC ≥ 0.85) could reach 80% power. A p above 0.05 here means *underpowered*,
  not *no effect*.
- **Thirty-six comparisons, uncorrected.** The Bonferroni column is in
  `groups.csv`; nothing survives it.
- **The published headline changed once already** (0.82 → 0.71). Treat that as
  the calibration for how much weight any single number here can carry.
- **Resected contacts that were never recorded cannot be scored.** In five
  temporal-lobe patients only 4 of 16 appear in the recording.
- **Retrospective, one centre, one surgical team.** The randomised
  [HFO Trial](https://www.thelancet.com/journals/laneur/article/PIIS1474-4422(22)00311-8/fulltext)
  (Lancet Neurology 2022) tested HFO-guided resection prospectively and did not
  find the benefit retrospective series report. This is a retrospective series.
- **Reproducing a retrospective result is evidence that the analysis is sound,
  not that the clinical claim is.**
""")
    st.info("Full design, every table, and the list of what this cannot support: "
            "`docs/OUTCOME.md`. Reproduce with `python -m onset_hfo.cli outcome` "
            "and `python -m onset_hfo.cli stability`.")
