# Roadmap

What to build next, in order, with the reason each item matters. Items near
the top change what the results *mean*; items lower down make the system nicer
to use.

Each item names the files it touches and roughly what is involved, so someone
joining can pick one up without a handover meeting.

---

## 1. Interictal recordings — the one that changes the science

**Why.** Everything in this repository is measured on ictal data, because that
is what `ds003029` publishes with signals attached. The clinical HFO
literature measures **interictal** rate, usually in slow-wave sleep. Until the
pipeline runs on interictal data, its numbers cannot be compared to that
literature at all.

**What.** Find a public interictal iEEG dataset with a high sampling rate
(candidates: other OpenNeuro iEEG datasets; the Montreal/Zurich HFO datasets
distributed with `mne-hfo`; institutional data under your own approvals). Add
a loader beside `fetch_slice` — the `Recording` dataclass is the only contract.

**Touches.** `onset_hfo/datasets.py`, `docs/DATA.md`.

---

## 2. Outcome as the reference standard

**Why.** `participants.tsv` in `ds003029` carries Engel and ILAE scores and
whether a resection happened. That makes one real question answerable: *do
channels this pipeline ranks highly fall inside the resected volume more often
in patients who became seizure free?* That is the reference standard the field
actually uses, and it is sitting in the archive unused.

**What.** Join participant outcome to per-channel ranks, restricted to
subjects where resection information can be recovered. Report by outcome
group, with confidence intervals and a permutation null. Expect a null result
on a first pass — and report it.

**Touches.** new `onset_hfo/cohort.py`, `docs/EVALUATION.md`.

---

## 3. Electrode geometry, so "neighbouring" means neighbouring

**Why.** Bipolar pairs are formed from consecutive contact *numbers*. On a
grid, numbering wraps at the end of a row, so some pairs join contacts that
are centimetres apart. Every rate computed on such a pair is suspect.

**What.** Read the BIDS `electrodes.tsv` where present, pair by Euclidean
distance with a maximum, and fall back to numbering when coordinates are
missing. Say which rule was used in the report's method section.

**Touches.** `onset_hfo/preprocess.py`, `onset_hfo/datasets.py`, `report.py`.

---

## 4. More than two detectors, and a proper agreement analysis

**Why.** Two detectors matching about half their events is a finding worth
taking seriously. Three or four would show whether the disagreement is
idiosyncratic or structural.

**What.** Add the Hilbert-envelope (MNI) and short-time-energy detectors —
both fit `detect_with_feature` almost unchanged. Then report agreement as a
matrix, and rank channels by the *number of detectors* that place them in the
top group, which is more robust than any single rate.

**Touches.** `onset_hfo/detectors/`, `metrics.py`, `report.py`.

---

## 5. A small hand-annotated benchmark

**Why.** The simulator can only measure what it simulates. Two hundred
expert-marked events on real data would turn "0.97 precision on synthetic
ripples" into a statement about recordings.

**What.** Build a review UI (the event figure already exists — it needs
keyboard shortcuts and a CSV writer), mark a few hundred candidate windows
from a public recording with two reviewers, publish the annotations, and score
against them. Include the inter-rater agreement: it is the ceiling on any
detector's measurable performance.

**Touches.** new `onset_hfo/review.py`, `notebooks/04_annotation.ipynb`.

---

## 6. Physiological versus epileptic ripples

**Why.** The single biggest scientific gap. A high ripple rate in healthy
occipital cortex is not a finding, and nothing in this prototype can tell the
two apart.

**What.** Start with what is measurable: co-occurrence with discharges (already
flagged), waveform morphology, spectral shape, and relation to sleep state
where sleep is annotated. Report the sub-populations separately rather than
merging them into one rate.

**Touches.** `onset_hfo/validate.py`, `metrics.py`, `report.py`.

---

## 7. Scaling: whole recordings instead of one-minute slices

**Why.** A minute is enough to demonstrate a method and not enough to measure a
patient. Rates in clinical studies come from ten-minute or hour-long
interictal windows.

**What.** Stream the byte-range loader in chunks with overlap handling; keep
memory flat; cache per-channel baselines rather than recomputing them.
Parallelise across channels — the detectors are embarrassingly parallel.

**Touches.** `onset_hfo/datasets.py`, `pipeline.py`.

---

## 8. The agent: more tools, and a measured evaluation of it

**Why.** The agent currently answers questions about one analysis. The obvious
next step — "compare this patient's ranking to their previous recording" —
needs multi-analysis tools, and a way to evaluate whether the agent's answers
are actually *useful* rather than merely verified.

**What.**
* Tools: rejected events and their reasons; comparison across two stores;
  a tool that returns the event figure as an image.
* An agent benchmark: a fixed question set with expected tool calls and
  expected refusals, scored per model. That turns "Qwen 7B is better than 1.5B
  at this" from an impression into a number, and it is cheap to build — the
  scripted backend already defines the shape of a correct trace.

**Touches.** `onset_agent/tools.py`, new `onset_agent/benchmark.py`.

---

## 9. Interface

**Why.** Everything here is a notebook or a CLI. The Onset project has a
Streamlit application; this pipeline should feed it.

**What.** A page that loads a saved results directory, shows the ranking, the
event figure on click, the disagreements, and the agent's chat box — with
every answer's citations resolving to the window the reader can see. Reuse
`ResultStore`; do not let the UI compute anything.

**Touches.** new `app/`, reusing `onset_hfo/store.py` and `viz.py`.

---

## Open questions worth someone's attention

* **Threshold choice.** On the simulator, 3–4 robust SDs beats the published 5
  on F1. Is that a property of the simulator's SNR distribution, or a real
  improvement? Answering it needs item 5.
* **Is the cycle-count criterion earning its place?** The ablation says the
  spectral check does nearly all the work. Keep, tighten, or delete?
* **Does the bipolar montage help or hurt for ripple *rate* specifically?**
  Easy experiment, currently unmeasured: run the whole pipeline in referential
  and bipolar montages and compare rankings.
* **How stable is the ranking across windows?** Ten one-minute windows from
  the same recording, same pipeline: how much does the top-5 move? This is the
  cheapest experiment in the list and possibly the most informative.
