# Roadmap

What to build next, in order, with the reason each item matters. Items near
the top change what the results *mean*; items lower down make the system nicer
to use.

Each item names the files it touches and roughly what is involved, so someone
joining can pick one up without a handover meeting.

> **Updated after the outcome study.** Item 1 (interictal recordings) is
> **done** — every headline number now comes from `ds003498`. Item 2 is done
> twice over, and what it produced is a **new item 10** which is now the top of
> the list: the outcome study's conclusion changed when the analysis window
> grew from 60 s to the whole 300 s run, so *ranking stability* displaced
> *detector quality* as the thing to fix. Item 7 (whole recordings) is done for
> the outcome study and not for the pipeline generally.
>
> **From the earlier round.** Three entries
> changed status. Item 2 (outcome as reference standard) is **done**: the
> archive publishes curated SOZ contacts, `onset_hfo/cohort.py` reads them,
> and the 22-subject cohort table is committed. Item 3 (electrode geometry) is
> **not possible on this dataset** — there is no `electrodes.tsv` for any
> subject. Item 8 (the agent) is largely built, including the falsification
> suite; what remains of it is a real language model driving the ladder, which
> nothing else can substitute for. See
> [`ORCHESTRATION.md`](ORCHESTRATION.md) §8 and
> [`LOCALIZATION.md`](LOCALIZATION.md) §6 for the current boundary.

---

## 1. Interictal recordings — *done*

**Status.** `ds003498` — Zurich interictal slow-wave sleep at 2000 Hz, with
expert HFO markings — is wired in, and every headline number in the project now
comes from it: the detector benchmark (`docs/EVALUATION.md` §0) and the outcome
study (`docs/OUTCOME.md`). The ictal dataset remains as the quickstart example.
Fast ripples are analysable there too, which they never were at 1000 Hz.

The original reasoning, kept because it explains why this was item 1:

**Why.** Everything in this repository was measured on ictal data, because that
is what `ds003029` publishes with signals attached. The clinical HFO
literature measures **interictal** rate, usually in slow-wave sleep. Until the
pipeline runs on interictal data, its numbers cannot be compared to that
literature at all.

**What.** Find a public interictal iEEG dataset with a high sampling rate
(candidates: other OpenNeuro iEEG datasets; the Montreal/Zurich HFO datasets
distributed with `mne-hfo`; SWEC-ETHZ; institutional data under your own
approvals). Add a loader beside `fetch_slice` — the `Recording` dataclass is
the only contract.

**Do not be misled by this archive's `task-interictal` files.** There are 25
of them and all but one are metadata with no signal attached; exactly one
interictal recording (`sub-umf002`, run 01) ships an `.eeg`. Checking this
takes one S3 listing and saves a week of building against a dataset that is
not there.

**Touches.** `onset_hfo/datasets.py`, `docs/DATA.md`.

---

## 2. Outcome as the reference standard — *done twice; the ladder across the cohort is not*

**The stronger version is now built, on the other dataset.** ds003498 ships
both halves of the question — which contacts were resected
(`sourcedata/clinical_ch_sheet_zurich.xlsx`) and whether the patient became
seizure-free (`participants.tsv`) — so the HFO map can be tested against
surgery rather than against a clinician's marker. [`OUTCOME.md`](OUTCOME.md)
has the design and the result: the expert markings reproduce the published
finding (busiest fast-ripple channel inside the resection in 12/13
seizure-free vs 2/7 recurrences, AUC 0.82, p = 0.007) and **our detector does
not reach significance on the same metric** (AUC 0.70, p = 0.13).

**That gap was mostly the analysis window, which is a more important finding
than the gap.** Re-running on whole 300-second runs instead of the first 60
seconds moved the expert arm from AUC 0.82 (p = 0.007) to 0.71 (p = 0.12) and
ours from 0.70 to 0.67 — the gap closed because the *expert* number came down,
not because ours went up, and now nothing in the study reaches p < 0.05. Two
patients account for all of it.

So the top open item is **not** "make the detector better", it is
**item&nbsp;10: is the channel ranking stable at all?** A `top_channel_resected`
claim is an argmax over 6–65 channels whose Poisson rate intervals overlap.
`metrics.py` already refuses to rank channels whose intervals overlap when it
reports rates; the outcome metric does not, and it should. Adding that, and
plotting AUC against window length (cheap — the recordings are cached after
the first run), comes before any detector work.

The detector still has no morphology criterion where the source study's did,
and that remains worth building — just not as the thing that would have
explained a gap the data says is mostly noise.

**The earlier, weaker version (ds003029, clinician SOZ contacts) is also
done:**

**What turned out to be true.** The archive carries more than outcome scores.
`sourcedata/clinical_data_summary.xlsx` gives **curated clinician SOZ
contacts** per patient, alongside Engel, ILAE, surgery type and clinical
centre. `onset_hfo/cohort.py` reads it, reconciles the subject ids, decodes
the `S`/`F` outcome trap, and falls back to a local CSV or the free-text
markers — recording which source it used. 32 of the 35 subjects with signals
have a row, and parsed contact names match `channels.tsv` exactly.

**The cohort run is done.** `onset_hfo/batch.py` analysed **22 of 31 planned
subjects — 1466 channels, 301 labelled SOZ** — and the table is committed at
`data/cohort/features.csv.gz`, so the modelling half runs with no download.
`docs/LOCALIZATION.md` has the results: the learned model barely beats the
rate it was built from (0.480 against 0.467 AUPRC), the within-subject ceiling
is far above both (0.709), and five clinician-labelled contacts recover 38% of
that gap.

The nine exclusions are findings in their own right, and anyone planning a
cross-site experiment on this archive needs them first: **four UMMC recordings
sample at 250 Hz**, where an 80–250 Hz ripple is not measurable at all, and
four UMF signal files are shorter than their own marked seizure time. So
"leave-one-site-out" here is really NIH (13) against JHH (6), with UMF and
UMMC contributing one and two subjects.

**What is still to do.** The S0–S3 ladder has only ever been run on one
patient. Running it across the 22 gives per-rung SOZ localization with an
interval around it instead of `sub-pt01`'s anecdote, and stratification by
`seizure_free` so the trustworthy positives (clinician named it *and* the
surgery worked) are scored apart from the ambiguous ones. It is cheap now that
the recordings are cached — but it is worth doing **after** a real model has
driven the ladder at all (item 8), because 22 patients' worth of scripted-
planner numbers measure the script, not the thesis.

**Touches.** `onset_agent/orchestrate.py` (a cohort loop over the ladder),
`docs/EVALUATION.md`. `batch.py`, `cohort.py`, `models.py`, `uncertainty.py`
and `onset_agent/scoring.py` already exist.

---

## 3. Electrode geometry — *blocked on this dataset; needs a different archive*

**Why it matters.** Bipolar pairs are formed from consecutive contact
*numbers*. On a grid, numbering wraps at the end of a row, so some pairs join
contacts that are centimetres apart. Every rate computed on such a pair is
suspect.

**Why it cannot be done here.** `ds003029` publishes **no `electrodes.tsv` for
any subject**. There are no coordinates in the archive at all. Anything that
needs geometry — distance-based pairing, distance-to-neighbour features,
source localisation — requires a different dataset.

**What to do instead.** Write the code against the BIDS `electrodes.tsv`
schema so it is ready, pair by Euclidean distance with a maximum where
coordinates exist, fall back to numbering where they do not, and state which
rule was used in the report's method section. Then validate it on an archive
that ships coordinates.

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

## 7. Scaling: whole recordings instead of one-minute slices — *done for the outcome study*

**Status.** `onset-hfo outcome` now defaults to the whole 300-second run rather
than a 60-second slice, and 20 of them fit in memory one at a time without any
streaming work. That was enough to discover that the answer *depends* on the
window (item 10), which is the reason this item mattered. What remains is the
general case: `run` and `benchmark` still take a slice, and nothing here
streams, so a ten-minute or hour-long recording from another archive would not
fit.

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

**What has since been built.** The agent can now *drive* the analysis rather
than read it: a frozen JSON tool contract with run ids, nine live tools that
each re-run the real pipeline at parameters the planner chooses, an
append-only evidence store, the S0–S3 ablation ladder, three stopping rules,
and two verifiers with a measured delta between them. See
[`ORCHESTRATION.md`](ORCHESTRATION.md).

**What remains, in order:**

* **Falsification tests.** Shuffled channel labels; the leading channel
  removed; a recording with no epileptiform activity. If the agent still
  produces a confident ranking, that is the result a reviewer will look
  hardest for — better found in week three than in month six. Three functions
  over `AnalysisSession`.
* **A real-model measurement.** Every ladder number published so far comes
  from the deterministic scripted planner. That is the control, not the
  result.
* **The model and quantization ladder.** Qwen3-4B/8B/14B at FP16/8-bit/4-bit,
  scored on planning quality, tool-call validity, unsupported-claim rate,
  tokens and wall-clock. Needs a GPU; the backends already exist.
* **A tool the rest of the system does not have**: rejected events with their
  reasons, and comparison across two recordings of the same patient.

**Touches.** new `onset_agent/falsify.py`, new `onset_agent/benchmark.py`.

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

## 10. Is the channel ranking stable? — *now the top item*

**Why.** The outcome study's conclusion changed between a 60-second window and
a 300-second one (§2), moving on two patients out of twenty. Every clinical
claim this pipeline could ever make is a statement about *which channels*, so a
ranking that moves when the window moves undermines all of them — and unlike
most items here, this one is cheap and entirely offline once the recordings are
cached.

**What to do, in order.**

1. **Window-length curve.** Run `outcome_study` at 30, 60, 120, 180, 240, 300 s
   and plot AUC against window length, per source and band. If it has not
   settled by 300 s, say so; the archive has 1–6 runs per subject, so the next
   step is combining runs rather than lengthening one.
2. **Top-k stability within a recording.** Ten non-overlapping windows from the
   same run, same pipeline: how far does the top-5 move? This was already in
   the open questions below as "the cheapest experiment in the list and
   possibly the most informative". It is no longer optional.
3. **Refuse to rank tied channels.** `metrics.rank_channels` already knows how
   to decline when Poisson intervals overlap. `outcome._top_channel_resected`
   and `_top_k_resected` take an argmax with no such check. Give them one, and
   report the fraction of patients for whom the busiest channel is *not*
   distinguishable from the runner-up — which on this cohort may well be most
   of them.

**Touches.** `onset_hfo/outcome.py`, `onset_hfo/metrics.py`, `docs/OUTCOME.md`,
a new figure.

---

## Open questions worth someone's attention

* **Is the robustness rule the right rule?** `rank_channels` multiplies a
  channel's survey rate by how well it survived a stricter threshold, capped
  at 1. That is a design choice, not a law, and it is the mechanism by which
  the re-planning rungs can differ from the fixed ones at all. On `sub-pt01`
  the leading channels are so tied that a 5% penalty reshuffles them, which
  means the rule is currently doing more than the evidence supports. Should
  the ranking refuse to order channels whose intervals overlap?
* **Threshold choice.** On the simulator, 3–4 robust SDs beats the published 5
  on F1. Is that a property of the simulator's SNR distribution, or a real
  improvement? Answering it needs item 5.
* **Is the cycle-count criterion earning its place?** The ablation says the
  spectral check does nearly all the work. Keep, tighten, or delete?
* **Does the bipolar montage help or hurt for ripple *rate* specifically?**
  Easy experiment, currently unmeasured: run the whole pipeline in referential
  and bipolar montages and compare rankings.
* ~~**How stable is the ranking across windows?**~~ **Promoted to item 10**,
  because the outcome study answered part of it by accident: the conclusion
  moved between a 60 s and a 300 s window. It was described here as "the
  cheapest experiment in the list and possibly the most informative", and that
  turned out to be right.
