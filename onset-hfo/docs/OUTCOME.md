# Does the HFO map point at the tissue whose removal cured the patient?

Every other number in this project compares an algorithm to another
algorithm, or to a human reading the same screen. This one compares it to
**what happened to the patient after surgery** — the only reference standard
in epilepsy surgery that is not another opinion.

Run it:

```bash
python -m onset_hfo.cli outcome            # 20 patients, whole recordings (~2.2 GB, ~25 min)
python -m onset_hfo.cli outcome --stop 60  # the first minute only -- a different answer
```

Everything below comes out of that command. Tables are in
`artifacts/results/outcome_ds003498/`.

---

## The headline, in one paragraph

**Nothing here separates the groups at p < 0.05, and the most important
finding is that an earlier version of this analysis did.** On the whole
recording (300 s per patient, every expert marking), the channel with the
most expert-marked fast ripples was inside the resection in 11 of 13 patients
who became seizure-free and 3 of 7 whose seizures returned — AUC 0.71, 95% CI
0.50–0.92, permutation p = 0.12. Our detector, same channels, same metric:
10 of 13 versus 3 of 7, AUC 0.67, p = 0.17.

On the **first 60 seconds** of the same recordings, the same code, the same
pre-specified metric, the expert arm gave AUC **0.82, p = 0.007** — which is
what this document reported first, and what got put on the project's landing
page. It does not survive using the other four minutes. Two patients account
for the whole difference (§ "The window changes the answer"), which is what a
20-patient study with a binary per-patient metric looks like when it is
underpowered: individual patients move the result.

So the honest reading is **three statements, in this order**:

1. **The published retrospective claim is directionally reproduced and
   statistically unsupported here.** Seizure-free patients are more likely to
   have had their busiest fast-ripple channel removed (0.85 vs 0.43), the
   effect is in the direction Fedele et al. 2017 predicts, and with 13 versus
   7 patients it does not reach significance. That is a power statement, not a
   refutation.
2. **Our detector is close to the expert arm, not far from it.** AUC 0.67
   against 0.71, with overlapping intervals. The gap the 60-second analysis
   appeared to show (0.70 vs 0.82) largely closed — and it closed because the
   *expert* number came down, not because ours went up.
3. **The ranking is window-dependent, and that is now the most concrete
   problem in this repository.** A conclusion that changes between minute one
   and minutes one-to-five is not yet a measurement.

## How the question is posed

| Ingredient | Where it comes from |
|---|---|
| Where the HFOs are | expert markings in `*_events.tsv`, **and** our detector, on the same channels |
| Which tissue was removed | `sourcedata/clinical_ch_sheet_zurich.xlsx`, parsed by `onset_hfo.clinical` |
| What happened to the patient | `participants.tsv`: `S` = seizure-free (13), `F` = recurrence (7) |

For each patient we compute how much of their HFO activity sat in tissue the
surgeon removed, and ask whether that is higher in the patients who became
seizure-free.

### Contacts, channels, and the resection margin

The clinical sheet names **contacts** (`ahr1-4` → AHR1…AHR4). The analysis
runs on **bipolar channels** (`AHR1-AHR2`), each of which sits between two
contacts. So a channel can be:

- **resected** — both contacts removed;
- **partial** — one contact removed. These straddle the resection edge;
- **spared** — neither removed.

`partial` is kept as its own label rather than folded into one of the
others, because folding it either way is a silent decision about the hardest
channels in the dataset. In the primary metric it counts in the denominator
and not the numerator, so a detector that fires along the margin gets no
credit for it. `share_in_rz_incl_partial` is reported alongside for anyone
who disagrees.

### What was dropped, and what could not be seen

- Contacts the source study excluded because electrical stimulation of them
  evoked motor or language responses are dropped (`--keep-eloquent` keeps
  them). 30 channels across 3 subjects.
- **Resected contacts that were never recorded cannot be scored.** In 15 of
  20 subjects every resected contact is present. In the other five — all
  temporal-lobe cases, where the source study kept only the three most mesial
  bipolar channels — only 4 of 16 listed resected contacts were recorded.
  `recordings.csv` carries `rz_coverage` per subject, and those five sit at
  0.25. Their numbers describe a quarter of a resection.
- The clinical sheet contains one typo (subject 15's excluded list says
  `1ll22-24` where every other token on the row says `tll`). The parser
  reports it rather than silently returning a smaller zone; three contacts
  are consequently not marked eloquent for that subject.

### Four metrics, because they disagree

| metric | question |
|---|---|
| `share_in_rz` | what fraction of this patient's HFO events were on resected channels? |
| `share_in_rz_incl_partial` | …counting margin channels as inside |
| `top_channel_resected` | was the single busiest channel removed? (0/1) |
| `top3_resected` | what fraction of the three busiest channels was removed? |

They disagree sharply, and that disagreement is a finding in itself — see
"What the numbers say" below.

---

## Results

Cohort: all 20 subjects, **the whole run** (300 s, 2000 Hz), every expert
marking. Detector: RMS, at the per-band operating points measured in
[EVALUATION.md](EVALUATION.md) (2.0 SD ripples, 5.0 SD fast ripples). Scope
`reviewed` = the channels the annotators marked; `all` = every channel
surviving preprocessing.

### `top_channel_resected` — the metric the source study's claim rests on

| source | band | scope | seizure-free | recurrence | AUC (95% CI) | p |
|---|---|---|---|---|---|---|
| expert | **fast ripple** | reviewed | **11/13 (0.85)** | **3/7 (0.43)** | **0.71 (0.50–0.92)** | 0.12 |
| rms | fast ripple | reviewed | 10/13 (0.77) | 3/7 (0.43) | 0.67 (0.45–0.89) | 0.17 |
| rms | ripple | reviewed | 8/13 (0.62) | 1/7 (0.14) | 0.74 (0.55–0.92) | 0.07 |
| expert | ripple | reviewed | 4/13 (0.31) | 2/7 (0.29) | 0.51 (0.29–0.73) | 1.00 |
| rms | fast ripple | all | 7/13 (0.54) | 3/7 (0.43) | 0.56 (0.34–0.78) | 1.00 |

### `share_in_rz` — the metric that carries nothing

| source | band | scope | median (free) | median (recur) | AUC (95% CI) | p |
|---|---|---|---|---|---|---|
| expert | fast ripple | reviewed | 0.50 | 0.41 | 0.48 (0.22–0.77) | 0.94 |
| rms | fast ripple | reviewed | 0.80 | 0.22 | 0.71 (0.46–0.91) | 0.14 |
| expert | ripple | reviewed | 0.19 | 0.30 | 0.41 (0.15–0.68) | 0.54 |
| rms | ripple | reviewed | 0.48 | 0.38 | 0.52 (0.24–0.79) | 0.94 |

`top3_resected` sits between the two (expert fast ripple AUC 0.63, p = 0.35).
Nothing in any arm reaches p < 0.05. The full table — every metric, source,
scope and band — is `groups.csv`.

---

## The window changes the answer

This is the finding worth more than any row above, and it exists only because
the analysis was re-run on the full recording rather than declared finished on
the first minute.

| | expert, fast ripple | rms, fast ripple |
|---|---|---|
| **first 60 s** | 12/13 vs 2/7 — AUC **0.82**, p = **0.007** | 10/12 vs 3/7 — AUC 0.70, p = 0.13 |
| **whole 300 s** | 11/13 vs 3/7 — AUC 0.71, p = 0.12 | 10/13 vs 3/7 — AUC 0.67, p = 0.17 |

Five times the data, a weaker result. Two patients account for all of it:

- **sub-18** (recurrence). In the first minute the busiest expert-marked
  fast-ripple channel was outside the resection; over five minutes it is
  inside. A recurrence patient whose HFO focus *was* removed counts against
  the hypothesis, so this flip costs twice.
- **sub-15** (seizure-free). The opposite: inside at 60 s, outside at 300 s.

Our detector moved on two patients as well (sub-14 flipped out; sub-10, which
produced no fast-ripple detections at all in 60 s, produced 15 in 300 s and
rejoins the cohort — which is why its `n` goes from 12 to 13).

Three things follow, and they are the actionable part of this document:

**The 60-second number should never have been the headline.** It was not
cherry-picked — 60 s was chosen for download size before any outcome data was
touched, and the metric and band were pre-specified — but a window short
enough to change the conclusion is not a defensible analysis window, and the
default is now the whole run. Anything published from the 60 s window is
superseded by the table above.

**"Which channel is busiest" is a fragile statistic on this much data.** It is
an argmax over 6–65 channels whose rates have overlapping confidence
intervals. `metrics.py` already refuses to rank channels whose Poisson
intervals overlap when it reports rates; this metric does not, and it should.
That is a specific, small piece of work.

**A 20-patient study cannot distinguish these two results from each other.**
`min_detectable_auc(13, 7)` = 0.85. Both 0.82 and 0.71 sit below that floor,
so neither run had the power to establish its own number. The difference
between them is noise of exactly the size this cohort produces.

---

## What the numbers say

**Fast ripples localise better than ripples in the expert arm; in ours the two
bands are indistinguishable.** For the experts, fast ripples give AUC 0.71
against 0.51 for ripples, on the same patients, channels and metric — matching
a decade of clinical literature holding that ripples are the less specific
marker. For our detector the ripple arm (0.74, p = 0.07) actually edges the
fast-ripple arm (0.67, p = 0.17), and the intervals overlap heavily. Read that
as "our detector does not reproduce the band distinction", not as "ripples are
better": with this cohort neither number is established, and taking the
larger one because it is larger is the error this document exists to avoid.
Either way, it is why these 2 kHz recordings matter — the project's other
dataset is 1 kHz and cannot support fast-ripple analysis at all.

**Concentration localises; proportion does not.** `share_in_rz` shows nothing
in the expert arm at all (AUC 0.48 — a coin flip), while
`top_channel_resected` on exactly the same events gives 0.71. The clinically
useful statement is not *most of this patient's HFOs were in the resection* —
that is largely a statement about how large the resection was — but *the one
place generating the most fast ripples was removed*. Anyone building a report
from this pipeline should show a ranking, not a percentage. This is the one
conclusion that held identically at both window lengths.

**Band-specific operating points are not a refinement, they are the
difference between a result and noise.** The first version of this analysis
used 2.0 SD in both bands, because that is the value the ripple benchmark
prefers. In the fast-ripple band 2.0 SD runs at precision 0.086 — a mean of
1,142 detections per 60 s against a mean of 228 expert-marked fast ripples —
and the detector's outcome arm was correspondingly flat. At the fast-ripple
band's own measured operating point (5.0 SD, rank ρ 0.610) the same code
recovers a usable ranking. Nothing about the outcome data was used to pick
either number; both come from channel-rank agreement with the experts, which
is a different question on data that says nothing about surgery.

**The event-starvation problem was real and the full run fixes it.** At 5.0 SD
in 60 s, five of twenty subjects yielded one or zero fast-ripple detections
and sub-10 yielded none at all, so a per-patient statistic was being computed
from a single event. Over 300 s every subject produces detections and the
cohort is complete. That was a genuine defect of the 60-second window — it is
just not the defect that was holding the detector back.

**The positive control did its job, and what it revealed was not what it was
built to reveal.** It was built to tell "our detector is worse than the
experts" apart from "this study is underpowered". The 60-second run looked
like the first. The full run says the second: expert 0.71 and ours 0.67, both
null, intervals almost entirely overlapping. A control that changes the
conclusion when you give it more data has earned its place.

## What this cannot support

- **Thirteen versus seven is a very small study.** `min_detectable_auc(13, 7)`
  returns **0.85**: with these group sizes, only a very large separation
  reaches 80% power. A p above 0.05 here means "underpowered", not "no
  effect". Every row carries an effect size and a bootstrap CI for that
  reason.
- **Twenty-four comparisons, uncorrected.** The Bonferroni column is in the
  table; on the full run nothing survives it, and nothing reaches p < 0.05
  uncorrected either. Read every row as hypothesis-generating.
- **The published headline changed once already.** The 60-second version of
  this analysis reported AUC 0.82, p = 0.007 and it reached the project's
  README and landing page before the full run was done. Both are now corrected
  to the table above. Treat that as the calibration for how much weight any
  single number in this document can carry.
- **One full run of one night** is not what the source study used: the
  patients contributed 1–6 nights each and it scored all of them. The archive
  has 1–6 runs per subject; combining them is the obvious next extension, and
  given how much the answer moved between one minute and five, it should be
  done before any number here is quoted as stable.
- **One run per patient, no cross-validation, no held-out set.** Nothing here
  is a model that was fitted, so there is nothing to hold out — but there is
  also nothing here that has been shown to generalise to another cohort.
- **Retrospective, single centre, one surgical team.** The HFO Trial
  (Jacobs et al., *Lancet Neurology* 2022) tested HFO-guided resection
  prospectively and did not find the benefit that retrospective series
  reported. This analysis is a retrospective series. See
  [LIMITATIONS.md](LIMITATIONS.md).

---

## Reproducing and extending

```bash
python -m onset_hfo.cli outcome                        # the table above (whole runs)
python -m onset_hfo.cli outcome --stop 60              # the first minute: a different answer
python -m onset_hfo.cli outcome --stop 120 --out a     # your own window-sensitivity check
python -m onset_hfo.cli outcome --threshold 3.0        # one threshold in every band
python -m onset_hfo.cli outcome --detector line_length
python -m onset_hfo.cli outcome --keep-eloquent        # keep stimulation-positive contacts
```

In Python:

```python
from onset_hfo.outcome import outcome_study

result = outcome_study()
print(result.summary("top_channel_resected"))
print(result.verdict(band="fast_ripple"))
result.save()
```

`result.channels` has every channel of every subject with its zone, its
expert event count and ours — which is where to look first when a subject's
number is surprising.

**If you extend one thing, extend this.** Run the study at several window
lengths and plot the AUC against window length. Everything above says that
curve has not settled by 300 s, and it is the cheapest experiment left:
the recordings are already cached after the first run, so every further window
is compute only.

### Files written

| file | contents |
|---|---|
| `subjects.csv` | one row per subject × source × scope × band: the per-patient metrics |
| `channels.csv` | per-channel zone, expert events, our events — the audit trail |
| `recordings.csv` | what was analysed per subject, including `rz_coverage` |
| `groups.csv` | the group comparison, every metric |
| `participants.csv` | outcome, ILAE class, follow-up, epilepsy type, as published |
| `run.json` | dataset, window, detector, thresholds, power floor, verdict |

### Source

Fedele T, Burnos S, Boran E, Krayenbühl N, Hilfiker P, Grunwald T, Sarnthein J.
*Resection of high frequency oscillations predicts seizure outcome in the
individual patient.* Scientific Reports 7:13836 (2017).
doi:[10.1038/s41598-017-13064-1](https://doi.org/10.1038/s41598-017-13064-1).
Data: OpenNeuro [ds003498](https://openneuro.org/datasets/ds003498), CC0.
