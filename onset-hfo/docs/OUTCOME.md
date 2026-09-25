# Does the HFO map point at the tissue whose removal cured the patient?

Every other number in this project compares an algorithm to another
algorithm, or to a human reading the same screen. This one compares it to
**what happened to the patient after surgery** — the only reference standard
in epilepsy surgery that is not another opinion.

Run it:

```bash
python -m onset_hfo.cli outcome            # ~5 minutes, 20 subjects, cached after the first run
```

Everything below comes out of that command. Tables are in
`artifacts/results/outcome_ds003498/`.

---

## The headline, in one paragraph

On 60 seconds of interictal sleep from each of 20 patients, **the single
channel with the most expert-marked fast ripples was inside the resection in
12 of 13 patients who became seizure-free, and in only 2 of 7 whose seizures
returned** (AUC 0.82, 95% CI 0.64–1.00, permutation p = 0.007). That is the
published claim of Fedele et al. 2017 — the study this dataset comes from —
reproduced here from one minute of recording per patient.

Our own detector, at its measured fast-ripple operating point, points the
same way and does not get there: 10 of 12 versus 3 of 7, AUC 0.70,
p = 0.13. **The gap between those two rows is the honest measure of how far
this prototype is from clinical usefulness**, and it is the most valuable
number in this repository.

Three things that gap is not:

- It is not a power problem *for the expert arm*. The expert arm cleared the
  bar on the same 20 patients.
- It is not hidden by the choice of metric. Every metric we computed is in
  the table below, including the ones that show nothing.
- It is not a result. p = 0.007 in a table of 24 comparisons is p = 0.17
  after Bonferroni. What makes the expert row worth reporting is that it was
  not found by searching: it is the specific claim of the paper the dataset
  accompanies, tested in the direction that paper predicts.

---

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

Cohort: all 20 subjects, first 60 s of run-01, 2000 Hz. Detector: RMS, at the
per-band operating points measured in [EVALUATION.md](EVALUATION.md) (2.0 SD
ripples, 5.0 SD fast ripples). Scope `reviewed` = the channels the annotators
marked; `all` = every channel surviving preprocessing.

### `top_channel_resected` — the metric that carries the signal

| source | band | scope | seizure-free | recurrence | AUC (95% CI) | p | Bonferroni |
|---|---|---|---|---|---|---|---|
| **expert** | **fast ripple** | reviewed | **12/13 (0.92)** | **2/7 (0.29)** | **0.82 (0.64–1.00)** | **0.007** | 0.17 |
| rms | fast ripple | reviewed | 10/12 (0.83) | 3/7 (0.43) | 0.70 (0.49–0.92) | 0.129 | 1.00 |
| expert | ripple | reviewed | 5/13 (0.38) | 0/7 (0.00) | 0.69 (0.58–0.85) | 0.114 | 1.00 |
| rms | ripple | reviewed | 6/13 (0.46) | 0/7 (0.00) | 0.73 (0.62–0.89) | 0.051 | 1.00 |
| rms | fast ripple | all | 5/12 (0.42) | 3/7 (0.43) | 0.49 (0.27–0.72) | 1.000 | 1.00 |

### `share_in_rz` — the metric that carries nothing

| source | band | scope | median (free) | median (recur) | AUC (95% CI) | p |
|---|---|---|---|---|---|---|
| expert | fast ripple | reviewed | 0.50 | 0.30 | 0.57 (0.26–0.87) | 0.643 |
| rms | fast ripple | reviewed | 0.63 | 0.18 | 0.73 (0.45–0.94) | 0.115 |
| expert | ripple | reviewed | 0.18 | 0.30 | 0.43 (0.16–0.70) | 0.643 |
| rms | ripple | reviewed | 0.39 | 0.36 | 0.54 (0.27–0.81) | 0.817 |

`top3_resected` sits between the two, with nothing significant in any arm.
The full table — every metric, source, scope and band — is `groups.csv`.

---

## What the numbers say

**Fast ripples localise; ripples do not.** Every arm that shows anything is a
fast-ripple arm. The expert ripple arm reaches AUC 0.69 and p = 0.11; the
expert fast-ripple arm reaches 0.82 and p = 0.007 on the same patients, the
same channels and the same metric. This matches the clinical literature,
which has held for a decade that ripples are the less specific marker, and it
is the reason these 2 kHz recordings matter: the project's other dataset is
sampled at 1 kHz and cannot support fast-ripple analysis at all.

**Concentration localises; proportion does not.** `share_in_rz` shows nothing
even in the expert arm, while `top_channel_resected` on the same events gives
AUC 0.82. The clinically useful statement is not *most of this patient's HFOs
were in the resection* — that is mostly a statement about how large the
resection was — but *the one place generating the most fast ripples was
removed*. Anyone building a report from this pipeline should show a ranking,
not a percentage.

**Band-specific operating points are not a refinement, they are the
difference between a result and noise.** The first version of this analysis
used 2.0 SD in both bands, because that is the value the ripple benchmark
prefers. In the fast-ripple band 2.0 SD runs at precision 0.086 — a mean of 1,142
detections per 60 s against a mean of 228 expert-marked fast ripples — and
the detector's
outcome arm was correspondingly flat (AUC 0.59, p = 0.64). At the
fast-ripple band's own measured operating point (5.0 SD, rank ρ 0.610) the
same code gives AUC 0.70. Nothing about the outcome data was used to pick
either number; both come from channel-rank agreement with the experts, which
is a different question on data that says nothing about surgery.

**Our detector is event-starved at that operating point.** Five of twenty
subjects yield one or zero fast-ripple detections in 60 s, and one (sub-10)
yields none at all and is dropped from that arm. A per-patient statistic
computed from a single event is not a measurement. This is a limit of the
60-second window, not of the threshold: the source study scored whole nights.

**A negative result that is about us.** On the metric where the experts
separate the groups cleanly, our detector does not. That is the one
configuration in this design that is evidence against the detector rather
than against the sample size — it is exactly what the expert positive control
was built to distinguish — and it is the concrete target for the next version
of the detector.

---

## What this cannot support

- **Thirteen versus seven is a very small study.** `min_detectable_auc(13, 7)`
  returns **0.85**: with these group sizes, only a very large separation
  reaches 80% power. A p above 0.05 here means "underpowered", not "no
  effect". Every row carries an effect size and a bootstrap CI for that
  reason.
- **Twenty-four comparisons, uncorrected.** The Bonferroni column is in the
  table. Read the expert fast-ripple row as a *replication of a
  pre-specified published claim*, and every other row as hypothesis-
  generating.
- **Sixty seconds of one night** is not what the source study used, and the
  patients contributed 1–6 nights each. Run `--stop 300` for five minutes per
  patient; the loader will fetch the extra bytes.
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
python -m onset_hfo.cli outcome                        # the table above
python -m onset_hfo.cli outcome --stop 300             # five minutes per patient
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
