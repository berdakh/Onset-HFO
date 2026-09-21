# Localization: a learned per-contact model, and what it is worth

This document covers the third part of the repository — the part that learns
rather than thresholds. It is the implementation of the localization thesis
(Gabdullina), deliberately simplified to the smallest version that can be
*evaluated* rather than merely run.

The question, stated so it can fail:

> Can a contact-level classifier trained on ictal features localize the
> seizure onset zone better than a fixed-threshold biomarker such as HFO rate
> alone?

Note the comparator. "Better than chance" is not the bar — the pipeline
already ranks channels by ripple rate for free. A model that beats chance
while losing to the rate it was built from has established nothing, so every
table here scores the untrained baselines alongside the models.

---

## 1. The cohort, and why it had to come first

A learned model cannot be evaluated on one recording. Leave-one-patient-out
needs patients, and a classifier fitted to 71 channels of a single subject has
an AUPRC that means nothing. `onset_hfo/batch.py` runs the existing pipeline
across the archive and emits **one row per analysed channel per subject**.

```bash
python -m onset_hfo.learn cohort --dry-run   # the plan: no signal downloaded
python -m onset_hfo.learn cohort             # ~24 MB per subject, resumable
```

Of 33 subjects with signals and a clinical row, **31 were planned and 22
analysed**, giving **1466 channels, 301 labelled SOZ (20.5% prevalence)**. The
nine exclusions are all real and all worth knowing:

| why | subjects |
|---|---|
| sampling rate 250 Hz — Nyquist 125 Hz cannot carry an 80–250 Hz ripple | 4 (UMMC) |
| sampling rate 500 Hz — the band reaches 0.9 × Nyquist | 1 |
| HTTP 416: the archive's signal file is shorter than its own marked seizure time | 4 (UMF) |

That first row is a finding rather than a nuisance: **most of the UMMC
recordings cannot support ripple analysis at all**, so the cross-site study is
really NIH (13) versus JHH (6), with UMF and UMMC contributing one and two
subjects. Anyone planning a generalization experiment on this archive needs
that number before they design it.

Three things in the runner matter more than they look:

* **The window follows the seizure, not the clock.** Every centre marks its
  seizures at a different point in the file, so a fixed 50–110 s slice lands
  mid-seizure for one subject and in flat baseline for another. The window is
  `[onset − 25 s, onset + 35 s]` per subject.
* **Marker wording differs per centre and silently drops half the cohort.**
  NIH writes `onset`, UMMC `sz onset`, UMF `eeg sz start`, JHH
  `SZ EVENT # (EEG SZ)`. Matching only the first two produces no error — those
  subjects simply report "no seizure marked" and every rate-change feature
  comes back empty. JHH also writes pushbutton events with the same prefix, so
  the electrographic marker is preferred and the kind is recorded in
  provenance: a pushbutton press is when someone *reacted*, which can trail
  the EEG change by many seconds.
* **Failure is recorded, not raised.** A cohort run that quietly analysed 22
  of 31 subjects and never said so is how a paper gets a number nobody can
  reproduce.

---

## 2. Three protocols, and the gap between them is the result

```bash
python -m onset_hfo.learn evaluate
```

| protocol | trains on | answers |
|---|---|---|
| **within-subject** | other contacts of the *same* patient | Are SOZ and non-SOZ contacts separable at all? The **ceiling**. |
| **LOPO** | every *other* patient | The deployable number. |
| **leave-one-site** | patients at the *other* centres | What survives a change of hospital. |

**Within-subject is an upper bound, not a deployable model.** Using it would
require already knowing some of that patient's SOZ contacts, which is the
thing you are trying to find. It earns its place by bounding everything else:
if contacts cannot be separated *inside* one recording — where electrodes,
amplifier, sedation and seizure are all held constant — no cross-patient model
is going to. It is also the personalisation arm, the same subject-specific
versus subject-independent gap that appears in every BCI decoding study, here
applied to contacts instead of trials.

Measured on the 22-subject cohort, prevalence 0.205:

| protocol | model | AUPRC | lift over prevalence | AUROC | precision@5 |
|---|---|---|---|---|---|
| — | **rms rate (untrained)** | 0.447 | 2.18× | 0.718 | 0.473 |
| — | **line length (untrained)** | 0.467 | 2.28× | 0.721 | 0.509 |
| — | spike rate (untrained) | 0.352 | 1.71× | 0.662 | 0.436 |
| LOPO | logistic | 0.455 | 2.22× | 0.744 | **0.573** |
| LOPO | gradient boosting | **0.480** | 2.34× | 0.735 | 0.555 |
| leave-one-site | gradient boosting | 0.476 | 2.32× | 0.705 | 0.536 |
| **within-subject** | gradient boosting | **0.709** | **3.45×** | **0.901** | 0.636 |

Three readings, in order of how much they should change what you do next.

**The learned model barely beats the rate it was built from.** LOPO AUPRC
0.480 against 0.467 for simply ranking channels by line length. Thirteen
features, two model families and a cross-validation harness buy about one
AUPRC point. Precision@5 moves more (0.509 → 0.573), which is the metric a
clinician would actually feel, but nobody should describe this as the learned
model working.

**The ceiling is far above both.** 0.709 within-subject against 0.480 LOPO.
The features *are* separable — there is a great deal of signal in this table —
and most of it does not survive the move to a new patient. That gap is the
transfer problem, measured, and it is the single most useful number here.

**Cross-site costs almost nothing on top of cross-patient.** 0.476 versus
0.480. Whatever fails to transfer between patients has already failed by the
time you change hospital. That is a mildly surprising result and it says the
normalisation work belongs at the patient level, not the site level.

### A caveat that inverts the reading if you miss it

A within-subject fold trains on one patient's other contacts — 40 to 120 rows
here, split four ways. That is ample for logistic regression and thin for a
boosted model. **On a cohort with fewer contacts per patient, the
subject-specific score can fall below the LOPO score purely from data
starvation**, and reading that as "the features are not there" would be
exactly backwards. The test suite pins both regimes.

---

## 2b. Personalisation: what does a clinician's handful of labels buy?

The within-subject ceiling is a real number and an unusable system: to train
on some of a patient's contacts you must already know which are SOZ. But there
is a non-circular version of the same idea, and it is clinically ordinary. A
reviewer looking at a new implantation can point at a few contacts they are
confident about. What does that buy?

```bash
python -m onset_hfo.learn personalize
```

Leave-one-patient-out, except the target patient first contributes `k`
labelled contacts, chosen **before anything is predicted** and scored only on
the contacts the model was *not* given. At `k = 0` it is exactly
leave-one-patient-out, which makes the two ends of the curve comparable by
construction rather than by assertion.

Logistic regression, raw features, five label draws per patient (which
contacts a clinician happens to label is a lottery, and with five of them it
is a wide one):

| labels | fraction of the implantation | AUPRC | gap closed | AUROC |
|---|---|---|---|---|
| 0 *(= LOPO)* | 0% | 0.455 | 0% | 0.744 |
| 1 | 1.5% | 0.470 | 8% | 0.748 |
| 2 | 3% | 0.490 | 17% | 0.761 |
| **5** | **7.5%** | **0.531** | **38%** | 0.781 |
| 10 | 15% | 0.555 | 50% | 0.797 |
| 20 | 30% | 0.590 | 67% | 0.822 |
| 40 | 60% | 0.655 | 100% | 0.856 |

**Five contacts — under a tenth of the implantation — closes 38% of the gap
between a model that has never seen the patient and one that has seen all of
them.** Ten closes half. That is the most encouraging result in this
repository, and it is the one that says where the effort should go: not into
squeezing the last transferable feature out of a cohort, but into a workflow
where the clinician's existing opinion enters the model cheaply.

Two cautions. `labelled_fraction` is there because "40 labels" sounds modest
until it is 60% of the electrodes. And **precision@5 is not monotone along
this curve** (0.573, 0.609, 0.609, 0.582, 0.609, 0.627, 0.546) — with 22
patients it is a noisy statistic and AUPRC is the one to read.

The labels are sampled **blind to the features**, stratified so the draw
usually contains at least one of each class. Sampling the highest-rate
contacts instead would leak the model's own opinion back into its training set
and inflate every point on the curve; a test asserts the sampler never sees a
feature value.

---

## 3. Calibration and conformal prediction

```bash
python -m onset_hfo.learn uncertainty
```

A ranking is not enough to act on. "These contacts, ranked" invites the reader
to draw their own line; "these six contacts, with 90% coverage" is a statement
with a guarantee attached.

**Calibration.** Expected calibration error on the LOPO predictions is 0.042
uncalibrated — better than a typical deep model, because a shallow boosted
model on tabular features is not badly overconfident to begin with. Isotonic
regression is applied anyway; it is monotone, so it changes what the numbers
claim, never their order.

**Split conformal**, α = 0.1, calibrated on 9 patients and tested on 13:

| | |
|---|---|
| nominal coverage | 0.900 |
| **empirical coverage** | **0.866** |
| mean set size | 1.12 |
| singletons | 87.7% |
| ambiguous (both labels) | 12.3% |

**It undercovers, and that is the interesting result.** Conformal guarantees
require calibration and test data to be *exchangeable*. Patients are not:
different electrodes, different pathology, different seizure. The guarantee
does not fail loudly — it produces sets that look tight and cover 3.4 points
less often than advertised.

`exchangeability_stress_test` makes the violation deliberate by calibrating on
some centres and testing on another, so the degradation is measured rather
than assumed. Distribution shift breaking the uncertainty estimate that was
supposed to protect against distribution shift is a finding, and it is the one
that joins the generalization and uncertainty questions into a single story.

**The split is by patient, never by contact.** Contacts in one recording share
everything, so a contact-level split puts near-duplicates on both sides and
reports a coverage that will not survive a new patient.

### The clinically meaningful object

`patient_candidate_set` reduces the per-contact sets to one number per
patient: **how many contacts cannot be ruled out.** It is brutal reading. On
the test patients the candidate set ranges from 0 contacts (the model ruled
out everything, and missed all 9 true SOZ contacts) to 32 of 51. A patient
whose set is four contacts has an actionable result; a patient whose set is
forty has not been localized, however confident any individual score looked.

---

## 4. The coupling: the model is a tool in the planner's registry

This is where the two halves of the system meet, and it is three lines:

```python
from onset_agent import AnalysisSession, Rung, run_rung
from onset_agent.planner import ConformalWidth
from onset_hfo.models import SozModel

session = AnalysisSession(recording, soz_model=SozModel.load("artifacts/models/soz.pkl"))
result  = run_rung(session, Rung.S2, stop_rule=ConformalWidth(max_width=5))
```

The planner does not know how the model works and does not need to: it calls
`estimate_soz_probability`, which returns JSON like every other analyzer. What
comes back is a calibrated probability per channel and the **candidate set**.
Its width is the stopping condition — the planner keeps gathering evidence
while the set is too wide to act on.

Without a model the tool refuses with a message the planner can act on, and
everything else still runs. A missing model degrades the agent; it does not
break it.

`SozModel` carries its own contract — which features, in which order, under
which normalisation, the conformal threshold and the alpha it was calibrated
at — because a tool returning a bare probability invites exactly the
misreading the caveat exists to prevent. `model_features()` builds the feature
table with the *same* code the cohort was built with, so a feature means the
same thing at training time and at prediction time.

### What it does on a held-out patient

`sub-pt01`, excluded from both fitting and calibration (`--holdout sub-pt01`),
71 channels, 8 of them touching a labelled SOZ contact:

| ranking | top 5 | hits@5 | chance | p | hits@10 | p |
|---|---|---|---|---|---|---|
| **learned model** | AD3-AD4, AD2-AD3, AD1-AD2, ATT2-ATT3, PD2-PD3 | **5** | 0.56 | **0.0002** | **8 of 8** | **0.0002** |
| rms rate | PST2-PST3, ATT7-ATT8, ATT6-ATT7, AST2-AST3, ATT5-ATT6 | 0 | 0.56 | 1.00 | 1 | 0.72 |

The true SOZ is `AD1–4, ATT1–2, PD1–4`. The model puts every one of the eight
SOZ channels in its top ten; the ripple rate finds one. This is the first
non-null localization result in the repository, and it is worth being precise
about what it is: **one favourable patient.** The cohort-level number in §2 —
0.480 against 0.467 — is the honest aggregate, and it says the typical patient
gains very little. Quote the aggregate.

---

## 5. Running it

```bash
pip install -e ".[dev,ml]"

# The cohort table is committed (data/cohort/features.csv.gz, 269 KB), so
# everything below except `cohort` runs with no download at all.
python -m onset_hfo.learn evaluate                # works immediately

python -m onset_hfo.learn cohort --dry-run        # plan, download nothing
python -m onset_hfo.learn cohort                  # build the feature table
python -m onset_hfo.learn evaluate                # the table in §2
python -m onset_hfo.learn personalize            # the label-budget curve in §2b
python -m onset_hfo.learn uncertainty             # calibration + conformal + stress test
python -m onset_hfo.learn fit --holdout sub-pt01 --out artifacts/models/soz.pkl
```

---

## 6. What this is not

* **Not a clinical model.** Fitted on 22 ictal recordings from three centres,
  scored against a retrospective record, on patients whose outcome is already
  known.
* **Not interictal.** The clinical HFO literature measures interictal rate;
  every feature here comes from a 60-second window around a seizure. See
  [`DATA.md`](DATA.md).
* **Not a seizure-onset-zone detector.** A high probability is a statement
  about ripple-band and discharge features. The agent's own guard still
  refuses SOZ questions; localization is an offline metric computed by code,
  never a claim the model is allowed to make.
* **Not calibrated across sites.** §3 measures the coverage loss; it does not
  repair it.

The nearest honest summary: the features carry real information about which
contacts a clinician named, most of it does not transfer between patients, a
handful of labels from the patient in front of you recovers a good part of
what is lost, and the machinery to measure all three of those facts now exists
and is tested.
