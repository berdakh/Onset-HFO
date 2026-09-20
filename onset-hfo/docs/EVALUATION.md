# Evaluation

## The honest summary

**On the public recording**, precision and recall are *not reported*, because
no one has marked every ripple in it. What can be measured there: event rates,
channel rankings, agreement between the two detectors, change across the
seizure, and a weak comparison against the contacts the clinician named.

**On synthetic data**, where every implanted event is known, the detectors are
scored properly. Those are the numbers below.

Reproduce everything with:

```bash
python -m onset_hfo.cli evaluate --seeds 1 7 42 --verbose
```

or run [`notebooks/03_validation_and_benchmark.ipynb`](../notebooks/03_validation_and_benchmark.ipynb).

---

## 1. Detector scores on synthetic data

Three recordings (seeds 1, 7, 42), 60 s each, 18 contacts → 15 bipolar
channels, 2000 Hz, ~200 implanted ripples, ~120 discharges, ~70 artifacts.

| detector | precision (mean ± sd) | recall | F1 |
|---|---|---|---|
| RMS energy (ripples) | 0.956 ± 0.017 | 0.528 ± 0.073 | 0.679 ± 0.065 |
| line length (ripples) | 0.957 ± 0.017 | 0.550 ± 0.078 | 0.697 ± 0.067 |
| interictal discharges | 0.998 ± 0.004 | 0.844 ± 0.037 | 0.914 ± 0.023 |

**How matching works.** A detection matches a truth event when they overlap in
time (20 ms tolerance) and the truth contact is one of the contacts the
detection's bipolar channel is built from — a pair `SA2-SA3` may legally claim
an event implanted on `SA2` or `SA3`. Recall counts truth events matched at
least once (an event found on two overlapping pairs counts once); precision
counts detections that match some truth event.

**Recall around 0.5 is the expected behaviour of a 5-SD threshold detector**,
not a defect. It finds events that are clearly above the background and misses
marginal ones. Section 3 shows exactly how that trades off.

---

## 2. What artifact rejection buys

Same recording, RMS detector, with and without the validation stage:

| stage | detections | precision | recall | false positives caused by |
|---|---|---|---|---|
| raw detector output | 238 | 0.630 | 0.614 | 86 artifacts, 2 spikes |
| after artifact rejection | 151 | 0.974 | 0.605 | 3 artifacts, 1 spike |

Precision 0.63 → 0.97 for one point of recall. The false positives it removes
are **filter ringing**: large sharp transients that an 80–250 Hz band-pass
turns into convincing ripples. This is the single most important stage in the
pipeline, and the reason the simulator implants artifacts at realistic
amplitudes (electrode pops in real recordings reach hundreds of microvolts).

### Which criterion actually does the work

| configuration | kept | precision | recall | F1 |
|---|---|---|---|---|
| no rejection at all | 238 | 0.630 | 0.614 | 0.622 |
| cycle count only | 237 | 0.629 | 0.610 | 0.619 |
| **spectral peak only** | 152 | **0.974** | 0.610 | 0.750 |
| both (the default) | 151 | 0.974 | 0.605 | 0.746 |

The spectral-peak check is responsible for essentially all of the gain; the
cycle count is a cheap guard that rarely fires. Both are kept — the cycle
count costs almost nothing and catches a failure mode the spectral check
cannot (an event too short to have a spectrum) — but nobody should believe the
cycle criterion is doing important work, and this table is here so nobody has
to take that on trust.

---

## 3. The threshold is a choice, not a fact

RMS detector, synthetic data, sweeping `threshold_sd`:

| threshold (robust SD) | detections | precision | recall | F1 |
|---|---|---|---|---|
| 3 | 215 | 0.916 | 0.719 | **0.806** |
| 4 | 185 | 0.968 | 0.662 | 0.786 |
| **5 (default)** | 151 | 0.974 | 0.605 | 0.746 |
| 6 | 113 | 0.973 | 0.486 | 0.648 |
| 7 | 69 | 0.957 | 0.338 | 0.500 |
| 8 | 39 | 0.949 | 0.210 | 0.343 |

**The default is not the best-scoring value here, on purpose.** A threshold of
3–4 SD scores higher F1 *on this simulator*, whose signal-to-noise
distribution is a guess. Tuning the default to it would be fitting the
detector to the fiction. Five robust SDs is Staba's published value; it is
what a reviewer will expect; and the curve is published here so anyone can
choose differently for their own data:

```python
from onset_hfo.config import PipelineConfig
cfg = PipelineConfig()
cfg.rms.threshold_sd = 4.0
```

---

## 4. How hard is the problem? Recall against SNR

`ripple_snr` is the implanted ripple's peak amplitude divided by the RMS the
ripple band already carries.

| SNR | precision | recall | F1 |
|---|---|---|---|
| 4 | 0.821 | 0.195 | 0.315 |
| 5 | 0.896 | 0.283 | 0.430 |
| 6 | 0.938 | 0.403 | 0.563 |
| 7 | 0.960 | 0.585 | 0.727 |
| 9 | 0.972 | 0.774 | 0.861 |
| 12 | 0.978 | 0.918 | 0.947 |

This is the curve to quote when someone asks "will it work on our
recordings?". The answer depends on their signal-to-noise ratio, not on ours.
Precision stays high throughout — when this detector fires, it is usually
right; what changes with SNR is how much it misses.

---

## 5. The spike detector's discontinuity check

Before adding it, the discharge detector scored precision ≈ 0.58, and **every**
false positive was an artifact step (a 5–60 Hz band-pass turns an electrode
pop into a textbook sharp wave). Measuring the maximum sample-to-sample jump
in the *unfiltered* signal, in robust SDs of that channel's derivative:

| population | 10th percentile | median | 90th percentile |
|---|---|---|---|
| genuine discharges | 2.8 | 3.6 | 4.9 |
| artifact steps | 24.6 | 45.5 | 63.9 |

A threshold of 10 separates them cleanly. Precision went to ≈ 1.00 with no
measurable loss of recall. The lesson generalises: **when a filter hides the
evidence, test on the raw samples.**

---

## 6. What can be checked on the real recording

`sub-pt01`, ictal run 01, 50–110 s, 71 bipolar channels, 1000 Hz. The whole
run takes about 7 seconds and produces 1732 RMS candidates (1549 accepted),
2387 line-length candidates (2188 accepted) and 490 interictal discharges.
Detected ripples have a median peak frequency of 146 Hz (10th–90th percentile
80–235 Hz) and a median spectral prominence of 8.9 dB above the recording's
own background.

**Rate change across the marked onset.** On the leading channels the ripple
rate is 0 before the clinician-marked onset and 144-146/min during the seizure
(`PST2-PST3`, `ATT7-ATT8`, `ATT6-ATT7`).
This is consistent with ictal HFOs, and it is also a reminder that this is
*ictal* data: the classical HFO literature measures interictal rate, which is
a different quantity (see `DATA.md`).

**Detector agreement.** The two detectors match about half their events
(Jaccard = 0.53 on this recording) and rank seven leading channels very
differently. The report names those channels rather than averaging them. Two
simple detectors on the same band disagreeing this much is itself a finding,
and the honest reading is: a single-detector rate table is less certain than
it looks.

**The clinician-marker check.** The events file names contacts near seizure
onset (`AD1-4, ATT1,2, G16, AST1,3, SLT1-3`). Permutation test on our ranking:

| | k = 5 | k = 10 |
|---|---|---|
| top channels touching a named contact | 1 | 2 |
| expected by chance | 1.05 | 2.15 |
| permutation p | 0.71 | 0.68 |

**No better than chance.** That result is reported here rather than buried,
and it is not evidence that the detector is broken. Reasons it is expected:
the markers are not a curated seizure-onset zone; the clinician was naming
what was visible at onset in the wideband signal, not where ripples were
densest; this is one 60-second ictal window; and ictal ripple energy spreads
far beyond the onset region. What it *does* establish is that **nothing in
this repository should be read as identifying a seizure-onset zone**.

The way to turn this into a real result is in the roadmap: interictal
recordings, several patients, and resection outcome as the reference.

---

## 7. Test suite

`pytest -q` — 58 tests, entirely offline, about six seconds. They cover the
primitives (robust scale, sliding features, threshold segmentation, bipolar
pairing), the detectors (hot channels found, events are oscillations, a flat
channel yields nothing, thresholds behave monotonically, reruns are
identical), validation (precision rises, recall survives, pops are rejected),
the report (no recommendation field, every finding carries evidence), and the
whole agent (tool schemas, argument validation, scope refusals, citation and
number verification, and the language-model path against a mock
OpenAI-compatible server that replies the way Qwen and Llama servers do —
including the two ways small models get it wrong).

## 8. What none of this establishes

Nothing here is a claim about clinical performance. Real ripples are not
Gaussian-windowed sinusoids; real artifacts are more varied; real recordings
contain physiological ripples in healthy tissue that this simulator does not
model at all. These numbers establish that the code does what it says on data
where the truth is known. They say nothing about patients.
