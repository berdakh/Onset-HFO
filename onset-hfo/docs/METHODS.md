# Methods

Every algorithm in the pipeline, the reference it comes from, the exact
parameter used, and the reason for any deviation. Parameters live in
[`onset_hfo/config.py`](../onset_hfo/config.py) and are documented at their
definition; this file explains the *thinking*.

---

## 0. Notation

* `x[n]` — one channel of the preprocessed signal, in microvolts, sampled at
  `fs`.
* **Robust SD** — `1.4826 × median(|x − median(x)|)`, the median-absolute-
  deviation estimate of the standard deviation. Used throughout instead of
  the plain SD.

> **Why robust statistics everywhere.** A channel packed with ripples inflates
> its own standard deviation, which raises its own threshold, which hides its
> own events. The detector would then be least sensitive exactly where the
> events are — the opposite of what is wanted. The MAD is almost unmoved by a
> few percent of outliers, so a busy channel keeps a sane threshold.
> `test_robust_scale_ignores_outliers` pins this behaviour down.

---

## 1. Preprocessing

`onset_hfo/preprocess.py`

| Step | Setting | Why |
|---|---|---|
| Channel selection | keep ECoG/SEEG/EEG, drop DC, trigger, ECG and names starting `DC`/`EKG`/… | those channels carry no brain signal but plenty of transients |
| Bad channels | drop everything the dataset flags `status = bad` | white matter, CSF, outside the brain, or noisy |
| High-pass | 1 Hz, zero-phase FIR | removes drift without touching anything we measure |
| Notch | mains frequency and harmonics below 0.9 × Nyquist, **2 Hz wide**, zero-phase | 180 Hz and 240 Hz sit *inside* the ripple band; a wide notch would remove real signal along with the interference |

The mains frequency is **taken from the dataset**, not configured: 60 Hz for
the US recordings in `ds003029`, 50 Hz for Zurich in `ds003498`. Setting
`PreprocessConfig.line_freq` overrides it deliberately; leaving it `None`
means "use the recording's". `Prepared.line_freq` then carries the value that
was actually filtered, so no later stage has to re-derive it — a quality
check that measures 60 Hz on a recording notched at 50 Hz is worse than no
check at all.
| Montage | bipolar: neighbouring contacts on the same electrode | a common reference shares its noise with every channel, producing "HFOs" that appear everywhere simultaneously |

**Zero-phase filtering** (forward–backward) is used everywhere because a
causal filter shifts events in time, and HFO analysis is about *when*
something happened.

**Bipolar caveat, stated in every report**: pairs are formed from consecutive
contact numbers. On a depth electrode or strip that is spatial adjacency. On a
rectangular grid the numbering wraps at the end of a row, so `G8-G9` may be two
contacts on opposite edges. Fixing this properly needs the electrode geometry,
which is in the roadmap.

---

## 2. HFO detection

`onset_hfo/detectors/engine.py` runs both HFO detectors; `rms.py` and
`line_length.py` differ *only* in one function. That is deliberate: it means
any disagreement between them is attributable to the feature, not to two
independent implementations drifting apart.

### The five steps

1. **Band-pass** to 80–250 Hz (zero-phase FIR, computed once and shared).
2. **Feature** in a sliding window of `rms_window_ms = 3 ms`:

   * RMS energy — `sqrt(mean(x²))` over the window (Staba et al., 2002);
   * line length — `mean(|x[n] − x[n−1]|)` over the window (Gardner et al., 2007).

3. **Threshold** at `median(feature) + threshold_sd × robustSD(feature)`,
   computed per channel. Default `threshold_sd = 5` for RMS (Staba's value),
   `3` for line length (its feature has a tighter distribution).
4. **Extent by hysteresis**: once a crossing exists, extend it outwards while
   the feature stays above `extend_sd = 2` robust SDs, merge crossings closer
   than `merge_gap_ms = 10 ms`, and require `min_duration_ms = 6 ms`.
   Without this the measured duration is only the loud middle of an
   oscillation, and every cycle count is an underestimate.
5. **Oscillation criterion**: at least `min_peaks = 6` rectified peaks above
   `peak_threshold_sd = 2` robust SDs of the band-passed signal. A ripple has
   roughly two rectified peaks per cycle, so six peaks is about three cycles.
   **This is the step that separates an oscillation from a single transient**,
   and it is the one most often omitted in quick implementations.

### Deviations from the original recipes, and why

| | Original | Here | Reason |
|---|---|---|---|
| Baseline | mean + 5 SD of a hand-picked quiet segment (Staba) | median + 5 robust SD of the whole channel | no hand-picking, no operator variance, no self-suppression on busy channels |
| Secondary threshold | 3 SD of the *baseline* | 2 robust SD of the whole channel | the whole-channel estimate is larger, so the multiplier is smaller; the criterion is the same idea |
| Band | 80–500 Hz | 80–250 Hz | the example recording is sampled at 1000 Hz (see `DATA.md`) |
| After detection | — | artifact validation (§4) | published detectors leave the ringing problem to the reader |

### Each event is then described

* **peak frequency** and **spectral prominence** — §3;
* **peak amplitude** of the band-passed signal (µV);
* **score** — how far the feature rose above its threshold, as a ratio. It is
  comparable within a detector and a channel. It is not a probability and not
  comparable across detectors, and the report never treats it as one;
* **cycles** — duration × peak frequency;
* **co-occurrence with a discharge** — flagged, never used to reject.

---

## 3. The event spectrum

`onset_hfo/spectral.py`

For each event, take the **unfiltered** signal in the event window plus 50 ms
either side and compute a single Hann-tapered, zero-padded periodogram.

Two deliberate choices:

* **Not Welch averaging.** An event is 30–100 ms long. Splitting it into
  overlapping Welch segments leaves 32-sample segments whose frequency grid is
  31 Hz wide — on which *every* ripple's "peak frequency" lands in the same
  bin (93.75 Hz at 1000 Hz sampling). That number is a property of the bin
  edges, not of the brain. This was visible in an early version of this
  pipeline and is the reason the estimator was changed.
* **Zero-padding to ≥ 8× the window length.** This interpolates the spectrum
  onto a fine grid. It does not create resolution the window cannot support —
  a 40 ms window still cannot separate 150 from 160 Hz — but the reported peak
  is now the peak of the actual spectrum. Verified against synthetic bursts: 100/150/200 Hz
  test oscillations are recovered at 102/148/201 Hz.

**Background fit.** Take `10*log10(PSD)` against `log10(f)` and fit a straight
line over 10 Hz–0.9×Nyquist, *excluding* the band of interest. The
**spectral prominence** is the largest amount, in dB, by which the in-band
spectrum exceeds that fitted line — and the **peak frequency** is where that
maximum occurs.

The second half of that sentence matters. Power falls as 1/f, so the raw
spectral maximum inside 80–250 Hz sits at the band's lower edge almost every
time: on the real recording, taking the raw maximum gave a median "peak
frequency" of 80 Hz for every channel, which says something about the band
and nothing about the oscillation. Measuring the peak as the maximum *excess
over background* moved that median to 146 Hz (10th–90th percentile
80–235 Hz), which is a ripple distribution.

---

## 4. Artifact rejection

`onset_hfo/validate.py`

The problem, in one sentence: **a sharp transient filtered at 80–250 Hz rings,
and the ringing looks exactly like a ripple.** Electrode pops, movement
artifacts and epileptiform spikes all do this. A detector that reports them is
not broken — it is measuring energy, and the energy is really there — but the
events are not oscillations.

Three checks per event:

1. **Cycle count** ≥ `min_cycles = 2` (duration × peak frequency). A floor
   against single transients.
2. **Spectral peak** ≥ `min_peak_prominence_db = 4` dB above the fitted
   background. This is the check that does the work: ringing has no peak of
   its own. A clean implanted ripple scores above 20 dB; on the real ictal
   recording the median accepted event scores 8.9 dB.
3. **Unmeasurable spectrum + large transient** → reject. When the window is
   too short to fit a background, an event coinciding with a very large
   low-frequency deflection is treated as probable ringing.

An HFO that **co-occurs with a discharge is not rejected** — ripples riding on
spikes are real and clinically interesting — it is flagged so the two
populations can be counted separately.

Rejected events stay in `events.csv` with `accepted = False` and a reason
string. Nothing is silently deleted.

Measured effect (synthetic data): precision 0.63 → 0.97, recall down about one
point. The ablation in notebook 3 shows the spectral check is
responsible for essentially all of that; the cycle check is a cheap guard that
rarely fires. Both are kept, and that is exactly the kind of claim this
repository expects you to re-check rather than trust.

---

## 5. Interictal discharge detection

`onset_hfo/detectors/spike.py`

1. Band-pass 5–60 Hz.
2. Peaks above `threshold_sd = 6` robust SDs, separated by at least
   `refractory_ms = 200 ms`.
3. **Sharpness**: the steepest slope inside the candidate must exceed
   `slope_sd = 5` robust SDs of the channel's first derivative — this is what
   separates a spike from a large slow wave.
4. Boundaries at half the peak amplitude; duration must be 10–200 ms.
   (Measured at half amplitude, so narrower than the 20–70 ms a clinician
   measures at the base.)
5. **Discontinuity check on the raw signal**: reject if the *unfiltered*
   sample-to-sample difference exceeds `max_raw_jump_sd = 10` robust SDs.
   After a 5–60 Hz band-pass an electrode pop looks like a textbook sharp
   wave; in the raw samples it is a step. Measured separation on synthetic
   data: genuine discharges score 3–5 on this feature, artifact steps 25–64.
   Adding it moved spike precision from 0.58 to ≈ 1.00 with no loss of recall.

It is not a trained classifier and does not try to be. It is a transparent
baseline a clinician can argue with, producing the quantity every HFO study
reports alongside HFO rate.

---

## 6. Rates, ranks and agreement

`onset_hfo/metrics.py`

* **Rate** = accepted events per minute per channel. Channels with zero events
  are kept in the table: "we looked and found nothing" is information.
* **Confidence interval**: a Poisson interval on the count. Sixty seconds
  turns a rate into a small count — 30/min *is* 30 events — and two channels
  whose intervals overlap are tied, not ranked. Every rate in the report
  carries one.
* **Rank**: position by rate, ties share a rank.
* **Detector agreement**: events are matched one-to-one by time overlap
  (20 ms tolerance) on the same channel; the report gives matched, only-A,
  only-B and a Jaccard ratio. It is deliberately *not* a kappa: there is no
  meaningful count of "events both detectors correctly did not detect", so a
  chance-corrected statistic would be invented rather than measured.
* **Disagreement**: ranks differing by ≥ `disagreement_ranks = 5` where at
  least one detector places the channel in its top `top_k = 5`. A disagreement
  deep in the list is not interesting; one about a leading channel is.
* **Rate change**: per-channel rate before versus during the marked seizure,
  reported with both counts, because a ratio built from three events and one
  event is a number, not a finding.

---

## 7. What is deliberately absent

* **No machine-learning classifier.** A threshold detector whose every step
  can be drawn on a whiteboard is the right first prototype; a learned model
  trained on 60 seconds of one patient would be a worse detector with better
  marketing.
* **No cross-patient normalisation.** Rates are compared within a recording
  only.
* **No source localisation, no electrode coordinates.** The archive ships
  them for some subjects; using them is in the roadmap.
* **No automatic operating point.** The threshold is a choice, stated in the
  report, with the curve in notebook 3.

---

## References

* Staba RJ, Wilson CL, Bragin A, Fried I, Engel J. *Quantitative analysis of
  high-frequency oscillations (80–500 Hz) recorded in human epileptic
  hippocampus and entorhinal cortex.* J Neurophysiol 88:1743–1752 (2002).
* Gardner AB, Worrell GA, Marsh E, Dlugos D, Litt B. *Human and automated
  detection of high-frequency oscillations in clinical intracranial EEG
  recordings.* Clin Neurophysiol 118:1134–1143 (2007).
* Zijlmans M, Jiruska P, Zelmann R, Leijten FSS, Jefferys JGR, Gotman J.
  *High-frequency oscillations as a new biomarker in epilepsy.*
  Ann Neurol 71:169–178 (2012).
* Worrell GA et al. *High-frequency oscillations in human temporal lobe:
  simultaneous microwire and clinical macroelectrode recordings.*
  Brain 131:928–937 (2008).
* Benar CG, Chauviere L, Bartolomei F, Wendling F. *Pitfalls of high-pass
  filtering for detecting epileptic oscillations: a technical note on "false"
  ripples.* Clin Neurophysiol 121:301–310 (2010). — the ringing problem, and
  the reason §4 exists.
* Li A et al. *Neural fragility as an EEG marker of the seizure onset zone.*
  bioRxiv 862797 (doi:10.1101/862797), published in Nature Neuroscience (2023)
  — the study behind the dataset; the archive's own `HowToAcknowledge` field
  asks for this citation.
