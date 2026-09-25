# Limitations

Read this before quoting any number from this repository to anyone.

## What this is

A **first prototype**, written to be readable and checkable, running classical
detectors on one minute of one public recording, plus an agent that can read
the results and is prevented from inventing them.

## What it is not

* **Not a medical device.** Not certified, not validated, not suitable for any
  clinical decision.
* **Not a diagnosis, and not a seizure-onset-zone finder.** High event rate is
  a measurement. Physiological ripples occur in healthy tissue — mesial
  temporal structures and occipital cortex particularly — and a channel with a
  high rate may simply be healthy tissue that ripples. On the real recording
  here, the top-ranked channels do **not** overlap the contacts the clinician
  named more than chance predicts (`EVALUATION.md` §6).
* **Not validated on patients.** The detectors are now scored against expert
  HFO markings on 20 real subjects (`docs/EVALUATION.md` §0), which is a real
  benchmark and still not clinical validation: it measures *agreement with
  another detector's validated output*, on 60 seconds per subject, with no
  link to surgical outcome.
* **Not tuned.** Thresholds are the published defaults, checked for sanity on
  synthetic data and deliberately *not* fitted to it.

## Specific limitations, and what each one means

| Limitation | Consequence |
|---|---|
| ~~**Ictal data only.**~~ **Resolved.** `ds003498` provides interictal slow-wave sleep at 2000 Hz with expert markings, and the benchmark runs on it. | The ictal quickstart still measures "where the ripple band is loudest during this seizure", which is not the interictal HFO rate. Use the interictal dataset for anything resembling a clinical claim. |
| **The reference is a detector, not a census.** ds003498's markings are the original study's Morphology detector output after human validation. | Our precision against it is a floor, not a measurement: an event we find that it never proposed counts against us whether or not it is real. Reported as *agreement*, never as accuracy. |
| **Only reviewed channels can be scored.** 6 to 65 of ~43 possible channels per subject, because the study kept the three most mesial bipolar channels in temporal-lobe cases. | Scores describe the reviewed subset. A detector could behave differently on the channels nobody read. |
| **60 seconds per subject, one run each.** The recordings are five minutes and each subject has 10–39 runs. | Rates and rankings from one minute are noisy; the benchmark is a measurement of the method, not of the patients. |
| **The shipped threshold is not the measured optimum.** The default stays at the literature's 5.0 SD; the data prefers 1.5–2.0 SD. | Anyone running the defaults on interictal data gets recall 0.12 and near-chance channel ranking unless they pass `--threshold interictal-agreement`. |
| **1000 Hz sampling in `ds003029`.** Nyquist is 500 Hz. | Ripples only in that dataset; the pipeline refuses to analyse fast ripples there. `ds003498` is 2000 Hz, so fast ripples *are* analysed there (F1 0.30, ρ 0.59). |
| **One patient for the ictal demo; 20 for the benchmark.** | The quickstart's numbers describe one recording. The benchmark's describe 20 subjects from one centre, one scanner, one annotation protocol. |
| **Small counts.** A 60 s window turns a rate into a count. | 30 events/min *is* 30 events. Confidence intervals are printed for this reason; overlapping intervals mean "tied", not "ranked". |
| **Bipolar pairs by contact number.** On a grid the numbering wraps at the end of a row. | `G8-G9` may be two contacts on opposite edges of the grid. Depth electrodes and strips are fine. Fixing this needs electrode coordinates (roadmap). |
| **No physiological-ripple discrimination.** | The pipeline cannot tell an epileptic ripple from a normal one. Nothing in it tries. |
| **Detector agreement is only moderate** (Jaccard ≈ 0.5 on real data). | Two reasonable detectors disagree about half the individual events. Any single-detector rate table is less certain than it looks. This is reported rather than hidden. |
| **Artifact rejection is not exhaustive.** | It catches filter ringing from large transients. It does not catch muscle artifact, stimulation, or the many creative ways real recordings go wrong. |
| **The agent's number check has a hole.** Bare integers ≤ 20 with no unit are allowed through. | A model could state "3 channels" when the tool said 4. Citations and unit-carrying numbers are checked; this class is not. |
| **A small model refuses a lot.** | With Qwen2.5-1.5B many answers fail verification and the agent declines. That is the system working, but it is not a pleasant demo — use a 7B model. |
| **Absolute amplitudes are only as good as the file header.** The archive declares 1 nV per stored unit; the resulting background is ~120 µV RMS, plausible but high. | Microvolt values in a report are not calibrated measurements. Detection is unaffected — all thresholds are in robust SDs of the channel itself. |
| **No security model.** No authentication, no audit log, no protection of the results directory. | Anyone who can write to `artifacts/results/` controls what the agent believes. Do not expose this to untrusted users. |
| **No privacy controls.** | The public data is already de-identified. If you point this at your own recordings, de-identification, ethics approval and data governance are entirely your responsibility — see `DATA.md`. |

## The evidence for HFOs themselves is contested

This software detects HFOs well enough to agree moderately with experts. That
says nothing about whether HFOs should guide surgery, and the best available
evidence is not encouraging: in [the HFO Trial](https://www.thelancet.com/journals/laneur/article/PIIS1474-4422(22)00311-8/fulltext)
(Lancet Neurology, 2022), 78 patients randomised to intraoperative
HFO-guided versus spike-guided tailoring, seizure freedom at one year was
**67% with HFO guidance against 90% with spikes** — non-inferiority not met.

The retrospective evidence is friendlier — [Fedele et al. (2017)](https://www.nature.com/articles/s41598-017-13064-1),
whose markings this repository is scored against, found that resecting HFO-
generating tissue predicted outcome in individual patients — but a randomised
trial outranks it.

What this means for anyone presenting or building on this work: the
contribution here is **measurement quality and provenance**, not a claim that
HFO rate identifies epileptogenic tissue. The pipeline detects interictal
discharges as well, and the trial above suggests spikes deserve at least
equal billing.

## Things that would change the conclusions

If you are deciding whether to build on this, these are the questions that
matter most, in order:

1. **Does the ranking hold on interictal data?** Until that is checked,
   nothing here can be compared to the HFO literature.
2. **Does it hold across windows and across patients?** A ranking that moves
   when you shift the window by a minute is not a finding.
3. **Does it relate to surgical outcome?** That is the only reference standard
   that means anything clinically, and this dataset has it (`participants.tsv`
   has Engel and ILAE scores) — it is the first serious study this codebase
   could support.

`ROADMAP.md` turns these into work items.

## A note on the agent

The guards in `onset_agent/guard.py` make it hard for the model to state a
number that no tool produced, and easy for it to refuse. They do **not** make
its answers correct: the model can still emphasise the wrong thing, miss a
caveat, or phrase a true number misleadingly. The report is the authoritative
artefact. The agent is a convenience for reading it.
