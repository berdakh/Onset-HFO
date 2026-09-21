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
* **Not validated on patients.** Precision and recall come from a simulator.
  There is no labelled clinical benchmark in this repository, and no outcome
  data.
* **Not tuned.** Thresholds are the published defaults, checked for sanity on
  synthetic data and deliberately *not* fitted to it.

## Specific limitations, and what each one means

| Limitation | Consequence |
|---|---|
| **Ictal data only.** The archive publishes recordings around seizures. Clinical HFO work uses interictal data, often in slow-wave sleep. | What is measured here is "where the ripple band is loudest during this seizure", not the interictal HFO rate that the literature associates with epileptogenic tissue. |
| **1000 Hz sampling.** Nyquist is 500 Hz. | Ripples (80–250 Hz) only. Fast ripples (250–500 Hz) — which some studies find more specific — cannot be analysed. The pipeline refuses to try. |
| **One patient, one minute.** | Nothing generalises. A different minute of the same recording may rank channels differently; check it, the code makes that easy. |
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
