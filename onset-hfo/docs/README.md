# Documentation index

Start wherever your question is.

## "Explain the whole thing to me once"

Open [**TUTORIAL.html**](TUTORIAL.html) in a browser — a standalone walkthrough
of the entire project in eleven parts: the clinical problem, what the signal
looks like, why filter ringing makes this hard, the cohort, the measured
transfer gap, what a handful of labels buys, calibration, the agent, and the
attempts to break it. Every section ends with what the result actually
justifies claiming. It is self-contained — one file, no build step.

## "I just want to click around in it"

```bash
pip install -e ".[app]"
streamlit run app/onset_app.py
```

The reading interface: a saved analysis on a page, where a rate leads to the
events behind it and an event leads to the signal it was measured on. See
[`app/README.md`](../app/README.md).

## "I just want to see it work"

Open [`notebooks/01_hfo_detection_quickstart.ipynb`](../notebooks/01_hfo_detection_quickstart.ipynb)
in Colab and run it top to bottom. Ten minutes, no setup, real data.

## "I want to understand what it does"

Read in this order:

1. [**ARCHITECTURE.md**](ARCHITECTURE.md) — the shape of the system: what the
   pipeline is, what the agent is, and why they are kept apart.
2. [**DATA.md**](DATA.md) — the recording: where it comes from, what its
   annotations mean, and how to point the loader at your own data.
3. [**METHODS.md**](METHODS.md) — every algorithm and every threshold, with
   the paper each came from and the reason for each deviation.
4. [**EVALUATION.md**](EVALUATION.md) — what has been measured, on what, and
   what the numbers do and do not support.
5. [**OUTCOME.md**](OUTCOME.md) — the one test whose reference standard is
   not another algorithm: did the HFO map point at the tissue whose removal
   made the patient seizure-free? Includes the expert positive control that
   tells an underpowered null apart from a detector that does not work, and
   the window study that measures whether any of it survives looking at a
   different minute of the same recording.
6. [**AGENT.md**](AGENT.md) — how a language model is allowed near clinical
   data at all: the tools, the guards, and the threat model.
7. [**ORCHESTRATION.md**](ORCHESTRATION.md) — the other direction: the model
   choosing and parameterising the analyses, the frozen tool contract, the
   evidence store, the S0–S3 ablation ladder, and what verification costs.
8. [**LOCALIZATION.md**](LOCALIZATION.md) — the learned half: the cohort, the
   per-contact model, calibration and conformal prediction sets, and the
   measured gap between a subject-specific model and one that has to work on
   a new patient.
9. [**LIMITATIONS.md**](LIMITATIONS.md) — read before quoting any number from
   this repository to anyone.

## "I am joining the project and need to do something useful"

* [**ROADMAP.md**](ROADMAP.md) — the next pieces of work, in order, each with
  the reason it matters and roughly what it involves.
* [**CONTRIBUTING.md**](CONTRIBUTING.md) — how to add a detector, a dataset or
  an agent tool without breaking the contracts the rest of the code relies on.
* [**GLOSSARY.md**](GLOSSARY.md) — if "bipolar montage", "ictal" or "ripple
  band" are not yet second nature.

## "Where is X?"

| Question | File |
|---|---|
| Which thresholds are used, and why those? | [`onset_hfo/config.py`](../onset_hfo/config.py) — every parameter is documented at its definition |
| How is the data downloaded without pulling 105 MB? | [`onset_hfo/datasets.py`](../onset_hfo/datasets.py) |
| Where does a detection actually get decided? | [`onset_hfo/detectors/engine.py`](../onset_hfo/detectors/engine.py) |
| How are filter-ringing false positives removed? | [`onset_hfo/validate.py`](../onset_hfo/validate.py) and [`onset_hfo/spectral.py`](../onset_hfo/spectral.py) |
| What exactly can the agent see? | [`onset_hfo/store.py`](../onset_hfo/store.py) and [`onset_agent/tools.py`](../onset_agent/tools.py) |
| What stops the model inventing a number? | [`onset_agent/guard.py`](../onset_agent/guard.py) |
| What does the report contain? | [`onset_hfo/report.py`](../onset_hfo/report.py) |
| Which contacts were resected, and how is that mapped to channels? | [`onset_hfo/clinical.py`](../onset_hfo/clinical.py) |
| How is the outcome study computed, and what are its statistics? | [`onset_hfo/outcome.py`](../onset_hfo/outcome.py) |
| What does the interface decide, and what does it only display? | [`app/panels.py`](../app/panels.py) |
| Does a result survive a change of analysis window? | [`onset_hfo/stability.py`](../onset_hfo/stability.py) |

## A note on how this repository is written

Every module starts with a docstring that explains *why it exists*, not just
what it contains, and every non-obvious parameter is documented where it is
defined rather than in a separate wiki that will drift. If you find code here
whose purpose is not explained in the file itself, that is a bug — please fix
it or open an issue.
