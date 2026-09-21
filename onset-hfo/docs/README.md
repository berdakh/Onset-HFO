# Documentation index

Start wherever your question is.

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
5. [**AGENT.md**](AGENT.md) — how a language model is allowed near clinical
   data at all: the tools, the guards, and the threat model.
6. [**ORCHESTRATION.md**](ORCHESTRATION.md) — the other direction: the model
   choosing and parameterising the analyses, the frozen tool contract, the
   evidence store, the S0–S3 ablation ladder, and what verification costs.
7. [**LIMITATIONS.md**](LIMITATIONS.md) — read before quoting any number from
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

## A note on how this repository is written

Every module starts with a docstring that explains *why it exists*, not just
what it contains, and every non-obvious parameter is documented where it is
defined rather than in a separate wiki that will drift. If you find code here
whose purpose is not explained in the file itself, that is a bug — please fix
it or open an issue.
