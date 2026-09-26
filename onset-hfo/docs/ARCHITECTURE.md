# Architecture

## The shape of it

```
                      ┌──────────────────────────────────────────┐
  public archive ───► │  onset_hfo   (deterministic, testable)   │
  or simulator        │                                          │
                      │  datasets/synthetic → preprocess →       │
                      │  detect → validate → measure → report    │
                      └───────────────┬──────────────────────────┘
                                      │ writes CSV + JSON
                                      ▼
                      ┌──────────────────────────────────────────┐
                      │  artifacts/results/<recording>/          │
                      │    events.csv  rates_*.csv  report.json  │
                      │    comparison.csv  provenance.json ...   │
                      └───────────────┬──────────────────────────┘
                                      │ read-only, via ResultStore
                         ┌────────────┴────────────┐
                         ▼                         ▼
       ┌──────────────────────────────┐  ┌────────────────────────────┐
       │  onset_agent (open-weight)   │  │  app/  (reading interface) │
       │  8 read-only tools · scope   │  │  ranking · evidence ·      │
       │  refusals · citation and     │  │  disagreements · agent ·   │
       │  number verification         │  │  report — shows, decides   │
       └──────────────────────────────┘  │  nothing                   │
                                         └────────────────────────────┘
```

Both consumers go through the same read-only boundary, and neither can change
a number. That is the whole reason `ResultStore` exists: a second reader is
where a project usually grows a second source of truth, and here it cannot.

## Why the two halves are separate

The pipeline is deterministic: the same recording and the same configuration
produce the same events, byte for byte. That is what makes it testable, what
makes a report reproducible months later, and what lets someone disagree with
a specific number and be answered with a specific window of signal.

A language model is none of those things. It is genuinely good at the other
job — deciding what to look up, noticing that two detectors disagree about a
channel, phrasing an answer for a person — and genuinely dangerous if it is
allowed anywhere near the arithmetic.

So the boundary is a **directory of files**. The pipeline writes it. The agent
reads it through `ResultStore` and cannot reach anything else: not the signal,
not a detector, not the network, not the filesystem. If you deleted
`onset_agent/` the pipeline would not notice; if you deleted the pipeline the
agent would have nothing to say. That is the intended relationship.

## Data flow, step by step

| Stage | Module | Input | Output | Notes |
|---|---|---|---|---|
| Load | `datasets.py` / `synthetic.py` | archive or seed | `Recording` | carries provenance, seizure markers, `t_offset` |
| Preprocess | `preprocess.py` | `Recording` | `Prepared` | µV, bipolar, filtered; logs every step |
| Band-pass | `detectors/base.py` | `Prepared` | array | computed **once** and shared by both HFO detectors |
| Detect | `detectors/engine.py` + `rms.py`/`line_length.py`/`spike.py` | `Prepared` | `list[Event]` | the two HFO detectors differ only in one function |
| Validate | `validate.py` | `list[Event]` | same list, marked | rejected events are kept with a reason |
| Measure | `metrics.py` | events | rates, ranks, agreement | Poisson intervals on every rate |
| Report | `report.py` | all of the above | `Report` | evidence-carrying, no recommendation field |
| Persist | `pipeline.py` | `PipelineResult` | CSV + JSON | the boundary |
| Serve | `store.py` | that directory | JSON-safe dicts | the agent's entire world |

## Key types

* **`Recording`** (`datasets.py`) — an MNE `Raw` plus provenance. The critical
  field is `t_offset`: we analyse a *slice* of a longer file, and every time
  this system reports is converted back to original-recording seconds, so a
  window in a report can be found again in the archive.
* **`Prepared`** (`preprocess.py`) — the microvolt array a detector sees, its
  channel names (bipolar pairs), and `steps`, a human-readable list of
  everything done to the signal. That list goes into the report.
* **`Event`** (`detectors/base.py`) — one detection: channel, start, stop,
  detector, score, peak frequency, spectral prominence, rectified-peak count,
  whether it was accepted, and if not, why.
* **`Report`** (`report.py`) — findings, disagreements, data quality, methods,
  limitations. Serialises to JSON for the agent and Markdown for a human.
* **`ResultStore`** (`store.py`) — the read-only query interface. Every method
  returns JSON-safe primitives, because each one is an agent tool's return
  value.

## Design decisions worth knowing about

**Two detectors, not one.** A single detector cannot disagree with anything.
Running an energy detector and a line-length detector on the same band and the
same preprocessing means a disagreement is attributable to the feature, and
the report can say "these two ways of looking at the same signal do not
agree about this channel" — which is information a clinician can use.

**Rejection is a stage, not a filter.** `validate.py` marks events rather than
deleting them. The report states how many were rejected and why; the CSV
contains all of them. A prototype that quietly discards two thirds of its
detections is not inspectable.

**The simulator is part of the system, not a test fixture.** It is the only
place where precision and recall can be measured, and it contains the
failure modes on purpose (see `docs/EVALUATION.md`).

**Configuration is one file.** `config.py` holds every band, threshold and
duration, each documented where it is defined. If a number appears anywhere
else in the codebase, that is a bug.

**Times are always original-recording seconds.** Every event, window and
figure. The slice offset is added once, at load, and never thought about
again.

## The interface

`app/` is a Streamlit page over one saved analysis:
`streamlit run app/onset_app.py`, after `pip install -e ".[app]"`.

It exists because every number this project produces already carried its
signal window — but only inside a JSON file, which makes "evidence-based" a
claim rather than something a reader can check. On the page a rate leads to
the events behind it, an event leads to the signal it was measured on, and an
agent answer's citations expand to both.

Everything the page decides lives in `app/panels.py`, which imports no
Streamlit and is therefore tested offline (`tests/test_app.py`). The page
itself was checked by driving Chromium against a running server — loading it,
opening each tab, rendering an event figure, asking the agent a scoped
question and an out-of-scope one, and expanding a citation to its window.
That is not a CI test; it is how the screenshots in the pull request were
produced, and it is the way to verify a change to `onset_app.py`.

`app/README.md` has the one rule the page is built on and its single
documented exception.

## Running it

| | Command |
|---|---|
| Pipeline, synthetic | `python -m onset_hfo.cli run --synthetic --figures` |
| Pipeline, public data | `python -m onset_hfo.cli run --subject sub-pt01 --start 50 --stop 110` |
| Scores against truth | `python -m onset_hfo.cli evaluate --seeds 1 7 42` |
| Agent, no model | `python -m onset_agent.cli --results <dir> --demo` |
| Agent, open weights | `python -m onset_agent.cli --results <dir> --backend ollama --chat` |
| Tests | `pytest -q` |

In Python:

```python
from onset_hfo.datasets import fetch_slice
from onset_hfo.pipeline import run_pipeline

recording = fetch_slice(subject="sub-pt01", t_start=50, t_stop=110)
result = run_pipeline(recording, save_to="artifacts/results")
print(result.report.to_markdown())
```
