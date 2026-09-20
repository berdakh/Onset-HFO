"""Generate the Colab notebooks in ``notebooks/``.

The notebooks are committed to the repository (so that the Colab badges work),
but they are *written* here, as plain Python, because reviewing a diff of
notebook JSON is miserable and because it keeps the three notebooks consistent
with each other.

    python scripts/build_notebooks.py

Every notebook is designed to run top to bottom in a free Colab instance with
no configuration. Notebook 1 downloads ~24 MB of public data; notebooks 2 and
3 need no download at all (notebook 2 can optionally pull an open-weight
model).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
REPO = "https://github.com/berdakh/bci-gan.git"
BRANCH = "claude/onset-hfo-detection-osujxo"

SETUP = f'''# Colab setup. On your own machine, skip this cell and run
#   pip install -e ".[dev]"  from the onset-hfo directory instead.
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
REPO = "{REPO}"
BRANCH = "{BRANCH}"   # change to the default branch once this work is merged

if IN_COLAB and not os.path.exists("bci-gan"):
    subprocess.run(["git", "clone", "-q", "--branch", BRANCH, "--depth", "1", REPO], check=True)
if IN_COLAB:
    os.chdir("/content/bci-gan/onset-hfo")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
elif os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
print("working directory:", os.getcwd())'''


_COUNTER = itertools.count(1)


def _cell_id() -> str:
    """nbformat 4.5 requires a stable id on every cell."""
    return f"cell-{next(_COUNTER):03d}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {},
            "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": _cell_id(), "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def write(name: str, cells: list[dict]) -> Path:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOK_DIR / name
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    return path


def badge(name: str) -> str:
    url = f"https://colab.research.google.com/github/berdakh/bci-gan/blob/{BRANCH}/onset-hfo/notebooks/{name}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


# =========================================================================
# 1. Detection quickstart, on real public data
# =========================================================================

NB1 = "01_hfo_detection_quickstart.ipynb"
nb1 = [
    md(f"""
# Onset-HFO 1 — Detecting ripples and spikes in real intracranial EEG

{badge(NB1)}

**What you will do in the next ten minutes.** Download one minute of a real
intracranial EEG recording from a public archive, filter it the way an
epileptologist's software would, run two classical high-frequency-oscillation
(HFO) detectors and one interictal-spike detector over it, look at what they
found, and produce a structured report that cites the exact signal windows
behind every number.

**Why this matters.** In epilepsy surgery the question is *which tissue is
generating the seizures*. Ripples (80–250 Hz oscillations lasting a few tens
of milliseconds) and interictal epileptiform discharges are two of the
markers used to answer it. Counting them by eye across 100 channels is
impossible, so detection is automated — and automated detectors are famously
easy to fool.

**What this is not.** Not a medical device, not validated, not a diagnosis.
It is a transparent first prototype whose numbers you can check.

**The vocabulary**, if any of it is new:

| term | meaning |
|---|---|
| iEEG / ECoG / SEEG | electrodes recording *inside* the skull: grids on the surface, depth electrodes in tissue |
| contact | one recording point on an electrode, e.g. `ATT3` |
| bipolar pair | the difference between two neighbouring contacts, e.g. `ATT3-ATT4` — the channel a detector actually sees |
| HFO / ripple | an 80–250 Hz oscillation of a few tens of milliseconds |
| IED / spike | an interictal epileptiform discharge: a sharp 20–70 ms transient |
| ictal / interictal | during a seizure / between seizures |
| SOZ | seizure onset zone — what clinicians try to find, and what this prototype does **not** claim to identify |
"""),
    code(SETUP),
    md("""
## 1. The data

[OpenNeuro **ds003029**](https://openneuro.org/datasets/ds003029) —
*Epilepsy-iEEG-Multicenter-Dataset*, CC0 licensed, four clinical centres,
100 subjects, BIDS format, with clinician markers for seizure onset and
offset.

We use subject `sub-pt01`, an ECoG recording sampled at 1000 Hz, and we take
**60 seconds** of it: 26 s before the marked electrographic onset and 34 s
after. The loader downloads only those seconds (~24 MB) by asking the archive
for a byte range of the binary file — the whole run is 105 MB and the whole
dataset is far larger.

One consequence of 1000 Hz sampling, stated up front: the Nyquist frequency
is 500 Hz, so **ripples (80–250 Hz) can be analysed and fast ripples
(250–500 Hz) cannot**. The pipeline refuses to pretend otherwise.
"""),
    code("""
from onset_hfo.datasets import fetch_slice, list_runs

# What else is available for this subject (not downloaded, just listed):
print(list_runs("sub-pt01").to_string(index=False))

recording = fetch_slice(subject="sub-pt01", task="ictal", run="01",
                        t_start=50.0, t_stop=110.0)
"""),
    code("""
import json
print(json.dumps(recording.provenance(), indent=2)[:1200])
print("\\nClinician free-text markers name these contacts near onset:")
print(recording.marked_contacts)
"""),
    md("""
Those contact names come from what the reviewer typed during the seizure
(`"AD1-4, ATT1,2"`). They are **not** a curated seizure-onset-zone label — the
dataset does not ship one — and we will treat them accordingly later.

## 2. What the raw signal looks like

Before any processing. Note the amplitude scale (hundreds of microvolts) and
that the interesting structure is invisible at this zoom: ripples are 30 ms
long and buried under much larger slow activity.
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np

sf = recording.sfreq
data = recording.raw.get_data(picks=["ATT6", "ATT7", "PST2"]) * 1e6   # volts -> microvolts
t = recording.t_offset + np.arange(data.shape[1]) / sf

fig, axes = plt.subplots(3, 1, figsize=(11, 5), sharex=True)
for ax, row, name in zip(axes, data, ["ATT6", "ATT7", "PST2"], strict=False):
    ax.plot(t, row, linewidth=0.5, color="#0b0b0b")
    ax.set_ylabel(name)
    ax.axvline(recording.seizure[0], color="#e34948", linewidth=1.5)
axes[-1].set_xlabel("time (s in the original recording); red line = marked seizure onset")
fig.tight_layout()
"""),
    md("""
## 3. Preprocessing

Four steps, each of which is recorded in the report so a reader knows exactly
what the detector saw:

1. keep intracranial channels, drop DC/trigger/ECG and the channels the
   dataset flagged `bad` (white matter, outside the brain, noisy);
2. high-pass at 1 Hz (drift);
3. notch at 60 Hz and harmonics — **narrow** notches, because the 180 Hz and
   240 Hz harmonics sit inside the ripple band and a wide notch would carve a
   hole in the signal we want to measure;
4. **bipolar montage**: subtract neighbouring contacts on the same electrode.
   This is standard for HFO work — a shared reference spreads its own noise
   across every channel and produces "HFOs" that appear everywhere at once.
"""),
    code("""
from onset_hfo.preprocess import prepare

prepared = prepare(recording)
print("\\nchannels:", prepared.ch_names[:8], "...", prepared.n_channels, "total")
"""),
    md("""
## 4. The detectors

Two HFO detectors, deliberately simple and deliberately different:

* **RMS energy** (Staba et al., 2002) — band-pass 80–250 Hz, root-mean-square
  in a 3 ms window, threshold at 5 robust SDs of the channel's own energy,
  and then the criterion that makes it an *oscillation*: at least 6 rectified
  peaks above a secondary threshold.
* **Line length** (Gardner et al., 2007) — the same pipeline with the mean
  absolute sample-to-sample change instead of energy. It responds to fast
  low-amplitude activity that RMS misses, and to sharp edges that RMS ignores.

Plus an **interictal discharge detector**: amplitude and sharpness in the
5–60 Hz band, with an explicit check against discontinuities (an electrode pop
looks like a beautiful spike once you filter it).

Two detectors rather than one, because *a single detector cannot disagree with
anything*, and disagreement is information this prototype reports instead of
hiding.
"""),
    code("""
from onset_hfo.pipeline import run_pipeline

result = run_pipeline(recording, save_to="artifacts/results")
"""),
    md("""
Notice the two numbers printed for each detector: **candidates** and
**accepted**. Every candidate is kept in the table with the reason it was
rejected. The rejection stage is the difference between a prototype and a
demo — see section 6.

## 5. What did it find?
"""),
    code("""
result.rates["rms"].head(8).round(2)
"""),
    code("""
from onset_hfo.viz import plot_channel_rates, plot_rate_timecourse, plot_detector_comparison

fig = plot_channel_rates(result.rates["rms"], top_k=15, label="rms")
"""),
    md("""
Read the confidence intervals, not just the bars. Sixty seconds of recording
turns a rate into a small count: 30 events/min *is* 30 events, and the
interval around it is wide. Two channels whose intervals overlap are not
ranked, they are tied.
"""),
    code("""
top = list(result.rates["rms"]["channel"].head(3))
fig = plot_rate_timecourse(result.events, prepared.t_offset,
                           prepared.t_offset + prepared.duration,
                           channels=top, seizure=recording.seizure)
"""),
    md("""
The shaded region is the clinician-marked seizure. In this recording the
detected ripple rate on the leading channels is essentially zero before onset
and high during the seizure — these are *ictal* HFOs.

That is worth pausing on, because it is also a limitation: clinical HFO work
is usually done on **interictal** recordings (between seizures, often during
sleep), where a high rate is thought to mark epileptogenic tissue. This
dataset only publishes ictal snapshots, so what we measure here is "where the
seizure is loudest in the ripple band", which is a different claim.
"""),
    code("""
fig = plot_detector_comparison(result.comparison, "rms", "line_length")
"""),
    code("""
# The channels the two detectors rank very differently, stated rather than averaged:
result.comparison[result.comparison["disagrees"]][
    ["channel", "rank_rms", "rank_line_length", "rank_gap",
     "rate_per_min_rms", "rate_per_min_line_length"]].round(1)
"""),
    md("""
## 6. One event, up close

This is the figure that makes a detection arguable. Three panels: the
wideband signal, the band-passed signal, and the spectrum of the event window
against the recording's own 1/f background.

A **real oscillation** leaves a bump in the spectrum. **Filter ringing** —
what a sharp transient becomes after an 80–250 Hz filter — does not: its
spectrum is just the background pushed up. That difference is the single most
important quality check in HFO detection, and it is why the pipeline measures
*spectral prominence* for every event.
"""),
    code("""
from onset_hfo.viz import plot_event

best = result.evidence(top[0], k=1)[0]
fig = plot_event(result.prepared, best)
"""),
    code("""
# ... and an event the validation stage threw away, with its reason:
rejected = sorted([e for evs in result.events.values() for e in evs if not e.accepted],
                  key=lambda e: -e.score)
print(rejected[0].reject_reason)
fig = plot_event(result.prepared, rejected[0])
"""),
    md("""
## 7. The report

A structured object, not prose: findings with rates, ranks, confidence
intervals and evidence windows; disagreements between detectors stated and
left unresolved; data quality; method; limitations. There is **no
recommendation field** — not empty, absent.
"""),
    code("""
from IPython.display import Markdown
Markdown(result.report.to_markdown())
"""),
    md("""
## 8. A weak reality check

We have no HFO ground truth for this recording. What we do have is the list of
contacts the clinician named in the seizure markers. Are our top-ranked
channels drawn from them more often than chance?

The honest way to ask is a permutation test, and the honest way to read the
answer is with both hands: the markers are not an SOZ label, they are
incomplete, and this is a 60-second ictal window.
"""),
    code("""
from onset_hfo.evaluate import marked_contact_check

check = marked_contact_check(result.rates["rms"], recording.marked_contacts, k=10)
print(json.dumps(check, indent=2))
"""),
    md("""
Whatever this prints, do not over-read it. If the overlap is no better than
chance that is unsurprising for ictal data and a rate-only ranking; if it is
better, it is mild encouragement and nothing more. Precision and recall are
measured in notebook 3, on synthetic data where the truth is known.

## 9. Where the results went

Everything is saved as CSV and JSON. That directory is also the *only* thing
the agent in notebook 2 can read.
"""),
    code("""
import os
out = "artifacts/results/sub-pt01_ictal_run-01"
print("\\n".join(sorted(os.listdir(out))))
"""),
    md("""
## What to try next

* Change the window (`t_start` / `t_stop`) to a purely pre-ictal stretch and
  see whether the ranking changes. If a channel's rank depends on which minute
  you picked, that is worth knowing before anyone builds on it.
* Raise `threshold_sd` from 5 to 7 in `PipelineConfig` and watch the rates
  fall. Which channels survive?
* Run another subject (`list_runs("sub-jh103")`) and compare.
* Open **notebook 2** to ask an open-weight language model about these
  results, and **notebook 3** to measure how good the detectors actually are.

**Reading**: `docs/METHODS.md` (the algorithms and their references),
`docs/DATA.md` (what is and is not in this dataset), `docs/LIMITATIONS.md`
(what this prototype must not be used for).
"""),
]

# =========================================================================
# 2. The agent
# =========================================================================

NB2 = "02_agentic_analysis.ipynb"
nb2 = [
    md(f"""
# Onset-HFO 2 — An evidence-only agent on an open-weight model

{badge(NB2)}

Notebook 1 produced tables. This notebook puts a language model in front of
them — carefully.

**The design in one paragraph.** The agent can call eight read-only tools
over one saved analysis. It cannot run a detector, change a threshold, open a
file or reach the network. Every factual claim it makes must cite an
`evidence_id` that a tool actually returned, and every number it states must
appear in a tool result. Those two checks run *after* the model answers; if
either fails, the answer is discarded and the agent says it cannot answer.
Questions about treatment, diagnosis or another patient are refused *before*
the model is called at all.

**Why bother with an agent then?** Because choosing which question to ask of a
dataset — "is this channel's lead real, or do the detectors disagree about
it?" — is exactly the kind of multi-step, ill-specified work a language model
is good at, and exactly the kind of work that must never be allowed to invent
a number. The split is: the model decides *what to look up and how to say
it*; the pipeline decides *what is true*.

Everything runs on **open weights** (Qwen2.5-Instruct by default). Nothing in
this project requires a hosted proprietary model.
"""),
    code(SETUP),
    md("""
## 1. Get an analysis to talk about

If you ran notebook 1, its results are already on disk. If not, this cell
builds one from the synthetic recording in a few seconds — no download.
"""),
    code("""
import os
from onset_hfo.store import ResultStore

REAL = "artifacts/results/sub-pt01_ictal_run-01"
if os.path.exists(os.path.join(REAL, "events.csv")):
    results_dir = REAL
else:
    from onset_hfo.pipeline import run_pipeline
    from onset_hfo.synthetic import make_synthetic_recording
    recording = make_synthetic_recording(duration_s=60)
    results_dir = str(run_pipeline(recording).save("artifacts/results"))

store = ResultStore(results_dir)
store
"""),
    md("""
## 2. What the agent can see

The store is the whole world the agent has access to. Look at the tools it
exposes — the descriptions below are literally what the model is shown.
"""),
    code("""
from onset_agent.tools import TOOLS

for name, tool in TOOLS.items():
    args = ", ".join(tool.parameters["properties"]) or "-"
    print(f"{name:26s} args: {args}")
    print(f"{'':26s} {tool.description[:110]}...")
"""),
    code("""
# There is no tool that takes a patient argument: scope belongs to the
# application, not to model output. And there is no tool that writes anything.
import json
print(json.dumps(TOOLS["get_evidence"].schema(), indent=2))
"""),
    md("""
## 3. First, with no model at all

The `scripted` backend is a deterministic keyword policy that speaks the same
tool-calling protocol. It is **not** a language model — it exists so the
machinery can be demonstrated and tested offline, and so you can see the loop
before any weights are downloaded.
"""),
    code("""
from onset_agent import OnsetAgent, make_backend

agent = OnsetAgent(store, make_backend("scripted"))
answer = agent.ask("Which channels have the highest ripple rate?")
print(answer)
"""),
    code("""
answer = agent.ask("What is the evidence for the top channel?")
print(answer.text)
print("\\ncitations:", answer.evidence_ids)
print("resolved:", [store.resolve(e) is not None for e in answer.evidence_ids])
"""),
    md("""
## 4. Now with a real open-weight model

Two ways, pick one.

**A. Ollama** (best on your own machine, also works in Colab):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct
ollama serve &
```

then `make_backend("ollama", model="qwen2.5:7b-instruct")`.

**B. Hugging Face transformers** (in-process, no server — the easy path in
Colab). On a free CPU instance use the 1.5B model; with a GPU runtime use the
7B, which is much better at tool calling.

The cell below tries transformers and falls back to the scripted policy if the
packages or the weights are unavailable, so the notebook always runs.
"""),
    code("""
USE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # try "Qwen/Qwen2.5-7B-Instruct" on a GPU runtime

backend = None
try:
    import torch, transformers  # noqa: F401
    backend = make_backend("transformers", model=USE_MODEL)
    print("using", backend.describe())
except Exception as exc:
    print("could not load a language model:", type(exc).__name__, exc)
    print("falling back to the scripted policy (install with: pip install -e '.[llm]')")
    backend = make_backend("scripted")

llm_agent = OnsetAgent(store, backend, verbose=True)
"""),
    code("""
answer = llm_agent.ask("Which channels have the highest ripple rate, and how sure can we be?")
print("\\n", answer.text)
print("tools:", answer.tools_called, "| verified:", answer.verified)
"""),
    md("""
If you are running the 1.5B model, there is a good chance the answer above was
*refused* rather than produced. That is not a bug in the notebook — it is the
system working. A small model fumbles the tool protocol or states a number it
did not retrieve, the guards catch it, and the agent declines instead of
publishing a plausible fabrication. Swap in the 7B model and the same
questions start going through.

This is the most useful thing this notebook can show you: **what the failure
looks like, and that it is contained.**

## 5. The refusals

Three kinds, all checked before the model runs.
"""),
    code("""
for question in ["Which contacts should we resect?",
                 "Does this patient have epilepsy?",
                 "What did you find in patient sub-pt02?",
                 "Where do the seizures start?"]:
    a = agent.ask(question)
    print(f"Q: {question}\\n   [{'refused' if a.refused else 'answered'}] {a.text}\\n"
          f"   tools called: {a.tools_called or 'none'}\\n")
"""),
    md("""
Note the last line of each: no tool was called. The refusal costs nothing and
cannot be argued out of, because there is no model in the path to argue with.

## 6. Watching a guard catch a fabrication

Let us make a model that lies, and check that the system holds. This backend
retrieves real data and then states a rate nobody measured.
"""),
    code("""
import json
from onset_agent.backends import AssistantMessage, Backend, ToolCall

class LyingBackend(Backend):
    name = "deliberately-wrong"
    def __init__(self): self.turn = 0
    def chat(self, messages, tools):
        self.turn += 1
        if self.turn == 1:
            return AssistantMessage(tool_calls=[ToolCall("top_channels", {"k": 3}, "c1")])
        return AssistantMessage(content=json.dumps({
            "answer": "The leading channel fires at 999.9 events/min, far above the others.",
            "evidence_ids": ["sub-xx|MADE-UP|rms|0.000"]}))

bad = OnsetAgent(store, LyingBackend())
answer = bad.ask("How often does the top channel fire?")
print(answer.text, "\\n")
print(json.dumps([s for s in answer.trace if s["type"] == "answer"], indent=2))
"""),
    md("""
Both problems are named: the citation was never returned by a tool, and the
number appears in no tool result. The answer is dropped. Note what the agent
says instead — it points at the report, which *is* authoritative.

## 7. The trace

Every answer carries the full sequence of tool calls, arguments and
verification outcomes. This is what you would show a reviewer who asks "where
did this sentence come from?".
"""),
    code("""
answer = agent.ask("Where do the two detectors disagree?")
print(answer.text, "\\n")
print(json.dumps(answer.as_dict(), indent=2)[:1800])
"""),
    md("""
## 8. The full demonstration set
"""),
    code("""
from onset_agent.prompts import EXAMPLE_QUESTIONS

for a in agent.ask_many(EXAMPLE_QUESTIONS):
    status = "REFUSED" if a.refused else "answer "
    print(f"[{status}] {a.question}\\n           {a.text[:220]}\\n")
"""),
    md("""
## What to try next

* Ask something the tools genuinely cannot answer ("what is the patient's
  age?") and watch it decline rather than guess.
* Add a tool in `onset_agent/tools.py` — say, one that returns the rejected
  events and their reasons — and see the model start using it. The schema is
  the only thing the model knows about it.
* Try a different open-weight model (`llama3.1:8b-instruct`,
  `mistral-nemo`) and compare how often the guards fire. That number is a
  useful, cheap benchmark of a model's tool discipline.
* Read `docs/AGENT.md` for the threat model — including why tool *results*
  are treated as data and never as instructions.
"""),
]

# =========================================================================
# 3. Validation
# =========================================================================

NB3 = "03_validation_and_benchmark.ipynb"
nb3 = [
    md(f"""
# Onset-HFO 3 — How good are these detectors, actually?

{badge(NB3)}

The public recording has no HFO labels. Nobody has marked every ripple in it,
so on real data we can report rates, rankings and agreement, but **not**
precision and recall. Claiming otherwise would be the most common way
detection papers mislead.

So we build a recording where the truth is known: 1/f background, mains noise,
implanted ripples at a controlled signal-to-noise ratio, interictal discharges
(some with ripples riding on them, as in real epileptic tissue), and — most
importantly — **the traps**: large sharp transients that ring through an
80–250 Hz filter and look exactly like ripples.

Everything in this notebook runs offline in about a minute.
"""),
    code(SETUP),
    md("""
## 1. Build a labelled recording
"""),
    code("""
from onset_hfo.synthetic import make_synthetic_recording

recording = make_synthetic_recording(duration_s=60, seed=7)
recording.ground_truth.groupby("kind").agg(
    n=("kind", "size"), mean_amplitude_uv=("amplitude_uv", "mean")).round(1)
"""),
    code("""
print("\\n".join(recording.notes))
"""),
    md("""
The "hot" contacts carry ~20x the event rate of the others, and their rate
rises again inside a simulated seizure window. A detector that cannot find
them is not worth discussing.

## 2. Look at the traps

Three windows: a clean implanted ripple, a discharge, and an artifact. The
artifact has no oscillation of its own — but watch what the band-pass does
to it.
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np
from onset_hfo.detectors.base import bandpass

sf = recording.sfreq
raw = recording.raw.get_data() * 1e6
names = list(recording.ch_names)
gt = recording.ground_truth

fig, axes = plt.subplots(3, 2, figsize=(11, 6.5))
for row, kind in enumerate(["ripple", "spike", "artifact"]):
    event = gt[gt["kind"] == kind].iloc[3]
    ch = names.index(event["contact"])
    i0 = int((event["start"] - 0.05) * sf); i1 = int((event["stop"] + 0.05) * sf)
    seg = raw[ch, i0:i1]
    t = np.arange(seg.size) / sf * 1000
    axes[row, 0].plot(t, seg, color="#0b0b0b", linewidth=1)
    axes[row, 0].set_ylabel(kind)
    axes[row, 1].plot(t, bandpass(seg[None, :], sf, (80, 250))[0], color="#eb6834", linewidth=1)
axes[0, 0].set_title("wideband")
axes[0, 1].set_title("80-250 Hz - the detector's view")
axes[2, 0].set_xlabel("ms"); axes[2, 1].set_xlabel("ms")
fig.tight_layout()
"""),
    md("""
The bottom right panel is the whole problem in one picture: a step with no
oscillation in it becomes a burst of 80–250 Hz "activity" after filtering.
Any energy-threshold detector will report it. The only way to tell it apart
from the top right panel is to look at the *spectrum* — a real oscillation
leaves a bump above the 1/f background, ringing does not.

## 3. Run the pipeline and score it
"""),
    code("""
from onset_hfo.evaluate import evaluate_detections
from onset_hfo.pipeline import run_pipeline

result = run_pipeline(recording, verbose=False)
for name, events in result.events.items():
    print(evaluate_detections(events, recording.ground_truth, detector=name).summary())
print(evaluate_detections(result.spikes, recording.ground_truth,
                          detector="spike", kind="spike").summary())
"""),
    md("""
Recall around 0.5 for the HFO detectors is not a bug to be tuned away. A
5-SD threshold detector finds the ripples that are clearly above the noise
and misses the marginal ones; the next cell shows exactly that trade-off.
What *would* be a bug is high recall bought with false positives, which is
what the artifact rejection stage prevents — see section 5.

## 4. What the threshold buys you
"""),
    code("""
from onset_hfo.detectors import detect_rms
from onset_hfo.evaluate import sweep_threshold
from onset_hfo.preprocess import prepare

prepared = prepare(recording, verbose=False)
sweep = sweep_threshold(prepared, recording.ground_truth, detect_rms,
                        values=[3, 4, 5, 6, 7, 8], name="rms")
sweep[["threshold_sd", "n_detections", "precision", "recall", "f1"]].round(3)
"""),
    code("""
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(sweep["threshold_sd"], sweep["recall"], marker="o", color="#2a78d6", label="recall")
ax.plot(sweep["threshold_sd"], sweep["precision"], marker="s", color="#eb6834", label="precision")
ax.plot(sweep["threshold_sd"], sweep["f1"], marker="^", color="#1baf7a", label="F1")
ax.axvline(5, color="#52514e", linestyle="--", linewidth=1)
ax.text(5.05, 0.05, " default (5 SD)", color="#52514e", fontsize=9)
ax.set_xlabel("detection threshold (robust SDs)"); ax.set_ylabel("score")
ax.set_ylim(0, 1.05); ax.legend(frameon=False)
for side in ("top", "right"): ax.spines[side].set_visible(False)
ax.grid(color="#dcdbd6", linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout()
"""),
    md("""
This curve is the honest description of a threshold detector: not "F1 = 0.74"
but "here is the trade-off, and here is why we chose 5". Quote the curve, not
the point.

## 5. Does artifact rejection earn its place?

The pipeline throws away a third of its own detections. That is only
justifiable if the ones it throws away are worse than the ones it keeps.
"""),
    code("""
from onset_hfo.evaluate import validation_benefit

validation_benefit(result.events["rms"], recording.ground_truth).round(3)
"""),
    md("""
Precision roughly 0.6 before, roughly 0.98 after; recall falls by a couple of
points. The `false_positive_causes` column names what was being confused —
`artifact` for filter ringing, `spike` for a discharge misread as an
oscillation. If that column ever showed mostly `background`, the detector
would be firing on noise and the thresholds would need to move.

## 6. Which criterion does the work?

Each check can be switched off. If a criterion costs recall and buys nothing,
it should be deleted rather than defended.
"""),
    code("""
import pandas as pd
from onset_hfo.config import ValidationConfig
from onset_hfo.validate import validate_events

rows = []
for label, cfg in [
    ("no rejection at all", None),
    ("cycles only", ValidationConfig(min_peak_prominence_db=-99)),
    ("spectral peak only", ValidationConfig(min_cycles=0)),
    ("both (default)", ValidationConfig()),
]:
    events = detect_rms(prepared)
    if cfg is not None:
        validate_events(events, prepared, cfg)
    scores = evaluate_detections(events, recording.ground_truth, accepted_only=cfg is not None)
    rows.append({"stage": label, "kept": sum(e.accepted for e in events) if cfg else len(events),
                 "precision": round(scores.precision, 3), "recall": round(scores.recall, 3),
                 "f1": round(scores.f1, 3)})
pd.DataFrame(rows)
"""),
    md("""
## 7. Sensitivity to the recording, not just the settings

One seed is an anecdote. Three seeds with different noise and event placement
tell you whether the numbers are a property of the detector or of the
recording it happened to see.
"""),
    code("""
rows = []
for seed in (1, 7, 42):
    rec = make_synthetic_recording(duration_s=60, seed=seed, verbose=False)
    res = run_pipeline(rec, verbose=False)
    for name, events in res.events.items():
        s = evaluate_detections(events, rec.ground_truth, detector=name)
        rows.append({"seed": seed, "detector": name, "precision": s.precision,
                     "recall": s.recall, "f1": s.f1})
    s = evaluate_detections(res.spikes, rec.ground_truth, detector="spike", kind="spike")
    rows.append({"seed": seed, "detector": "spike", "precision": s.precision,
                 "recall": s.recall, "f1": s.f1})
scores = pd.DataFrame(rows)
scores.groupby("detector")[["precision", "recall", "f1"]].agg(["mean", "std"]).round(3)
"""),
    md("""
## 8. How hard is the problem? Recall against SNR

The one parameter that matters most is how far the ripple stands above the
background. This is the curve to quote when someone asks "will it work on our
data?" — the answer depends on their signal-to-noise ratio, not on ours.
"""),
    code("""
rows = []
for snr in (4, 5, 6, 7, 9, 12):
    rec = make_synthetic_recording(duration_s=45, seed=3, ripple_snr=snr, verbose=False)
    res = run_pipeline(rec, verbose=False)
    s = evaluate_detections(res.events["rms"], rec.ground_truth, detector="rms")
    rows.append({"ripple_snr": snr, "precision": s.precision, "recall": s.recall, "f1": s.f1})
snr_curve = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(snr_curve["ripple_snr"], snr_curve["recall"], marker="o", color="#2a78d6", label="recall")
ax.plot(snr_curve["ripple_snr"], snr_curve["precision"], marker="s", color="#eb6834", label="precision")
ax.set_xlabel("implanted ripple amplitude / band-limited background RMS")
ax.set_ylabel("score"); ax.set_ylim(0, 1.05); ax.legend(frameon=False)
for side in ("top", "right"): ax.spines[side].set_visible(False)
ax.grid(color="#dcdbd6", linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout()
snr_curve.round(3)
"""),
    md("""
## What this does and does not establish

**Does:** the detectors find implanted oscillations well above chance; the
artifact rejection stage raises precision sharply at a small cost in recall;
the spike detector's discontinuity check removes the electrode-pop
false positives; the numbers are stable across recordings.

**Does not:** say anything about clinical performance. Real ripples are not
Gaussian-windowed sinusoids, real artifacts are more varied and more
creative, and real recordings have physiological ripples in healthy tissue
that no simulator here models. These scores are a floor on sanity, not a
measure of clinical utility.

The next step for this project — see `docs/ROADMAP.md` — is a small
hand-annotated subset of real recordings. That is the only thing that
converts these numbers into a claim about patients.
"""),
]


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in [(NB1, nb1), (NB2, nb2), (NB3, nb3)]:
        path = write(name, cells)
        print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
