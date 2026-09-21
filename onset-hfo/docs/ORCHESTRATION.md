# Orchestration: letting the model choose the analysis

This document covers the second half of the repository — the part where a
language model decides *what to measure* rather than describing what was
already measured. It is the implementation of the agentic-orchestration thesis
(Nurmanova), built so that its central claim can be tested rather than
asserted.

The claim, stated narrowly enough to be falsifiable:

> An agent that selects analyses, sets their parameters, inspects intermediate
> results and re-plans accordingly can characterise a recording better — and
> with a more defensible evidence trail — than a fixed pipeline **using the
> same analyzers**.

"Using the same analyzers" is the load-bearing phrase. All four configurations
call the identical tool registry against the identical recording. The only
variable is who decides what runs.

---

## 1. What had to change first

The original agent (`onset_agent/agent.py`, `tools.py`) is a set of read-only
queries over a **saved** analysis. It can report that a channel had 14.2
ripples/min at a threshold somebody chose hours ago. It cannot ask the
question a reviewer asks next — *does that survive a stricter threshold?* —
because there is no tool that runs anything.

That made the thesis's worked example impossible by construction. Its step 4,
which the proposal calls "the entire contribution", is:

> The two branches disagree for `RH2--RH3`. The agent calls
> `detect_hfo(RH2--RH3, threshold=6SD)` to test robustness → rate falls to
> 2.1/min. Likely a threshold artifact; deprioritized.

`onset_agent/analysis.py` makes that call real. Every tool in it executes the
same code path as `python -m onset_hfo.cli run`, at parameters the planner
chooses at call time.

**Making it fast enough to plan with.** Naively, re-running detection means
preprocessing and band-passing 71 channels again — several seconds per call.
Two caches fix it, each keyed on exactly what changes the answer:

| cached | why it is safe |
|---|---|
| the prepared montage (selection, high-pass, notch, bipolar) | cannot depend on a detector threshold |
| the band-passed signal, per band | the filter is the expensive step; thresholding the feature trace is cheap |

So the first whole-recording survey costs ~3 s and the robustness re-test that
follows it costs ~0.05 s — which is the call pattern re-planning actually
produces. An identical repeat call is served from a memo cache and recorded as
a fresh run id with `cached: true`, so the audit trail still shows that the
planner asked twice.

---

## 2. The frozen tool contract

`onset_agent/contract.py`. Every analyzer is a function with a strict JSON
input and output, and every call is recorded as a `ToolRun`:

```json
{
  "tool": "detect_hfo",
  "input":  {"channels": ["ATT6-ATT7"], "detector": "rms", "threshold_sd": 7.0, "k": 5},
  "output": {"channels": {"ATT6-ATT7": {"rate_per_min": 61.0, "n_events": 61}}, "...": "..."},
  "run_id": "hfo_002",
  "runtime_s": 0.051
}
```

Two rules follow, and both are enforced in the dispatcher rather than
requested in a prompt:

1. **The model never touches raw signal.** It sees numbers produced by
   deterministic, separately tested code. This is the answer to "why not feed
   the EEG to a multimodal model": nothing here would be auditable.
2. **Every number carries a run id.** A claim in the report is admissible only
   if it resolves back to one.

`CONTRACT_VERSION` is stamped into every record. Bump it when an input or
output *shape* changes, never when an implementation detail does — the point
is that the orchestration half and the analysis half develop independently
against a fixed interface.

Argument handling is deliberately asymmetric: a wrong **type** or an unknown
**name** is refused, while an out-of-range **value** is clamped to the nearest
legal one, and the clamped value is what appears in the record. A planner
exploring `threshold_sd = 99` should be corrected, not stopped.

### The nine tools

| tool | recomputes | what it does |
|---|---|---|
| `get_recording_metadata` | | subject, sampling rate, analysed window, seizure markers |
| `channel_qc` | | analysable channels, dataset-flagged bads, preprocessing, a line-noise index |
| `detect_hfo` | ✓ | **runs** an HFO detector at a chosen threshold, band, window and channel set |
| `detect_spikes` | ✓ | **runs** the discharge detector |
| `spectral_power` | ✓ | relative band power — is a channel oscillating or just broadband-noisy? |
| `rate_change` | ✓ | rate before vs during the marked seizure, with both raw counts |
| `compare_detectors` | ✓ | both detectors, and where they rank channels differently |
| `propagation_lead` | ✓ | order of first appearance, and each channel's lead in ms |
| `get_event_evidence` | ✓ | the strongest events on one channel, each with a citable `evidence_id` |

No tool writes a file, opens a URL, executes code, or takes a subject
argument. The patient is fixed when the session is opened, so no model output
can change which recording is being analysed.

---

## 3. The evidence store

`onset_agent/evidence.py`. An append-only ledger answering three questions:

* **What has been done?** `digest()` — the compact state the planner re-reads
  between steps. Keeping it small is not cosmetic: a 4B model handed 30 kB of
  raw JSON stops planning and starts pattern-matching. Channel order is
  preserved (never `sort_keys`), because sorting them alphabetically would
  hand the planner a ranking that is not the ranking the tool computed.
* **Where did this number come from?** `find_number(x)` → the run ids whose
  output contains it, within a 1% tolerance. Empty means unsupported.
* **What did it cost?** `cost()` — calls, failures and wall-clock; the x-axis
  of a cost/accuracy curve.

Failed runs are kept. A trace in which the third call errored and the planner
recovered is a better audit record than one in which it silently vanished.

---

## 4. The ladder

`onset_agent/planner.py`.

| rung | control | what it is |
|---|---|---|
| **S0** | none | A hard-coded sequence of tool calls. No model at all. The prior-work baseline and the number to beat. |
| **S1** | single-shot | The model chooses the analyses, they run, it writes the report. It is never shown what came back, so it cannot revise. |
| **S2** | re-planning | The model sees each result before choosing the next call, and stops when a stopping rule fires. |
| **S3** | + verifier | As S2, then every claim must resolve to a run id or be struck. |

### Where a difference can actually show up

A ladder whose rungs cannot produce different answers measures nothing. The
mechanism is `rank_channels`:

* a channel's **survey rate** is its rate at the *lowest* threshold it was
  measured at — the broad look;
* its **robustness** is the worst ratio `rate_at_stricter / survey_rate` over
  every stricter threshold it was re-tested at, **capped at 1.0**;
* its **score** is `survey_rate × robustness`.

The cap is deliberate: a re-test can lower confidence in a channel, never
raise it. A channel nobody re-tested keeps robustness 1.0, which means
*unchallenged*, not *verified* — and that distinction is the entire difference
between the fixed rungs and the re-planning ones. S0 and S1 cannot produce a
robustness below 1, because they never re-test.

This is a design choice, not a law. It is written down here so a reviewer can
disagree with it and change one function.

### Stopping rules

Three, compared on accuracy and cost:

| rule | stops when |
|---|---|
| `FixedBudget(k)` | after *k* tool calls. The honest floor. |
| `ModelJudged()` | the model says the evidence is sufficient — exactly as trustworthy as the model's self-assessment, which is why it is measured rather than assumed. |
| `TiedSetWidth(max_width)` | the set of channels statistically tied for the lead is small enough. |

`TiedSetWidth` is the seam for the calibrated-classifier half of the project.
Replace its `width()` with the size of a conformal prediction set at `1 − α`
and it becomes the uncertainty-driven stopping criterion, with everything
around it unchanged. What ships here is a **tied-rank set** built from the
Poisson intervals the pipeline already computes. It is not a conformal set and
carries no coverage guarantee; the attribute `guarantees_coverage = False`
says so to anyone who asks the object.

A planner that says "done" while its stopping rule declines is asked once for
an analysis that would narrow the answer, then stopped with the reason *"the
model had nothing further to run"*. Spinning through the step limit producing
no evidence would otherwise look, in the cost column, like diligence.

### The scripted planner

`ScriptedPlanner` follows the strategy the prompt describes — survey, then
challenge the leaders at a stricter threshold — with no language model in it.
It reads only what a model would read: the text of the messages it is given.
Two things this buys:

* the whole ladder runs in CI, offline, in seconds, with no weights;
* any improvement a real model shows has to beat a competent script rather
  than a straw man.

---

## 5. Verification, and what it costs

`onset_agent/verifier.py`. The rule is absolute: **a sentence stating a number
is admissible only if that number appears in some tool's output.** A sentence
that fails is removed and recorded with its reason.

Two verifiers, and the point is to have both:

* **`DeterministicVerifier`** resolves numbers by arithmetic. Exact, instant,
  reproducible, and cannot be talked out of a verdict. It also catches
  miscitation — citing `hfo_001` for a figure only `compare_001` reported.
* **`LLMVerifier`** is a second model instance that judges each sentence, as
  the proposal specifies. It can catch something arithmetic cannot — a
  sentence whose numbers are all real but whose *claim* about them is wrong
  ("the rate doubled" when it halved) — and it fails in ways arithmetic does
  not.

Running only one answers the wrong question. Verification removes
hallucinations by construction; the research question is **what it costs**,
because a verifier also strikes true statements it cannot resolve.
`compare_verifiers()` measures exactly that:

| cell | meaning |
|---|---|
| `struck_by_llm_only` | true statements the model verifier removed — the coverage cost |
| `struck_by_arithmetic_only` | unsupported numbers the model verifier let through |

Neither verifier is treated as ground truth. The deterministic one is ground
truth *about arithmetic*, which is a smaller claim and the only one available
without an annotator.

A sentence with no numbers in it is not a claim and is never struck.
Verification must not quietly delete the caveats.

A malformed verifier response is treated as "supported": making report quality
a function of decoding luck would be worse than the failure it prevents.

---

## 6. Scoring, and what it does not mean

`onset_agent/scoring.py` compares a ranking against the clinician SOZ contacts
in `onset_hfo/cohort.py` (see [`DATA.md`](DATA.md) for where those come from).

The question is **not** "did the system find the seizure onset zone". It is:

> Do the channels this configuration ranks highest overlap the contacts a
> clinician named, more than a random ranking of the same channels would?

Chance is estimated by permuting the ranking rather than assumed, because SOZ
contacts are a large and uneven fraction of the implanted contacts — on some
subjects here a third of all channels touch a named one, and "3 of the top 5"
would then be unremarkable.

Every score carries `label_source`, `trustworthy` and `is_placeholder`, so a
number computed against weak free-text markers — or against generated stand-in
labels — can never be mistaken for one computed against a curated label.

**Before the real labels exist**, two stand-ins keep this machinery runnable
(see [`DATA.md`](DATA.md)): exact labels derived from a synthetic recording's
implanted events, and arbitrary seeded placeholders for un-curated real data.
Both fail `trustworthy`, and their warning travels into every score's JSON.
`--write-label-template` emits the one CSV a clinical centre fills in;
passing it back with `--labels-csv` replaces the stand-in everywhere at once.

**Nothing in this module identifies a seizure onset zone for a patient.** It
scores a ranking against a record, retrospectively, on data whose outcome is
already known. The agent's own guard still refuses to answer SOZ questions;
localization is an offline metric computed by code, never a claim the model is
allowed to make.

---

## 7. Running it

```bash
# offline, no download, no model -- the whole ladder in ~20 s
python -m onset_agent.orchestrate --synthetic

# the public recording, scored against the archive's clinician SOZ labels
python -m onset_agent.orchestrate --subject sub-pt01 --task ictal --run 01 \
    --start 50 --stop 110 --score

# a real open-weight model driving the planner
ollama pull qwen2.5:7b-instruct && ollama serve &
python -m onset_agent.orchestrate --synthetic --backend ollama

# compare the three stopping rules, five repeats each, on the re-planning rung
python -m onset_agent.orchestrate --synthetic --rungs S2 \
    --stop-rule budget,model,tied --repeats 5
```

Per rung, into `--out`: `evidence.json` (the audit trail — every call, its
parameters, its result, its run id), `result.json` (ranking, report,
verification), and a combined `ladder.json`.

---

## 8. What is measured today, and what is not

Implemented and tested (47 offline tests in `tests/test_orchestration.py`):

* the frozen contract, the evidence store, and the nine live tools;
* all four rungs, three stopping rules, and run-to-run stability over repeats;
* both verifiers and the delta between them;
* SOZ scoring with a permutation null, against curated public labels.

Not yet, and each is a self-contained next piece of work:

* **The cohort run.** One patient at a time today. The 32 subjects in
  `ds003029` that carry both signals and SOZ labels would turn the ladder into
  a table with confidence intervals — and, grouped by the four clinical
  centres, into a leave-one-site-out generalization study.
* **The falsification tests.** Shuffled channel labels; the leading channel
  removed; a recording with no epileptiform activity. If the agent still
  produces a confident ranking, that is the result a reviewer will look
  hardest for. The plumbing is here — these are three functions over
  `AnalysisSession`.
* **The model and quantization ladder.** Qwen3-4B/8B/14B at FP16/8-bit/4-bit,
  scored on planning quality, tool-call validity, unsupported-claim rate and
  wall-clock. Needs a GPU; the backends already exist.
* **A real-model measurement of the ladder.** Every number quoted here comes
  from the deterministic scripted planner. That is the control, not the
  result.
