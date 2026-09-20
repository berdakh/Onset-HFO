# Release validation — Onset-HFO 0.1.0

Everything below was run and observed, not estimated. Re-run any of it with the
commands given.

## Environment

Python 3.11, MNE 1.13.2, NumPy 2.4.6, SciPy, pandas, scikit-learn, Matplotlib,
ruff, pytest. Linux, CPU only, no GPU, no model weights downloaded.

## What passed

| Check | Command | Result |
|---|---|---|
| Lint | `ruff check .` | clean |
| Tests | `pytest -q` | **58 passed** in ~6 s, fully offline |
| Pipeline on synthetic data | `python -m onset_hfo.cli run --synthetic --figures` | completes in ~3 s; 6 figures written |
| Pipeline on public data | `python -m onset_hfo.cli run --subject sub-pt01 --task ictal --run 01 --start 50 --stop 110 --figures` | 23.5 MB downloaded, 98 → 71 bipolar channels, **7.4 s** end to end |
| Detector scores | `python -m onset_hfo.cli evaluate --seeds 1 7 42` | see below |
| Agent, no model | `python -m onset_agent.cli --results <dir> --demo` | 5 answered, 3 refused, all citations resolve |
| Agent, LLM protocol | `pytest tests/test_agent.py` | structured tool calls, tool calls in content, hallucinated number, fabricated citation, unknown tool — all handled as designed |
| Notebooks | `jupyter nbconvert --execute` on all three | all execute top to bottom with no errors |

## Measured detector performance (synthetic ground truth, seeds 1, 7, 42)

| detector | precision | recall | F1 |
|---|---|---|---|
| RMS energy (ripples) | 0.956 ± 0.017 | 0.528 ± 0.073 | 0.679 ± 0.065 |
| line length (ripples) | 0.957 ± 0.017 | 0.550 ± 0.078 | 0.697 ± 0.067 |
| interictal discharges | 0.998 ± 0.004 | 0.844 ± 0.037 | 0.914 ± 0.023 |

Artifact rejection: precision 0.630 → 0.974, recall 0.614 → 0.605 (seed 7).
The ablation in `docs/EVALUATION.md` shows the spectral-peak check is
responsible for essentially all of that.

## Observed on the public recording (`sub-pt01`, ictal run 01, 50–110 s)

* 1732 RMS candidates → 1549 accepted; 2387 line-length candidates → 2188
  accepted; 490 interictal discharges.
* Detected ripples: median peak frequency 146 Hz (10th–90th percentile
  80–235 Hz), median spectral prominence 8.9 dB.
* Detector agreement: Jaccard 0.53; seven leading channels ranked very
  differently by the two detectors, all named in the report.
* Ripple rate on the leading channels: 0/min before the clinician-marked
  onset, 144–146/min during the seizure.
* Clinician-marker permutation check: 1 of the top 5 channels touches a named
  contact (1.05 expected by chance, p = 0.71); 2 of the top 10 (2.15 expected,
  p = 0.68). **No better than chance**, reported as such.

## Repairs made during development, and why they mattered

* **Peak frequency was pinned to the band edge.** Welch averaging over a 40 ms
  event leaves 32-sample segments with a 31 Hz grid, so every event's "peak"
  landed in the same bin (93.75 Hz). Replaced with a Hann-tapered,
  zero-padded periodogram, and the peak is now taken where the spectrum most
  exceeds its fitted 1/f background rather than where it is largest. Median
  peak frequency on real data moved from 80 Hz to 146 Hz.
* **Cycle-count rejection was discarding real ripples.** Measured duration is
  the time above threshold, which under-counts cycles; the threshold was
  lowered to a floor and hysteresis added so durations are measured properly.
* **The discharge detector was reporting electrode pops.** After a 5–60 Hz
  band-pass a step looks like a sharp wave. Added a discontinuity test on the
  *unfiltered* samples: precision 0.58 → ≈ 1.00 with no loss of recall.
* **The agent's number check misread time ranges.** "45.898-45.93 s" parsed as
  a negative number and failed verification. Fixed with a lookbehind, and the
  check was then strengthened to cover bare numbers above 20.

## Limits of this validation

This validates the software, not a clinical claim. Precision and recall come
from a simulator; the public recording has no HFO labels; one patient, one
minute, ictal data only. `docs/LIMITATIONS.md` is the full list and should be
read before quoting any number here.
