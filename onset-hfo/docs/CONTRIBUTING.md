# Contributing

## Ground rules

1. **Every number in the output must be traceable to signal.** If you add a
   quantity, it must be computed in the pipeline, written to the result
   directory, and carry the window it came from.
2. **Never add a recommendation.** Not to the report schema, not to the
   agent's prompt, not as a "suggested" field. This is a hard rule and the
   test suite enforces it.
3. **Keep the two halves apart.** The agent reads `ResultStore` and nothing
   else. A tool that computes something new belongs in the pipeline instead.
4. **Document where the code lives.** Module docstrings explain *why the
   module exists*; parameters are documented at their definition in
   `config.py`. Do not start a separate wiki — it will drift.
5. **Offline tests.** The suite must run with no network and no model weights.
   Use the simulator and the mock server in `tests/test_agent.py`.

## Setup

```bash
cd onset-hfo
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # 58 tests, ~6 s
ruff check .
```

## Adding a detector

1. Write the feature function — it takes `(signal, window_samples)` and
   returns a trace of the same length.
2. Add a thin module in `onset_hfo/detectors/` that calls
   `detect_with_feature` with it, plus a docstring naming the paper and the
   parameters you changed.
3. Register it in `detectors/__init__.py` and in `pipeline.HFO_DETECTORS`.
4. Add a config dataclass field if your detector needs its own thresholds.
5. Tests: it finds the hot channels on synthetic data; it returns nothing on a
   flat channel; more events at a lower threshold than a higher one.
6. Measure it: `python -m onset_hfo.cli evaluate --seeds 1 7 42` and put the
   numbers in `docs/EVALUATION.md`. A detector with no measured scores does
   not go in.

## Adding a dataset

Build a `Recording` (see `docs/DATA.md` for the three routes). The pipeline
needs a preloaded MNE `Raw` in volts, channel names as `LETTERS + NUMBER`, the
right mains frequency, and `t_offset` if you loaded a slice. Set `citation` —
every report prints it.

## Adding an agent tool

See `docs/AGENT.md` § *Adding a tool*. In short: a `ResultStore` method that
returns JSON-safe primitives, a strict schema with
`additionalProperties: false`, a description written for someone who has never
seen the pipeline, and tests for a valid call, an invalid call, and (if it
returns evidence) that the ids resolve.

## Changing a threshold

Change it in `config.py`, update the docstring where it is defined, re-run
`python -m onset_hfo.cli evaluate --seeds 1 7 42`, and update the numbers in
`docs/EVALUATION.md` in the same commit. A parameter change without a
re-measurement is not reviewable.

## Notebooks

The three notebooks are generated:

```bash
python scripts/build_notebooks.py     # writes notebooks/*.ipynb without outputs
```

Edit `scripts/build_notebooks.py`, not the JSON. The committed copies include
executed outputs so that they read well on GitHub; regenerate and re-execute
them when you change something they demonstrate:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Style

* Comments explain *why*, not *what*. The code already says what.
* No parameter appears outside `config.py`.
* Times are always original-recording seconds.
* Prefer a boring, readable implementation to a clever one: this codebase's
  main job is to be argued with.

## Review checklist

- [ ] `pytest -q` and `ruff check .` pass
- [ ] new numbers are measured, not asserted, and appear in `docs/EVALUATION.md`
- [ ] new parameters are documented in `config.py`
- [ ] the report still contains no recommendation, and every finding still
      carries evidence
- [ ] if the agent surface changed: schemas are strict, and refusal and
      verification tests still pass
- [ ] `docs/LIMITATIONS.md` updated if the change alters what the results mean
