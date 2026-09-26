# `app/` — the reading interface

```bash
pip install -e ".[app]"
streamlit run app/Home.py
```

Works on a fresh clone with **no download**: a real 60-second analysis ships in
`data/example_analysis/`, and the cohort studies read tables committed to
`data/stability/`.

## Same pages as the Onset prototype, real data underneath

This mirrors the page order of [`berdakh/onset`](https://github.com/berdakh/onset)
(the [Onset prototype](https://berdakh-onset.streamlit.app/)) so the two can be
read side by side and eventually merged. The difference is what is behind them:
that app builds a **synthetic** five-patient cohort at startup; this one reads
analyses and cohort studies computed from **public recordings of real
patients** — with expert HFO markings, the contacts the surgeon removed, and
whether the patient became seizure-free.

| page | Onset prototype (synthetic) | here (real) |
|---|---|---|
| **Home** | what it is / is not | the same, plus what the real cohort contains |
| **Recording** (*Patient*) | two models' ranks, disagreement in amber, evidence window | two **detectors**' ranks, disagreement in amber, the **signal** behind any event, and "does anything actually stand out?" |
| **Report** | structured, cited, no recommendation | the same schema, from a real recording |
| **Assistant** | rule-based over synthetic evidence | the agent: scripted offline, or a real open-weight model; every citation opens to its window |
| **Detectors** (*Models*) | LOPO, precision@3, first-hit rank on synthetic SOZ | precision/recall/F1 and channel-rank ρ against **expert HFO markings on 20 patients** |
| **Outcome** | — | **new:** did the map point at the tissue whose removal cured the patient? Plus the two results that did not hold up |
| **Data** | how the synthetic cohort is made | the two archives, what each carries, and the byte-range trick |
| **Architecture** | full system vs prototype | the same, for this pipeline |
| **Research** | lab, theses, principles | lab, the measured state of each claim, what is open |

## The rule it is built on

**The page shows; it does not decide.** Everything comes from
`onset_hfo.store.ResultStore` through `app/panels.py` — the same read-only
boundary the agent is held to. The page has no thresholds, no parameters and no
analysis of its own, so it cannot disagree with the report it displays.

One deliberate exception, documented in `panels.py`: `leader_note` runs
`metrics.leader_separation` on the stored rate table. That is the missing null
hypothesis — *does any channel stand out, or are they all tied?* — and a page
that showed a ranking without it would be the most misleading thing this
project could ship. Treat a second exception as a bug.

Re-drawing a signal window (`app/signal.py`) is not an exception either. The
event's times, frequency, prominence and amplitude all come from the store; the
recording is fetched again only so the window can be seen. Nothing is
re-detected, and `panels.event_from_record` carries every stored field into the
figure so its subtitle cannot disagree with the table above it.

## Three things it deliberately does not do

- **No recommendation.** Not an empty panel, not a greyed-out button: the
  concept does not exist, exactly as it does not exist in the report schema.
- **No resolving of disagreements.** Both ranks are shown; neither is preferred.
- **No hiding of refusals.** Ask the Assistant what to resect — it refuses, and
  the page shows the refusal and the reason, because that is the system working.

## Layout

```
app/
  Home.py           the landing page
  common.py         banner, analysis picker, committed-study loader
  panels.py         everything the page decides — imports no Streamlit, so it is tested
  signal.py         the one place the interface touches the recording again
  pages/            1_Recording · 2_Report · 3_Assistant · 4_Detectors
                    5_Outcome · 6_Data · 7_Architecture · 8_Research
```

## Offline and non-archive analyses

A synthetic run has no archive to fetch from, and a machine without network
access cannot re-load a public recording. In both cases the page shows the
stored measurements and says why the signal is not drawn, rather than offering
a control that cannot work.

## Testing

`app/panels.py` holds everything the page decides and imports no Streamlit, so
`tests/test_app.py` covers it offline with no browser. The pages themselves
were verified by driving Chromium against a running server — every page loaded,
the evidence figure rendered, the agent asked a scoped question and an
out-of-scope one. See `docs/ARCHITECTURE.md`.
