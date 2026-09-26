# `app/` — the reading interface

```bash
pip install -e ".[app]"
streamlit run app/onset_app.py
```

Then pick a saved analysis in the sidebar. If you have not produced one yet:

```bash
python -m onset_hfo.cli run --synthetic --figures     # offline, ~10 seconds
```

## What it is for

Every number this project produces already carries the signal window behind
it. Until this page existed that was true only inside a JSON file — a reader
could not *look* at the window without writing code, which makes
"evidence-based" a claim rather than a property.

The page closes that gap. A rate leads to the events behind it, an event leads
to the signal it was measured on, and an answer from the agent leads to both:

| tab | what it shows |
|---|---|
| **Ranking** | channels by rate, **with their confidence intervals**, and the "does anything actually stand out?" verdict above the table |
| **Evidence** | a channel's citable events, and the three-panel figure for the one you pick |
| **Disagreements** | where the two detectors rank a channel differently — both ranks, neither preferred |
| **Ask the agent** | a question box; every citation in the answer expands to its stored record *and its signal window* |
| **Report** | the structured report, as written, with a download |

## The rule it is built on

**The page shows; it does not decide.** Everything comes from
`onset_hfo.store.ResultStore` through `app/panels.py` — the same read-only
boundary the agent is held to. The page has no thresholds, no parameters and
no analysis of its own, so it cannot disagree with the report it displays.

There is one deliberate exception, documented at length in `panels.py`:
`leader_note` runs `metrics.leader_separation` on the stored rate table. That
is the missing null hypothesis — *does any channel stand out, or are they all
tied?* — and a page that showed a ranking without it would be the most
misleading thing this project could ship. Treat a second exception as a bug.

Re-drawing a signal window is not an exception either. The event's times,
frequency, prominence and amplitude all come from the store; the recording is
fetched again only so the window can be seen. Nothing is re-detected, and
`panels.event_from_record` copies every stored field into the figure so the
subtitle cannot disagree with the table above it.

## Three things it deliberately does not do

- **No recommendation.** Not an empty panel, not a greyed-out button: the
  concept does not exist here, exactly as it does not exist in the report
  schema.
- **No resolving of disagreements.** Both ranks are shown; neither is
  preferred.
- **No hiding of refusals.** When the agent declines a question, the page
  shows the refusal and the reason, because that is the system working.

## Offline and non-archive analyses

A synthetic run has no archive to fetch from, and a machine without network
access cannot re-load a public recording. In both cases the page shows the
stored measurements and says why the signal is not drawn, rather than
offering a control that cannot work.

## Testing

`app/panels.py` holds everything the page decides and imports no Streamlit, so
`tests/test_app.py` covers it offline with no browser. The page itself was
verified by driving a real browser against a real server — see
`docs/ARCHITECTURE.md`.
