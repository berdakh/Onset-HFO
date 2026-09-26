"""The reading interface's view model.

Offline, and deliberately free of Streamlit: everything the page decides
lives in :mod:`app.panels`, so all of it can be tested without a browser.
What the page itself does — arranging these results — is verified by driving
a real browser against a real server, which is not something to run in CI.

The rule under test throughout is the one in that module's docstring: the
interface shows what the pipeline wrote and does not compute a second answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app import panels

# -- finding analyses ------------------------------------------------------

def test_a_directory_without_events_is_not_an_analysis(tmp_path, store):
    """The cohort studies live in the same results folder and are not analyses.

    Offering `outcome_ds003498` in the picker and then failing to load it is
    worse than not offering it.
    """
    (tmp_path / "outcome_ds003498").mkdir()
    (tmp_path / "outcome_ds003498" / "groups.csv").write_text("a,b\n1,2\n")
    real = tmp_path / "sub-01_run-01"
    real.mkdir()
    for name in ("events.csv", "provenance.json", "report.json", "config.json"):
        (real / name).write_text("{}" if name.endswith(".json") else "channel\n")
    found = panels.find_results(tmp_path)
    assert [p.name for p in found] == ["sub-01_run-01"]


def test_a_missing_folder_is_empty_not_an_error(tmp_path):
    assert panels.find_results(tmp_path / "nope") == []


# -- the header ------------------------------------------------------------

def test_metadata_rows_name_the_window_that_was_analysed(store):
    rows = dict(panels.metadata_rows(store))
    assert rows["Subject"] == store.subject
    assert "s" in rows["Window analysed"]
    assert rows["Pipeline version"] != "?"


# -- the ranking, and the null hypothesis attached to it -------------------

def test_ranking_carries_the_intervals_not_just_the_rate(store):
    """Two channels whose intervals overlap are tied, so the page must show them."""
    table = panels.ranking_table(store)
    assert {"channel", "rank", "rate_per_min", "ci_low", "ci_high"} <= set(table.columns)
    assert table["rank"].tolist() == sorted(table["rank"].tolist())


def test_ranking_of_an_unknown_detector_is_empty_not_an_error(store):
    assert panels.ranking_table(store, "no_such_detector").empty


def test_the_leader_note_is_available_and_says_which_way(store):
    """The one computation the page is allowed, and the reason it is allowed."""
    note = panels.leader_note(store)
    assert note["available"] is True
    assert isinstance(note["distinguishable"], bool)
    assert note["statement"]
    assert note["n_tied_with_leader"] <= note["n_channels"]


def test_the_leader_note_degrades_rather_than_raising(store):
    assert panels.leader_note(store, "no_such_detector")["available"] is False


# -- evidence, and rebuilding the event behind it --------------------------

def test_evidence_rows_carry_a_citable_id(store):
    channel = store.list_channels()[0]["channel"]
    table = panels.evidence_table(store, channel, k=3)
    if not len(table):
        pytest.skip("no accepted events on the leading channel of this fixture")
    assert table["evidence_id"].map(store.resolve).notna().all()


def test_the_rebuilt_event_carries_the_stored_measurements(store):
    """The figure prints these in its subtitle; losing them prints 'nan'.

    This is not hypothetical — the first version of the page built the event
    from times alone and drew ``peak nan Hz, nan dB over background`` directly
    beneath a panel showing 192 Hz and 12.5 dB.
    """
    channel = store.list_channels()[0]["channel"]
    table = panels.evidence_table(store, channel, k=1)
    if not len(table):
        pytest.skip("no accepted events on the leading channel of this fixture")
    record = table.iloc[0].to_dict()
    event = panels.event_from_record(store, record)
    assert event.channel == record["channel"]
    assert event.start == pytest.approx(record["start"])
    assert event.peak_frequency_hz == pytest.approx(record["peak_frequency_hz"])
    assert event.spectral_prominence_db == pytest.approx(record["spectral_prominence_db"])
    assert not np.isnan(event.peak_frequency_hz)


def test_the_rebuilt_event_uses_the_band_the_detector_used(store):
    """Guessing the band would filter the trace differently from the measurement."""
    detector = store.detectors()[0]
    band = panels.detector_band(store, detector)
    assert band[0] < band[1]
    channel = store.list_channels(detector)[0]["channel"]
    table = panels.evidence_table(store, channel, detector=detector, k=1)
    if not len(table):
        pytest.skip("no accepted events for this detector in the fixture")
    assert panels.event_from_record(store, table.iloc[0].to_dict()).band == band


def test_a_null_in_the_record_does_not_become_a_crash(store):
    """Stored CSVs carry NaN, and the store turns those into None."""
    channel = store.list_channels()[0]["channel"]
    table = panels.evidence_table(store, channel, k=1)
    if not len(table):
        pytest.skip("no accepted events in the fixture")
    record = {**table.iloc[0].to_dict(), "peak_frequency_hz": None, "n_peaks": None}
    event = panels.event_from_record(store, record)
    assert np.isnan(event.peak_frequency_hz) and event.n_peaks == 0


# -- citations -------------------------------------------------------------

def test_a_real_citation_resolves(store):
    channel = store.list_channels()[0]["channel"]
    table = panels.evidence_table(store, channel, k=1)
    if not len(table):
        pytest.skip("no accepted events in the fixture")
    rows = panels.citations(store, [table.iloc[0]["evidence_id"]])
    assert rows[0]["resolved"] is True and rows[0]["channel"] == channel


def test_an_invented_citation_is_flagged_rather_than_dropped(store):
    """A row that does not resolve is a guard failure worth seeing on the page."""
    rows = panels.citations(store, ["sub-xx|MADE-UP|rms|0.000"])
    assert rows[0]["resolved"] is False


# -- what ships with the source -------------------------------------------

def test_panels_never_imports_streamlit():
    """The whole reason `panels.py` exists, asserted rather than assumed.

    CI installs the `dev` extra, which has no Streamlit. A path constant put
    in `common.py` (which does import it) failed there while passing locally,
    because the local environment happened to have Streamlit installed. This
    is the guard that stops that recurring.
    """
    import ast

    tree = ast.parse(Path(panels.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "streamlit" not in imported, \
        "app/panels.py must import no Streamlit: the tests run where it is absent"

def test_the_example_analysis_ships_and_loads():
    """Every page must work on a fresh clone, with no download.

    The Onset prototype builds a synthetic cohort at startup; this one cannot
    generate real recordings, so a real analysis is committed instead.
    """
    from app.panels import EXAMPLE
    from onset_hfo.store import ResultStore

    assert EXAMPLE.exists(), "the shipped example analysis is missing"
    shipped = ResultStore(EXAMPLE)
    assert len(shipped.events) and shipped.detectors()
    assert str(shipped.provenance.get("source", "")).startswith("openneuro:"), \
        "the shipped example must be real data, not a simulation"


def test_the_committed_studies_are_readable():
    """The Outcome page reads these rather than rerunning a 90-minute sweep."""
    import pandas as pd

    from app.panels import STUDIES

    groups = pd.read_csv(STUDIES / "outcome_groups_300s.csv")
    assert {"source", "metric", "auc", "p_permutation", "p_bonferroni"} <= set(groups.columns)
    # Nothing survives correction. This assertion exists because the docs said
    # "nothing reaches p < 0.05" for two PRs after a metric was added that made
    # one row do so uncorrected; the claim the pages actually rest on is this
    # one, so it is the one under test.
    assert not (groups["p_bonferroni"] < 0.05).any(), \
        "a result now survives Bonferroni; every page claiming otherwise must change"
    uncorrected = int((groups["p_permutation"] < 0.05).sum())
    assert uncorrected <= len(groups) * 0.05 + 2, \
        "more uncorrected hits than chance explains; the pages must say so"


def test_a_gzipped_analysis_loads_like_a_plain_one(tmp_path, store):
    """The example ships gzipped, so the store has to read both."""
    import gzip
    import shutil

    from onset_hfo.store import ResultStore

    target = tmp_path / "gzipped"
    target.mkdir()
    for path in Path(store.dir).iterdir():
        if path.suffix == ".csv":
            with path.open("rb") as src, gzip.open(target / f"{path.name}.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
        elif path.is_file():
            shutil.copy(path, target / path.name)
    zipped = ResultStore(target)
    assert len(zipped.events) == len(store.events)
    assert sorted(zipped.rates) == sorted(store.rates)


# -- re-loading the signal -------------------------------------------------

def test_a_synthetic_analysis_offers_no_reload(store):
    """There is no archive behind simulated data, so the page must not offer one."""
    spec = panels.reload_spec(store)
    if str(store.provenance.get("source", "")).startswith("openneuro:"):
        assert spec is not None and spec["t_start"] < spec["t_stop"]
    else:
        assert spec is None


def test_reload_spec_names_everything_fetch_slice_needs(store, monkeypatch):
    monkeypatch.setitem(store.provenance, "source", "openneuro:ds003029")
    monkeypatch.setitem(store.provenance, "slice_start_s", 50.0)
    monkeypatch.setitem(store.provenance, "slice_stop_s", 110.0)
    spec = panels.reload_spec(store)
    assert set(spec) == {"dataset", "subject", "run", "session", "task", "acq",
                         "t_start", "t_stop"}
    assert spec["dataset"] == "ds003029"
