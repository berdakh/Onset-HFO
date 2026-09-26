"""Window-stability bookkeeping: the plan, the cohort guard, the spread.

Offline. The expensive part -- actually running the outcome study at eleven
windows -- is not tested here; what is tested is everything that could make
that run's *output* wrong without anyone noticing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from onset_hfo.outcome import OutcomeResult
from onset_hfo.stability import (
    DISJOINT_LENGTH,
    GROWING_WINDOWS,
    StabilityResult,
    _check_cohort,
    _top_channels,
)

# -- the cohort guard ------------------------------------------------------

def test_identical_cohorts_pass():
    out = _check_cohort({"0-60": {"sub-01", "sub-02"}, "0-300": {"sub-01", "sub-02"}})
    assert out["consistent"] and out["n_subjects"] == 2 and out["missing"] == {}


def test_a_window_missing_one_patient_is_flagged():
    """One dropped download must not pass as a change in the curve.

    This is not hypothetical: a transient TLS failure dropped sub-10 from the
    30 s window on the first real run of this study.
    """
    out = _check_cohort({"0-30": {"sub-01"}, "0-60": {"sub-01", "sub-02"}})
    assert not out["consistent"]
    assert out["missing"] == {"0-30": ["sub-02"]}


def test_no_windows_is_not_an_inconsistency():
    assert _check_cohort({})["consistent"]


# -- the retry that makes the cohort guard rarely fire ---------------------

@pytest.fixture
def online(monkeypatch):
    """Let ``_http_get`` past its offline guard, with the transport stubbed out.

    The test suite runs with ``ONSET_HFO_OFFLINE`` set, and that guard fires
    before anything else in ``_http_get`` -- which is the guarantee working,
    and which these three tests have to step around deliberately because the
    retry loop lives on the other side of it. Nothing reaches the network:
    ``_http_get_once`` is replaced in every one of them.
    """
    import time

    from onset_hfo import datasets

    monkeypatch.setenv(datasets.OFFLINE_ENV, "0")
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    return datasets


def test_a_transient_transport_failure_is_retried(online, monkeypatch):
    """A dropped TLS connection must not silently shrink a cohort."""
    datasets = online
    calls = []

    def flaky(url, headers, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise OSError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF in violation of protocol")
        return b"payload"

    monkeypatch.setattr(datasets, "_http_get_once", flaky)
    assert datasets._http_get("https://example.invalid/x") == b"payload"
    assert len(calls) == 3


def test_a_missing_file_is_not_retried(online, monkeypatch):
    """A 404 will not succeed on the second try; failing slowly helps nobody."""
    datasets = online
    calls = []

    def missing(url, headers, timeout):
        calls.append(url)
        raise RuntimeError("HTTP 404 for " + url)

    monkeypatch.setattr(datasets, "_http_get_once", missing)
    with pytest.raises(RuntimeError, match="404"):
        datasets._http_get("https://example.invalid/x")
    assert len(calls) == 1


def test_retries_are_exhausted_and_reported(online, monkeypatch):
    datasets = online

    def always(url, headers, timeout):
        raise OSError("connection reset")

    monkeypatch.setattr(datasets, "_http_get_once", always)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        datasets._http_get("https://example.invalid/x", retries=2)


def test_the_offline_guard_still_wins_over_the_retry_loop(monkeypatch):
    """Retries must never become a way around ``ONSET_HFO_OFFLINE``."""
    from onset_hfo import datasets

    monkeypatch.setenv(datasets.OFFLINE_ENV, "1")
    with pytest.raises(datasets.OfflineError):
        datasets._http_get("https://example.invalid/x")


# -- the top-channel extraction -------------------------------------------

def _channels(rows):
    return pd.DataFrame(rows, columns=["subject", "band", "channel", "reviewed",
                                       "eloquent", "expert_events", "rms_events"])


def _result(frame):
    return OutcomeResult(subjects=pd.DataFrame(), channels=frame, meta=pd.DataFrame(),
                         groups=pd.DataFrame(), participants=pd.DataFrame())


def test_busiest_reviewed_channel_is_picked_per_source():
    frame = _channels([
        ("sub-01", "ripple", "A1-A2", True, False, 10.0, 2.0),
        ("sub-01", "ripple", "B1-B2", True, False, 3.0, 9.0),
    ])
    out = _top_channels(_result(frame), "rms").set_index("source")
    assert out.loc["expert", "top_channel"] == "A1-A2"
    assert out.loc["rms", "top_channel"] == "B1-B2"


def test_unreviewed_and_eloquent_channels_are_not_eligible():
    """The outcome metric ignores them, so the stability of it must too."""
    frame = _channels([
        ("sub-01", "ripple", "A1-A2", True, False, 4.0, 4.0),
        ("sub-01", "ripple", "X1-X2", False, False, 99.0, 99.0),   # not reviewed
        ("sub-01", "ripple", "E1-E2", True, True, 99.0, 99.0),     # eloquent
    ])
    out = _top_channels(_result(frame), "rms")
    assert set(out["top_channel"]) == {"A1-A2"}


def test_a_tie_resolves_the_same_way_in_every_window():
    """Otherwise this module would manufacture instability that is not there."""
    rows = [("sub-01", "ripple", "B1-B2", True, False, 5.0, 5.0),
            ("sub-01", "ripple", "A1-A2", True, False, 5.0, 5.0)]
    first = _top_channels(_result(_channels(rows)), "rms")
    second = _top_channels(_result(_channels(rows[::-1])), "rms")
    assert first["top_channel"].tolist() == second["top_channel"].tolist()


def test_a_silent_channel_set_yields_no_winner():
    frame = _channels([("sub-01", "ripple", "A1-A2", True, False, 0.0, 0.0)])
    out = _top_channels(_result(frame), "rms")
    assert out["top_channel"].isna().all()


# -- the views -------------------------------------------------------------

def _groups(rows):
    return pd.DataFrame(rows, columns=["arm", "t_start", "t_stop", "window_s", "source",
                                       "scope", "band", "metric", "auc", "auc_lo",
                                       "auc_hi", "p_permutation", "n_seizure_free",
                                       "n_recurrence", "mean_seizure_free",
                                       "mean_recurrence"])


@pytest.fixture
def result():
    rows = []
    for window, auc in [(60.0, 0.82), (300.0, 0.71)]:
        rows.append(("growing", 0.0, window, window, "expert", "reviewed",
                     "fast_ripple", "top_channel_resected", auc, auc - 0.2, auc + 0.1,
                     0.05, 13, 7, 0.9, 0.4))
    for start, auc in [(0.0, 0.82), (60.0, 0.55), (120.0, 0.61)]:
        rows.append(("disjoint", start, start + 60.0, 60.0, "expert", "reviewed",
                     "fast_ripple", "top_channel_resected", auc, auc - 0.2, auc + 0.1,
                     0.3, 13, 7, 0.8, 0.5))
    channels = pd.DataFrame([
        ("disjoint", "fast_ripple", "sub-01", "expert", "A1-A2"),
        ("disjoint", "fast_ripple", "sub-01", "expert", "A1-A2"),
        ("disjoint", "fast_ripple", "sub-01", "expert", "A1-A2"),
        ("disjoint", "fast_ripple", "sub-02", "expert", "A1-A2"),
        ("disjoint", "fast_ripple", "sub-02", "expert", "B1-B2"),
        ("disjoint", "fast_ripple", "sub-02", "expert", "C1-C2"),
    ], columns=["arm", "band", "subject", "source", "top_channel"])
    return StabilityResult(groups=_groups(rows), subjects=pd.DataFrame(),
                           channels=channels)


def test_curve_is_ordered_by_window_length(result):
    curve = result.curve()
    assert curve["window_s"].tolist() == [60.0, 300.0]


def test_spread_reports_the_range_between_best_and_worst_minute(result):
    spread = result.spread().set_index("source")
    assert spread.loc["expert", "auc_min"] == pytest.approx(0.55)
    assert spread.loc["expert", "auc_max"] == pytest.approx(0.82)
    assert spread.loc["expert", "auc_range"] == pytest.approx(0.27)


def test_modal_share_separates_a_stable_patient_from_an_unstable_one(result):
    table = result.top_channel_stability().set_index("subject")
    assert table.loc["sub-01", "modal_share"] == pytest.approx(1.0)
    assert table.loc["sub-01", "n_distinct_channels"] == 1
    assert table.loc["sub-02", "modal_share"] == pytest.approx(1 / 3)
    assert table.loc["sub-02", "n_distinct_channels"] == 3


def test_verdict_names_both_arms(result):
    text = result.verdict()
    assert "0.82" in text and "0.71" in text and "disjoint" in text


def test_views_of_an_arm_that_was_not_run_come_back_empty(result):
    """A study that ran only windows must not crash when asked about runs.

    The two studies write different arms, and ``save()`` asks for both. An
    empty result with no columns raised a KeyError the first time this was
    tried on real output.
    """
    assert result.decision_stability(arm="run").empty
    assert result.spread(arm="run").empty
    assert list(result.decision_stability(arm="run").columns)[:2] == ["subject", "source"]


def test_two_studies_do_not_share_an_output_directory(tmp_path, result):
    """The across-runs study overwrote the window study's tables exactly once."""
    windows = result.save(tmp_path)
    runs = StabilityResult(groups=result.groups, subjects=result.subjects,
                           channels=result.channels, label="runs").save(tmp_path)
    assert windows != runs


# -- pooling a subject's runs ---------------------------------------------

def _run_channels(rows):
    return pd.DataFrame(rows, columns=["subject", "band", "channel", "run", "zone",
                                       "eloquent", "reviewed", "expert_events",
                                       "rms_events"])


@pytest.fixture
def resections():
    from onset_hfo.clinical import Resection

    return {"sub-01": Resection(subject="sub-01", resected=("A1", "A2"), eloquent=())}


def test_pooling_adds_the_runs_up(resections):
    """Rates add, so two runs of a channel are one longer recording of it."""
    from onset_hfo.stability import pool_runs

    frame = _run_channels([
        ("sub-01", "ripple", "A1-A2", "01", "resected", False, True, 10.0, 4.0),
        ("sub-01", "ripple", "A1-A2", "02", "resected", False, True, 6.0, 2.0),
        ("sub-01", "ripple", "B1-B2", "01", "spared", False, True, 1.0, 1.0),
        ("sub-01", "ripple", "B1-B2", "02", "spared", False, True, 1.0, 1.0),
    ])
    pooled = pool_runs(frame, resections, "rms", 300.0, ("ripple",)).set_index("source")
    assert pooled.loc["expert", "n_events"] == pytest.approx(18.0)
    assert pooled.loc["rms", "n_events"] == pytest.approx(8.0)
    assert pooled.loc["expert", "n_runs"] == 2


def test_a_channel_reviewed_in_any_run_counts_as_reviewed(resections):
    """The annotators marked events; a quiet segment is not an unreviewed one."""
    from onset_hfo.stability import pool_runs

    frame = _run_channels([
        ("sub-01", "ripple", "A1-A2", "01", "resected", False, True, 10.0, 4.0),
        ("sub-01", "ripple", "A1-A2", "02", "resected", False, False, 0.0, 1.0),
    ])
    pooled = pool_runs(frame, resections, "rms", 300.0, ("ripple",))
    assert len(pooled) and pooled["n_channels"].iloc[0] == 1


def test_pooling_drops_eloquent_channels(resections):
    from onset_hfo.stability import pool_runs

    frame = _run_channels([
        ("sub-01", "ripple", "A1-A2", "01", "resected", False, True, 10.0, 4.0),
        ("sub-01", "ripple", "E1-E2", "01", "spared", True, True, 99.0, 99.0),
    ])
    pooled = pool_runs(frame, resections, "rms", 300.0, ("ripple",))
    assert pooled["n_channels"].iloc[0] == 1


def test_pooling_narrows_the_candidate_set(resections):
    """More recording separates channels a single run could not tell apart.

    The same counts over one run leave two candidates; summed over five runs
    the Poisson intervals no longer overlap and the set is one.
    """
    from onset_hfo.outcome import candidate_channels
    from onset_hfo.stability import pool_runs

    one = pd.Series({"A1-A2": 30.0, "B1-B2": 20.0})
    assert len(candidate_channels(one, 5.0)) == 2

    rows = []
    for run in range(1, 6):
        rows += [("sub-01", "ripple", "A1-A2", f"{run:02d}", "resected", False, True,
                  30.0, 0.0),
                 ("sub-01", "ripple", "B1-B2", f"{run:02d}", "spared", False, True,
                  20.0, 0.0)]
    pooled = pool_runs(_run_channels(rows), resections, "rms", 300.0, ("ripple",))
    expert = pooled[pooled["source"] == "expert"].iloc[0]
    assert expert["n_runs"] == 5 and expert["n_candidates"] == 1


def test_pooling_an_empty_table_is_not_an_error(resections):
    from onset_hfo.stability import pool_runs

    assert pool_runs(pd.DataFrame(), resections, "rms", 300.0, ("ripple",)).empty


# -- the plan --------------------------------------------------------------

def test_the_default_windows_end_at_the_full_run():
    from onset_hfo.outcome import FULL_RUN_S

    assert GROWING_WINDOWS[-1] == FULL_RUN_S
    assert list(GROWING_WINDOWS) == sorted(GROWING_WINDOWS)
    assert FULL_RUN_S % DISJOINT_LENGTH == 0, "disjoint windows must tile the run exactly"
