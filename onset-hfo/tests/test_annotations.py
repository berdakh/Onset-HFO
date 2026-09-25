"""Expert-annotation handling: label parsing, channel matching, reviewed scope.

All offline. The parsing fixtures are copied from real ``ds003498`` rows, so
these tests fail if the archive's label grammar is ever misread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from onset_hfo.benchmark import _rank_agreement
from onset_hfo.datasets import _spec, _stem, parse_hfo_annotations
from onset_hfo.detectors.base import Event
from onset_hfo.evaluate import evaluate_detections


def _events(rows):
    return pd.DataFrame(rows, columns=["onset", "duration", "trial_type"])


# -- label parsing ---------------------------------------------------------

def test_bipolar_labels_expand_to_montage_names():
    """``ripple_HL2-3`` means the pair HL2-HL3, which is how we name channels."""
    truth, reviewed = parse_hfo_annotations(_events([("0.05", "0.06", "ripple_HL2-3")]))
    assert truth.loc[0, "channel"] == "HL2-HL3"
    assert truth.loc[0, "contact"] == "HL2" and truth.loc[0, "contact_b"] == "HL3"
    assert reviewed == ["HL2-HL3"]


def test_multi_letter_lead_names_survive():
    truth, _ = parse_hfo_annotations(_events([("0.0", "0.02", "fr_IAR4-5")]))
    assert truth.loc[0, "channel"] == "IAR4-IAR5"
    assert truth.loc[0, "kind"] == "fast_ripple"


def test_a_fast_ripple_on_a_ripple_is_scored_in_both_bands():
    """``frandr`` carries evidence in two bands; one row each, not one row."""
    truth, _ = parse_hfo_annotations(_events([("1.0", "0.05", "frandr_AHR3-4")]))
    assert sorted(truth["kind"]) == ["fast_ripple", "ripple"]
    assert set(truth["channel"]) == {"AHR3-AHR4"}
    assert truth["start"].nunique() == 1        # the same event, twice scored


def test_window_filters_events_but_not_reviewed_channels():
    """A channel stays 'reviewed' even when its events fall outside the slice.

    Otherwise a slice with no events on a channel would look unreviewed, and
    detections there would be silently discarded from scoring.
    """
    truth, reviewed = parse_hfo_annotations(
        _events([("5.0", "0.05", "ripple_HL2-3"), ("900.0", "0.05", "ripple_AR1-2")]),
        window=(0.0, 60.0))
    assert list(truth["channel"]) == ["HL2-HL3"]
    assert reviewed == ["AR1-AR2", "HL2-HL3"]


@pytest.mark.parametrize("label", ["", "noise", "ripple", "ripple_", "seizure_onset", "ripple_HL"])
def test_labels_that_are_not_hfo_markings_are_ignored(label):
    truth, reviewed = parse_hfo_annotations(_events([("1.0", "0.05", label)]))
    assert truth.empty and reviewed == []


def test_no_events_file_is_not_an_error():
    truth, reviewed = parse_hfo_annotations(None)
    assert truth.empty and reviewed == []
    assert list(truth.columns)[:3] == ["kind", "channel", "label"]


# -- dataset plumbing ------------------------------------------------------

def test_paths_use_only_the_entities_each_archive_has():
    assert _stem(_spec("ds003029"), "sub-pt01", "01").endswith(
        "sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01")
    assert _stem(_spec("ds003498"), "sub-01", "03").endswith(
        "sub-01_ses-interictalsleep_run-03")


def test_datasets_carry_their_own_mains_frequency():
    assert _spec("ds003029").line_freq == 60.0     # US
    assert _spec("ds003498").line_freq == 50.0     # Zurich


def test_unknown_dataset_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown dataset"):
        _spec("ds000000")


# -- matching and scope ----------------------------------------------------

def _truth_row(channel, start, stop, kind="ripple"):
    return {"kind": kind, "channel": channel, "label": f"{kind}_{channel}",
            "contact": channel.split("-")[0], "contact_b": channel.split("-")[-1],
            "start": start, "stop": stop, "duration_ms": (stop - start) * 1000}


def test_an_adjacent_pair_cannot_claim_its_neighbours_event():
    """HL1-HL2 and HL2-HL3 share a contact but are different channels."""
    truth = pd.DataFrame([_truth_row("HL2-HL3", 1.0, 1.05)])
    on_neighbour = [Event("HL1-HL2", 1.0, 1.05, "rms", (80, 250), contacts=["HL1", "HL2"])]
    on_channel = [Event("HL2-HL3", 1.0, 1.05, "rms", (80, 250), contacts=["HL2", "HL3"])]
    assert evaluate_detections(on_neighbour, truth).n_true_positives == 0
    assert evaluate_detections(on_channel, truth).n_true_positives == 1


def test_detections_on_unreviewed_channels_are_unjudged_not_wrong():
    """The reviewer did not look there, so it is not a false positive."""
    truth = pd.DataFrame([_truth_row("HL2-HL3", 1.0, 1.05)])
    events = [Event("HL2-HL3", 1.0, 1.05, "rms", (80, 250), contacts=["HL2", "HL3"]),
              Event("ZZ1-ZZ2", 2.0, 2.05, "rms", (80, 250), contacts=["ZZ1", "ZZ2"])]
    unrestricted = evaluate_detections(events, truth)
    restricted = evaluate_detections(events, truth, channels=["HL2-HL3"])
    assert unrestricted.precision == 0.5          # the unreviewed one counts against us
    assert restricted.precision == 1.0            # ... and should not
    assert restricted.n_detections == 1


def test_synthetic_truth_still_matches_through_contacts():
    """Implanted events name a contact, which two bipolar pairs legitimately see."""
    truth = pd.DataFrame([{"kind": "ripple", "contact": "SA3", "start": 1.0, "stop": 1.05}])
    events = [Event("SA2-SA3", 1.0, 1.05, "rms", (80, 250), contacts=["SA2", "SA3"])]
    assert evaluate_detections(events, truth).n_true_positives == 1


# -- rank agreement --------------------------------------------------------

def test_perfect_ranking_agreement_scores_one():
    expert = pd.Series({"a": 10, "b": 5, "c": 1})
    result = _rank_agreement(expert, expert.copy(), top_k=2)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["top2_overlap"] == 2


def test_reversed_ranking_scores_minus_one():
    expert = pd.Series({"a": 10, "b": 5, "c": 1})
    ours = pd.Series({"a": 1, "b": 5, "c": 10})
    assert _rank_agreement(expert, ours, top_k=1)["spearman_rho"] == pytest.approx(-1.0)


def test_a_detector_that_found_nothing_gives_no_correlation():
    """All-zero detections have no ordering; the answer is nan, not zero."""
    expert = pd.Series({"a": 10, "b": 5, "c": 1})
    result = _rank_agreement(expert, pd.Series({"a": 0, "b": 0, "c": 0}), top_k=2)
    assert np.isnan(result["spearman_rho"])


# -- benchmark aggregation (offline: fabricated rows, no download) ---------

def _benchmark_fixture():
    from onset_hfo.benchmark import BenchmarkResult

    scores = pd.DataFrame([
        {"subject": s, "band": "ripple", "detector": d, "threshold_sd": t,
         "precision": p, "recall": r, "f1": 2 * p * r / (p + r), "n_detections": 100}
        for s in ("sub-01", "sub-02")
        for d in ("rms",)
        for t, p, r in ((3.0, 0.70, 0.40), (5.0, 0.85, 0.20))
    ])
    ranking = pd.DataFrame([
        {"subject": s, "band": "ripple", "detector": "rms", "threshold_sd": t,
         "spearman_rho": rho, "top5_overlap": k, "p_value": 0.01, "n_channels": 20}
        for s in ("sub-01", "sub-02") for t, rho, k in ((3.0, 0.6, 3), (5.0, 0.2, 1))
    ])
    subjects = pd.DataFrame([{"subject": "sub-01", "n_expert_events": 800},
                             {"subject": "sub-02", "n_expert_events": 600}])
    return BenchmarkResult(scores=scores, ranking=ranking, subjects=subjects)


def test_summary_averages_over_subjects_and_keeps_both_measures():
    table = _benchmark_fixture().summary("ripple")
    assert set(table["threshold_sd"]) == {3.0, 5.0}
    assert {"precision", "recall", "f1", "rank_rho", "top5_overlap"} <= set(table.columns)
    low = table[table["threshold_sd"] == 3.0].iloc[0]
    assert low["precision"] == pytest.approx(0.70) and low["rank_rho"] == pytest.approx(0.6)


def test_the_two_criteria_can_disagree_and_both_are_reported():
    """F1 and rank agreement are different questions; the API answers each."""
    result = _benchmark_fixture()
    assert result.best_threshold("ripple", "f1")["threshold_sd"] == 3.0
    assert result.best_threshold("ripple", "rank")["threshold_sd"] == 3.0


def test_saving_writes_the_tables_a_reader_can_check(tmp_path):
    out = _benchmark_fixture().save(tmp_path)
    written = {p.name for p in out.iterdir()}
    assert {"scores.csv", "ranking.csv", "subjects.csv", "summary_ripple.csv",
            "run.json"} <= written
    import json
    meta = json.loads((out / "run.json").read_text())
    assert meta["n_subjects"] == 2 and meta["thresholds"] == [3.0, 5.0]
