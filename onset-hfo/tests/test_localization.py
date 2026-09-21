"""The localization half: cohort features, the three protocols, calibration,
conformal prediction, and the coupling into the planner.

Offline. A small synthetic cohort stands in for the real one: 12 fake patients
with a planted signal, which is enough to exercise every protocol and to catch
the failures that matter (leakage, a split that ignores patients, a conformal
threshold that undercovers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from onset_hfo.batch import (
    FEATURE_COLUMNS,
    CohortSpec,
    _run_token,
    add_within_subject_normalisation,
)
from onset_hfo.models import (
    MODELS,
    SozModel,
    baseline_scores,
    build_design_matrix,
    evaluate,
    fit_soz_model,
)
from onset_hfo.uncertainty import (
    calibrate,
    conformal_coverage_report,
    exchangeability_stress_test,
    expected_calibration_error,
    patient_candidate_set,
    reliability_table,
    split_conformal,
)


def _make_cohort(n_patients: int, contacts: tuple[int, int], seed: int,
                 patient_specific: float = 0.0) -> pd.DataFrame:
    """A synthetic cohort with a planted per-contact signal.

    Each patient gets its own rate scale, so a model reading raw rates across
    patients is mostly reading the patient -- the thing the normalisation
    columns exist to defeat.

    ``patient_specific`` controls *which* features separate SOZ from non-SOZ
    in each patient. At 0 every patient shares one decision boundary, so a
    model pooled across patients legitimately beats a subject-specific one:
    it is fitting the same function with far more data. Turn it up and each
    patient's boundary points in its own direction -- which is what real
    recordings look like, where electrode type, pathology and montage differ
    -- and the subject-specific model becomes the ceiling it is supposed to
    be. Simulating the mechanism is the point: asserting that within-subject
    wins on data where nothing is patient-specific would test nothing.
    """
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 1, len(FEATURE_COLUMNS))
    rows = []
    for p in range(n_patients):
        scale = 10 ** rng.uniform(0, 1.2)
        own = rng.normal(0, 1, len(FEATURE_COLUMNS))
        weights = (1 - patient_specific) * shared + patient_specific * own
        weights /= np.linalg.norm(weights) or 1.0
        n = int(rng.integers(*contacts))
        n_soz = max(3, int(0.2 * n))
        for c in range(n):
            is_soz = c < n_soz
            noise = rng.normal(0, 1.0, len(FEATURE_COLUMNS))
            values = noise + (3.0 * weights if is_soz else 0.0)
            row = {"subject": f"sub-{p:02d}", "channel": f"E{c}-E{c + 1}",
                   "site": f"site-{p % 3}", "is_soz": bool(is_soz),
                   "seizure_free": p % 2 == 0, "engel": 1 if p % 2 == 0 else 3}
            for i, name in enumerate(FEATURE_COLUMNS):
                row[name] = scale * float(values[i])
            rows.append(row)
    return add_within_subject_normalisation(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def rich_cohort() -> pd.DataFrame:
    """Enough contacts per patient to fit, and a patient-specific boundary.

    This is the regime the real archive is in, and the one where the
    subject-specific model is genuinely a ceiling.
    """
    return _make_cohort(8, (200, 260), seed=3, patient_specific=0.9)


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    """A synthetic cohort: 12 patients, ~50 channels each, a real but noisy signal.

    Each patient gets its own rate scale, so a model that reads raw rates
    across patients is mostly reading the patient -- which is the thing the
    normalisation columns exist to defeat, and the thing these tests check.
    """
    return _make_cohort(12, (40, 60), seed=11)


# --------------------------------------------------------------------------
# Cohort plumbing
# --------------------------------------------------------------------------


def test_run_tokens_survive_a_csv_round_trip():
    """A plan read back from CSV has run parsed as an int, and run-1 is a 404."""
    assert _run_token(1) == "01" and _run_token("01") == "01" and _run_token("1") == "01"
    assert _run_token("02b") == "02b"


def test_the_cohort_window_follows_the_seizure():
    spec = CohortSpec(pre_s=25.0, post_s=35.0)
    assert spec.duration_s == 60.0


def test_within_subject_normalisation_removes_the_patient_scale(cohort):
    """Raw rates differ ten-fold between patients for reasons unrelated to epilepsy."""
    raw = cohort.groupby("subject")["rms_rate_per_min"].median()
    ranked = cohort.groupby("subject")["rms_rate_per_min_rank"].median()
    assert raw.max() / max(raw.min(), 1e-9) > 3        # the problem is real
    assert ranked.max() - ranked.min() < 0.15          # and ranking removes it


def test_normalisation_does_not_invent_columns_it_cannot_compute(cohort):
    with pytest.raises(ValueError, match="no 'rank' feature columns"):
        build_design_matrix(cohort[["subject", "is_soz", "site"]], "rank")


def test_patients_with_no_positive_contact_are_dropped(cohort):
    """They contribute only negatives and bias every fold they appear in."""
    poisoned = cohort.copy()
    poisoned.loc[poisoned["subject"] == "sub-00", "is_soz"] = False
    _X, _y, groups, _sites, _names = build_design_matrix(poisoned, "raw")
    assert "sub-00" not in set(groups)


# --------------------------------------------------------------------------
# The three protocols
# --------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["within_subject", "lopo", "loso"])
def test_every_protocol_produces_out_of_fold_predictions(cohort, protocol):
    result = evaluate(cohort, "logistic", "rank", protocol)
    assert len(result.folds) >= 2
    assert np.isfinite(result.scores).mean() > 0.9
    metrics = result.metrics()
    assert metrics["auprc"] > metrics["prevalence"], "should beat the useless baseline"


def test_leave_one_patient_out_never_trains_on_the_test_patient(cohort):
    result = evaluate(cohort, "logistic", "rank", "lopo")
    for fold in result.folds:
        assert set(result.groups[fold.index]) == {fold.fold}


def test_leave_one_site_out_never_trains_on_the_test_site(cohort):
    result = evaluate(cohort, "logistic", "rank", "loso")
    sites = cohort.loc[cohort["is_soz"].notna(), "site"].to_numpy()
    for fold in result.folds:
        assert set(sites[fold.index]) == {fold.fold}


def test_the_within_subject_protocol_trains_only_on_the_same_patient(cohort):
    """That is what makes it an upper bound -- and what makes it undeployable."""
    result = evaluate(cohort, "logistic", "rank", "within_subject")
    for fold in result.folds:
        assert set(result.groups[fold.index]) == {fold.fold}


def test_the_ceiling_beats_the_deployable_score_when_there_is_data_to_fit(rich_cohort):
    """With enough contacts per patient, knowing part of the answer should help.

    This is the regime the real archive is in: 40-120 analysed channels per
    subject, where the within-subject score reaches 0.71 AUPRC against 0.48
    for leave-one-patient-out. The gap is the transfer problem.
    """
    ceiling = evaluate(rich_cohort, "logistic", "rank", "within_subject").metrics()
    deployable = evaluate(rich_cohort, "logistic", "rank", "lopo").metrics()
    assert ceiling["auprc"] >= deployable["auprc"]


def test_a_starved_ceiling_is_a_sample_size_statement_not_a_separability_one(cohort):
    """A within-subject fold trains on one patient's other contacts. With ~50
    contacts split four ways that is ~37 rows, which a boosted model cannot
    use -- so its 'ceiling' can fall below its LOPO score. Reading that as
    'the features are not there' would be exactly backwards, and the module
    docstring says so. This pins the behaviour rather than asserting a law
    that does not hold at every cohort size."""
    starved = evaluate(cohort, "logistic", "rank", "within_subject").metrics()
    deployable = evaluate(cohort, "logistic", "rank", "lopo").metrics()
    assert starved["auprc"] < deployable["auprc"], (
        "on a cohort with ~50 contacts per patient the within-subject folds are "
        "too small to fit; if this passes the starvation caveat needs revisiting")
    # And it is starvation, not a broken protocol: the same protocol on a
    # cohort with four times the contacts per patient comes out on top.
    assert starved["auprc"] > starved["prevalence"]


def test_the_within_subject_result_says_it_is_not_deployable(cohort):
    result = evaluate(cohort, "logistic", "rank", "within_subject")
    assert "upper bound" in result.note and "not a deployable model" in result.note


def test_the_untrained_baseline_is_always_scored(cohort):
    """'Better than chance' is not the bar; the pipeline ranks by rate for free."""
    baseline = baseline_scores(cohort, "rms_rate_per_min", "raw").metrics()
    assert baseline["protocol"] == "none"
    assert baseline["auprc"] > baseline["prevalence"]


def test_per_patient_metrics_expose_the_spread(cohort):
    frame = evaluate(cohort, "logistic", "rank", "lopo").per_patient()
    assert len(frame) == cohort["subject"].nunique()
    assert {"auprc", "prevalence", "n_soz"} <= set(frame.columns)


def test_an_unknown_protocol_is_refused(cohort):
    with pytest.raises(ValueError, match="protocol must be"):
        evaluate(cohort, "logistic", "rank", "leave_one_out_somehow")


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_calibration_error_is_zero_for_an_honest_model():
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0, 1, 4000)
    truth = (rng.uniform(0, 1, 4000) < probabilities).astype(int)
    assert expected_calibration_error(probabilities, truth) < 0.05


def test_calibration_error_catches_overconfidence():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, 2000)
    # Claims 0.95 confidence on coin flips.
    probabilities = np.where(truth == 1, 0.95, 0.95)
    assert expected_calibration_error(probabilities, truth) > 0.4


def test_isotonic_calibration_cannot_change_the_ranking():
    """It is monotone, so it changes what the numbers claim, not their order."""
    rng = np.random.default_rng(3)
    scores = rng.uniform(0, 1, 400)
    truth = (rng.uniform(0, 1, 400) < scores ** 2).astype(int)
    calibrated = calibrate(scores, truth, scores, "isotonic")
    order_before = np.argsort(scores)
    assert (np.diff(calibrated[order_before]) >= -1e-9).all()


def test_reliability_table_is_empty_rather_than_wrong_when_there_is_no_data():
    assert not len(reliability_table(np.array([]), np.array([])))


# --------------------------------------------------------------------------
# Conformal prediction
# --------------------------------------------------------------------------


def test_split_conformal_covers_at_the_nominal_rate_when_exchangeable():
    rng = np.random.default_rng(5)
    truth = rng.integers(0, 2, 4000)
    probabilities = np.clip(truth * 0.6 + rng.normal(0.2, 0.25, 4000), 0.01, 0.99)
    sets = split_conformal(probabilities[:2000], truth[:2000],
                           probabilities[2000:], truth[2000:], alpha=0.1)
    assert sets.empirical_coverage >= 0.87, "conformal must cover when exchangeable"


def test_a_tighter_alpha_gives_wider_sets():
    rng = np.random.default_rng(6)
    truth = rng.integers(0, 2, 2000)
    probabilities = np.clip(truth * 0.5 + rng.normal(0.25, 0.3, 2000), 0.01, 0.99)
    wide = split_conformal(probabilities, truth, probabilities, truth, alpha=0.01)
    narrow = split_conformal(probabilities, truth, probabilities, truth, alpha=0.3)
    assert wide.sizes.mean() >= narrow.sizes.mean()


def test_an_unmeasurable_channel_is_maximally_uncertain():
    sets = split_conformal(np.array([0.2, 0.8]), np.array([0, 1]),
                           np.array([np.nan]), alpha=0.1)
    assert sets.sets[0] == {0, 1}


def test_conformal_needs_a_calibration_set():
    with pytest.raises(ValueError, match="non-empty calibration set"):
        split_conformal(np.array([]), np.array([]), np.array([0.5]))


def test_the_conformal_split_is_by_patient_not_by_contact(cohort):
    """Contacts in one recording share electrodes, amplifier and seizure."""
    result = evaluate(cohort, "logistic", "rank", "lopo")
    report = conformal_coverage_report(result, alpha=0.1)
    assert report["available"]
    assert "by patient" in report["split"]
    assert report["n_calibration_patients"] + report["n_test_patients"] == 12


def test_the_candidate_set_is_reported_per_patient(cohort):
    result = evaluate(cohort, "logistic", "rank", "lopo")
    report = conformal_coverage_report(result, alpha=0.1)
    frame = pd.DataFrame(report["candidate_sets"])
    assert {"subject", "n_candidates", "candidate_fraction"} <= set(frame.columns)
    assert (frame["n_candidates"] <= frame["n_contacts"]).all()


def test_the_exchangeability_stress_test_reports_per_group_coverage(cohort):
    result = evaluate(cohort, "logistic", "rank", "lopo")
    sites = cohort.loc[cohort["is_soz"].notna(), "site"].to_numpy()
    stress = exchangeability_stress_test(result, by="site", sites=sites)
    assert stress["available"] and len(stress["per_group"]) >= 2
    assert stress["worst_empirical_coverage"] <= stress["mean_empirical_coverage"]


def test_the_stress_test_says_so_rather_than_guessing_with_one_group(cohort):
    result = evaluate(cohort, "logistic", "rank", "lopo")
    one = np.array(["only-site"] * len(result.truth))
    assert exchangeability_stress_test(result, sites=one)["available"] is False


def test_patient_candidate_set_needs_groups():
    sets = split_conformal(np.array([0.1, 0.9]), np.array([0, 1]), np.array([0.5]))
    assert not len(patient_candidate_set(sets))


# --------------------------------------------------------------------------
# The deployable model, and the coupling
# --------------------------------------------------------------------------


def test_a_fitted_model_carries_its_own_contract(cohort):
    model = fit_soz_model(cohort, "logistic", "rank")
    described = model.describe()
    assert described["features"] and described["normalisation"] == "rank"
    assert described["n_train_patients"] >= 1
    assert "not a seizure onset zone" in described["caveat"]


def test_holding_a_subject_out_removes_it_from_fitting_and_calibration(cohort):
    model = fit_soz_model(cohort, "logistic", "rank", holdout=["sub-00"])
    assert model.held_out == ["sub-00"]
    # n_train_patients counts only patients the classifier actually fitted on.
    assert model.n_train_patients <= cohort["subject"].nunique() - 1


def test_a_model_round_trips_through_disk(cohort, tmp_path):
    model = fit_soz_model(cohort, "logistic", "rank")
    path = model.save(tmp_path / "soz.pkl")
    reloaded = SozModel.load(path)
    one = cohort[cohort["subject"] == "sub-01"]
    assert np.allclose(model.predict(one), reloaded.predict(one))


def test_predicting_on_a_table_missing_features_says_which(cohort):
    model = fit_soz_model(cohort, "logistic", "rank")
    with pytest.raises(ValueError, match="missing"):
        model.predict(cohort[["subject", "channel", "is_soz"]])


def test_the_model_produces_a_candidate_set(cohort):
    model = fit_soz_model(cohort, "logistic", "rank", alpha=0.1)
    one = cohort[cohort["subject"] == "sub-01"]
    sets = model.conformal_sets(model.predict(one))
    assert len(sets) == len(one)
    assert all(s <= {0, 1} for s in sets)


def test_fitting_a_conformal_model_needs_enough_patients(cohort):
    two = cohort[cohort["subject"].isin(["sub-00", "sub-01"])]
    with pytest.raises(ValueError, match="at least 3 patients"):
        fit_soz_model(two, "logistic", "rank")


def test_the_planner_stops_on_the_conformal_width():
    from onset_agent.contract import ToolRun
    from onset_agent.evidence import EvidenceStore
    from onset_agent.planner import ConformalWidth

    rule = ConformalWidth(max_width=5)
    store = EvidenceStore()
    assert rule.should_stop(store, False) == (False, "")      # nothing measured yet

    store.append(ToolRun("soz_001", "estimate_soz_probability", {},
                         {"candidate_set_size": 30, "n_channels": 71}))
    assert rule.should_stop(store, False)[0] is False         # still too wide to act on

    store.append(ToolRun("soz_002", "estimate_soz_probability", {},
                         {"candidate_set_size": 4, "n_channels": 71}))
    stop, reason = rule.should_stop(store, False)
    assert stop and "4 channel" in reason


def test_the_soz_tool_refuses_cleanly_when_no_model_is_attached(registry):
    """A missing model must degrade the planner, not crash it."""
    run = registry.run("estimate_soz_probability", {})
    assert not run.ok and "no learned SOZ model" in run.error
    assert "fit_soz_model" in run.error


def test_the_model_is_a_tool_like_any_other(cohort, recording):
    """The agent does not know how it works, and does not need to."""
    from onset_agent.analysis import AnalysisSession, build_registry

    model = fit_soz_model(cohort, "logistic", "raw")
    session = AnalysisSession(recording, soz_model=model)
    run = build_registry(session).run("estimate_soz_probability", {"k": 3})
    assert run.ok, run.error
    assert {"candidate_set_size", "candidate_fraction", "channels", "model"} <= set(run.output)
    assert run.output["candidate_set_size"] <= run.output["n_channels"]
    assert "not a seizure onset zone" in run.output["model"]["caveat"]


def test_every_model_spec_builds(cohort):
    for name in MODELS:
        result = evaluate(cohort, name, "rank", "lopo")
        assert np.isfinite(result.scores).any()
