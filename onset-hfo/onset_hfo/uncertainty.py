"""Calibration and conformal prediction: turning a ranking into a set.

A ranking is not enough to act on. "These contacts, ranked" invites the reader
to draw their own line; "these six contacts, with 90% coverage" is a statement
with a guarantee attached. This module does two things to get there.

**Calibration.** A classifier's raw score is not a probability. Modern models
are systematically overconfident, and a contact scored 0.9 should be in the
SOZ about 90% of the time or the number is decoration.
:func:`expected_calibration_error` measures the gap and
:func:`calibrate` closes it with isotonic or Platt scaling.

**Split conformal prediction.** Given a held-out calibration set, find the
score threshold at which the true label is retained (1 - alpha) of the time,
then apply it to new contacts. The result is a *prediction set* per contact,
and the guarantee is distribution-free: it needs no assumption about the model
being right, only that calibration and test data are **exchangeable**.

The assumption is the interesting part
--------------------------------------
Exchangeability is exactly what a cross-site study breaks. Calibrate on three
hospitals, test on a fourth, and the coverage guarantee is void -- not
approximately, but in the sense that the theorem no longer applies.
:func:`conformal_coverage_report` therefore always reports **empirical**
coverage beside the nominal level, and :func:`exchangeability_stress_test`
deliberately calibrates on one set of patients and tests on another to measure
how far the guarantee degrades. Distribution shift breaking the uncertainty
estimate that was supposed to protect against distribution shift is a finding,
not a bug to hide.

The coupling
------------
:func:`patient_candidate_set` turns per-contact prediction sets into one
number per patient: how many contacts cannot be ruled out at ``1 - alpha``.
That number is what
:class:`onset_agent.planner.ConformalWidth` consumes as a stopping rule -- the
planner keeps gathering evidence while the set is too wide to act on. It is
the one place the two halves of the system actually meet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["expected_calibration_error", "reliability_table", "calibrate",
           "ConformalSets", "split_conformal", "conformal_coverage_report",
           "patient_candidate_set", "exchangeability_stress_test"]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def expected_calibration_error(probabilities: np.ndarray, truth: np.ndarray,
                               n_bins: int = 10) -> float:
    """Average gap between confidence and accuracy, weighted by bin count.

    Zero is perfect. A model at 0.15 is claiming 90% confidence on contacts
    that are positive 75% of the time, which a clinician acting on the number
    would experience as being lied to.
    """
    table = reliability_table(probabilities, truth, n_bins)
    if not len(table):
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["mean_predicted"] - table["observed"]).abs()).sum())


def reliability_table(probabilities: np.ndarray, truth: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed frequency, per confidence bin."""
    probabilities = np.asarray(probabilities, dtype=float)
    truth = np.asarray(truth, dtype=int)
    finite = np.isfinite(probabilities)
    probabilities, truth = probabilities[finite], truth[finite]
    if not len(probabilities):
        return pd.DataFrame(columns=["bin_low", "bin_high", "n", "mean_predicted",
                                     "observed"])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = index == b
        if not mask.any():
            continue
        rows.append({"bin_low": round(float(edges[b]), 3),
                     "bin_high": round(float(edges[b + 1]), 3),
                     "n": int(mask.sum()),
                     "mean_predicted": round(float(probabilities[mask].mean()), 4),
                     "observed": round(float(truth[mask].mean()), 4)})
    return pd.DataFrame(rows)


def calibrate(scores_fit: np.ndarray, truth_fit: np.ndarray, scores_apply: np.ndarray,
              method: str = "isotonic") -> np.ndarray:
    """Map raw scores onto calibrated probabilities.

    ``isotonic`` is non-parametric and strictly monotone, so it cannot change
    the ranking -- only what the numbers claim. ``platt`` (a logistic fit) has
    two parameters and is the safer choice on a small calibration set, where
    isotonic will happily memorise noise.
    """
    finite = np.isfinite(scores_fit)
    scores_fit, truth_fit = np.asarray(scores_fit)[finite], np.asarray(truth_fit)[finite]
    if len(set(truth_fit.tolist())) < 2:
        return np.asarray(scores_apply, dtype=float)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(scores_fit, truth_fit)
        return np.clip(model.predict(np.asarray(scores_apply, dtype=float)), 0.0, 1.0)
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
        model.fit(scores_fit.reshape(-1, 1), truth_fit)
        return model.predict_proba(np.asarray(scores_apply, dtype=float).reshape(-1, 1))[:, 1]
    raise ValueError("method must be 'isotonic' or 'platt'")


# --------------------------------------------------------------------------
# Split conformal prediction
# --------------------------------------------------------------------------


@dataclass
class ConformalSets:
    """Per-contact prediction sets at a nominal 1 - alpha coverage."""

    alpha: float
    quantile: float
    n_calibration: int
    #: For each contact, the labels that cannot be ruled out. Binary problem,
    #: so each entry is a subset of {0, 1}: a singleton is a decision, a
    #: two-element set is an explicit "cannot tell", and an empty set means
    #: neither label is plausible at this confidence -- which is information,
    #: not a bug.
    sets: list[set] = field(default_factory=list)
    truth: np.ndarray | None = None
    groups: np.ndarray | None = None

    @property
    def sizes(self) -> np.ndarray:
        return np.array([len(s) for s in self.sets])

    @property
    def empirical_coverage(self) -> float:
        """Fraction of contacts whose set contains the true label."""
        if self.truth is None or not len(self.sets):
            return float("nan")
        return float(np.mean([int(t) in s for s, t in zip(self.sets, self.truth,
                                                          strict=False)]))

    def as_dict(self) -> dict:
        return {"alpha": self.alpha, "nominal_coverage": round(1 - self.alpha, 4),
                "empirical_coverage": round(self.empirical_coverage, 4),
                "quantile": round(self.quantile, 4),
                "n_calibration": self.n_calibration, "n_test": len(self.sets),
                "mean_set_size": round(float(self.sizes.mean()), 4) if len(self.sets) else None,
                "frac_singleton": round(float((self.sizes == 1).mean()), 4) if len(self.sets) else None,
                "frac_ambiguous": round(float((self.sizes == 2).mean()), 4) if len(self.sets) else None,
                "frac_empty": round(float((self.sizes == 0).mean()), 4) if len(self.sets) else None}


def split_conformal(probabilities_cal: np.ndarray, truth_cal: np.ndarray,
                    probabilities_test: np.ndarray, truth_test: np.ndarray | None = None,
                    alpha: float = 0.1, groups_test: np.ndarray | None = None
                    ) -> ConformalSets:
    """Standard split-conformal sets for a binary problem.

    The score is ``s = 1 - p(true label)`` on the calibration set. The
    threshold is its ``ceil((n+1)(1-alpha))/n`` empirical quantile -- the
    finite-sample correction matters at the cohort sizes here, where ``n`` is
    hundreds of contacts and using the plain quantile undercovers.

    A contact's prediction set is every label whose ``1 - p(label)`` falls at
    or below that threshold.
    """
    probabilities_cal = np.asarray(probabilities_cal, dtype=float)
    truth_cal = np.asarray(truth_cal, dtype=int)
    finite = np.isfinite(probabilities_cal)
    probabilities_cal, truth_cal = probabilities_cal[finite], truth_cal[finite]

    n = len(truth_cal)
    if n == 0:
        raise ValueError("split_conformal needs a non-empty calibration set")
    p_true = np.where(truth_cal == 1, probabilities_cal, 1.0 - probabilities_cal)
    conformity = 1.0 - p_true
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    quantile = float(np.quantile(conformity, level, method="higher"))

    sets = []
    for p in np.asarray(probabilities_test, dtype=float):
        if not np.isfinite(p):
            sets.append({0, 1})          # unmeasured is maximally uncertain
            continue
        candidate = set()
        if 1.0 - p <= quantile:
            candidate.add(1)
        if p <= quantile:
            candidate.add(0)
        sets.append(candidate)
    return ConformalSets(alpha=alpha, quantile=quantile, n_calibration=n, sets=sets,
                         truth=None if truth_test is None else np.asarray(truth_test, int),
                         groups=groups_test)


def conformal_coverage_report(result, alpha: float = 0.1, calibration_fraction: float = 0.4,
                              seed: int = 0, calibrate_method: str | None = "isotonic"
                              ) -> dict:
    """Split an :class:`~onset_hfo.models.EvaluationResult` and report coverage.

    The split is **by patient**, never by contact. Contacts from one patient
    are strongly dependent -- same electrodes, same amplifier, same seizure --
    so a random contact-level split puts near-duplicates on both sides and
    reports a coverage that will not survive contact with a new patient.
    """
    scores, truth, groups = result.scores, result.truth, result.groups
    patients = sorted(set(groups))
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(patients))
    n_cal = max(1, int(round(calibration_fraction * len(shuffled))))
    cal_patients = set(shuffled[:n_cal])
    if len(cal_patients) >= len(patients):
        return {"available": False,
                "reason": "need at least two patients to split calibration from test"}

    cal = np.array([g in cal_patients for g in groups])
    test = ~cal
    if not cal.any() or not test.any():
        return {"available": False, "reason": "empty calibration or test split"}

    p_cal, p_test = scores[cal], scores[test]
    if calibrate_method:
        p_cal_fit = calibrate(scores[cal], truth[cal], scores[cal], calibrate_method)
        p_test = calibrate(scores[cal], truth[cal], scores[test], calibrate_method)
        p_cal = p_cal_fit

    sets = split_conformal(p_cal, truth[cal], p_test, truth[test], alpha=alpha,
                           groups_test=groups[test])
    payload = {"available": True, **sets.as_dict(),
               "n_calibration_patients": len(cal_patients),
               "n_test_patients": len(patients) - len(cal_patients),
               "calibration": calibrate_method or "none",
               "ece_before": round(expected_calibration_error(scores[test], truth[test]), 4),
               "ece_after": round(expected_calibration_error(p_test, truth[test]), 4),
               "split": "by patient (contacts within a patient are not exchangeable)"}
    payload["candidate_sets"] = patient_candidate_set(sets).to_dict("records")
    return payload


def patient_candidate_set(sets: ConformalSets) -> pd.DataFrame:
    """Per patient: how many contacts cannot be ruled out, and whether SOZ is in there.

    This is the clinically meaningful object and the one the planner consumes.
    A patient whose candidate set is four contacts has an actionable result; a
    patient whose set is forty has not been localized, however confident any
    individual score looked.
    """
    if sets.groups is None:
        return pd.DataFrame(columns=["subject", "n_contacts", "n_candidates",
                                     "candidate_fraction", "n_true_soz", "n_soz_covered"])
    rows = []
    groups = np.asarray(sets.groups)
    contains_soz = np.array([1 in s for s in sets.sets])
    for subject in sorted(set(groups.tolist())):
        mask = groups == subject
        row = {"subject": subject, "n_contacts": int(mask.sum()),
               "n_candidates": int(contains_soz[mask].sum())}
        row["candidate_fraction"] = round(row["n_candidates"] / max(1, row["n_contacts"]), 4)
        if sets.truth is not None:
            truth = np.asarray(sets.truth)[mask]
            row["n_true_soz"] = int(truth.sum())
            row["n_soz_covered"] = int((contains_soz[mask] & (truth == 1)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def exchangeability_stress_test(result, alpha: float = 0.1,
                                by: str = "site", sites: np.ndarray | None = None) -> dict:
    """Calibrate on one group and test on another: does the guarantee survive?

    Conformal coverage is guaranteed only when calibration and test data are
    exchangeable. Calibrating on three hospitals and testing on a fourth
    breaks that, and this measures by how much. A drop from a nominal 0.90 to
    an empirical 0.7 is the result that joins the generalization and
    uncertainty questions into one story: distribution shift breaks not only
    accuracy but the uncertainty estimate that was supposed to protect
    against it.
    """
    key = np.asarray(sites if sites is not None else result.groups)
    levels = sorted(set(key.tolist()))
    if len(levels) < 2:
        return {"available": False,
                "reason": f"need at least two {by}s to test exchangeability"}
    rows = []
    for held_out in levels:
        test = key == held_out
        cal = ~test
        if not test.any() or not cal.any() or len(set(result.truth[cal])) < 2:
            continue
        p_cal = calibrate(result.scores[cal], result.truth[cal], result.scores[cal])
        p_test = calibrate(result.scores[cal], result.truth[cal], result.scores[test])
        sets = split_conformal(p_cal, result.truth[cal], p_test, result.truth[test],
                               alpha=alpha, groups_test=result.groups[test])
        rows.append({"held_out": str(held_out), "n_test": int(test.sum()),
                     "nominal_coverage": round(1 - alpha, 3),
                     "empirical_coverage": round(sets.empirical_coverage, 4),
                     "mean_set_size": round(float(sets.sizes.mean()), 3)})
    if not rows:
        return {"available": False, "reason": "no usable held-out group"}
    frame = pd.DataFrame(rows)
    return {"available": True, "by": by, "alpha": alpha,
            "nominal_coverage": round(1 - alpha, 3),
            "mean_empirical_coverage": round(float(frame["empirical_coverage"].mean()), 4),
            "worst_empirical_coverage": round(float(frame["empirical_coverage"].min()), 4),
            "per_group": frame.to_dict("records"),
            "reading": ("coverage below the nominal level means exchangeability was "
                        "violated: the guarantee does not transfer across this grouping")}
