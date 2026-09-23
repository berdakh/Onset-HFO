"""A learned per-contact SOZ model, and the three ways to evaluate it.

The question this module exists to answer is RQ1 of the localization thesis:

> Can a contact-level classifier trained on ictal features localize the
> seizure onset zone more accurately than a fixed-threshold biomarker such as
> HFO rate alone?

Note the comparator. "Better than chance" is not the bar -- the pipeline
already ranks channels by ripple rate for free, so a model that merely beats
chance while losing to the rate it was built from has established nothing.
:func:`evaluate` always scores the rate baselines alongside the models.

Three evaluation protocols, and the gap between them is the result
------------------------------------------------------------------

============== ====================================== =========================
protocol       train on                               what it answers
============== ====================================== =========================
within-subject other contacts of the *same* patient   Are SOZ and non-SOZ
                                                      contacts separable at
                                                      all? The **ceiling**.
LOPO           every *other* patient                  The deployable number.
leave-one-site every patient at the *other* centres   What survives a change
                                                      of hospital.
============== ====================================== =========================

**Within-subject is an upper bound, not a deployable model.** To use it you
would have to already know some of that patient's SOZ contacts, which is the
thing you are trying to find. It is reported because it bounds everything
else: if a model cannot separate contacts *inside* one recording, where the
electrodes, amplifier, sedation and seizure are all held constant, then no
cross-patient model is going to. It is also the personalisation arm -- the
same subject-specific-versus-subject-independent gap that shows up in every
BCI decoding study, here applied to contacts instead of trials.

So the headline is not one number but a drop: ceiling, then deployable, then
cross-site. Where the drop happens says what the problem is. A high ceiling
with a poor LOPO score means the features are real but do not transfer, which
is a normalisation problem. A low ceiling means the features are not there,
and no amount of domain adaptation will help.

**The ceiling is only a ceiling when there is enough of it to fit.** A
within-subject fold trains on one patient's other contacts -- on this archive,
40 to 120 rows, split four ways. That is ample for logistic regression and
thin for a boosted model, and on a cohort with fewer contacts per patient the
subject-specific score can fall *below* the leave-one-patient-out score purely
from data starvation. When that happens it is a statement about sample size,
not about separability, and reading it as "the features are not there" would
be exactly backwards. Compare the two models before concluding anything: if
logistic beats boosting within-subject but loses to it under LOPO, the
within-subject folds are too small.

Class imbalance
---------------
Roughly 10-20% of contacts are positive, so accuracy is meaningless and AUROC
is optimistic. **AUPRC against the prevalence baseline** is the number to
read, and every result carries the prevalence next to it so the reader can
see what a useless model would have scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["ModelSpec", "FoldResult", "EvaluationResult", "MODELS", "NORMALISATIONS",
           "build_design_matrix", "evaluate", "baseline_scores", "SozModel",
           "fit_soz_model", "evaluate_personalized", "label_budget_curve",
           "evaluate_active_learning", "active_learning_comparison",
           "ACQUISITION"]

#: How features are scaled before a model sees them. The choice matters more
#: than the choice of classifier: raw rates differ ten-fold between subjects
#: for reasons unrelated to epilepsy, so a model given raw values across
#: patients mostly learns to recognise the patient.
NORMALISATIONS = {
    "raw": "",            # the feature as measured
    "z": "_z",            # robust z-score within the subject
    "rank": "_rank",      # percentile within the subject
}

#: The rate columns the models have to beat. These need no training at all.
BASELINE_COLUMNS = ["rms_rate_per_min", "ll_rate_per_min", "spike_rate_per_min"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: object
    description: str


def _logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))


def _gradient_boosting():
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Shallow and heavily regularised on purpose: a few hundred contacts per
    # fold is not enough data to justify a deep model, and an overfitted
    # ceiling is worse than no ceiling.
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.06,
        min_samples_leaf=10, l2_regularization=1.0, random_state=0)


MODELS: dict[str, ModelSpec] = {
    "logistic": ModelSpec("logistic", _logistic,
                          "L2 logistic regression, class-weighted. The floor: if the "
                          "boosted model cannot beat this, the extra capacity is noise."),
    "gradient_boosting": ModelSpec("gradient_boosting", _gradient_boosting,
                                   "Shallow histogram gradient boosting. Strong on tabular "
                                   "contact-level features and it handles NaN natively."),
}


# --------------------------------------------------------------------------
# Design matrix
# --------------------------------------------------------------------------


def build_design_matrix(features: pd.DataFrame, normalisation: str = "rank",
                        columns: list[str] | None = None):
    """``(X, y, groups, sites, names)`` for one normalisation scheme.

    Rows with no label, and subjects with no positive contact at all, are
    dropped: a patient whose labelled electrode was excluded as bad
    contributes only negatives, which biases every fold it appears in.
    """
    from onset_hfo.batch import FEATURE_COLUMNS

    if normalisation not in NORMALISATIONS:
        raise ValueError(f"normalisation must be one of {', '.join(NORMALISATIONS)}")
    suffix = NORMALISATIONS[normalisation]
    base = columns or FEATURE_COLUMNS
    names = [f"{c}{suffix}" for c in base if f"{c}{suffix}" in features.columns]
    if not names:
        raise ValueError(f"no {normalisation!r} feature columns found; run "
                         "onset_hfo.batch.add_within_subject_normalisation first")

    frame = features[features["is_soz"].notna()].copy()
    positives = frame.groupby("subject")["is_soz"].transform("sum")
    frame = frame[positives > 0]

    X = frame[names].astype(float).to_numpy()
    X = np.where(np.isfinite(X), X, np.nan)
    y = frame["is_soz"].astype(int).to_numpy()
    groups = frame["subject"].astype(str).to_numpy()
    sites = frame["site"].astype(str).to_numpy() if "site" in frame else np.array([""] * len(y))
    return X, y, groups, sites, names


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class FoldResult:
    fold: str
    n_train: int
    n_test: int
    n_positive: int
    scores: np.ndarray
    truth: np.ndarray
    index: np.ndarray


@dataclass
class EvaluationResult:
    """Out-of-fold predictions for one (model, normalisation, protocol)."""

    model: str
    normalisation: str
    protocol: str
    scores: np.ndarray
    truth: np.ndarray
    groups: np.ndarray
    folds: list[FoldResult] = field(default_factory=list)
    note: str = ""

    @property
    def prevalence(self) -> float:
        return float(self.truth.mean()) if len(self.truth) else float("nan")

    def metrics(self, ks=(1, 3, 5)) -> dict:
        """AUPRC, AUROC, and per-patient precision@k -- with the useless baseline."""
        out = {"model": self.model, "normalisation": self.normalisation,
               "protocol": self.protocol, "n": int(len(self.truth)),
               "n_patients": int(len(set(self.groups))),
               "prevalence": round(self.prevalence, 4)}
        out.update(_ranking_metrics(self.scores, self.truth))
        out["auprc_lift_over_prevalence"] = (
            round(out["auprc"] / self.prevalence, 3) if self.prevalence else None)
        for k in ks:
            hits, total = [], 0
            for subject in sorted(set(self.groups)):
                mask = self.groups == subject
                if mask.sum() < k:
                    continue
                order = np.argsort(-self.scores[mask])[:k]
                hits.append(float(self.truth[mask][order].sum()) / k)
                total += 1
            out[f"precision_at_{k}"] = round(float(np.mean(hits)), 4) if hits else None
            out[f"n_patients_at_{k}"] = total
        if self.note:
            out["note"] = self.note
        return out


    def per_patient(self) -> pd.DataFrame:
        """AUPRC and AUROC computed separately for each patient.

        The pooled number hides two things a reviewer will ask about. First,
        patients differ in prevalence -- one with 33 SOZ contacts out of 73
        and one with 6 out of 120 are not the same problem, and pooling lets
        the easy patient carry the hard one. Second, contacts within a patient
        are strongly dependent, so the effective sample size is closer to the
        number of patients than the number of rows. The spread of this column
        is the honest error bar.
        """
        rows = []
        for subject in sorted(set(self.groups)):
            mask = self.groups == subject
            row = {"subject": subject, "n_contacts": int(mask.sum()),
                   "n_soz": int(self.truth[mask].sum())}
            row["prevalence"] = round(row["n_soz"] / max(1, row["n_contacts"]), 4)
            row.update(_ranking_metrics(self.scores[mask], self.truth[mask]))
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self, ks=(1, 3, 5)) -> dict:
        """Pooled metrics plus the per-patient spread."""
        out = self.metrics(ks)
        frame = self.per_patient()
        usable = frame[frame["auprc"].notna()]
        if len(usable):
            out["auprc_per_patient_median"] = round(float(usable["auprc"].median()), 4)
            out["auprc_per_patient_iqr"] = [
                round(float(usable["auprc"].quantile(0.25)), 4),
                round(float(usable["auprc"].quantile(0.75)), 4)]
            out["n_patients_scored"] = int(len(usable))
        return out


def _ranking_metrics(scores: np.ndarray, truth: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    finite = np.isfinite(scores)
    scores, truth = scores[finite], truth[finite]
    if len(truth) == 0 or truth.min() == truth.max():
        return {"auprc": None, "auroc": None}
    return {"auprc": round(float(average_precision_score(truth, scores)), 4),
            "auroc": round(float(roc_auc_score(truth, scores)), 4)}


# --------------------------------------------------------------------------
# The three protocols
# --------------------------------------------------------------------------


def _fit_predict(spec: ModelSpec, X_train, y_train, X_test) -> np.ndarray:
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    model = spec.build()
    model.fit(imputer.fit_transform(X_train), y_train)
    return model.predict_proba(imputer.transform(X_test))[:, 1]


def evaluate(features: pd.DataFrame, model: str = "gradient_boosting",
             normalisation: str = "rank", protocol: str = "lopo",
             n_splits: int = 4, seed: int = 0) -> EvaluationResult:
    """Out-of-fold predictions under one protocol.

    ``protocol`` is ``"within_subject"``, ``"lopo"`` or ``"loso"``.
    """
    spec = MODELS[model]
    X, y, groups, sites, _names = build_design_matrix(features, normalisation)
    scores = np.full(len(y), np.nan)
    folds: list[FoldResult] = []
    note = ""

    if protocol == "within_subject":
        from sklearn.model_selection import StratifiedKFold

        note = ("subject-specific: trained on other contacts of the SAME patient. An "
                "upper bound, not a deployable model -- using it would require already "
                "knowing some of that patient's SOZ contacts.")
        for subject in sorted(set(groups)):
            mask = np.flatnonzero(groups == subject)
            y_sub = y[mask]
            k = min(int(n_splits), int(y_sub.sum()), int((y_sub == 0).sum()))
            if k < 2:
                continue        # too few of one class to split at all
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            for train_idx, test_idx in splitter.split(np.zeros(len(mask)), y_sub):
                predicted = _fit_predict(spec, X[mask[train_idx]], y_sub[train_idx],
                                         X[mask[test_idx]])
                scores[mask[test_idx]] = predicted
                folds.append(FoldResult(subject, len(train_idx), len(test_idx),
                                        int(y_sub[test_idx].sum()), predicted,
                                        y_sub[test_idx], mask[test_idx]))
    elif protocol in ("lopo", "loso"):
        key = groups if protocol == "lopo" else sites
        note = ("left-out patient" if protocol == "lopo" else
                "left-out recording centre: nothing from the test site is in training")
        for held_out in sorted(set(key)):
            test = np.flatnonzero(key == held_out)
            train = np.flatnonzero(key != held_out)
            if not len(test) or len(set(y[train])) < 2:
                continue
            predicted = _fit_predict(spec, X[train], y[train], X[test])
            scores[test] = predicted
            folds.append(FoldResult(str(held_out), len(train), len(test),
                                    int(y[test].sum()), predicted, y[test], test))
    else:
        raise ValueError("protocol must be 'within_subject', 'lopo' or 'loso'")

    return EvaluationResult(model=model, normalisation=normalisation, protocol=protocol,
                            scores=scores, truth=y, groups=groups, folds=folds, note=note)


def baseline_scores(features: pd.DataFrame, column: str = "rms_rate_per_min",
                    normalisation: str = "rank") -> EvaluationResult:
    """The untrained comparator: rank channels by one measured rate.

    No model, no folds, no fitting. This is what the pipeline already does,
    and any learned model has to beat it to have earned its complexity.
    """
    suffix = NORMALISATIONS[normalisation]
    name = f"{column}{suffix}" if f"{column}{suffix}" in features.columns else column
    frame = features[features["is_soz"].notna()].copy()
    positives = frame.groupby("subject")["is_soz"].transform("sum")
    frame = frame[positives > 0]
    return EvaluationResult(
        model=f"baseline:{column}", normalisation=normalisation, protocol="none",
        scores=frame[name].astype(float).to_numpy(),
        truth=frame["is_soz"].astype(int).to_numpy(),
        groups=frame["subject"].astype(str).to_numpy(),
        note="untrained: channels ranked by a single measured rate")


# --------------------------------------------------------------------------
# A fitted model that can be saved, loaded, and called as an agent tool
# --------------------------------------------------------------------------


@dataclass
class SozModel:
    """A fitted classifier plus everything needed to use it on a new recording.

    This is the object that crosses the boundary into the orchestration half.
    The agent does not know how it works and does not need to: it calls a tool
    that returns JSON, exactly like every other analyzer. What the object has
    to carry with it is therefore not just the weights but the *contract*:
    which features, in which order, under which normalisation, and the
    conformal threshold calibrated at a stated alpha.

    It also carries where it came from. A model fitted on 22 ictal recordings
    from two centres is not a general SOZ classifier, and a tool that returned
    a bare probability would invite exactly that misreading, so
    :meth:`describe` travels with every prediction.
    """

    model: object
    imputer: object
    feature_names: list[str]
    normalisation: str
    #: Isotonic/Platt map fitted on held-out patients, or ``None``.
    calibrator: object | None = None
    #: Split-conformal threshold on ``1 - p(label)``, at ``alpha``.
    conformal_quantile: float | None = None
    alpha: float = 0.1
    n_train_patients: int = 0
    n_train_contacts: int = 0
    train_prevalence: float = float("nan")
    sites: list[str] = field(default_factory=list)
    #: Subjects excluded from both fitting and calibration, so the model can be
    #: demonstrated on them without a caveat.
    held_out: list[str] = field(default_factory=list)
    protocol_note: str = ""

    def describe(self) -> dict:
        return {"features": list(self.feature_names), "normalisation": self.normalisation,
                "n_train_patients": self.n_train_patients,
                "n_train_contacts": self.n_train_contacts,
                "train_prevalence": (None if not np.isfinite(self.train_prevalence)
                                     else round(self.train_prevalence, 4)),
                "train_sites": list(self.sites), "held_out_subjects": list(self.held_out),
                "alpha": self.alpha,
                "conformal_quantile": (None if self.conformal_quantile is None
                                       else round(self.conformal_quantile, 4)),
                "caveat": ("fitted on ictal recordings from this cohort only. A high "
                           "probability is a statement about ripple-band and discharge "
                           "features, not a seizure onset zone.")}

    # -- prediction -------------------------------------------------------
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities for a per-channel feature table."""
        X = self._matrix(frame)
        raw = self.model.predict_proba(self.imputer.transform(X))[:, 1]
        if self.calibrator is None:
            return raw
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def conformal_sets(self, probabilities: np.ndarray) -> list[set]:
        """Prediction sets at ``1 - alpha``. Requires a calibrated quantile."""
        if self.conformal_quantile is None:
            raise ValueError("this model has no conformal quantile; fit it with "
                             "calibration patients held out")
        q = self.conformal_quantile
        out = []
        for p in np.asarray(probabilities, dtype=float):
            if not np.isfinite(p):
                out.append({0, 1})
                continue
            candidate = set()
            if 1.0 - p <= q:
                candidate.add(1)
            if p <= q:
                candidate.add(0)
            out.append(candidate)
        return out

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise ValueError(
                f"the feature table is missing {len(missing)} column(s) this model needs: "
                f"{', '.join(missing[:5])}. Build it with onset_hfo.batch and, for a "
                f"normalised model, add_within_subject_normalisation.")
        X = frame[self.feature_names].astype(float).to_numpy()
        return np.where(np.isfinite(X), X, np.nan)

    # -- persistence ------------------------------------------------------
    def save(self, path) -> object:
        import pickle
        from pathlib import Path as _Path

        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        return path

    @staticmethod
    def load(path) -> SozModel:
        import pickle
        from pathlib import Path as _Path

        with _Path(path).open("rb") as handle:
            return pickle.load(handle)


def fit_soz_model(features: pd.DataFrame, model: str = "gradient_boosting",
                  normalisation: str = "raw", alpha: float = 0.1,
                  calibration_fraction: float = 0.3, seed: int = 0,
                  holdout: list[str] | str | None = None) -> SozModel:
    """Fit a deployable model on a cohort, with a conformal threshold.

    Patients are split into a fitting set and a calibration set. The
    calibration patients are never seen by the classifier, which is what makes
    the conformal threshold and the isotonic map honest: both are estimated on
    data the model did not fit.

    The split is by patient, never by contact. Contacts from one recording
    share electrodes, amplifier and seizure, so a contact-level split leaks
    and produces a threshold that looks tight and does not hold.

    ``holdout`` removes named subjects from **both** the fitting and the
    calibration sets. Use it for any patient you intend to demonstrate the
    model on. Leaving a demo subject in the calibration set does not corrupt
    the ranking -- the classifier still never saw it -- but it does make that
    patient's probabilities and candidate-set width optimistic, and a figure
    captioned "the model found these contacts" should not need a footnote
    about which half of it was honest.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression

    from onset_hfo.uncertainty import split_conformal

    spec = MODELS[model]
    if holdout:
        excluded = {holdout} if isinstance(holdout, str) else set(holdout)
        features = features[~features["subject"].astype(str).isin(excluded)]
    X, y, groups, sites, names = build_design_matrix(features, normalisation)
    groups = np.asarray([str(g) for g in groups])
    patients = sorted(set(groups.tolist()))
    if len(patients) < 3:
        raise ValueError(f"fitting a model with a conformal threshold needs at least 3 "
                         f"patients; this cohort has {len(patients)}")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(patients))
    n_cal = max(1, int(round(calibration_fraction * len(shuffled))))
    cal_patients = {str(s) for s in shuffled[:n_cal]}
    cal = np.array([g in cal_patients for g in groups])
    fit = ~cal

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    estimator = spec.build()
    estimator.fit(imputer.fit_transform(X[fit]), y[fit])
    raw_cal = estimator.predict_proba(imputer.transform(X[cal]))[:, 1]

    calibrator = None
    calibrated_cal = raw_cal
    if len(set(y[cal].tolist())) == 2:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_cal, y[cal])
        calibrated_cal = np.clip(calibrator.predict(raw_cal), 0.0, 1.0)

    sets = split_conformal(calibrated_cal, y[cal], calibrated_cal, y[cal], alpha=alpha)
    return SozModel(
        model=estimator, imputer=imputer, feature_names=names,
        normalisation=normalisation, calibrator=calibrator,
        conformal_quantile=sets.quantile, alpha=alpha,
        n_train_patients=int(fit.sum() and len(set(groups[fit]))),
        n_train_contacts=int(fit.sum()),
        train_prevalence=float(y[fit].mean()) if fit.any() else float("nan"),
        sites=sorted({s for s in sites[fit].tolist() if s}),
        held_out=sorted({holdout} if isinstance(holdout, str) else set(holdout or [])),
        protocol_note=(f"fitted on {len(patients) - len(cal_patients)} patients, "
                       f"calibrated on {len(cal_patients)} held out at alpha={alpha}"))


# --------------------------------------------------------------------------
# Personalisation: how many labels does a new patient actually need?
# --------------------------------------------------------------------------


def evaluate_personalized(features: pd.DataFrame, n_labels: int = 5,
                          model: str = "logistic", normalisation: str = "rank",
                          target_weight: float = 20.0, n_repeats: int = 5,
                          seed: int = 0) -> EvaluationResult:
    """Leave-one-patient-out, but the clinician first labels ``n_labels`` contacts.

    This is the deployable form of a subject-specific model, and the
    difference from :func:`evaluate` with ``protocol="within_subject"`` is the
    whole point.

    A within-subject fold trains on a random split of the target patient's
    contacts, which presumes you already know which of them are SOZ -- the
    question you are asking. It is a ceiling, not a system. Here the target
    patient contributes only ``n_labels`` contacts, chosen before anything is
    predicted, and the model is scored on **every contact it was not given**.
    Nothing circular happens: a clinician looking at a new implantation can
    genuinely point at a few contacts, and this measures what that buys.

    ``n_labels=0`` is exactly leave-one-patient-out, which makes the two ends
    of the curve comparable by construction rather than by assertion.

    Parameters
    ----------
    target_weight:
        How much each labelled contact from the target patient counts
        relative to a contact from the cohort. It has to be large: five
        contacts against a thousand would otherwise vanish into the average,
        and the experiment would measure nothing. Twenty is not tuned -- it is
        the order of magnitude that makes the target patient's handful weigh
        about as much as one other patient.
    n_repeats:
        Which contacts the clinician happens to label is a lottery, and with
        five of them it is a wide one. Each patient is re-sampled this many
        times and the scores averaged, so the curve reports a typical
        clinician rather than a lucky one.
    """
    from sklearn.impute import SimpleImputer

    spec = MODELS[model]
    X, y, groups, _sites, _names = build_design_matrix(features, normalisation)
    accumulated = np.zeros(len(y))
    counts = np.zeros(len(y))
    folds: list[FoldResult] = []
    rng = np.random.default_rng(seed)

    for target in sorted(set(groups)):
        target_rows = np.flatnonzero(groups == target)
        cohort_rows = np.flatnonzero(groups != target)
        if len(set(y[cohort_rows].tolist())) < 2:
            continue
        for repeat in range(max(1, n_repeats) if n_labels else 1):
            given = _sample_labels(y[target_rows], n_labels, rng)
            held = np.setdiff1d(np.arange(len(target_rows)), given)
            if not len(held):
                continue
            train = np.concatenate([cohort_rows, target_rows[given]]) if len(given) \
                else cohort_rows
            weights = np.concatenate([
                np.ones(len(cohort_rows)),
                np.full(len(given), float(target_weight))]) if len(given) \
                else np.ones(len(cohort_rows))

            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            estimator = spec.build()
            _fit_with_weights(estimator, imputer.fit_transform(X[train]), y[train], weights)
            predicted = estimator.predict_proba(imputer.transform(X[target_rows[held]]))[:, 1]

            accumulated[target_rows[held]] += predicted
            counts[target_rows[held]] += 1
            folds.append(FoldResult(f"{target}#{repeat}", len(train), len(held),
                                    int(y[target_rows[held]].sum()), predicted,
                                    y[target_rows[held]], target_rows[held]))

    scores = np.divide(accumulated, counts, out=np.full(len(y), np.nan), where=counts > 0)
    return EvaluationResult(
        model=f"{model}+{n_labels}labels", normalisation=normalisation,
        protocol=f"personalized_{n_labels}", scores=scores, truth=y, groups=groups,
        folds=folds,
        note=(f"leave-one-patient-out plus {n_labels} contact label(s) from the target "
              f"patient, averaged over {n_repeats} draws; scored only on contacts the "
              f"model was not given" if n_labels else
              "leave-one-patient-out (zero labels from the target patient)"))


def _sample_labels(y_target: np.ndarray, n_labels: int, rng) -> np.ndarray:
    """Pick the contacts a clinician labels: a realistic, not a helpful, sample.

    Stratified so the sample usually contains at least one of each class --
    a clinician pointing at contacts would not hand over five negatives and
    call it a day -- but otherwise uniform. Deliberately *not* chosen by
    feature value: sampling the highest-rate contacts would leak the model's
    own opinion back into its training set and inflate every point on the
    curve.
    """
    if n_labels <= 0:
        return np.array([], dtype=int)
    positives = np.flatnonzero(y_target == 1)
    negatives = np.flatnonzero(y_target == 0)
    if n_labels >= len(y_target):
        return np.arange(len(y_target))
    n_pos = max(1, int(round(n_labels * len(positives) / max(1, len(y_target)))))
    n_pos = min(n_pos, len(positives), n_labels - 1 if len(negatives) else n_labels)
    n_neg = min(n_labels - n_pos, len(negatives))
    chosen = np.concatenate([
        rng.choice(positives, size=n_pos, replace=False) if n_pos > 0 else np.array([], int),
        rng.choice(negatives, size=n_neg, replace=False) if n_neg > 0 else np.array([], int)])
    return chosen.astype(int)


def _fit_with_weights(estimator, X, y, weights) -> None:
    """Fit with sample weights, through a pipeline if necessary."""
    try:
        if hasattr(estimator, "steps"):          # sklearn Pipeline
            final = estimator.steps[-1][0]
            estimator.fit(X, y, **{f"{final}__sample_weight": weights})
        else:
            estimator.fit(X, y, sample_weight=weights)
    except (TypeError, ValueError):
        estimator.fit(X, y)                      # model does not support weights


def label_budget_curve(features: pd.DataFrame, budgets=(0, 2, 5, 10, 20),
                       model: str = "logistic", normalisation: str = "rank",
                       n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Score against the number of contacts the clinician labels first.

    The curve's two ends are the two numbers that already exist: at zero
    labels it *is* leave-one-patient-out, and as the budget approaches the
    whole implantation it approaches the within-subject ceiling. What is new
    is the middle -- the part that says whether the gap between them is
    reachable with an amount of clinician attention anyone would actually
    give.
    """
    contacts_per_patient = float(
        features.groupby("subject").size().median()) if len(features) else float("nan")
    # The right-hand end of the curve. When the within-subject folds are too
    # small to fit, this comes out below the zero-label score and there is no
    # gap to report a fraction of -- see the starvation caveat above.
    ceiling = evaluate(features, model, normalisation, "within_subject").metrics()["auprc"]
    floor = None
    rows = []
    for budget in budgets:
        result = evaluate_personalized(features, n_labels=int(budget), model=model,
                                       normalisation=normalisation,
                                       n_repeats=n_repeats, seed=seed)
        summary = result.summary()
        summary["n_labels"] = int(budget)
        # How much of the implantation the clinician is being asked to label.
        # 40 contacts sounds modest until it is 60% of the electrodes.
        summary["labelled_fraction"] = (round(budget / contacts_per_patient, 3)
                                        if contacts_per_patient else None)
        if floor is None:
            floor = summary["auprc"]
        # Explicitly None rather than absent when there is no gap to close:
        # on a cohort whose within-subject folds are starved the ceiling sits
        # *below* the floor, and a silently missing column reads as a bug.
        summary["gap_closed"] = (
            round((summary["auprc"] - floor) / (ceiling - floor), 3)
            if ceiling and floor is not None and ceiling > floor else None)
        rows.append(summary)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Active learning: does it matter WHICH contacts the clinician labels?
# --------------------------------------------------------------------------

#: How the ``n_labels`` contacts are chosen from the patient in front of you.
#: ``random`` is the control -- the assumption the label-budget curve makes.
ACQUISITION = {
    "random": "a stratified random draw: what the label-budget curve assumes",
    "uncertainty": "the contacts the cohort model is least sure about (p nearest 0.5) -- "
                   "textbook active learning",
    "confident": "the contacts the cohort model ranks highest: what a clinician handed a "
                 "ranking would look at first",
    "rate": "the contacts with the highest ripple rate -- available today, needs no model",
}


def _acquire(strategy: str, pool: np.ndarray, probabilities: np.ndarray,
             rates: np.ndarray, truth: np.ndarray, n_labels: int, rng) -> np.ndarray:
    """Choose which contacts of the target patient get labelled."""
    if n_labels <= 0 or not len(pool):
        return np.array([], dtype=int)
    n_labels = min(int(n_labels), len(pool))
    if strategy == "random":
        return _sample_labels(truth, n_labels, rng)
    if strategy == "uncertainty":
        order = np.argsort(np.abs(probabilities - 0.5))
    elif strategy == "confident":
        order = np.argsort(-probabilities)
    elif strategy == "rate":
        order = np.argsort(-np.nan_to_num(rates, nan=-np.inf))
    else:
        raise ValueError(f"strategy must be one of {', '.join(ACQUISITION)}")
    return order[:n_labels]


def evaluate_active_learning(features: pd.DataFrame, strategy: str = "random",
                             n_labels: int = 5, model: str = "logistic",
                             normalisation: str = "raw", eval_fraction: float = 0.5,
                             target_weight: float = 20.0, n_repeats: int = 3,
                             rate_column: str = "rms_rate_per_min",
                             seed: int = 0) -> EvaluationResult:
    """Leave-one-patient-out plus ``n_labels`` contacts chosen by ``strategy``.

    The confound this is built around
    --------------------------------
    :func:`evaluate_personalized` scores the model on every contact it was not
    given, which is correct when the labelled contacts are drawn at random.
    The moment a *strategy* picks them it stops being correct, because each
    strategy removes different contacts from what remains. Uncertainty
    sampling takes the hard ones and leaves an easier test set; ranking by
    probability takes the obvious positives and leaves a harder one. Comparing
    strategies scored on their own leftovers compares four different exams.

    So each patient's contacts are split once into a **fixed evaluation pool**
    and a labelling pool, from the repeat's seed alone -- never from the
    strategy. Every strategy chooses its labels from the same pool and is
    scored on the same held-out contacts. The split is what makes the
    comparison mean anything, and ``test_every_strategy_is_scored_on_the_same
    _contacts`` pins it.

    The cohort model that ranks the candidates is trained **without** the
    target patient, so choosing what to label never sees that patient's
    answers.
    """
    from sklearn.impute import SimpleImputer

    if strategy not in ACQUISITION:
        raise ValueError(f"strategy must be one of {', '.join(ACQUISITION)}")
    spec = MODELS[model]
    X, y, groups, _sites, _names = build_design_matrix(features, normalisation)
    frame = features[features["is_soz"].notna()].copy()
    frame = frame[frame.groupby("subject")["is_soz"].transform("sum") > 0]
    rates_all = (frame[rate_column].to_numpy(dtype=float)
                 if rate_column in frame.columns else np.full(len(y), np.nan))

    accumulated = np.zeros(len(y))
    counts = np.zeros(len(y))
    folds: list[FoldResult] = []
    cohort_cache: dict[str, tuple] = {}

    for repeat in range(max(1, n_repeats)):
        rng = np.random.default_rng(seed + repeat)
        for target in sorted(set(groups)):
            rows = np.flatnonzero(groups == target)
            others = np.flatnonzero(groups != target)
            if len(set(y[others].tolist())) < 2 or len(rows) < 4:
                continue

            # The candidate-ranking model never sees the target patient.
            if target not in cohort_cache:
                imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                estimator = spec.build()
                estimator.fit(imputer.fit_transform(X[others]), y[others])
                cohort_cache[target] = (estimator, imputer)
            estimator, imputer = cohort_cache[target]

            # One split per (patient, repeat), identical for every strategy.
            split_rng = np.random.default_rng(hash((target, repeat, seed)) % (2**32))
            shuffled = split_rng.permutation(len(rows))
            n_eval = max(2, int(round(eval_fraction * len(rows))))
            eval_local = np.sort(shuffled[:n_eval])
            pool_local = np.sort(shuffled[n_eval:])
            if not len(pool_local) or len(set(y[rows[eval_local]].tolist())) < 2:
                continue

            pool_rows = rows[pool_local]
            probabilities = estimator.predict_proba(imputer.transform(X[pool_rows]))[:, 1]
            chosen = _acquire(strategy, pool_local, probabilities, rates_all[pool_rows],
                              y[pool_rows], n_labels, rng)
            given = pool_rows[chosen] if len(chosen) else np.array([], dtype=int)

            train = np.concatenate([others, given]) if len(given) else others
            weights = np.concatenate([np.ones(len(others)),
                                      np.full(len(given), float(target_weight))]) \
                if len(given) else np.ones(len(others))
            fold_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            fold_model = spec.build()
            _fit_with_weights(fold_model, fold_imputer.fit_transform(X[train]),
                              y[train], weights)
            held = rows[eval_local]
            predicted = fold_model.predict_proba(fold_imputer.transform(X[held]))[:, 1]

            accumulated[held] += predicted
            counts[held] += 1
            folds.append(FoldResult(f"{target}#{repeat}", len(train), len(held),
                                    int(y[held].sum()), predicted, y[held], held))

    scores = np.divide(accumulated, counts, out=np.full(len(y), np.nan), where=counts > 0)
    keep = counts > 0
    return EvaluationResult(
        model=f"{model}+{n_labels}@{strategy}", normalisation=normalisation,
        protocol=f"active_{strategy}_{n_labels}", scores=scores[keep], truth=y[keep],
        groups=groups[keep], folds=folds,
        note=(f"{n_labels} label(s) chosen by '{strategy}' ({ACQUISITION[strategy]}); "
              f"scored on a fixed held-out pool identical across strategies"))


def active_learning_comparison(features: pd.DataFrame, budgets=(2, 5, 10),
                               strategies=tuple(ACQUISITION), model: str = "logistic",
                               normalisation: str = "raw", n_repeats: int = 3,
                               seed: int = 0) -> pd.DataFrame:
    """Score every (strategy, budget) on the same held-out contacts.

    ``lift_over_random`` is the column the experiment exists for: whether
    choosing *which* contacts to label beats asking for any five. A strategy
    that does not beat random is not worth the workflow it would require.
    """
    rows = []
    for budget in budgets:
        for strategy in strategies:
            result = evaluate_active_learning(features, strategy=strategy,
                                              n_labels=int(budget), model=model,
                                              normalisation=normalisation,
                                              n_repeats=n_repeats, seed=seed)
            summary = result.summary()
            summary["strategy"] = strategy
            summary["n_labels"] = int(budget)
            rows.append(summary)
    table = pd.DataFrame(rows)
    baseline = table[table["strategy"] == "random"].set_index("n_labels")["auprc"]
    table["lift_over_random"] = [
        round(row["auprc"] - baseline.get(row["n_labels"], float("nan")), 4)
        if row["auprc"] is not None else None for _, row in table.iterrows()]
    return table.sort_values(["n_labels", "auprc"], ascending=[True, False])
