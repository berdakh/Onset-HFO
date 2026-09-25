"""Measuring a detector against known truth.

Truth comes from two places, and they support different claims.

On the **synthetic** recording (:mod:`onset_hfo.synthetic`) every implanted
event is known by construction, so precision and recall here mean *accuracy*.

On **ds003498** the events were marked by the authors of the original study,
so the same arithmetic means *agreement with a reference detector that a human
validated* -- a weaker claim and a more useful one. :mod:`onset_hfo.benchmark`
runs that comparison across the cohort.

On a recording with neither (the ictal dataset), neither can be computed, and
this package refuses to print numbers that look like them: what is measurable
there is rate, ranking, detector agreement and change over time.

Definitions used here
---------------------
* A detection **matches** a truth event when they overlap in time (within a
  tolerance) and the truth event's contact is one of the contacts the
  detection's channel is built from -- a bipolar pair ``SA2-SA3`` can legally
  claim an event implanted on either ``SA2`` or ``SA3``.
* **Recall** = fraction of truth events matched by at least one detection.
  One truth event detected on two overlapping bipolar pairs still counts once.
* **Precision** = fraction of detections that match at least one truth event.
* False positives are **explained** when they match an implanted spike or
  artifact instead: that tells you *what* the detector is confusing, which is
  more useful than the single number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from onset_hfo.config import DetectorConfig
from onset_hfo.detectors.base import Event
from onset_hfo.preprocess import Prepared

__all__ = ["EvaluationResult", "evaluate_detections", "sweep_threshold", "validation_benefit",
           "marked_contact_check"]


@dataclass
class EvaluationResult:
    """Scores for one detector against one synthetic ground truth."""

    detector: str
    kind: str
    n_truth: int
    n_detections: int
    n_true_positives: int
    n_false_positives: int
    n_missed: int
    precision: float
    recall: float
    f1: float
    false_positive_causes: dict[str, int] = field(default_factory=dict)
    tolerance_s: float = 0.02
    accepted_only: bool = True

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"{self.detector}: recall {self.recall:.2f} "
                f"({self.n_true_positives}/{self.n_truth} {self.kind}s), "
                f"precision {self.precision:.2f} "
                f"({self.n_true_positives}/{self.n_detections} detections), "
                f"F1 {self.f1:.2f}")


def _truth_rows(truth: pd.DataFrame, kind: str | None = None) -> pd.DataFrame:
    df = truth if kind is None else truth[truth["kind"] == kind]
    return df.reset_index(drop=True)


def evaluate_detections(events: list[Event], truth: pd.DataFrame, detector: str | None = None,
                        kind: str = "ripple", tolerance: float = 0.02,
                        accepted_only: bool = True,
                        channels: list[str] | None = None) -> EvaluationResult:
    """Score one detector's events against a ground truth.

    Parameters
    ----------
    channels:
        Restrict scoring to these channels. **Essential for expert-annotated
        real data**, where only some channels were reviewed: a detection on an
        unreviewed channel is not a false positive, it is unjudged, and
        counting it as wrong would understate precision for no reason other
        than how much of the recording somebody had time to read.
        ``None`` scores everything, which is right for synthetic data where
        every channel's truth is known.
    """
    dets = [e for e in events
            if (detector is None or e.detector == detector) and (e.accepted or not accepted_only)]
    if channels is not None:
        allowed = set(channels)
        dets = [e for e in dets if e.channel in allowed]
        if "channel" in truth.columns:
            truth = truth[truth["channel"].isin(allowed)]
    target = _truth_rows(truth, kind)
    others = {k: _truth_rows(truth, k) for k in truth["kind"].unique() if k != kind}

    matched_truth: set[int] = set()
    fp_causes: dict[str, int] = {}
    n_tp = 0
    for det in dets:
        contacts = set(det.contacts or [det.channel])
        hits = _overlapping(target, det, contacts, tolerance)
        if hits:
            n_tp += 1
            matched_truth.update(hits)
            continue
        cause = "background"
        for other_kind, frame in others.items():
            if _overlapping(frame, det, contacts, tolerance):
                cause = other_kind
                break
        fp_causes[cause] = fp_causes.get(cause, 0) + 1

    n_det = len(dets)
    n_fp = n_det - n_tp
    n_truth = len(target)
    recall = len(matched_truth) / n_truth if n_truth else float("nan")
    precision = n_tp / n_det if n_det else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and np.isfinite(precision) and np.isfinite(recall) and
          (precision + recall) > 0 else 0.0)
    return EvaluationResult(
        detector=detector or "all", kind=kind, n_truth=n_truth, n_detections=n_det,
        n_true_positives=len(matched_truth), n_false_positives=n_fp,
        n_missed=n_truth - len(matched_truth), precision=float(precision), recall=float(recall),
        f1=float(f1), false_positive_causes=dict(sorted(fp_causes.items(), key=lambda kv: -kv[1])),
        tolerance_s=tolerance, accepted_only=accepted_only)


def _overlapping(frame: pd.DataFrame, det: Event, contacts: set[str], tol: float) -> list[int]:
    """Rows of ``frame`` that overlap this detection in time, on its channel.

    Two ways to decide "same channel", because the two ground truths express
    it differently. Expert annotations name a bipolar channel outright
    (``HL2-HL3``), so they are matched by channel name -- exactly, since
    ``HL1-HL2`` and ``HL2-HL3`` share a contact but are different channels and
    one must not claim the other's events. Implanted synthetic events are
    placed on a single *contact*, which legitimately appears in the two
    bipolar pairs built from it, so those are matched by contact membership.
    """
    if frame.empty:
        return []
    same_channel = (frame["channel"] == det.channel) if "channel" in frame.columns \
        else frame["contact"].isin(contacts)
    mask = same_channel & (frame["start"] - tol < det.stop) & (det.start - tol < frame["stop"])
    return list(np.flatnonzero(mask.to_numpy()))


def sweep_threshold(prep: Prepared, truth: pd.DataFrame, detector_fn, values: list[float],
                    cfg: DetectorConfig | None = None, kind: str = "ripple",
                    validate: bool = True, name: str = "detector") -> pd.DataFrame:
    """Precision/recall as the detection threshold moves.

    This is the honest way to describe a threshold detector: not "it gets
    F1 = 0.8" but "here is the curve, and here is the operating point we
    chose and why". Used by ``notebooks/03_validation_and_benchmark.ipynb``.
    """
    from onset_hfo.validate import validate_events

    base = cfg or DetectorConfig()
    rows = []
    for value in values:
        trial = DetectorConfig(**{**base.as_dict(), "threshold_sd": float(value)})
        events = detector_fn(prep, trial)
        if validate:
            validate_events(events, prep)
        res = evaluate_detections(events, truth, kind=kind, accepted_only=validate)
        row = res.as_dict()
        row.update({"threshold_sd": float(value), "detector": name})
        rows.append(row)
    return pd.DataFrame(rows)


def validation_benefit(events: list[Event], truth: pd.DataFrame, kind: str = "ripple",
                       detector: str | None = None) -> pd.DataFrame:
    """What artifact rejection bought: scores with and without it.

    If the "with validation" row does not show higher precision, the
    validation stage is not earning its place and should be re-tuned or
    removed -- which is exactly the kind of claim a prototype should be able
    to check rather than assert.
    """
    rows = []
    for accepted_only in (False, True):
        res = evaluate_detections(events, truth, detector=detector, kind=kind,
                                  accepted_only=accepted_only)
        row = res.as_dict()
        row["stage"] = "after artifact rejection" if accepted_only else "raw detector output"
        rows.append(row)
    return pd.DataFrame(rows)[["stage", "n_detections", "n_true_positives", "n_false_positives",
                               "precision", "recall", "f1", "false_positive_causes"]]


# --------------------------------------------------------------------------
# A weak check that can be run on real data
# --------------------------------------------------------------------------


def marked_contact_check(rates: pd.DataFrame, marked_contacts: list[str], k: int = 5,
                         n_permutations: int = 2000, seed: int = 0) -> dict:
    """Do the top-ranked channels touch the contacts the clinician named?

    The public recording has no HFO labels, but its events file contains free
    text the reviewer typed during the seizure ("AD1-4, ATT1,2"). Those are
    the contacts a human was looking at. This function asks one narrow
    question: **is the overlap between our top-k channels and those contacts
    larger than chance?** -- with the chance level estimated by permuting the
    ranking rather than assumed.

    What it is not: a measure of accuracy. The markers are not a curated
    seizure-onset zone, they are incomplete, and they name contacts for
    reasons this pipeline does not model. A high overlap is mild encouragement;
    a low one means very little. It is included because a prototype that only
    ever reports numbers with no external referent is easy to fool yourself
    with.
    """
    marked = {c.upper() for c in marked_contacts}
    if not marked or not len(rates):
        return {"available": False,
                "reason": "the recording has no clinician-named contacts"}
    order = rates.sort_values("rate_per_min", ascending=False)["channel"].tolist()

    def touches(channel: str) -> bool:
        return bool({part.upper() for part in str(channel).split("-")} & marked)

    observed = sum(touches(ch) for ch in order[:k])
    rng = np.random.default_rng(seed)
    permuted = np.array([
        sum(touches(ch) for ch in rng.permutation(order)[:k]) for _ in range(n_permutations)])
    p_value = float((np.sum(permuted >= observed) + 1) / (n_permutations + 1))
    return {
        "available": True,
        "k": int(k),
        "marked_contacts": sorted(marked),
        "top_channels": order[:k],
        "n_top_touching_marked": int(observed),
        "expected_by_chance": float(permuted.mean()),
        "permutation_p_value": p_value,
        "interpretation": ("the clinician's free-text markers are not a curated seizure-onset "
                           "zone; treat agreement as mild encouragement and disagreement as "
                           "uninformative"),
    }
