"""Scoring a rung's channel ranking against the clinician's SOZ labels.

This is the Tier 1 metric of both thesis proposals -- *localization against
the record that already exists* -- and it costs no clinician time. The labels
come from :mod:`onset_hfo.cohort`; the ranking comes from
:func:`onset_agent.planner.rank_channels`; this module only compares them.

What is being measured, stated precisely so nobody overclaims from it
---------------------------------------------------------------------
The question is **not** "did the system find the seizure onset zone". It is:

    Do the channels this configuration ranks highest overlap the contacts a
    clinician named as seizure onset, more than a random ranking of the same
    channels would?

The chance level is estimated by permuting the ranking rather than assumed,
because the SOZ contacts are a large and unevenly distributed fraction of the
implanted contacts -- on some patients here, a third of all channels touch a
named contact, and "3 of the top 5" would then be unremarkable.

Two properties of the labels limit what any number here can mean, and both
are carried into the output so they cannot be lost downstream:

* a contact is a trustworthy positive only when the patient became seizure
  free, so ``labels.trustworthy`` is reported alongside every score;
* these are **ictal** recordings. Ripple energy during a seizure spreads far
  beyond the onset region, so a null result is the expected result and is not
  evidence that the detector is broken. ``docs/EVALUATION.md`` says the same
  thing about the original ranking, and it remains true here.

Stand-in labels
---------------
:func:`onset_hfo.cohort.placeholder_labels` and
:func:`~onset_hfo.cohort.labels_from_ground_truth` exist so this machinery can
run before a clinical centre has sent anything. Scores computed against them
are meaningless as results -- they demonstrate that the scorer works. Every
:class:`RankingScore` therefore carries ``label_source``, ``trustworthy`` and
``is_placeholder``, and a ``warning`` string that travels into every JSON file
the score is written to. A stand-in that reaches a results table unmarked
makes the table worthless in a way nobody can detect afterwards, so the flag
is carried rather than checked once at the top.

Nothing in this module, and nothing built on it, identifies a seizure onset
zone for a patient. It scores a ranking against a record, retrospectively, on
data whose outcome is already known.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from onset_hfo.cohort import SozLabels

__all__ = ["RankingScore", "score_ranking", "score_result", "compare_rungs"]

DEFAULT_KS = (1, 3, 5, 10)


@dataclass
class RankingScore:
    """Overlap between a ranking and the labelled contacts, with a null."""

    subject: str
    label_source: str
    trustworthy: bool
    n_ranked: int
    n_soz_channels: int
    prevalence: float
    at_k: dict[int, dict] = field(default_factory=dict)
    note: str = ""
    #: True when the labels were generated rather than observed. Carried into
    #: every serialised score: a placeholder that reaches a results table
    #: unmarked makes the table worthless in a way nobody can detect later.
    is_placeholder: bool = False
    warning: str = ""

    def as_dict(self) -> dict:
        payload = {"subject": self.subject, "label_source": self.label_source,
                   "trustworthy_label": self.trustworthy,
                   "is_placeholder": self.is_placeholder,
                   "n_ranked_channels": self.n_ranked,
                   "n_soz_channels": self.n_soz_channels,
                   "soz_prevalence": round(self.prevalence, 4),
                   "at_k": {str(k): v for k, v in self.at_k.items()}, "note": self.note}
        if self.warning:
            payload["warning"] = self.warning
        return payload


def score_ranking(channels: list[str], labels: SozLabels, ks=DEFAULT_KS,
                  n_permutations: int = 2000, seed: int = 0) -> RankingScore:
    """Score one ordered channel list against one patient's SOZ labels.

    Parameters
    ----------
    channels:
        Channel names, best first. A bipolar channel counts as SOZ when
        *either* of its contacts is labelled -- the same permissive rule
        :func:`onset_hfo.cohort.label_channels` uses, stated there.
    n_permutations:
        Size of the permutation null. The p-value is the fraction of random
        rankings of *these same channels* whose top-k overlap is at least as
        large as the observed one.
    """
    soz = labels.soz_contacts
    score = RankingScore(subject=labels.subject, label_source=labels.source,
                         trustworthy=labels.trustworthy, n_ranked=len(channels),
                         n_soz_channels=0, prevalence=0.0,
                         is_placeholder=labels.is_placeholder, warning=labels.warning)
    if not soz or not channels:
        score.note = ("no labelled contacts for this subject, or no channels were ranked; "
                      "nothing can be scored")
        return score

    def touches(channel: str) -> bool:
        return bool({p.upper() for p in str(channel).split("-")} & soz)

    hits = np.array([touches(ch) for ch in channels])
    score.n_soz_channels = int(hits.sum())
    score.prevalence = float(hits.mean())
    if score.n_soz_channels == 0:
        score.note = ("none of the analysed channels touches a labelled contact -- usually "
                      "the labelled electrode was dropped as bad, or the montage renamed it")
        return score

    rng = np.random.default_rng(seed)
    order = np.arange(len(channels))
    permutations = np.array([rng.permutation(order) for _ in range(n_permutations)])
    for k in ks:
        if k > len(channels):
            continue
        observed = int(hits[:k].sum())
        null = hits[permutations[:, :k]].sum(axis=1)
        score.at_k[int(k)] = {
            "n_hits": observed,
            "precision_at_k": round(observed / k, 4),
            "recall_at_k": round(observed / score.n_soz_channels, 4),
            "expected_by_chance": round(float(null.mean()), 3),
            "permutation_p": round(float((np.sum(null >= observed) + 1) /
                                         (n_permutations + 1)), 4),
        }
    score.note = ("permutation null over the same channel set; a p-value near 1 means the "
                  "ranking is no better than chance, which on ictal data is the expected "
                  "result and not a detector failure")
    return score


def score_result(result, labels: SozLabels, ks=DEFAULT_KS, **kwargs) -> RankingScore:
    """Score a :class:`~onset_agent.planner.PlannerResult`."""
    return score_ranking(result.top_channels, labels, ks=ks, **kwargs)


def compare_rungs(results: dict, labels: SozLabels, k: int = 5) -> list[dict]:
    """One row per rung: what it cost, what it challenged, and what it scored.

    This is the shape of the ablation table. It deliberately reports cost and
    the number of channels re-tested beside the score, because a rung that
    scores the same for three times the tool calls has lost, and a rung that
    never re-tested anything cannot have gained from re-planning whatever its
    score says.
    """
    rows = []
    for name, result in results.items():
        score = score_result(result, labels)
        at_k = score.at_k.get(k, {})
        cost = result.cost or {}
        rows.append({
            "rung": name,
            "label_source": score.label_source,
            "is_placeholder": score.is_placeholder,
            "n_tool_calls": cost.get("n_tool_calls", 0),
            "runtime_s": cost.get("runtime_s", 0.0),
            "n_channels_retested": result.n_retested,
            f"n_hits_at_{k}": at_k.get("n_hits"),
            f"precision_at_{k}": at_k.get("precision_at_k"),
            f"expected_by_chance_at_{k}": at_k.get("expected_by_chance"),
            f"permutation_p_at_{k}": at_k.get("permutation_p"),
            "unsupported_claim_rate": (result.verification or {}).get(
                "unsupported_claim_rate"),
            "top_channels": result.top_k(k),
            "stop_reason": result.stop_reason,
        })
    return rows
