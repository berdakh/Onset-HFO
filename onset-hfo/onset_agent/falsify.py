"""Falsification: try hard to make the system produce a confident wrong answer.

Every number elsewhere in this repository is a measurement of how well
something works. These are measurements of whether it can be *broken*, and
both thesis proposals say to run them early for the same reason:

    Give the agent a patient with channel labels randomly shuffled, a
    recording with no epileptiform activity, and a patient with the true
    HFO-dominant channel removed. If it still produces a confident answer,
    you have found the result reviewers will look hardest for. Better that
    you find it in November than that a reviewer finds it in March.

Each test below states its expectation **before** it runs, and reports a
verdict against that expectation rather than a number to be interpreted
afterwards. A test that can be passed by any outcome is not a test.

The four
--------

**1. Anonymised channel names.** Rename every channel to ``CH01``, ``CH02``,
and re-run. The ranking must not move. A rate is computed from samples, so for
the pipeline this is trivially true -- but a *language model* planner may know
that ``AD`` is an amygdala depth electrode and that mesial temporal contacts
are where seizures often start. If its ranking changes when the names do, it
is partly reciting priors rather than reading the evidence, and that is
invisible in every other measurement here.

**2. Shuffled name-to-signal mapping.** Permute which name sits on which
signal. The SOZ score must collapse to chance, because the labels attach to
names and the names now point at the wrong signals. A score that survives is
not coming from the correspondence it claims to measure.

**3. The leading channel removed.** Drop the top-ranked channel and re-run.
The reported leading rate must *fall*. A system that promotes the runner-up
and reports it with the same confidence is describing a ranking, not a
finding.

**4. No pathology.** A recording simulated with no epileptic contacts at all.
The system must not name a confident leader. This is the one most likely to
fail, because every ranking function will happily sort noise.

What a failure means
--------------------
Not that the code is wrong. A detector that ranks channels in a recording with
nothing in it is behaving exactly as specified -- the specification is the
problem. The value of running these is that the specification's gap becomes a
number in a table instead of a question at a defence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from onset_agent.analysis import AnalysisSession
from onset_agent.planner import Rung, rank_channels, run_rung

__all__ = ["FalsificationResult", "anonymise_channels", "shuffle_channel_mapping",
           "drop_channel", "falsify_anonymised_names", "falsify_shuffled_mapping",
           "falsify_leading_channel_removed", "falsify_no_pathology",
           "falsify_run_to_run", "run_falsification_suite"]


@dataclass
class FalsificationResult:
    """One attempt to break the system, with its verdict against a stated rule."""

    name: str
    expectation: str
    passed: bool | None
    measure: dict = field(default_factory=dict)
    reading: str = ""

    @property
    def verdict(self) -> str:
        return {True: "PASS", False: "FAIL", None: "INCONCLUSIVE"}[self.passed]

    def as_dict(self) -> dict:
        return {"test": self.name, "verdict": self.verdict, "expectation": self.expectation,
                "measure": self.measure, "reading": self.reading}

    def __str__(self) -> str:
        return f"[{self.verdict}] {self.name}: {self.reading}"


# --------------------------------------------------------------------------
# Ways to damage a recording
# --------------------------------------------------------------------------


def _letters(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA. An electrode id with no digits in it."""
    out = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        out = chr(ord("A") + remainder) + out
    return out


def _copy_recording(recording):
    from dataclasses import replace as dc_replace

    return dc_replace(recording, raw=recording.raw.copy())


def anonymise_channels(recording) -> tuple[object, dict[str, str]]:
    """Strip the meaning out of channel names, keeping the electrode structure.

    ``AD1, AD2, ATT1`` becomes ``EA1, EA2, EB1``: the electrode grouping and
    the contact numbering survive, the clinical meaning does not.

    The names must still be LETTERS followed by a NUMBER, because that is the
    convention ``preprocess.bipolar_pairs`` uses to decide what is adjacent to
    what. A first attempt used ``E01C1``, which the contact pattern does not
    match at all, so nothing paired and 71 bipolar channels became 85
    unpaired ones -- a test failure that looked exactly like the system
    reading channel names.

    Preserving the structure is the whole difficulty. A first version renamed
    everything to ``CH01..CH98`` and the test failed on real data -- not
    because the system was reading names, but because the bipolar montage
    pairs consecutive contacts *on the same electrode*, so fusing 14 electrodes
    into one called ``CH`` changed 71 analysed channels into 79 and compared
    two different analyses. The signal must be identical for this test to mean
    anything, and that means the naming scheme has to carry the same
    information the montage uses.
    """
    import re as _re

    out = _copy_recording(recording)
    pattern = _re.compile(r"^([A-Za-z]+[A-Za-z']*?)(\d{1,3})$")
    electrodes: dict[str, str] = {}
    mapping = {}
    for name in out.raw.ch_names:
        match = pattern.match(name)
        if not match:
            mapping[name] = f"Z{len(mapping) + 1}"
            continue
        prefix, number = match.group(1), match.group(2)
        if prefix not in electrodes:
            electrodes[prefix] = "E" + _letters(len(electrodes))
        mapping[name] = f"{electrodes[prefix]}{int(number)}"
    out.raw.rename_channels(mapping)
    out.bads = [mapping.get(b, b) for b in (out.bads or [])]
    out.raw.info["bads"] = [b for b in out.bads if b in out.raw.ch_names]
    if out.channels is not None and "name" in out.channels.columns:
        out.channels = out.channels.copy()
        out.channels["name"] = out.channels["name"].map(lambda n: mapping.get(str(n), n))
    out.marked_contacts = [mapping.get(c, c) for c in (out.marked_contacts or [])]
    return out, mapping


def shuffle_channel_mapping(recording, seed: int = 0) -> tuple[object, dict[str, str]]:
    """Permute which name sits on which signal. Returns the permutation.

    Names are reused, not invented, so the montage still pairs things that
    look like neighbouring contacts -- the spatial structure of the *analysis*
    survives and only the correspondence between signal and label is
    destroyed. That is the narrow thing this test is for.
    """
    out = _copy_recording(recording)
    names = list(out.raw.ch_names)
    permuted = list(np.random.default_rng(seed).permutation(names))
    mapping = dict(zip(names, permuted, strict=False))
    # Rename via placeholders: MNE refuses a rename whose target already exists.
    out.raw.rename_channels({n: f"__tmp{i}__" for i, n in enumerate(names)})
    out.raw.rename_channels({f"__tmp{i}__": mapping[n] for i, n in enumerate(names)})
    out.raw.reorder_channels(names)
    out.bads = [mapping.get(b, b) for b in (out.bads or [])]
    out.raw.info["bads"] = [b for b in out.bads if b in out.raw.ch_names]
    return out, mapping


def drop_channel(recording, channel: str):
    """Remove one channel (or the contacts behind one bipolar pair)."""
    out = _copy_recording(recording)
    contacts = [c for c in str(channel).split("-") if c in out.raw.ch_names]
    targets = contacts or ([channel] if channel in out.raw.ch_names else [])
    if not targets:
        raise ValueError(f"{channel!r} is not in this recording")
    out.raw.drop_channels(targets)
    out.bads = [b for b in (out.bads or []) if b in out.raw.ch_names]
    out.raw.info["bads"] = list(out.bads)
    return out


# --------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------


def _ranking(recording, rung: Rung, backend=None, model=None, **kwargs) -> list[str]:
    session = AnalysisSession(recording, soz_model=model)
    result = run_rung(session, rung, backend=backend, **kwargs)
    return [r.channel for r in rank_channels(result.store)], result


def falsify_anonymised_names(recording, rung: Rung = Rung.S2, backend=None,
                             model=None, k: int = 5, **kwargs) -> FalsificationResult:
    """Renaming channels must not move the ranking."""
    expectation = ("the top-k channels must be the same signals before and after "
                   "renaming: a rate is computed from samples, not from a name")
    original, _ = _ranking(recording, rung, backend, model, **kwargs)
    anonymised_recording, mapping = anonymise_channels(recording)
    renamed, _ = _ranking(anonymised_recording, rung, backend, model, **kwargs)

    # Map the anonymised bipolar names back through the contact-level mapping.
    back = {v: k_ for k_, v in mapping.items()}
    recovered = ["-".join(back.get(p, p) for p in ch.split("-")) for ch in renamed]
    overlap = len(set(original[:k]) & set(recovered[:k]))
    passed = overlap == min(k, len(original), len(recovered))
    return FalsificationResult(
        name="anonymised channel names", expectation=expectation, passed=passed,
        measure={"top_k": k, "overlap": overlap,
                 "ranking_before": original[:k], "ranking_after_recovered": recovered[:k]},
        reading=("the ranking is driven by the signal" if passed else
                 "the ranking changed when only the NAMES changed: something in the loop "
                 "is using what channels are called, not what they contain"))


def falsify_shuffled_mapping(recording, labels, rung: Rung = Rung.S2, backend=None,
                             model=None, k: int = 5, seed: int = 0,
                             **kwargs) -> FalsificationResult:
    """Breaking the name-to-signal correspondence must collapse the SOZ score."""
    from onset_agent.scoring import score_ranking

    expectation = ("with names permuted across signals, the SOZ score must fall to "
                   "chance: the labels attach to names, which now point at the wrong "
                   "signals")
    if not labels.usable:
        return FalsificationResult("shuffled name-to-signal mapping", expectation, None,
                                   reading="no SOZ labels for this subject; nothing to score")
    intact, _ = _ranking(recording, rung, backend, model, **kwargs)
    shuffled_recording, _ = shuffle_channel_mapping(recording, seed=seed)
    shuffled, _ = _ranking(shuffled_recording, rung, backend, model, **kwargs)

    before = score_ranking(intact, labels, ks=(k,), n_permutations=2000).at_k.get(k, {})
    after = score_ranking(shuffled, labels, ks=(k,), n_permutations=2000).at_k.get(k, {})
    hits_after = after.get("n_hits", 0)
    chance_after = after.get("expected_by_chance", 0.0)
    # "Collapsed" means not distinguishable from the permutation null.
    passed = after.get("permutation_p", 1.0) > 0.05
    return FalsificationResult(
        name="shuffled name-to-signal mapping", expectation=expectation, passed=passed,
        measure={"hits_at_k_intact": before.get("n_hits"),
                 "p_intact": before.get("permutation_p"),
                 "hits_at_k_shuffled": hits_after, "expected_by_chance": chance_after,
                 "p_shuffled": after.get("permutation_p")},
        reading=("the SOZ score collapsed to chance when the correspondence was broken, "
                 "which is what it should do" if passed else
                 "the SOZ score survived having the signal-to-name correspondence "
                 "destroyed, so it is not measuring that correspondence"))


def falsify_leading_channel_removed(recording, rung: Rung = Rung.S2, backend=None,
                                    model=None, tolerance: float = 1.0,
                                    **kwargs) -> FalsificationResult:
    """Removing the leader must not change what the other channels measure.

    The obvious version of this test -- "the new leading rate must be lower" --
    is wrong, and the real recording is what showed it. A bipolar channel
    cannot be removed in isolation: dropping the contacts behind
    ``PST2-PST3`` also destroys ``PST1-PST2`` and ``PST3-PST4``, because those
    pairs share a contact with it. So "the same recording minus one channel"
    does not exist, and comparing leaders across the two is comparing two
    different montages.

    What *can* be asked, and is diagnostic, is whether the channels that
    survive unchanged still measure the same thing. Each channel's rate is
    computed from its own samples against its own baseline, so removing an
    unrelated channel must not move it. If it does, something is leaking
    between channels -- a shared threshold, a global normalisation -- and
    every rate in every report is contingent on which other channels happened
    to be included.
    """
    expectation = ("channels untouched by the removal must report exactly the same rate: "
                   "a rate is computed per channel, so removing a different channel "
                   "cannot move it. Leaders cannot be compared across the two runs, "
                   "because removing a bipolar channel destroys its neighbours too")
    _, result = _ranking(recording, rung, backend, model, **kwargs)
    ranking = rank_channels(result.store)
    if len(ranking) < 2:
        return FalsificationResult("leading channel removed", expectation, None,
                                   reading="fewer than two channels were ranked")
    leader = ranking[0]
    reduced = drop_channel(recording, leader.channel)
    _, after_result = _ranking(reduced, rung, backend, model, **kwargs)
    after = {r.channel: r for r in rank_channels(after_result.store)}
    if not after:
        return FalsificationResult("leading channel removed", expectation, None,
                                   reading="no channels ranked after removal")

    survivors = [r for r in ranking if r.channel in after]
    if not survivors:
        return FalsificationResult("leading channel removed", expectation, None,
                                   reading="no channel survived the removal unchanged")
    drifts = {r.channel: abs(after[r.channel].survey_rate_per_min - r.survey_rate_per_min)
              for r in survivors}
    worst_channel = max(drifts, key=drifts.get)
    worst = drifts[worst_channel]
    passed = worst <= tolerance
    new_leader = max(after.values(), key=lambda r: r.survey_rate_per_min)
    return FalsificationResult(
        name="leading channel removed", expectation=expectation, passed=passed,
        measure={"removed": leader.channel,
                 "removed_rate_per_min": round(leader.survey_rate_per_min, 2),
                 "n_channels_before": len(ranking), "n_channels_after": len(after),
                 "n_surviving": len(survivors),
                 "worst_rate_drift_per_min": round(worst, 3),
                 "worst_drifting_channel": worst_channel,
                 "tolerance_per_min": tolerance,
                 "new_leader": new_leader.channel,
                 "new_leader_rate_per_min": round(new_leader.survey_rate_per_min, 2)},
        reading=(f"{len(survivors)} surviving channels measured the same rate to within "
                 f"{worst:.2f}/min, so rates are per-channel and not contingent on the "
                 f"rest of the montage" if passed else
                 f"{worst_channel} moved by {worst:.2f}/min when an unrelated channel was "
                 f"removed: something is leaking between channels"))


def falsify_no_pathology(rung: Rung = Rung.S2, backend=None, model=None,
                         seed: int = 3, duration_s: float = 30.0,
                         **kwargs) -> FalsificationResult:
    """A recording with nothing in it must not produce a confident leader.

    The test is deliberately *not* "is the top rate below some number". Any
    threshold I picked would be one I tuned until this passed, and a recording
    with a noisier baseline would sail over it. What is checked instead is
    whether the system **says so**: ``leader_stands_out`` is false when the
    leader's confidence interval overlaps the median channel's, which is a
    statement about intervals and tightens by itself as the window grows.

    This test failed when it was first written. The pipeline ranked a channel
    at 6/min on a simulation containing no epileptic contacts at all and
    nothing in its output said the recording was empty. The fix was not a
    threshold but the missing null hypothesis --
    :func:`onset_hfo.metrics.leader_separation` -- now surfaced in the
    ``detect_hfo`` tool and checked here.
    """
    from onset_hfo.synthetic import make_synthetic_recording

    expectation = ("on a simulation with no epileptic contacts, the system must report "
                   "that no channel stands out, rather than ranking the noise")
    recording = make_synthetic_recording(seed=seed, duration_s=duration_s,
                                         hot_leads=0, verbose=False)
    _, result = _ranking(recording, rung, backend, model, **kwargs)
    ranking = rank_channels(result.store)
    surveys = [r for r in result.store.runs("detect_hfo", ok_only=True)
               if (r.output or {}).get("leader_separation", {}).get("available")]
    if not surveys:
        return FalsificationResult("no pathology", expectation, None,
                                   reading="no survey run to inspect")
    separation = surveys[0].output["leader_separation"]
    passed = separation["distinguishable"] is False
    return FalsificationResult(
        name="no pathology", expectation=expectation, passed=passed,
        measure={"leader": separation["leader"],
                 "leader_rate_per_min": separation["leader_rate_per_min"],
                 "leader_ci": separation["leader_ci"],
                 "median_rate_per_min": separation["median_rate_per_min"],
                 "n_tied_with_leader": separation["n_tied_with_leader"],
                 "n_channels": separation["n_channels"],
                 "leader_stands_out": separation["distinguishable"],
                 "n_ranked": len(ranking)},
        reading=("the system reported that no channel stands out, which is the right "
                 "answer for a recording with nothing in it" if passed else
                 f"the system named {separation['leader']} as a leader on a recording "
                 f"containing no epileptic activity: the ranking sorts noise and nothing "
                 f"in the output says so"))


def falsify_run_to_run(recording, rung: Rung = Rung.S2, backend=None, model=None,
                       repeats: int = 5, k: int = 5, **kwargs) -> FalsificationResult:
    """The same question, asked five times, must get the same answer."""
    expectation = (f"the top-{k} channels must be identical across {repeats} repeated "
                   "runs: a surgeon who asked twice and got two answers would be right "
                   "to stop asking")
    tops = []
    for _ in range(repeats):
        ranking, _ = _ranking(recording, rung, backend, model, **kwargs)
        tops.append(tuple(ranking[:k]))
    identical = len(set(tops)) == 1
    overlaps = [len(set(tops[0]) & set(t)) / max(1, len(tops[0])) for t in tops[1:]]
    return FalsificationResult(
        name="run-to-run stability", expectation=expectation, passed=identical,
        measure={"repeats": repeats, "identical": identical,
                 "mean_top_k_overlap": round(float(np.mean(overlaps)), 3) if overlaps else 1.0,
                 "distinct_answers": len(set(tops))},
        reading=("every run gave the same top-k" if identical else
                 f"{len(set(tops))} different answers across {repeats} runs"))


def run_falsification_suite(recording, labels=None, rung: Rung = Rung.S2, backend=None,
                            model=None, repeats: int = 3, seed: int = 0,
                            **kwargs) -> list[FalsificationResult]:
    """Run every test. Order is cheapest-to-most-expensive, not most-to-least important."""
    results = [
        falsify_no_pathology(rung, backend, model, **kwargs),
        falsify_anonymised_names(recording, rung, backend, model, **kwargs),
        falsify_leading_channel_removed(recording, rung, backend, model, **kwargs),
        falsify_run_to_run(recording, rung, backend, model, repeats=repeats, **kwargs),
    ]
    if labels is not None:
        results.append(falsify_shuffled_mapping(recording, labels, rung, backend, model,
                                                seed=seed, **kwargs))
    return results
