"""The orchestration half: labels, contract, evidence store, live tools,
the S0-S3 ladder, the two verifiers, and SOZ scoring.

Everything here is offline. No download, no model weights, no GPU: the
language-model paths are exercised against small fake backends that reply the
way a served Qwen or Llama does, including the ways they get it wrong.
"""

from __future__ import annotations

import json

import pytest

from onset_agent.analysis import AnalysisSession, build_registry
from onset_agent.backends import AssistantMessage, Backend
from onset_agent.contract import ToolRun
from onset_agent.evidence import EvidenceStore
from onset_agent.planner import (
    FixedBudget,
    ModelJudged,
    Rung,
    ScriptedPlanner,
    TiedSetWidth,
    rank_channels,
    run_rung,
)
from onset_agent.scoring import score_ranking
from onset_agent.verifier import (
    DeterministicVerifier,
    LLMVerifier,
    claim_numbers,
    compare_verifiers,
    split_sentences,
)
from onset_hfo.cohort import (
    SozLabels,
    decode_outcome,
    expand_contacts,
    label_channels,
    labels_from_ground_truth,
    load_labels_csv,
    normalize_subject,
    placeholder_labels,
    soz_labels,
    write_label_template,
)

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def test_expand_contacts_handles_the_clinicians_shorthand():
    assert expand_contacts("TT1-3; AST1, mst2") == {"TT1", "TT2", "TT3", "AST1", "MST2"}
    assert expand_contacts("PST1-4: AST1-2, MST1-2") == {
        "PST1", "PST2", "PST3", "PST4", "AST1", "AST2", "MST1", "MST2"}
    # A trailing newline and an empty token must not produce a phantom contact.
    assert expand_contacts("G4, G5,\n") == {"G4", "G5"}
    assert expand_contacts(None) == set()
    assert expand_contacts(float("nan")) == set()


def test_expand_contacts_refuses_an_absurd_range():
    """A range wider than an electrode is a parsing accident, not 900 contacts."""
    assert expand_contacts("A1-9999") == set()


def test_normalize_subject_reconciles_archive_and_spreadsheet_ids():
    # The S3 tree says sub-pt01; the spreadsheet says pt1. Both must agree.
    assert normalize_subject("sub-pt01") == normalize_subject("pt1") == "pt1"
    assert normalize_subject("sub-umf001") == normalize_subject("umf001") == "umf1"
    assert normalize_subject("sub-ummc002") == "ummc2"
    assert normalize_subject("jh101") == "jh101"


def test_outcome_code_s_means_seizure_free():
    """The trap: 'F' is failure, not free. Inverting this inverts every label."""
    assert decode_outcome("S") is True
    assert decode_outcome("F") is False
    assert decode_outcome("NR") is None
    assert decode_outcome(None) is None


def test_a_bipolar_channel_is_soz_when_either_contact_is():
    labels = SozLabels(subject="x", soz_contacts={"AD1", "ATT1"})
    frame = label_channels(["AD1-AD2", "AD3-AD4", "ATT1-ATT2"], labels)
    assert frame.set_index("channel")["is_soz"].to_dict() == {
        "AD1-AD2": True, "AD3-AD4": False, "ATT1-ATT2": True}


def test_trustworthy_requires_a_curated_label_and_a_working_surgery():
    curated = SozLabels("x", {"A1"}, source="clinical_summary", seizure_free=True)
    failed = SozLabels("x", {"A1"}, source="clinical_summary", seizure_free=False)
    weak = SozLabels("x", {"A1"}, source="events_markers", seizure_free=True)
    assert curated.trustworthy and not failed.trustworthy and not weak.trustworthy


def test_local_label_csv_is_a_drop_in_replacement(tmp_path):
    path = tmp_path / "soz.csv"
    path.write_text("subject,soz_contacts,engel,seizure_free,site\n"
                    "anon-01,\"LA1-3; LH2\",1,True,our-centre\n")
    labels = load_labels_csv(path)["anon-01"]
    assert labels.soz_contacts == {"LA1", "LA2", "LA3", "LH2"}
    assert labels.source == "local" and labels.engel == 1


def test_a_label_csv_without_the_required_columns_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("subject,contacts\nanon-01,LA1\n")
    with pytest.raises(ValueError, match="soz_contacts"):
        load_labels_csv(path)


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


def test_unknown_tools_and_arguments_are_refused_before_anything_runs(registry):
    assert not registry.run("no_such_tool").ok
    run = registry.run("detect_hfo", {"nonsense": 1})
    assert not run.ok and "does not accept nonsense" in run.error


def test_a_wrong_type_is_refused_but_an_out_of_range_value_is_clamped(registry):
    assert not registry.run("detect_hfo", {"detector": 5}).ok
    assert not registry.run("detect_hfo", {"detector": "wavelet"}).ok
    # Clamping, not rejecting: a planner exploring parameters is corrected,
    # and the record shows what actually ran.
    run = registry.run("detect_hfo", {"threshold_sd": 99, "k": 2})
    assert run.ok and run.input["threshold_sd"] == 12.0


def test_a_handler_error_meant_for_the_model_is_passed_through(registry):
    run = registry.run("detect_hfo", {"channels": ["NOSUCH1-NOSUCH2"]})
    assert not run.ok
    assert "unknown channel" in run.error and "channel_qc" in run.error


def test_run_ids_are_sequential_and_name_their_tool(registry):
    assert registry.run("get_recording_metadata").run_id == "meta_001"
    assert registry.run("get_recording_metadata").run_id == "meta_002"
    assert registry.run("channel_qc").run_id == "qc_001"


def test_a_run_record_has_the_shape_the_contract_promises(registry):
    record = registry.run("detect_hfo", {"k": 2}).as_record()
    assert set(record) == {"tool", "input", "output", "run_id", "runtime_s"}


# --------------------------------------------------------------------------
# The evidence store
# --------------------------------------------------------------------------


def test_a_number_resolves_to_the_run_that_produced_it():
    store = EvidenceStore()
    store.append(ToolRun("hfo_001", "detect_hfo", {}, {"A-B": {"rate_per_min": 14.23}}))
    assert store.find_number(14.23) == ["hfo_001"]
    assert store.find_number(14.2) == ["hfo_001"]      # honest rounding
    assert store.find_number(15.5) == []               # not a rounding of anything
    assert store.find_number(1423) == []


def test_a_failed_run_is_still_recorded_but_contributes_no_numbers():
    store = EvidenceStore()
    store.append(ToolRun("hfo_001", "detect_hfo", {}, {"error": "boom"}, ok=False,
                         error="boom"))
    assert len(store) == 1 and store.cost()["n_failed"] == 1
    assert store.all_numbers() == set()


def test_duplicate_run_ids_are_a_programming_error():
    store = EvidenceStore()
    store.append(ToolRun("hfo_001", "detect_hfo", {}, {}))
    with pytest.raises(ValueError, match="already in the store"):
        store.append(ToolRun("hfo_001", "detect_hfo", {}, {}))


def test_the_digest_keeps_the_ranking_the_tool_computed():
    """Sorting the channel keys would hand the planner a different ranking."""
    store = EvidenceStore()
    store.append(ToolRun("hfo_001", "detect_hfo", {},
                         {"channels": {"ZZ1-ZZ2": {"rate_per_min": 90.0},
                                       "AA1-AA2": {"rate_per_min": 1.0}}}))
    assert store.digest().index("ZZ1-ZZ2") < store.digest().index("AA1-AA2")


# --------------------------------------------------------------------------
# Live tools
# --------------------------------------------------------------------------


def test_a_stricter_threshold_finds_fewer_events(registry):
    loose = registry.run("detect_hfo", {"threshold_sd": 4, "k": 5})
    strict = registry.run("detect_hfo", {"threshold_sd": 8, "k": 5})
    assert loose.output["n_accepted"] > strict.output["n_accepted"]


def test_the_same_call_twice_gives_the_same_numbers_and_is_cached(registry):
    first = registry.run("detect_hfo", {"threshold_sd": 5, "k": 3})
    second = registry.run("detect_hfo", {"threshold_sd": 5, "k": 3})
    assert first.output["channels"] == second.output["channels"]
    assert second.output["cached"] is True and first.run_id != second.run_id


def test_a_window_restricts_the_analysis_to_that_time(registry):
    run = registry.run("detect_hfo", {"window_s": [10, 20], "k": 3})
    assert run.ok and run.output["window_s"] == [10.0, 20.0]
    evidence = registry.run("get_event_evidence",
                            {"channel": run.output["channels"] and
                             next(iter(run.output["channels"])), "k": 1})
    assert evidence.ok


def test_an_unusably_high_band_is_refused_with_the_reason(recording):
    """Fast ripples against the Nyquist edge are not measurable, and saying so
    is better than returning numbers nobody should trust."""
    from dataclasses import replace as dc_replace

    slow = dc_replace(recording)
    slow.raw = recording.raw.copy().resample(1000)
    run = build_registry(AnalysisSession(slow)).run("detect_hfo", {"band": "fast_ripple"})
    assert not run.ok and "Nyquist" in run.error


def test_every_tool_returns_json_serialisable_output(registry, session):
    top = next(iter(registry.run("detect_hfo", {"k": 2}).output["channels"]))
    calls = [("get_recording_metadata", {}), ("channel_qc", {"max_channels": 2}),
             ("detect_spikes", {"k": 2}), ("spectral_power", {"channels": [top]}),
             ("rate_change", {"k": 2}), ("compare_detectors", {}),
             ("propagation_lead", {"k": 2}), ("get_event_evidence", {"channel": top})]
    for name, args in calls:
        run = registry.run(name, args)
        assert run.ok, f"{name} failed: {run.error}"
        json.dumps(run.as_record(), default=str)


def test_no_tool_can_change_which_patient_is_analysed(registry):
    for spec in registry.specs.values():
        assert "subject" not in spec.parameters["properties"]
        assert "patient" not in spec.parameters["properties"]


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def _store_with(rates):
    store = EvidenceStore()
    for i, (threshold, channels) in enumerate(rates, start=1):
        store.append(ToolRun(f"hfo_{i:03d}", "detect_hfo", {},
                             {"threshold_sd": threshold,
                              "channels": {c: {"rate_per_min": r} for c, r in channels}}))
    return store


def test_a_channel_that_collapses_under_scrutiny_is_demoted():
    store = _store_with([(5.0, [("A-B", 100.0), ("C-D", 80.0)]),
                         (7.0, [("A-B", 10.0)])])
    ranked = {r.channel: r for r in rank_channels(store)}
    assert ranked["A-B"].robustness == pytest.approx(0.1)
    assert ranked["C-D"].robustness == 1.0
    assert [r.channel for r in rank_channels(store)] == ["C-D", "A-B"]


def test_a_retest_can_never_promote_a_channel():
    """Robustness is capped at 1: scrutiny lowers confidence, never raises it."""
    store = _store_with([(5.0, [("A-B", 10.0)]), (7.0, [("A-B", 999.0)])])
    assert rank_channels(store)[0].robustness == 1.0


def test_an_unchallenged_channel_is_unchallenged_not_verified():
    store = _store_with([(5.0, [("A-B", 10.0)])])
    ranked = rank_channels(store)[0]
    assert ranked.robustness == 1.0 and ranked.retested_at == []


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def test_the_fixed_pipeline_never_challenges_itself(session):
    session.reset_memo()
    result = run_rung(session, Rung.S0)
    assert result.n_retested == 0
    assert result.ranking and result.backend.startswith("none")
    assert "no mechanism for challenging" in result.report


def test_single_shot_is_never_shown_its_results(session):
    """S1 must not be able to re-plan: that is what makes it the S1 rung."""
    session.reset_memo()
    result = run_rung(session, Rung.S1, stop_rule=ModelJudged(max_calls=8))
    assert result.n_retested == 0


def test_replanning_challenges_the_leaders_and_can_reorder_them(session):
    session.reset_memo()
    fixed = run_rung(session, Rung.S0)
    session.reset_memo()
    planned = run_rung(session, Rung.S2, stop_rule=ModelJudged(max_calls=12))
    assert planned.n_retested > 0
    challenged = {r.channel for r in planned.ranking if r.retested_at}
    assert challenged <= set(fixed.top_k(5)), "re-tests should target the leaders"


def test_every_rung_uses_the_same_analyzers(session):
    for rung in Rung:
        session.reset_memo()
        result = run_rung(session, rung, stop_rule=FixedBudget(k=5))
        for step in result.steps:
            if step.get("type") == "tool":
                assert step["tool"] in build_registry(session).names()


def test_the_stopping_rules_stop_for_different_reasons(session):
    session.reset_memo()
    budget = run_rung(session, Rung.S2, stop_rule=FixedBudget(k=3))
    assert budget.cost["n_tool_calls"] == 3 and "budget" in budget.stop_reason
    session.reset_memo()
    tied = run_rung(session, Rung.S2, stop_rule=TiedSetWidth(max_width=2, max_calls=12))
    # On this recording many channels are statistically tied, so the width
    # rule never fires and the planner runs out of things to try. That is a
    # legitimate outcome and must be reported as one, not as a step limit.
    assert "tied" in tied.stop_reason or "nothing further" in tied.stop_reason
    assert "step limit" not in tied.stop_reason


def test_the_tied_set_rule_does_not_claim_a_coverage_guarantee():
    """It is a tied-rank set from Poisson intervals, not a conformal set."""
    assert TiedSetWidth().guarantees_coverage is False


def test_a_malformed_plan_does_not_end_the_session(session):
    class Babbling(Backend):
        name = "babbling"

        def chat(self, messages, tools):
            return AssistantMessage(content="Sure! Let me think about that.")

    session.reset_memo()
    result = run_rung(session, Rung.S2, backend=Babbling(), max_steps=3)
    assert any(s["type"] == "format_error" for s in result.steps)
    assert result.ranking == []


def test_the_audit_trail_survives_a_round_trip(session, tmp_path):
    session.reset_memo()
    result = run_rung(session, Rung.S2, stop_rule=FixedBudget(k=4))
    path = result.store.save(tmp_path / "evidence.json")
    payload = json.loads(path.read_text())
    assert payload["contract_version"] and len(payload["runs"]) == 4
    assert all("run_id" in r for r in payload["runs"])


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


@pytest.fixture
def verified_store():
    store = EvidenceStore()
    store.append(ToolRun("hfo_001", "detect_hfo", {"threshold_sd": 5},
                         {"channels": {"A-B": {"rate_per_min": 14.2}}}))
    store.append(ToolRun("compare_001", "compare_detectors", {},
                         {"agreement": {"jaccard": 0.53}}))
    return store


def test_sentences_split_without_breaking_decimals():
    assert split_sentences("A had 14.2 ripples/min [hfo_001]. B was quiet.") == [
        "A had 14.2 ripples/min [hfo_001].", "B was quiet."]


def test_channel_names_and_small_counts_are_not_claims():
    assert claim_numbers("The top 3 channels were AD1-AD2 and G16-G17.") == []
    assert claim_numbers("AD1-AD2 had 14.2 ripples/min.") == [14.2]


def test_a_fabricated_number_is_struck(verified_store):
    report = ("A-B had 14.2 ripples/min [hfo_001]. A-B had 99.9 ripples/min [hfo_001].")
    result = DeterministicVerifier()(report, verified_store)
    assert result.n_struck == 1 and "99.9" in result.struck[0]["reason"]
    assert "99.9" not in result.kept_text and "14.2" in result.kept_text
    assert result.provenance_coverage == pytest.approx(0.5)


def test_prose_and_caveats_are_never_struck(verified_store):
    report = "Rates come from one short window and physiological ripples occur in healthy tissue."
    result = DeterministicVerifier()(report, verified_store)
    assert result.n_struck == 0 and result.kept_text == report


def test_citing_the_wrong_run_is_caught(verified_store):
    report = "A-B had 14.2 ripples/min [compare_001]."
    result = DeterministicVerifier()(report, verified_store)
    assert result.n_struck == 1 and "did not produce" in result.struck[0]["reason"]


def test_citing_a_run_that_does_not_exist_is_caught(verified_store):
    result = DeterministicVerifier()("A-B had 14.2 ripples/min [hfo_777].", verified_store)
    assert result.n_struck == 1 and "do not exist" in result.struck[0]["reason"]


class _FakeVerifierBackend(Backend):
    """A model verifier that strikes whichever sentence indices it was told to."""

    name = "fake-verifier"

    def __init__(self, strike=(), broken=False):
        self.strike = set(strike)
        self.broken = broken

    def chat(self, messages, tools):
        if self.broken:
            return AssistantMessage(content="I think most of it looks fine, honestly.")
        verdicts = [{"index": i, "supported": i not in self.strike,
                     "run_ids": [], "reason": "cannot resolve"} for i in range(6)]
        return AssistantMessage(content=json.dumps({"verdicts": verdicts}))


def test_the_model_verifier_strikes_what_it_judges_unsupported(verified_store):
    report = "A-B had 14.2 ripples/min [hfo_001]. A-B had 99.9 ripples/min [hfo_001]."
    result = LLMVerifier(_FakeVerifierBackend(strike={1}))(report, verified_store)
    assert result.n_struck == 1 and "99.9" not in result.kept_text


def test_a_broken_verifier_response_keeps_the_report(verified_store):
    """Report quality must not be a function of decoding luck."""
    report = "A-B had 14.2 ripples/min [hfo_001]."
    result = LLMVerifier(_FakeVerifierBackend(broken=True))(report, verified_store)
    assert result.n_struck == 0 and result.kept_text == report


def test_comparing_the_verifiers_measures_what_verification_costs(verified_store):
    # Sentence 0 is true and supported; sentence 1 is fabricated. The model
    # verifier strikes the true one and misses the fabricated one -- both
    # failure modes at once, which is what the comparison is for.
    report = "A-B had 14.2 ripples/min [hfo_001]. A-B had 99.9 ripples/min [hfo_001]."
    delta = compare_verifiers(report, verified_store, _FakeVerifierBackend(strike={0}))
    assert delta["struck_by_llm_only"] == [0]
    assert delta["struck_by_arithmetic_only"] == [1]
    assert delta["coverage_cost"] == 1 and delta["hallucinations_missed_by_llm"] == 1


def test_s3_strikes_an_unsupported_claim_from_the_report(session):
    class Fabricating(ScriptedPlanner):
        name = "fabricating-planner"

        def _report(self, prompt):
            return {"report": "A-B had 12345.6 ripples/min [hfo_001].",
                    "run_ids": ["hfo_001"]}

    session.reset_memo()
    result = run_rung(session, Rung.S3, backend=Fabricating(),
                      stop_rule=FixedBudget(k=4))
    assert result.struck and "12345.6" not in result.report
    assert result.verification["unsupported_claim_rate"] == 1.0


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_a_perfect_ranking_beats_chance():
    labels = SozLabels("x", {"A1", "A2", "B1"}, source="clinical_summary", seizure_free=True)
    channels = ["A1-A2", "B1-B2"] + [f"Z{i}-Z{i + 1}" for i in range(1, 20)]
    score = score_ranking(channels, labels, ks=(2,), n_permutations=500)
    assert score.at_k[2]["precision_at_k"] == 1.0
    assert score.at_k[2]["permutation_p"] < 0.05


def test_an_unlabelled_subject_scores_nothing_rather_than_zero():
    score = score_ranking(["A1-A2"], SozLabels("x"), ks=(1,))
    assert score.at_k == {} and "nothing can be scored" in score.note


def test_the_label_source_is_carried_into_every_score():
    """A number computed against weak labels must never look like a curated one."""
    weak = SozLabels("x", {"A1"}, source="events_markers")
    score = score_ranking(["A1-A2", "B1-B2"], weak, ks=(1,), n_permutations=200)
    assert score.label_source == "events_markers" and score.trustworthy is False


# --------------------------------------------------------------------------
# Stand-in labels, for before the real ones exist
# --------------------------------------------------------------------------


def test_synthetic_labels_recover_the_contacts_events_were_implanted_on(recording):
    """Not an estimate: the simulator knows exactly where it put the ripples."""
    labels = labels_from_ground_truth(recording)
    assert labels.soz_contacts == {c.upper() for c in recording.marked_contacts}
    assert labels.source == "synthetic_truth"


def test_the_ground_truth_rule_is_scale_free_not_an_absolute_floor():
    """An absolute floor sweeps in every background contact that got two events,
    pushing label prevalence to a third of all channels."""
    import pandas as pd

    class _Rec:
        subject = "sim"
        ground_truth = pd.DataFrame({
            "kind": ["ripple"] * 6,
            "contact": ["HOT1"] * 3 + ["HOT2"] * 2 + ["COLD1"]})

    labels = labels_from_ground_truth(_Rec(), min_events=1, min_fraction=0.5)
    assert labels.soz_contacts == {"HOT1", "HOT2"}


def test_artifacts_are_never_a_label(recording):
    """The simulator implants them so the detector can be caught reporting them."""
    ripples = labels_from_ground_truth(recording, kind="ripple").soz_contacts
    artifacts = labels_from_ground_truth(recording, kind="artifact").soz_contacts
    assert ripples and ripples != artifacts


def test_a_real_recording_gets_no_invented_labels():
    """soz_labels never makes anything up; asking for a stand-in is explicit."""
    class _Rec:
        subject = "sub-nobody-999"
        marked_contacts: list = []
        ground_truth = None

    labels = soz_labels("sub-nobody-999", recording=_Rec())
    assert not labels.usable and labels.source == "none"


def test_placeholder_labels_are_reproducible_and_drawn_from_real_channels():
    channels = ["AD1-AD2", "AD2-AD3", "ATT1-ATT2", "G16-G17"]
    first = placeholder_labels("anon-01", channels, n_contacts=3, seed=0)
    again = placeholder_labels("anon-01", channels, n_contacts=3, seed=0)
    assert first.soz_contacts == again.soz_contacts and len(first.soz_contacts) == 3
    assert first.soz_contacts <= {"AD1", "AD2", "AD3", "ATT1", "ATT2", "G16", "G17"}


def test_placeholder_labels_ask_for_more_contacts_than_exist():
    labels = placeholder_labels("anon-01", ["A1-A2"], n_contacts=50)
    assert labels.soz_contacts == {"A1", "A2"}


@pytest.mark.parametrize("source", ["placeholder", "synthetic_truth"])
def test_stand_ins_are_never_trustworthy_and_always_carry_a_warning(source):
    labels = SozLabels("x", {"A1"}, source=source, seizure_free=True)
    assert labels.is_placeholder and not labels.trustworthy and labels.warning


def test_the_warning_travels_into_the_serialised_score():
    """A stand-in reaching a results table unmarked is undetectable later."""
    labels = placeholder_labels("anon-01", ["A1-A2", "B1-B2", "C1-C2"], n_contacts=1)
    payload = score_ranking(["A1-A2", "B1-B2"], labels, ks=(1,),
                            n_permutations=200).as_dict()
    assert payload["is_placeholder"] is True
    assert "PLACEHOLDER" in payload["warning"]
    assert json.dumps(payload)      # survives being written to a results file


def test_the_label_template_round_trips_into_real_labels(tmp_path):
    """Filling the template in and passing it back is the whole swap."""
    channels = ["AD1-AD2", "ATT1-ATT2"]
    stand_in = placeholder_labels("anon-01", channels, n_contacts=2, seed=0)
    path = write_label_template("anon-01", channels, tmp_path / "soz.csv", labels=stand_in)

    text = path.read_text()
    assert "AD1, AD2, ATT1, ATT2" in text      # the exact spelling to use
    assert "PLACEHOLDER" in text               # and that these are not real

    # A centre overwrites soz_contacts and sends it back; the comments stay.
    path.write_text(text.replace(
        f"anon-01,{'; '.join(sorted(stand_in.soz_contacts))}",
        'anon-01,"AD1-2; ATT1"'))
    real = load_labels_csv(path)["anon-01"]
    assert real.soz_contacts == {"AD1", "AD2", "ATT1"}
    assert real.source == "local" and not real.is_placeholder


def test_scoring_synthetic_ground_truth_beats_chance(session):
    """The one case where a non-null score proves the scorer, not the brain."""
    session.reset_memo()
    result = run_rung(session, Rung.S0)
    labels = labels_from_ground_truth(session.recording)
    score = score_ranking(result.top_channels, labels, ks=(5,), n_permutations=1000)
    assert score.at_k[5]["n_hits"] > score.at_k[5]["expected_by_chance"]
    assert score.at_k[5]["permutation_p"] < 0.1
